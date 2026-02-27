"""Dequantize quantized model layers for weight surgery.

Most mlx-community models ship as 4-bit or 8-bit quantized weights
(stored as uint32). The ``cut()`` operation does rank-1 projection
removal on float matrices — applying it to quantized weights produces
garbage silently. This module detects and replaces quantized layers
with their float equivalents before any weight modification.

On the PyTorch backend, models are always loaded in float — both
``is_quantized()`` and ``dequantize_model()`` are no-ops.
"""

from typing import TYPE_CHECKING

from vauban.types import CausalLM

if TYPE_CHECKING:
    from vauban._backend import get_backend as _get_backend

    _BACKEND = _get_backend()
else:
    from vauban._backend import get_backend

    _BACKEND = get_backend()


if TYPE_CHECKING or _BACKEND == "mlx":
    import mlx.core as _mx  # _mx.dequantize (MLX-specific, no ops equivalent)
    import mlx.nn as nn  # nn.QuantizedLinear isinstance checks

    from vauban import _ops as ops
    from vauban._forward import force_eval

    def is_quantized(model: CausalLM) -> bool:
        """Check if a model contains any quantized parameters.

        Scans all parameters for ``ops.uint32`` dtype, which indicates
        quantized weights (4-bit or 8-bit packed into uint32).

        Args:
            model: A causal language model.

        Returns:
            True if any parameter has dtype ``ops.uint32``.
        """
        flat = dict(ops.tree_flatten(model.parameters()))  # type: ignore[attr-defined]
        return any(v.dtype == ops.uint32 for v in flat.values())

    def dequantize_model(model: CausalLM) -> bool:
        """Replace all quantized layers with dequantized float equivalents.

        Handles both ``nn.QuantizedLinear`` (standard dense layers) and
        ``QuantizedSwitchLinear`` (MoE batched experts from mlx-lm).
        Operates in-place on the module tree.

        Args:
            model: A causal language model to dequantize.

        Returns:
            True if any layer was replaced, False if model was already float.
        """
        try:
            from mlx_lm.models.switch_layers import (
                QuantizedSwitchLinear,
                SwitchLinear,
            )
        except ImportError:
            QuantizedSwitchLinear = None  # type: ignore[assignment, misc]  # noqa: N806
            SwitchLinear = None  # type: ignore[assignment, misc]  # noqa: N806

        changed = False

        def _recurse(mod: object) -> None:
            nonlocal changed
            if not isinstance(mod, nn.Module):
                return
            for key in list(mod.keys()):
                child = mod[key]
                if isinstance(child, nn.QuantizedLinear):
                    mod[key] = _dequantize_linear(child)
                    changed = True
                elif (
                    QuantizedSwitchLinear is not None
                    and SwitchLinear is not None
                    and isinstance(child, QuantizedSwitchLinear)
                ):
                    mod[key] = _dequantize_switch(child, SwitchLinear)
                    changed = True
                elif isinstance(child, nn.Module):
                    _recurse(child)
                elif isinstance(child, list):
                    for item in child:
                        if isinstance(item, nn.Module):
                            _recurse(item)

        _recurse(model)
        return changed

    def _dequantize_linear(child: nn.QuantizedLinear) -> nn.Linear:
        """Convert a single QuantizedLinear to Linear."""
        w = _mx.dequantize(
            child.weight, child.scales, child.biases,
            child.group_size, child.bits,
        )
        has_bias = hasattr(child, "bias") and child.bias is not None
        linear = nn.Linear(w.shape[1], w.shape[0], bias=has_bias)
        linear.weight = w
        if has_bias:
            linear.bias = child.bias
        force_eval(linear.parameters())
        return linear

    def _dequantize_switch(
        child: nn.Module,
        switch_cls: type[nn.Module],
    ) -> nn.Module:
        """Convert a single QuantizedSwitchLinear to SwitchLinear."""
        w = _mx.dequantize(
            child.weight,
            child.scales,
            child.biases,
            child.group_size,
            child.bits,
        )
        has_bias = "bias" in child
        # switch_cls is SwitchLinear at runtime; typed as nn.Module for fallback.
        # Construct via positional args matching SwitchLinear(input, output, experts).
        ctor: object = switch_cls
        switch: nn.Module = ctor(
            child.input_dims, child.output_dims, child.num_experts, bias=has_bias,  # type: ignore[too-many-positional-arguments, unknown-argument]
        )
        switch.weight = w
        if has_bias:
            switch.bias = child.bias
        force_eval(switch.parameters())
        return switch

elif _BACKEND == "torch":
    def is_quantized(model: CausalLM) -> bool:
        """Check if a model contains quantized parameters.

        PyTorch models loaded via HF Transformers are always float —
        returns False unconditionally.
        """
        return False

    def dequantize_model(model: CausalLM) -> bool:
        """No-op on PyTorch backend — models are already float.

        Returns:
            False (no layers replaced).
        """
        return False

else:
    msg = f"Unknown backend: {_BACKEND!r}"
    raise ValueError(msg)
