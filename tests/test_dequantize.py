# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for vauban.dequantize: quantization detection and dequantization."""

import builtins
import importlib
from types import ModuleType
from typing import TYPE_CHECKING, cast

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

import vauban.dequantize as dequantize_module  # noqa: E402
from tests.conftest import MockCausalLM  # noqa: E402
from vauban._backend import get_backend  # noqa: E402
from vauban.dequantize import dequantize_model, is_quantized  # noqa: E402

if TYPE_CHECKING:
    from vauban.types import CausalLM

pytestmark = pytest.mark.skipif(
    get_backend() != "mlx",
    reason="MLX-only: tests mlx.nn.QuantizedLinear",
)

# QuantizedLinear needs input_dims divisible by group_size (min 32).
# Create a slightly larger model for quantization tests.
_Q_D_MODEL = 64
_Q_NUM_LAYERS = 2
_Q_VOCAB_SIZE = 32
_Q_NUM_HEADS = 2


def _make_quantized_model() -> MockCausalLM:
    """Create a model with dimensions large enough for quantization."""
    model = MockCausalLM(_Q_D_MODEL, _Q_NUM_LAYERS, _Q_VOCAB_SIZE, _Q_NUM_HEADS)
    mx.eval(model.parameters())
    return model


def _make_quantized_linear(in_dim: int, out_dim: int) -> nn.QuantizedLinear:
    """Create a QuantizedLinear with default group_size=32."""
    return nn.QuantizedLinear(
        in_dim, out_dim, bias=False, group_size=32, bits=4,
    )


class TestIsQuantized:
    def test_normal_model_not_quantized(
        self, mock_model: MockCausalLM,
    ) -> None:
        assert is_quantized(mock_model) is False

    def test_model_with_quantized_linear(self) -> None:
        """A model containing nn.QuantizedLinear is detected as quantized."""
        model = _make_quantized_model()

        quantized = _make_quantized_linear(_Q_D_MODEL, _Q_D_MODEL)
        mx.eval(quantized.parameters())
        model.model.layers[0].self_attn.o_proj = quantized

        assert is_quantized(model) is True


class TestDequantizeModel:
    def test_returns_false_when_nothing_quantized(
        self, mock_model: MockCausalLM,
    ) -> None:
        assert dequantize_model(mock_model) is False

    def test_non_module_root_returns_false(self) -> None:
        """Non-module roots should be ignored by the recursive walker."""
        assert dequantize_model(cast("CausalLM", object())) is False

    def test_replaces_quantized_linear(self) -> None:
        """QuantizedLinear layers are replaced with Linear, weights become float."""
        model = _make_quantized_model()

        quantized = _make_quantized_linear(_Q_D_MODEL, _Q_D_MODEL)
        mx.eval(quantized.parameters())
        model.model.layers[0].self_attn.o_proj = quantized

        assert is_quantized(model) is True

        changed = dequantize_model(model)
        assert changed is True

        # The replaced layer should be nn.Linear now
        new_layer = model.model.layers[0].self_attn.o_proj
        assert isinstance(new_layer, nn.Linear)
        assert new_layer.weight.dtype != mx.uint32

        # Model should no longer be quantized
        assert is_quantized(model) is False

    def test_import_error_falls_back_to_quantized_linear_only(self) -> None:
        """Missing switch-layer support should not block linear dequantization."""
        model = _make_quantized_model()
        quantized = _make_quantized_linear(_Q_D_MODEL, _Q_D_MODEL)
        mx.eval(quantized.parameters())
        model.model.layers[0].self_attn.o_proj = quantized

        original_import = builtins.__import__

        def _fake_import(
            name: str,
            globals_dict: dict[str, object] | None = None,
            locals_dict: dict[str, object] | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> object:
            del globals_dict, locals_dict, fromlist, level
            if name == "mlx_lm.models.switch_layers":
                raise ImportError("switch layers unavailable")
            return original_import(name)

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(builtins, "__import__", _fake_import)
            changed = dequantize_model(model)

        assert changed is True
        assert isinstance(model.model.layers[0].self_attn.o_proj, nn.Linear)

    def test_replaces_quantized_switch_module(self) -> None:
        """Quantized switch modules should be routed through the switch helper."""

        class FakeQuantizedSwitchLinear(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = mx.zeros((2, 2, 2), dtype=mx.uint32)
                self.scales = mx.ones((1,), dtype=mx.float32)
                self.biases = mx.zeros((1,), dtype=mx.float32)
                self.bias = mx.zeros((2, 2), dtype=mx.float32)
                self.group_size = 32
                self.bits = 4
                self.input_dims = 2
                self.output_dims = 2
                self.num_experts = 2

        class FakeSwitchLinear(nn.Module):
            def __init__(
                self,
                input_dims: int,
                output_dims: int,
                num_experts: int,
                *,
                bias: bool = False,
            ) -> None:
                super().__init__()
                del input_dims, output_dims, num_experts, bias

        model = _make_quantized_model()
        model.model.layers[0].mlp = FakeQuantizedSwitchLinear()
        replacement = nn.Linear(_Q_D_MODEL, _Q_D_MODEL, bias=False)
        mx.eval(replacement.parameters())
        fake_switch_module = ModuleType("mlx_lm.models.switch_layers")
        fake_switch_module.QuantizedSwitchLinear = FakeQuantizedSwitchLinear  # type: ignore[attr-defined]
        fake_switch_module.SwitchLinear = FakeSwitchLinear  # type: ignore[attr-defined]
        original_import = builtins.__import__

        def _fake_import(
            name: str,
            globals_dict: dict[str, object] | None = None,
            locals_dict: dict[str, object] | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> object:
            del globals_dict, locals_dict, fromlist, level
            if name == "mlx_lm.models.switch_layers":
                return fake_switch_module
            return original_import(name)

        with (
            pytest.MonkeyPatch.context() as monkeypatch,
            pytest.MonkeyPatch.context() as patch_import,
        ):
            monkeypatch.setattr(
                dequantize_module,
                "_dequantize_switch",
                lambda child, switch_cls: replacement,
            )
            patch_import.setattr(builtins, "__import__", _fake_import)
            changed = dequantize_model(model)

        assert changed is True
        assert model.model.layers[0].mlp is replacement


class TestDequantizeHelpers:
    def test_dequantize_linear_helper_preserves_bias(self) -> None:
        """The linear helper should keep a quantized layer's bias."""

        class FakeQuantizedLinear:
            def __init__(self) -> None:
                self.weight = mx.zeros((3, 4), dtype=mx.uint32)
                self.scales = mx.ones((1,), dtype=mx.float32)
                self.biases = mx.zeros((1,), dtype=mx.float32)
                self.group_size = 32
                self.bits = 4
                self.bias = mx.arange(3, dtype=mx.float32)

        child = FakeQuantizedLinear()
        dequantized = mx.ones((3, 4), dtype=mx.float32)

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                dequantize_module._mx,
                "dequantize",
                lambda *args: dequantized,
            )
            monkeypatch.setattr(dequantize_module, "force_eval", lambda params: None)
            linear = dequantize_module._dequantize_linear(
                cast("nn.QuantizedLinear", child),
            )

        assert linear.weight is dequantized
        assert linear.bias is child.bias

    def test_dequantize_switch_helper_sets_weight_and_bias(self) -> None:
        """The switch helper should preserve the dequantized weights and bias."""

        class FakeQuantizedSwitch:
            def __init__(self) -> None:
                self.weight = mx.zeros((2, 3, 4), dtype=mx.uint32)
                self.scales = mx.ones((1,), dtype=mx.float32)
                self.biases = mx.zeros((1,), dtype=mx.float32)
                self.group_size = 32
                self.bits = 4
                self.input_dims = 4
                self.output_dims = 3
                self.num_experts = 2
                self.bias = mx.ones((2, 3), dtype=mx.float32)

            def __contains__(self, key: str) -> bool:
                return key == "bias"

        class FakeSwitch:
            def __init__(
                self,
                input_dims: int,
                output_dims: int,
                num_experts: int,
                *,
                bias: bool = False,
            ) -> None:
                self.input_dims = input_dims
                self.output_dims = output_dims
                self.num_experts = num_experts
                self.bias = None if not bias else mx.zeros((2, 3), dtype=mx.float32)
                self.weight = mx.zeros((2, 3, 4), dtype=mx.float32)

            def parameters(self) -> list[mx.array]:
                params = [self.weight]
                if self.bias is not None:
                    params.append(self.bias)
                return params

        child = FakeQuantizedSwitch()
        dequantized = mx.ones((2, 3, 4), dtype=mx.float32)

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                dequantize_module._mx,
                "dequantize",
                lambda *args: dequantized,
            )
            monkeypatch.setattr(dequantize_module, "force_eval", lambda params: None)
            switch = dequantize_module._dequantize_switch(
                cast("nn.Module", child),
                cast("type[nn.Module]", FakeSwitch),
            )

        fake_switch = cast("FakeSwitch", switch)
        assert fake_switch.input_dims == 4
        assert fake_switch.output_dims == 3
        assert fake_switch.num_experts == 2
        assert fake_switch.weight is dequantized
        assert fake_switch.bias is child.bias


class TestReloadedBackendBranches:
    def test_torch_backend_branches_return_false(
        self,
    ) -> None:
        """Reloading under torch should bind the no-op backend branch."""
        backend_module = importlib.import_module("vauban._backend")
        module = dequantize_module
        original_backend = backend_module.get_backend()

        try:
            with pytest.MonkeyPatch.context() as monkeypatch:
                monkeypatch.setattr(backend_module, "get_backend", lambda: "torch")
                reloaded = importlib.reload(module)
                model = cast("CausalLM", object())
                assert reloaded.is_quantized(model) is False
                assert reloaded.dequantize_model(model) is False
        finally:
            with pytest.MonkeyPatch.context() as monkeypatch:
                monkeypatch.setattr(
                    backend_module,
                    "get_backend",
                    lambda: original_backend,
                )
                importlib.reload(module)

    def test_unknown_backend_raises_on_reload(self) -> None:
        """Unsupported backends should fail fast at import time."""
        backend_module = importlib.import_module("vauban._backend")
        module = dequantize_module
        original_backend = backend_module.get_backend()

        try:
            with pytest.MonkeyPatch.context() as monkeypatch:
                monkeypatch.setattr(backend_module, "get_backend", lambda: "bogus")
                with pytest.raises(ValueError, match="Unknown backend"):
                    importlib.reload(module)
        finally:
            with pytest.MonkeyPatch.context() as monkeypatch:
                monkeypatch.setattr(
                    backend_module,
                    "get_backend",
                    lambda: original_backend,
                )
                importlib.reload(module)
