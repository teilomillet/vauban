"""Shared forward-pass primitives — single point of change for MLX-specific APIs.

Consolidates boilerplate that was duplicated across probe, cast, evaluate,
surface, depth, sic, and softprompt modules: KV cache creation, embed + mask
setup, LM head application, and logit extraction.
"""

from typing import TYPE_CHECKING

from vauban._array import Array
from vauban.types import CausalLM, LayerCache, TransformerModel

if TYPE_CHECKING:
    from vauban._backend import get_backend as _get_backend

    _BACKEND = _get_backend()
else:
    from vauban._backend import get_backend

    _BACKEND = get_backend()


def get_transformer(model: CausalLM) -> TransformerModel:
    """Access the inner transformer (model.model)."""
    return model.model


if TYPE_CHECKING or _BACKEND == "mlx":
    import mlx.core as mx
    import mlx.nn as nn

    def make_cache(model: CausalLM) -> list[LayerCache]:
        """Create a KV cache for the model.

        Uses model.make_cache() if available (real mlx-lm and mock),
        otherwise falls back to importing from mlx_lm.
        """
        if hasattr(model, "make_cache"):
            return model.make_cache()  # type: ignore[no-any-return]
        from mlx_lm.models.cache import make_prompt_cache

        return make_prompt_cache(model)  # type: ignore[no-any-return]

    def lm_head_forward(model: CausalLM, h: Array) -> Array:
        """Apply the language model head (lm_head or tied embeddings)."""
        if hasattr(model, "lm_head"):
            lm_head: nn.Module = model.lm_head  # type: ignore[attr-defined]
            return lm_head(h)
        return model.model.embed_tokens.as_linear(h)

    def extract_logits(result: Array | tuple[Array, ...]) -> Array:
        """Extract logits tensor from model output (handles tuple or bare array)."""
        if isinstance(result, tuple):
            return result[0]
        return result

    def embed_and_mask(
        transformer: TransformerModel,
        token_ids: Array,
    ) -> tuple[Array, Array]:
        """Embed tokens and create causal mask.

        Returns:
            Tuple of (hidden_states, causal_mask).
        """
        h = transformer.embed_tokens(token_ids)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
        mask = mask.astype(h.dtype)
        return h, mask

    def embed_and_mask_with_prefix(
        transformer: TransformerModel,
        prefix_embeds: Array,
        token_ids: Array,
    ) -> tuple[Array, Array]:
        """Embed tokens, prepend prefix embeddings, and create causal mask.

        Returns:
            Tuple of (hidden_states, causal_mask) where hidden_states is
            [prefix_embeds | embed(token_ids)].
        """
        prompt_embeds = transformer.embed_tokens(token_ids)
        h = mx.concatenate([prefix_embeds, prompt_embeds], axis=1)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
        mask = mask.astype(h.dtype)
        return h, mask

    # -------------------------------------------------------------------
    # Lazy evaluation wrapper
    # -------------------------------------------------------------------

    def force_eval(*args: Array) -> None:
        """Force evaluation of lazy arrays. No-op on eager backends."""
        mx.eval(*args)

    # -------------------------------------------------------------------
    # Stable numerics — CPU-stream wrappers for SVD/QR
    # -------------------------------------------------------------------

    def svd_stable(matrix: Array) -> tuple[Array, Array, Array]:
        """SVD with numerical stability (CPU stream on MLX)."""
        return mx.linalg.svd(matrix, stream=mx.cpu)  # type: ignore[arg-type]

    def qr_stable(matrix: Array) -> tuple[Array, Array]:
        """QR with numerical stability (CPU stream on MLX)."""
        return mx.linalg.qr(matrix, stream=mx.cpu)  # type: ignore[arg-type]


elif _BACKEND == "torch":
    from collections.abc import Callable

    def force_eval(*args: Array) -> None:
        """No-op — PyTorch is eager, no lazy evaluation needed."""

    def _torch_not_implemented(name: str) -> Callable[..., object]:
        """Create a stub that raises NotImplementedError for a given function name."""
        def _stub(*args: object, **kwargs: object) -> object:
            raise NotImplementedError(
                f"{name}() not yet implemented for PyTorch backend"
            )
        _stub.__name__ = name
        _stub.__doc__ = f"PyTorch stub for {name} (not yet implemented)."
        return _stub

    make_cache = _torch_not_implemented("make_cache")
    lm_head_forward = _torch_not_implemented("lm_head_forward")
    extract_logits = _torch_not_implemented("extract_logits")
    embed_and_mask = _torch_not_implemented("embed_and_mask")
    embed_and_mask_with_prefix = _torch_not_implemented("embed_and_mask_with_prefix")
    svd_stable = _torch_not_implemented("svd_stable")
    qr_stable = _torch_not_implemented("qr_stable")

else:
    msg = f"Unknown backend: {_BACKEND!r}"
    raise ValueError(msg)
