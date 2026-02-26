"""Shared forward-pass primitives — single point of change for MLX-specific APIs.

Consolidates boilerplate that was duplicated across probe, cast, evaluate,
surface, depth, sic, and softprompt modules: KV cache creation, embed + mask
setup, LM head application, and logit extraction.
"""

import mlx.core as mx
import mlx.nn as nn

from vauban._array import Array
from vauban.types import CausalLM, LayerCache, TransformerModel


def get_transformer(model: CausalLM) -> TransformerModel:
    """Access the inner transformer (model.model)."""
    return model.model


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


# ---------------------------------------------------------------------------
# Lazy evaluation wrapper
# ---------------------------------------------------------------------------


def force_eval(*args: Array) -> None:
    """Force evaluation of lazy arrays. No-op on eager backends."""
    mx.eval(*args)


# ---------------------------------------------------------------------------
# Stable numerics — CPU-stream wrappers for SVD/QR
# ---------------------------------------------------------------------------


def svd_stable(matrix: Array) -> tuple[Array, Array, Array]:
    """SVD with numerical stability (CPU stream on MLX)."""
    return mx.linalg.svd(matrix, stream=mx.cpu)  # type: ignore[arg-type]


def qr_stable(matrix: Array) -> tuple[Array, Array]:
    """QR with numerical stability (CPU stream on MLX)."""
    return mx.linalg.qr(matrix, stream=mx.cpu)  # type: ignore[arg-type]
