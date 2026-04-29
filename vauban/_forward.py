# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Shared forward-pass primitives — single point of change for MLX-specific APIs.

Consolidates boilerplate that was duplicated across probe, cast, evaluate,
surface, depth, sic, and softprompt modules: KV cache creation, embed + mask
setup, LM head application, and logit extraction.
"""

from typing import TYPE_CHECKING, Protocol, cast

from vauban._array import Array
from vauban.types import CausalLM, LayerCache, Tokenizer, TransformerModel

if TYPE_CHECKING:
    from vauban._backend import get_backend as _get_backend

    _BACKEND = _get_backend()
else:
    from vauban._backend import get_backend

    _BACKEND = get_backend()


class LayerModule(Protocol):
    """Callable transformer layer interface used by the forward helpers."""

    def __call__(
        self,
        h: Array,
        mask: Array | None,
        *,
        cache: LayerCache | None = None,
    ) -> Array: ...


class HeadModule(Protocol):
    """Callable language-model head interface used by forward helpers."""

    def __call__(self, h: Array) -> Array: ...


class HasLMHead(Protocol):
    """Model shape for architectures with an explicit language-model head."""

    lm_head: HeadModule


def get_transformer(model: CausalLM) -> TransformerModel:
    """Access the inner transformer, auto-detecting architecture."""
    from vauban._arch import get_inner_model, normalize_transformer

    inner = get_inner_model(model)
    return normalize_transformer(inner)  # type: ignore[return-value]


def encode_chat_prompt(
    tokenizer: Tokenizer,
    messages: list[dict[str, str]],
) -> Array:
    """Apply the chat template and encode messages into batched token IDs.

    Returns a ``(1, seq_len)`` int array suitable for model input.
    """
    from vauban import _ops as ops

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(messages, tokenize=False)
    if not isinstance(text, str):
        msg = "apply_chat_template must return str when tokenize=False"
        raise TypeError(msg)
    return ops.array(tokenizer.encode(text))[None, :]


def encode_user_prompt(tokenizer: Tokenizer, prompt: str) -> Array:
    """Encode a single user message into batched token IDs.

    Convenience wrapper around :func:`encode_chat_prompt` for the most
    common case: a single ``{"role": "user"}`` message.
    """
    return encode_chat_prompt(tokenizer, [{"role": "user", "content": prompt}])


def make_ssm_mask(transformer: TransformerModel, h: Array) -> Array | None:
    """Create a boolean SSM mask for hybrid architectures.

    Returns ``None`` when all layers use standard attention.
    For architectures like Qwen3.5 that mix GatedDeltaNet (SSM) and
    standard attention layers, returns a ``(B, T)`` boolean mask where
    all positions are valid (no cache).
    """
    from vauban import _ops as ops

    if any(getattr(layer, "is_linear", False) for layer in transformer.layers):
        return ops.ones((h.shape[0], h.shape[1]), dtype=ops.bool_)
    return None


def select_mask(
    layer: object, attn_mask: Array | None, ssm_mask: Array | None,
) -> Array | None:
    """Pick the correct mask for a layer (attention vs SSM/linear)."""
    if ssm_mask is not None and getattr(layer, "is_linear", False):
        return ssm_mask
    return attn_mask


def run_transformer_layers(
    transformer: TransformerModel,
    h: Array,
    mask: Array,
    cache: list[LayerCache] | None = None,
    *,
    differentiable: bool = False,
) -> Array:
    """Run all transformer layers with hybrid mask support.

    Handles architectures like Qwen3.5 that mix standard attention and
    SSM/linear layers, each requiring a different mask format.

    Args:
        transformer: The transformer model with ``layers`` attribute.
        h: Hidden states, shape ``(B, T, D)``.
        mask: Causal attention mask for standard layers.
        cache: Optional KV cache list (one per layer).
        differentiable: If True, use straight-through estimator for SSM
            layers to allow gradient flow. Required for GCG/EGD.

    Returns:
        Hidden states after all layers.
    """
    if differentiable and cache is not None:
        msg = (
            "Cannot use both differentiable=True and cache: "
            "STE layers do not support KV cache"
        )
        raise ValueError(msg)

    ssm_mask = make_ssm_mask(transformer, h)
    for i, layer in enumerate(transformer.layers):
        typed_layer = cast("LayerModule", layer)
        if differentiable:
            h = ste_layer_forward(typed_layer, h, mask, ssm_mask)
        elif cache is not None:
            h = typed_layer(
                h, select_mask(layer, mask, ssm_mask), cache=cache[i],
            )
        else:
            h = typed_layer(h, select_mask(layer, mask, ssm_mask))
    return h


def ste_layer_forward(
    layer: LayerModule,
    h: Array,
    mask: Array,
    ssm_mask: Array | None,
) -> Array:
    """Forward through a layer with straight-through estimator for SSM layers.

    SSM/linear layers (e.g. GatedDeltaNet in Qwen3.5) use custom kernels
    that don't support backward differentiation (VJP). This function uses
    a straight-through estimator: the forward pass computes the real SSM
    output, but the backward pass treats the layer as identity, allowing
    gradients to flow through for GCG/EGD optimization.

    Standard attention layers are differentiated normally.
    """
    from vauban import _ops as ops

    if ssm_mask is not None and getattr(layer, "is_linear", False):
        h_ssm = layer(ops.stop_gradient(h), ssm_mask)
        return h + ops.stop_gradient(h_ssm - h)
    return layer(h, select_mask(layer, mask, ssm_mask))


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
            return cast("HasLMHead", model).lm_head(h)
        return get_transformer(model).embed_tokens.as_linear(h)

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
        token_position: str = "prefix",
        infix_split: int | None = None,
    ) -> tuple[Array, Array]:
        """Embed tokens, combine with soft embeddings, and create causal mask.

        Supports prefix (default), suffix, and infix placement of soft
        embeddings relative to prompt token embeddings.

        Returns:
            Tuple of (hidden_states, causal_mask).
        """
        prompt_embeds = transformer.embed_tokens(token_ids)
        if token_position == "suffix":
            h = mx.concatenate([prompt_embeds, prefix_embeds], axis=1)
        elif token_position == "infix" and infix_split is not None:
            part1 = prompt_embeds[:, :infix_split, :]
            part2 = prompt_embeds[:, infix_split:, :]
            h = mx.concatenate([part1, prefix_embeds, part2], axis=1)
        else:
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
        """SVD with numerical stability on CPU, restoring vector dtypes."""
        m = matrix.astype(mx.float32) if matrix.dtype != mx.float32 else matrix
        u, s, vt = mx.linalg.svd(m, stream=mx.cpu)  # type: ignore[arg-type]
        if matrix.dtype == mx.float32:
            return u, s, vt
        return u.astype(matrix.dtype), s, vt.astype(matrix.dtype)

    def qr_stable(matrix: Array) -> tuple[Array, Array]:
        """QR with numerical stability (CPU stream on MLX)."""
        return mx.linalg.qr(matrix, stream=mx.cpu)  # type: ignore[arg-type]


elif _BACKEND == "torch":
    import torch as _torch

    def force_eval(*args: Array) -> None:
        """No-op — PyTorch is eager, no lazy evaluation needed."""

    def make_cache(model: CausalLM) -> list[LayerCache]:
        """Create a KV cache via the wrapper's make_cache()."""
        return model.make_cache()

    def lm_head_forward(model: CausalLM, h: Array) -> Array:
        """Apply the language model head (lm_head or tied embeddings)."""
        if hasattr(model, "lm_head"):
            return model.lm_head(h)
        # Tied embeddings: project through embedding weight matrix
        import torch.nn.functional as _f

        return _f.linear(h, get_transformer(model).embed_tokens.weight)

    def extract_logits(result: Array | tuple[Array, ...]) -> Array:
        """Extract logits tensor from model output."""
        if isinstance(result, tuple):
            return result[0]
        if hasattr(result, "logits"):
            return result.logits
        return result

    def embed_and_mask(
        transformer: TransformerModel,
        token_ids: Array,
    ) -> tuple[Array, Array | None]:
        """Embed tokens. HF layers handle causal masking internally.

        Returns:
            Tuple of (hidden_states, None).
        """
        # Move token_ids to the same device as the embedding weights
        dev = transformer.embed_tokens.weight.device
        token_ids = token_ids.to(dev)
        h = transformer.embed_tokens(token_ids)
        return h, None

    def embed_and_mask_with_prefix(
        transformer: TransformerModel,
        prefix_embeds: Array,
        token_ids: Array,
        token_position: str = "prefix",
        infix_split: int | None = None,
    ) -> tuple[Array, Array | None]:
        """Embed tokens, combine with soft embeddings.

        Supports prefix (default), suffix, and infix placement.

        Returns:
            Tuple of (hidden_states, None).
        """
        dev = transformer.embed_tokens.weight.device
        token_ids = token_ids.to(dev)
        prefix_embeds = prefix_embeds.to(dev)
        prompt_embeds = transformer.embed_tokens(token_ids)
        if token_position == "suffix":
            h = _torch.cat([prompt_embeds, prefix_embeds], dim=1)
        elif token_position == "infix" and infix_split is not None:
            part1 = prompt_embeds[:, :infix_split, :]
            part2 = prompt_embeds[:, infix_split:, :]
            h = _torch.cat([part1, prefix_embeds, part2], dim=1)
        else:
            h = _torch.cat([prefix_embeds, prompt_embeds], dim=1)
        return h, None

    def svd_stable(matrix: Array) -> tuple[Array, Array, Array]:
        """SVD with numerical stability (CPU, float32)."""
        cpu_m = matrix.to("cpu").float()
        u, s, vt = _torch.linalg.svd(cpu_m)
        dev = matrix.device
        return u.to(dev), s.to(dev), vt.to(dev)

    def qr_stable(matrix: Array) -> tuple[Array, Array]:
        """QR with numerical stability (CPU, float32)."""
        cpu_m = matrix.to("cpu").float()
        q, r = _torch.linalg.qr(cpu_m)
        dev = matrix.device
        return q.to(dev), r.to(dev)

else:
    msg = f"Unknown backend: {_BACKEND!r}"
    raise ValueError(msg)
