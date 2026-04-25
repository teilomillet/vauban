# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Shared forward-pass primitives — single point of change for MLX-specific APIs.

Consolidates boilerplate that was duplicated across probe, cast, evaluate,
surface, depth, sic, and softprompt modules: KV cache creation, embed + mask
setup, LM head application, and logit extraction.
"""

from dataclasses import dataclass
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


class ExtendedLayerModule(Protocol):
    """Callable layer interface for decoders with extra runtime state.

    Gemma 4 style decoder layers accept shared KV state, per-layer inputs,
    and return a tuple carrying updated shared KV information.
    """

    def __call__(
        self,
        h: Array,
        mask: Array | None,
        *,
        cache: LayerCache | None = None,
        per_layer_input: Array | None = None,
        shared_kv: object | None = None,
        offset: object | None = None,
    ) -> object: ...


@dataclass(slots=True)
class LayerRuntime:
    """Mutable runtime state for manual layer-by-layer decoding."""

    hidden: Array
    masks: list[Array | None]
    cache: list[LayerCache | None]
    per_layer_inputs: list[Array | None]
    previous_kvs: list[int] | None = None
    intermediates: list[tuple[object | None, object | None]] | None = None


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


def _scale_input_embeddings(
    transformer: TransformerModel,
    input_embeddings: Array,
) -> Array:
    """Apply architecture-specific input scaling when present."""
    embed_scale = getattr(transformer, "embed_scale", None)
    if embed_scale is None:
        return input_embeddings
    return input_embeddings * embed_scale


def _expand_layer_cache(
    transformer: TransformerModel,
    cache: list[LayerCache] | None,
) -> list[LayerCache | None]:
    """Pad caches to one entry per layer.

    Some architectures such as Gemma 4 only materialize caches for the
    non-shared KV layers and pad the remainder with ``None`` internally.
    """
    num_layers = len(transformer.layers)
    if cache is None:
        return [None] * num_layers
    return [
        cache[i] if i < len(cache) else None
        for i in range(num_layers)
    ]


def _build_layer_masks(
    transformer: TransformerModel,
    hidden: Array,
    cache: list[LayerCache | None],
) -> list[Array | None]:
    """Build one mask per layer.

    Uses architecture-provided mask builders when available, falling back
    to the generic causal-mask + SSM handling used elsewhere in Vauban.
    """
    make_masks = getattr(transformer, "_make_masks", None)
    if callable(make_masks):
        raw_masks = make_masks(hidden, cache)
        if not isinstance(raw_masks, list):
            msg = "_make_masks() must return list[Array | None]"
            raise TypeError(msg)
        if len(raw_masks) != len(transformer.layers):
            msg = (
                "_make_masks() returned wrong number of masks: "
                f"{len(raw_masks)} != {len(transformer.layers)}"
            )
            raise ValueError(msg)
        return [cast("Array | None", mask) for mask in raw_masks]

    attn_mask = create_attention_mask(hidden)
    ssm_mask = make_ssm_mask(transformer, hidden)
    return [
        select_mask(layer, attn_mask, ssm_mask)
        for layer in transformer.layers
    ]


def _build_per_layer_inputs(
    transformer: TransformerModel,
    token_ids: Array,
    input_embeddings: Array,
    hidden: Array,
) -> list[Array | None]:
    """Build per-layer inputs for architectures that require them."""
    num_layers = len(transformer.layers)
    hidden_size_per_layer_input = getattr(
        transformer, "hidden_size_per_layer_input", None,
    )
    if hidden_size_per_layer_input in (None, 0):
        return [None] * num_layers

    get_per_layer_inputs = getattr(transformer, "_get_per_layer_inputs", None)
    project_per_layer_inputs = getattr(
        transformer, "_project_per_layer_inputs", None,
    )
    if not callable(get_per_layer_inputs) or not callable(project_per_layer_inputs):
        return [None] * num_layers

    raw_inputs = get_per_layer_inputs(token_ids, input_embeddings)
    projected_inputs = project_per_layer_inputs(hidden, raw_inputs)
    if projected_inputs is None:
        return [None] * num_layers

    return [
        projected_inputs[:, :, i, :]
        for i, _layer in enumerate(transformer.layers)
    ]


def _combine_input_embeddings(
    prompt_embeddings: Array,
    prefix_embeddings: Array,
    *,
    token_position: str = "prefix",
    infix_split: int | None = None,
) -> Array:
    """Combine prompt and prefix embeddings in the requested position."""
    from vauban import _ops as ops

    if token_position == "suffix":
        return ops.concatenate([prompt_embeddings, prefix_embeddings], axis=1)
    if token_position == "infix" and infix_split is not None:
        part1 = prompt_embeddings[:, :infix_split, :]
        part2 = prompt_embeddings[:, infix_split:, :]
        return ops.concatenate([part1, prefix_embeddings, part2], axis=1)
    return ops.concatenate([prefix_embeddings, prompt_embeddings], axis=1)


def _build_prefixed_per_layer_inputs(
    transformer: TransformerModel,
    token_ids: Array,
    prompt_embeddings: Array,
    prefix_embeddings: Array,
    *,
    token_position: str = "prefix",
    infix_split: int | None = None,
) -> list[Array | None]:
    """Build per-layer inputs for prompt tokens and pad synthetic prefix slots.

    Architectures like Gemma 4 E2B/E4B use additional per-layer token inputs.
    Synthetic prefix embeddings do not correspond to token IDs, so their
    auxiliary inputs are zero-filled while prompt-token inputs are computed
    exactly as the model would for the original prompt.
    """
    from vauban import _ops as ops

    num_layers = len(transformer.layers)
    hidden_size_per_layer_input = getattr(
        transformer, "hidden_size_per_layer_input", None,
    )
    if hidden_size_per_layer_input in (None, 0):
        return [None] * num_layers

    get_per_layer_inputs = getattr(transformer, "_get_per_layer_inputs", None)
    project_per_layer_inputs = getattr(
        transformer, "_project_per_layer_inputs", None,
    )
    if not callable(get_per_layer_inputs) or not callable(project_per_layer_inputs):
        return [None] * num_layers

    prompt_hidden = _scale_input_embeddings(transformer, prompt_embeddings)
    raw_prompt_inputs = get_per_layer_inputs(token_ids, prompt_embeddings)
    projected_prompt_inputs = project_per_layer_inputs(
        prompt_hidden, raw_prompt_inputs,
    )
    if projected_prompt_inputs is None:
        return [None] * num_layers

    prefix_len = prefix_embeddings.shape[1]
    prefix_inputs = ops.zeros(
        (
            projected_prompt_inputs.shape[0],
            prefix_len,
            projected_prompt_inputs.shape[2],
            projected_prompt_inputs.shape[3],
        ),
        dtype=projected_prompt_inputs.dtype,
    )
    combined_inputs = _combine_input_embeddings(
        projected_prompt_inputs,
        prefix_inputs,
        token_position=token_position,
        infix_split=infix_split,
    )
    return [
        combined_inputs[:, :, i, :]
        for i, _layer in enumerate(transformer.layers)
    ]


def prepare_layer_runtime(
    transformer: TransformerModel,
    token_ids: Array,
    cache: list[LayerCache] | None = None,
) -> LayerRuntime:
    """Prepare runtime state for manual layer stepping.

    This handles several architecture-specific behaviors that the older
    helpers assumed away:
    - scaled token embeddings,
    - per-layer mask selection,
    - shortened cache lists for shared-KV decoders,
    - per-layer token inputs used by Gemma 4 E2B/E4B,
    - shared-KV bookkeeping for tuple-returning decoder layers.
    """
    input_embeddings = transformer.embed_tokens(token_ids)
    hidden = _scale_input_embeddings(transformer, input_embeddings)
    expanded_cache = _expand_layer_cache(transformer, cache)
    masks = _build_layer_masks(transformer, hidden, expanded_cache)
    per_layer_inputs = _build_per_layer_inputs(
        transformer, token_ids, input_embeddings, hidden,
    )

    raw_previous_kvs = getattr(transformer, "previous_kvs", None)
    previous_kvs: list[int] | None = None
    intermediates: list[tuple[object | None, object | None]] | None = None
    if isinstance(raw_previous_kvs, list):
        if len(raw_previous_kvs) != len(transformer.layers):
            msg = (
                "previous_kvs length must match number of layers: "
                f"{len(raw_previous_kvs)} != {len(transformer.layers)}"
            )
            raise ValueError(msg)
        previous_kvs = [int(idx) for idx in raw_previous_kvs]
        intermediates = [(None, None)] * len(transformer.layers)

    return LayerRuntime(
        hidden=hidden,
        masks=masks,
        cache=expanded_cache,
        per_layer_inputs=per_layer_inputs,
        previous_kvs=previous_kvs,
        intermediates=intermediates,
    )


def prepare_prefixed_layer_runtime(
    transformer: TransformerModel,
    prefix_embeddings: Array,
    token_ids: Array,
    cache: list[LayerCache] | None = None,
    *,
    token_position: str = "prefix",
    infix_split: int | None = None,
) -> LayerRuntime:
    """Prepare runtime state for manual stepping with synthetic embeddings.

    This is the prefix-aware analogue of :func:`prepare_layer_runtime`. It is
    used when callers prepend or splice embeddings that do not come from token
    IDs, such as defensive prefixes or soft prompts.
    """
    prompt_embeddings = transformer.embed_tokens(token_ids)
    input_embeddings = _combine_input_embeddings(
        prompt_embeddings,
        prefix_embeddings,
        token_position=token_position,
        infix_split=infix_split,
    )
    hidden = _scale_input_embeddings(transformer, input_embeddings)
    expanded_cache = _expand_layer_cache(transformer, cache)
    masks = _build_layer_masks(transformer, hidden, expanded_cache)
    per_layer_inputs = _build_prefixed_per_layer_inputs(
        transformer,
        token_ids,
        prompt_embeddings,
        prefix_embeddings,
        token_position=token_position,
        infix_split=infix_split,
    )

    raw_previous_kvs = getattr(transformer, "previous_kvs", None)
    previous_kvs: list[int] | None = None
    intermediates: list[tuple[object | None, object | None]] | None = None
    if isinstance(raw_previous_kvs, list):
        if len(raw_previous_kvs) != len(transformer.layers):
            msg = (
                "previous_kvs length must match number of layers: "
                f"{len(raw_previous_kvs)} != {len(transformer.layers)}"
            )
            raise ValueError(msg)
        previous_kvs = [int(idx) for idx in raw_previous_kvs]
        intermediates = [(None, None)] * len(transformer.layers)

    return LayerRuntime(
        hidden=hidden,
        masks=masks,
        cache=expanded_cache,
        per_layer_inputs=per_layer_inputs,
        previous_kvs=previous_kvs,
        intermediates=intermediates,
    )


def _unpack_layer_result(
    result: object,
) -> tuple[Array, object | None, object | None]:
    """Normalize layer outputs to ``(hidden, shared_kv, offset)``."""
    if isinstance(result, tuple):
        if len(result) == 0:
            msg = "Layer returned an empty tuple"
            raise ValueError(msg)
        hidden = cast("Array", result[0])
        shared_kv = result[1] if len(result) > 1 else None
        offset = result[2] if len(result) > 2 else None
        return hidden, shared_kv, offset
    return cast("Array", result), None, None


def advance_layer(
    transformer: TransformerModel,
    runtime: LayerRuntime,
    layer_index: int,
) -> Array:
    """Advance one transformer layer using ``runtime`` state."""
    layer = transformer.layers[layer_index]
    mask = runtime.masks[layer_index]
    cache = runtime.cache[layer_index]
    per_layer_input = runtime.per_layer_inputs[layer_index]

    if runtime.previous_kvs is None and per_layer_input is None:
        typed_layer = cast("LayerModule", layer)
        if cache is None:
            result = typed_layer(runtime.hidden, mask)
        else:
            result = typed_layer(runtime.hidden, mask, cache=cache)
        hidden, _shared_kv, _offset = _unpack_layer_result(result)
        runtime.hidden = hidden
        return hidden

    shared_kv: object | None = None
    offset: object | None = None
    if runtime.previous_kvs is not None and runtime.intermediates is not None:
        previous_idx = runtime.previous_kvs[layer_index]
        shared_kv, offset = runtime.intermediates[previous_idx]

    typed_layer = cast("ExtendedLayerModule", layer)
    result = typed_layer(
        runtime.hidden,
        mask,
        cache=cache,
        per_layer_input=per_layer_input,
        shared_kv=shared_kv,
        offset=offset,
    )
    hidden, next_shared_kv, next_offset = _unpack_layer_result(result)
    runtime.hidden = hidden
    if runtime.intermediates is not None:
        runtime.intermediates[layer_index] = (next_shared_kv, next_offset)
    return hidden


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

    def create_attention_mask(hidden: Array) -> Array:
        """Create the default additive causal mask for MLX layers."""
        mask = nn.MultiHeadAttention.create_additive_causal_mask(hidden.shape[1])
        return mask.astype(hidden.dtype)

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
        mask = create_attention_mask(h)
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
        mask = create_attention_mask(h)
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

    def create_attention_mask(hidden: Array) -> Array | None:
        """HF decoder layers build causal masks internally."""
        del hidden
        return None

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
