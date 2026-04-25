# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace Transformers model wrappers for PyTorch backend.

Wraps HF ``AutoModelForCausalLM`` objects so they satisfy the same
structural protocols (CausalLM, TransformerModel, LayerCache) that
MLX models do. This lets all consumer code (probe, cast, evaluate,
cut, etc.) call ``model(token_ids, cache=cache)``,
``model.parameters()``, ``model.load_weights(items)``, and
``layer(h, mask, cache=cache[i])`` without knowing which backend
is active.

Important: wrapper layers do NOT use ``torch.no_grad()`` so that
autograd flows through for softprompt ``value_and_grad``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping

    from vauban._array import Array


class _TorchTensor(Protocol):
    """Subset of torch Tensor behavior used by the wrapper."""

    shape: tuple[int, ...]
    device: object

    def to(self, device: object) -> _TorchTensor:
        """Move tensor to a device."""

    def unsqueeze(self, dim: int) -> _TorchTensor:
        """Insert a tensor dimension."""


class _TorchParameter(Protocol):
    """Subset of torch Parameter behavior used by the wrapper."""

    device: object


class _DynamicCache(Protocol):
    """Subset of HuggingFace DynamicCache used by Vauban."""

    key_cache: list[Array]
    value_cache: list[Array]

    def get_seq_length(self, layer_idx: int) -> int:
        """Return the cached sequence length for a layer."""

    def update(self, keys: Array, values: Array, layer_idx: int) -> object:
        """Update a layer cache."""


class _RotaryEmbedding(Protocol):
    """Callable rotary embedding helper exposed by some HF models."""

    def __call__(self, h: Array, position_ids: _TorchTensor) -> object:
        """Return positional embeddings."""


class _HfLayer(Protocol):
    """Subset of a HuggingFace decoder layer used by Vauban."""

    def __call__(
        self, h: Array, *, attention_mask: object, **kwargs: object,
    ) -> object:
        """Run a decoder layer."""


class _HfOutput(Protocol):
    """Subset of HuggingFace model output used by Vauban."""

    logits: Array


class _HfCausalLM(Protocol):
    """Subset of a HuggingFace causal LM used by Vauban."""

    def __call__(self, **kwargs: object) -> _HfOutput:
        """Run the causal LM."""

    def parameters(self) -> Iterator[_TorchParameter]:
        """Return model parameters."""

    def state_dict(self) -> Mapping[str, object]:
        """Return model weights."""

    def load_state_dict(self, state: dict[str, Array], *, strict: bool) -> object:
        """Load model weights."""


class TorchLayerCache:
    """Per-layer KV cache proxy wrapping HuggingFace DynamicCache."""

    def __init__(self, shared_cache: _DynamicCache, layer_idx: int) -> None:
        self._shared_cache = shared_cache
        self._layer_idx = layer_idx

    @property
    def offset(self) -> int:
        """Current sequence length stored in this layer's cache."""
        return self._shared_cache.get_seq_length(self._layer_idx)

    def update_and_fetch(
        self, keys: Array, values: Array,
    ) -> tuple[Array, Array]:
        """Update the cache and return full key/value tensors."""
        self._shared_cache.update(keys, values, self._layer_idx)
        return (
            self._shared_cache.key_cache[self._layer_idx],
            self._shared_cache.value_cache[self._layer_idx],
        )


class TorchLayerWrapper:
    """Wraps HF decoder layer to match MLX ``layer(h, mask, cache=...)`` convention."""

    def __init__(
        self,
        hf_layer: _HfLayer,
        layer_idx: int,
        rotary_emb: _RotaryEmbedding | None = None,
    ) -> None:
        self._hf_layer = hf_layer
        self._layer_idx = layer_idx
        self._rotary_emb = rotary_emb

    def __call__(
        self,
        h: Array,
        mask: Array | None = None,
        cache: TorchLayerCache | None = None,
    ) -> Array:
        """Forward pass translating MLX convention to HF convention."""
        import torch

        seq_start = cache.offset if cache is not None else 0
        h_tensor = cast("_TorchTensor", h)
        seq_len = h_tensor.shape[1]
        device = cast("torch.device | str | int | None", h_tensor.device)
        position_ids = cast(
            "_TorchTensor",
            torch.arange(
                seq_start, seq_start + seq_len,
                device=device,
                dtype=torch.long,
            ).unsqueeze(0),
        )

        kwargs: dict[str, object] = {"position_ids": position_ids}
        if cache is not None:
            kwargs["past_key_value"] = cache._shared_cache
            kwargs["use_cache"] = True
        # Newer HF transformers computes rotary embeddings at model level
        if self._rotary_emb is not None:
            kwargs["position_embeddings"] = self._rotary_emb(h, position_ids)
        # Let HF handle causal masking internally (attention_mask=None)
        outputs = self._hf_layer(h, attention_mask=None, **kwargs)
        # Some HF layers return a tuple (hidden_states, ...), others a bare tensor
        if isinstance(outputs, tuple):
            return cast("Array", outputs[0])
        return cast("Array", outputs)

    def __getattr__(self, name: str) -> object:
        """Proxy attribute access for weight inspection (self_attn, mlp, etc.)."""
        return getattr(self._hf_layer, name)


class TorchTransformerWrapper:
    """Wraps HF inner model to match TransformerModel protocol."""

    def __init__(self, hf_inner: object) -> None:
        from vauban._arch import _EMBED_ATTRS, _LAYERS_ATTRS, _NORM_ATTRS, _find_attr

        self.embed_tokens = _find_attr(hf_inner, _EMBED_ATTRS)
        rotary_emb = cast(
            "_RotaryEmbedding | None",
            getattr(hf_inner, "rotary_emb", None),
        )
        raw_layers = cast("Iterable[_HfLayer]", _find_attr(hf_inner, _LAYERS_ATTRS))
        self.layers: list[TorchLayerWrapper] = [
            TorchLayerWrapper(layer, i, rotary_emb)
            for i, layer in enumerate(raw_layers)
        ]
        self.norm = _find_attr(hf_inner, _NORM_ATTRS)


class TorchCausalLMWrapper:
    """Wraps HF CausalLM to match vauban CausalLM protocol."""

    def __init__(self, hf_model: object) -> None:
        from vauban._arch import _INNER_ATTRS, _find_attr

        self._hf_model = cast("_HfCausalLM", hf_model)
        hf_inner = _find_attr(hf_model, _INNER_ATTRS)
        self.model = TorchTransformerWrapper(hf_inner)
        if hasattr(hf_model, "lm_head"):
            self.lm_head = hf_model.lm_head

    @property
    def device(self) -> object:
        """Return the device the underlying HF model lives on."""
        return next(self._hf_model.parameters()).device

    def __call__(
        self,
        token_ids: Array,
        cache: list[TorchLayerCache] | None = None,
    ) -> Array:
        """Forward pass delegating to the wrapped HF model."""
        token_tensor = cast("_TorchTensor", token_ids).to(self.device)
        kwargs: dict[str, object] = {"input_ids": token_tensor}
        if cache is not None:
            # cache is list[TorchLayerCache] — extract shared DynamicCache
            kwargs["past_key_values"] = cache[0]._shared_cache
            kwargs["use_cache"] = True
        outputs = self._hf_model(**kwargs)
        return outputs.logits

    def parameters(self) -> dict[str, object]:
        """Return state_dict — compatible with tree_flatten()."""
        return dict(self._hf_model.state_dict())

    def load_weights(self, items: list[tuple[str, Array]]) -> None:
        """Load weights in MLX format: list of (key, tensor) pairs."""
        self._hf_model.load_state_dict(dict(items), strict=False)

    def make_cache(self) -> list[TorchLayerCache]:
        """Create KV cache (list of per-layer proxies)."""
        from transformers import DynamicCache

        shared = cast("_DynamicCache", DynamicCache())
        n_layers = len(self.model.layers)
        return [TorchLayerCache(shared, i) for i in range(n_layers)]
