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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vauban._array import Array


class TorchLayerCache:
    """Per-layer KV cache proxy wrapping HuggingFace DynamicCache."""

    def __init__(self, shared_cache: object, layer_idx: int) -> None:
        self._shared_cache = shared_cache
        self._layer_idx = layer_idx

    @property
    def offset(self) -> int:
        """Current sequence length stored in this layer's cache."""
        return self._shared_cache.get_seq_length(self._layer_idx)  # type: ignore[union-attr]

    def update_and_fetch(
        self, keys: Array, values: Array,
    ) -> tuple[Array, Array]:
        """Update the cache and return full key/value tensors."""
        self._shared_cache.update(keys, values, self._layer_idx)  # type: ignore[union-attr]
        return (
            self._shared_cache.key_cache[self._layer_idx],  # type: ignore[union-attr]
            self._shared_cache.value_cache[self._layer_idx],  # type: ignore[union-attr]
        )


class TorchLayerWrapper:
    """Wraps HF decoder layer to match MLX ``layer(h, mask, cache=...)`` convention."""

    def __init__(
        self, hf_layer: object, layer_idx: int, rotary_emb: object | None = None,
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
        seq_len = h.shape[1]
        position_ids = torch.arange(
            seq_start, seq_start + seq_len,
            device=h.device,  # type: ignore[union-attr]
            dtype=torch.long,
        ).unsqueeze(0)

        kwargs: dict[str, object] = {"position_ids": position_ids}
        if cache is not None:
            kwargs["past_key_value"] = cache._shared_cache
            kwargs["use_cache"] = True
        # Newer HF transformers computes rotary embeddings at model level
        if self._rotary_emb is not None:
            kwargs["position_embeddings"] = self._rotary_emb(h, position_ids)  # type: ignore[call-non-callable]
        # Let HF handle causal masking internally (attention_mask=None)
        outputs = self._hf_layer(h, attention_mask=None, **kwargs)  # type: ignore[call-non-callable]
        # Some HF layers return a tuple (hidden_states, ...), others a bare tensor
        if isinstance(outputs, tuple):
            return outputs[0]
        return outputs

    def __getattr__(self, name: str) -> object:
        """Proxy attribute access for weight inspection (self_attn, mlp, etc.)."""
        return getattr(self._hf_layer, name)


class TorchTransformerWrapper:
    """Wraps HF inner model to match TransformerModel protocol."""

    def __init__(self, hf_inner: object) -> None:
        from vauban._arch import _EMBED_ATTRS, _LAYERS_ATTRS, _NORM_ATTRS, _find_attr

        self.embed_tokens = _find_attr(hf_inner, _EMBED_ATTRS)
        rotary_emb = getattr(hf_inner, "rotary_emb", None)
        raw_layers = _find_attr(hf_inner, _LAYERS_ATTRS)
        self.layers: list[TorchLayerWrapper] = [
            TorchLayerWrapper(layer, i, rotary_emb)
            for i, layer in enumerate(raw_layers)  # type: ignore[union-attr]
        ]
        self.norm = _find_attr(hf_inner, _NORM_ATTRS)


class TorchCausalLMWrapper:
    """Wraps HF CausalLM to match vauban CausalLM protocol."""

    def __init__(self, hf_model: object) -> None:
        from vauban._arch import _INNER_ATTRS, _find_attr

        self._hf_model = hf_model
        hf_inner = _find_attr(hf_model, _INNER_ATTRS)
        self.model = TorchTransformerWrapper(hf_inner)
        if hasattr(hf_model, "lm_head"):
            self.lm_head = hf_model.lm_head

    @property
    def device(self) -> object:
        """Return the device the underlying HF model lives on."""
        return next(self._hf_model.parameters()).device  # type: ignore[union-attr]

    def __call__(
        self,
        token_ids: Array,
        cache: list[TorchLayerCache] | None = None,
    ) -> Array:
        """Forward pass delegating to the wrapped HF model."""
        token_ids = token_ids.to(self.device)  # type: ignore[union-attr]
        kwargs: dict[str, object] = {"input_ids": token_ids}
        if cache is not None:
            # cache is list[TorchLayerCache] — extract shared DynamicCache
            kwargs["past_key_values"] = cache[0]._shared_cache
            kwargs["use_cache"] = True
        outputs = self._hf_model(**kwargs)  # type: ignore[call-non-callable]
        return outputs.logits

    def parameters(self) -> dict[str, object]:
        """Return state_dict — compatible with tree_flatten()."""
        return dict(self._hf_model.state_dict())  # type: ignore[union-attr]

    def load_weights(self, items: list[tuple[str, Array]]) -> None:
        """Load weights in MLX format: list of (key, tensor) pairs."""
        self._hf_model.load_state_dict(dict(items), strict=False)  # type: ignore[union-attr]

    def make_cache(self) -> list[TorchLayerCache]:
        """Create KV cache (list of per-layer proxies)."""
        from transformers import DynamicCache

        shared = DynamicCache()
        n_layers = len(self.model.layers)
        return [TorchLayerCache(shared, i) for i in range(n_layers)]
