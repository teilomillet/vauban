# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Fake-module tests for ``vauban._model_torch`` without a torch install."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import TYPE_CHECKING, cast

from vauban._model_torch import (
    TorchCausalLMWrapper,
    TorchLayerCache,
    TorchLayerWrapper,
    TorchTransformerWrapper,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pytest

    from vauban._array import Array


class FakeTensor:
    """Minimal tensor-like object for wrapper tests."""

    def __init__(self, shape: tuple[int, ...], device: object = "cpu") -> None:
        self.shape = shape
        self.device = device
        self.to_calls: list[object] = []

    def to(self, device: object) -> FakeTensor:
        """Record the move and return ``self``."""
        self.to_calls.append(device)
        self.device = device
        return self


class FakeRangeTensor:
    """Return value of ``torch.arange`` for position IDs."""

    def __init__(
        self,
        start: int,
        end: int,
        device: object,
        dtype: object,
    ) -> None:
        self.start = start
        self.end = end
        self.device = device
        self.dtype = dtype
        self.unsqueeze_dims: list[int] = []

    def unsqueeze(self, dim: int) -> FakeRangeTensor:
        """Record the unsqueeze dimension and return ``self``."""
        self.unsqueeze_dims.append(dim)
        return self


class FakeTorchModule(ModuleType):
    """Typed fake ``torch`` module."""

    def __init__(self) -> None:
        super().__init__("torch")
        self.long = object()
        self.arange_calls: list[FakeRangeTensor] = []

    def arange(
        self,
        start: int,
        end: int,
        *,
        device: object,
        dtype: object,
    ) -> FakeRangeTensor:
        """Create a fake position-id tensor."""
        result = FakeRangeTensor(start, end, device, dtype)
        self.arange_calls.append(result)
        return result


class FakeDynamicCache:
    """Minimal stand-in for ``transformers.DynamicCache``."""

    def __init__(self) -> None:
        self.key_cache: list[FakeTensor] = []
        self.value_cache: list[FakeTensor] = []

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return cached sequence length for one layer."""
        if layer_idx < len(self.key_cache):
            return self.key_cache[layer_idx].shape[-2]
        return 0

    def update(self, keys: FakeTensor, values: FakeTensor, layer_idx: int) -> None:
        """Store cache entries for one layer."""
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(FakeTensor((0, 0)))
            self.value_cache.append(FakeTensor((0, 0)))
        self.key_cache[layer_idx] = keys
        self.value_cache[layer_idx] = values


class FakeTransformersModule(ModuleType):
    """Typed fake ``transformers`` module."""

    def __init__(self) -> None:
        super().__init__("transformers")
        self.DynamicCache = FakeDynamicCache


class FakeDecoderLayer:
    """Callable decoder-layer stub."""

    def __init__(self, output: object) -> None:
        self.output = output
        self.calls: list[dict[str, object]] = []
        self.some_attr = "proxied"

    def __call__(
        self,
        hidden: object,
        attention_mask: object | None = None,
        **kwargs: object,
    ) -> object:
        """Record call arguments and return the configured output."""
        self.calls.append(
            {"hidden": hidden, "attention_mask": attention_mask, **kwargs},
        )
        return self.output


class FakeInnerCanonical:
    """Canonical inner model with ``embed_tokens``, ``layers``, and ``norm``."""

    def __init__(self, layers: list[object], rotary_emb: object | None = None) -> None:
        self.embed_tokens = object()
        self.layers = layers
        self.norm = object()
        self.rotary_emb = rotary_emb


class FakeInnerGpt2:
    """GPT-2-style inner model using alternative attribute names."""

    def __init__(self, layers: list[object]) -> None:
        self.wte = object()
        self.h = layers
        self.ln_f = object()


class FakeParameter:
    """Parameter stub exposing only ``device``."""

    def __init__(self, device: object) -> None:
        self.device = device


class FakeOutputs:
    """Wrapper for model logits."""

    def __init__(self, logits: object) -> None:
        self.logits = logits


class FakeHFModel:
    """Minimal HF-style causal LM for wrapper tests."""

    def __init__(
        self,
        inner: object,
        *,
        inner_attr: str = "model",
        with_lm_head: bool = True,
    ) -> None:
        setattr(self, inner_attr, inner)
        if with_lm_head:
            self.lm_head = object()
        self._parameters = [FakeParameter("cuda:1")]
        self._state = {"weight": object()}
        self.forward_calls: list[dict[str, object]] = []
        self.loaded_state: tuple[dict[str, object], bool] | None = None
        self.logits = object()

    def parameters(self) -> Iterator[FakeParameter]:
        """Yield fake parameters."""
        return iter(self._parameters)

    def __call__(self, **kwargs: object) -> FakeOutputs:
        """Record kwargs and return fake logits."""
        self.forward_calls.append(kwargs)
        return FakeOutputs(self.logits)

    def state_dict(self) -> dict[str, object]:
        """Expose fake state dict."""
        return dict(self._state)

    def load_state_dict(self, state: dict[str, object], strict: bool = False) -> None:
        """Record load requests."""
        self.loaded_state = (state, strict)


def _install_fake_modules(monkeypatch: pytest.MonkeyPatch) -> FakeTorchModule:
    """Install fake ``torch`` and ``transformers`` modules for one test."""
    fake_torch = FakeTorchModule()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", FakeTransformersModule())
    return fake_torch


class TestTorchLayerCacheFake:
    """Tests for ``TorchLayerCache`` without torch."""

    def test_offset_and_update_and_fetch(self) -> None:
        cache = FakeDynamicCache()
        layer_cache = TorchLayerCache(cache, 1)

        assert layer_cache.offset == 0

        keys = FakeTensor((1, 4, 3, 8))
        values = FakeTensor((1, 4, 3, 8))
        fetched_keys, fetched_values = layer_cache.update_and_fetch(
            cast("Array", keys),
            cast("Array", values),
        )

        assert fetched_keys is keys
        assert fetched_values is values
        assert layer_cache.offset == 3


class TestTorchLayerWrapperFake:
    """Tests for ``TorchLayerWrapper`` without torch."""

    def test_call_without_cache_returns_first_tuple_item(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_torch = _install_fake_modules(monkeypatch)
        output = object()
        hf_layer = FakeDecoderLayer((output, "extra"))
        wrapper = TorchLayerWrapper(hf_layer, 0)
        hidden = FakeTensor((1, 5, 16), device="cuda:0")

        result = wrapper(cast("Array", hidden))

        call = hf_layer.calls[0]
        position_ids = cast("FakeRangeTensor", call["position_ids"])
        assert result is output
        assert call["attention_mask"] is None
        assert position_ids.start == 0
        assert position_ids.end == 5
        assert position_ids.unsqueeze_dims == [0]
        assert position_ids is fake_torch.arange_calls[0]
        assert "past_key_value" not in call
        assert "use_cache" not in call
        assert "position_embeddings" not in call
        assert wrapper.some_attr == "proxied"

    def test_call_with_cache_and_rotary_uses_hf_conventions(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_torch = _install_fake_modules(monkeypatch)
        shared_cache = FakeDynamicCache()
        shared_cache.update(FakeTensor((1, 4, 7, 8)), FakeTensor((1, 4, 7, 8)), 2)
        layer_cache = TorchLayerCache(shared_cache, 2)
        rotary_calls: list[tuple[object, object]] = []

        def _fake_rotary(hidden: object, position_ids: object) -> str:
            rotary_calls.append((hidden, position_ids))
            return "rotary"

        output = FakeTensor((1, 4, 16), device="cuda:2")
        hf_layer = FakeDecoderLayer(output)
        wrapper = TorchLayerWrapper(hf_layer, 2, _fake_rotary)
        hidden = FakeTensor((1, 4, 16), device="cuda:2")

        result = wrapper(cast("Array", hidden), cache=layer_cache)

        call = hf_layer.calls[0]
        position_ids = cast("FakeRangeTensor", call["position_ids"])
        assert result is output
        assert position_ids.start == 7
        assert position_ids.end == 11
        assert position_ids is fake_torch.arange_calls[0]
        assert call["past_key_value"] is shared_cache
        assert call["use_cache"] is True
        assert call["position_embeddings"] == "rotary"
        assert rotary_calls == [(hidden, position_ids)]


class TestTorchTransformerWrapperFake:
    """Tests for ``TorchTransformerWrapper`` without torch."""

    def test_wraps_canonical_inner_model(self) -> None:
        inner = FakeInnerCanonical([FakeDecoderLayer(object())], rotary_emb="rotary")

        wrapper = TorchTransformerWrapper(inner)

        assert wrapper.embed_tokens is inner.embed_tokens
        assert wrapper.norm is inner.norm
        assert len(wrapper.layers) == 1
        assert isinstance(wrapper.layers[0], TorchLayerWrapper)
        assert wrapper.layers[0]._rotary_emb == "rotary"

    def test_wraps_gpt2_style_inner_model(self) -> None:
        inner = FakeInnerGpt2([FakeDecoderLayer(object())])

        wrapper = TorchTransformerWrapper(inner)

        assert wrapper.embed_tokens is inner.wte
        assert wrapper.norm is inner.ln_f
        assert len(wrapper.layers) == 1


class TestTorchCausalLMWrapperFake:
    """Tests for ``TorchCausalLMWrapper`` without torch."""

    def test_init_device_and_parameters(self) -> None:
        hf_model = FakeHFModel(FakeInnerCanonical([FakeDecoderLayer(object())]))

        wrapper = TorchCausalLMWrapper(hf_model)

        assert wrapper.lm_head is hf_model.lm_head
        assert wrapper.device == "cuda:1"
        assert wrapper.parameters() == {"weight": hf_model._state["weight"]}

    def test_call_without_cache_moves_inputs_and_returns_logits(self) -> None:
        hf_model = FakeHFModel(FakeInnerCanonical([FakeDecoderLayer(object())]))
        wrapper = TorchCausalLMWrapper(hf_model)
        token_ids = FakeTensor((1, 3), device="cpu")

        result = wrapper(cast("Array", token_ids))

        assert result is hf_model.logits
        assert token_ids.to_calls == ["cuda:1"]
        assert hf_model.forward_calls == [{"input_ids": token_ids}]

    def test_load_weights_and_make_cache_and_call_with_cache(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_fake_modules(monkeypatch)
        hf_model = FakeHFModel(
            FakeInnerCanonical(
                [FakeDecoderLayer(object()), FakeDecoderLayer(object())],
            ),
            inner_attr="transformer",
        )
        wrapper = TorchCausalLMWrapper(hf_model)
        token_ids = FakeTensor((1, 2), device="cpu")
        cache = wrapper.make_cache()
        weight_value = FakeTensor((1,))

        wrapper.load_weights([("patched", cast("Array", weight_value))])
        result = wrapper(cast("Array", token_ids), cache=cache)

        assert len(cache) == 2
        assert all(isinstance(item, TorchLayerCache) for item in cache)
        assert hf_model.loaded_state is not None
        assert hf_model.loaded_state[0] == {"patched": weight_value}
        assert hf_model.loaded_state[1] is False
        assert result is hf_model.logits
        assert hf_model.forward_calls[0]["input_ids"] is token_ids
        assert hf_model.forward_calls[0]["past_key_values"] is cache[0]._shared_cache
        assert hf_model.forward_calls[0]["use_cache"] is True
