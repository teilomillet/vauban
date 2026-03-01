"""Tests for vauban._model_io, _model_torch, and _nn lazy dispatch."""

import pytest

# ── _model_io dispatch ───────────────────────────────────────────────


class TestModelIoDispatch:
    def test_load_model_is_callable(self) -> None:
        """load_model should be importable and callable (MLX backend)."""
        from vauban._model_io import load_model

        assert callable(load_model)


# ── _model_torch (only if torch installed) ───────────────────────────
# importorskip is placed AFTER TestModelIoDispatch so that the MLX-only
# test still runs when torch is not installed.

torch = pytest.importorskip("torch")


class _FakeDynamicCache:
    """Minimal stand-in for transformers.DynamicCache."""

    def __init__(self) -> None:
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx < len(self.key_cache):
            return self.key_cache[layer_idx].shape[-2]
        return 0

    def update(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        layer_idx: int,
    ) -> None:
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(torch.empty(0))
            self.value_cache.append(torch.empty(0))
        self.key_cache[layer_idx] = keys
        self.value_cache[layer_idx] = values


class _MockLayer:
    """Minimal mock for a single decoder layer."""


class TestTorchLayerCache:
    def test_offset_empty(self) -> None:
        from vauban._model_torch import TorchLayerCache

        cache = _FakeDynamicCache()
        lc = TorchLayerCache(cache, 0)
        assert lc.offset == 0

    def test_update_and_fetch(self) -> None:
        from vauban._model_torch import TorchLayerCache

        cache = _FakeDynamicCache()
        lc = TorchLayerCache(cache, 0)
        keys = torch.randn(1, 4, 3, 8)
        values = torch.randn(1, 4, 3, 8)
        k, v = lc.update_and_fetch(keys, values)
        assert k is keys
        assert v is values
        assert lc.offset == 3  # seq dim


class TestTorchTransformerWrapper:
    def test_wraps_standard_model(self) -> None:
        from vauban._model_torch import TorchTransformerWrapper

        class _MockInner:
            def __init__(self) -> None:
                self.embed_tokens = torch.nn.Embedding(100, 16)
                self.layers = [_MockLayer()]
                self.norm = torch.nn.LayerNorm(16)
                self.rotary_emb = None

        inner = _MockInner()
        wrapper = TorchTransformerWrapper(inner)
        assert wrapper.embed_tokens is inner.embed_tokens
        assert len(wrapper.layers) == 1
        assert wrapper.norm is inner.norm

    def test_gpt2_style_wrapping(self) -> None:
        from vauban._model_torch import TorchTransformerWrapper

        class _MockInner:
            def __init__(self) -> None:
                self.wte = torch.nn.Embedding(100, 16)
                self.h = [_MockLayer()]
                self.ln_f = torch.nn.LayerNorm(16)

        inner = _MockInner()
        wrapper = TorchTransformerWrapper(inner)
        assert wrapper.embed_tokens is inner.wte
        assert len(wrapper.layers) == 1
        assert wrapper.norm is inner.ln_f


def _make_mock_hf_model(n_layers: int = 1) -> torch.nn.Module:
    """Build a minimal mock HF model for TorchCausalLMWrapper tests."""

    class _MockInner:
        def __init__(self) -> None:
            self.embed_tokens = torch.nn.Embedding(100, 16)
            self.layers = [_MockLayer() for _ in range(n_layers)]
            self.norm = torch.nn.LayerNorm(16)
            self.rotary_emb = None

    class _MockModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = _MockInner()
            self.lm_head = torch.nn.Linear(16, 100, bias=False)

    return _MockModel()


class TestTorchCausalLMWrapper:
    def test_parameters_returns_dict(self) -> None:
        from vauban._model_torch import TorchCausalLMWrapper

        wrapper = TorchCausalLMWrapper(_make_mock_hf_model())
        params = wrapper.parameters()
        assert isinstance(params, dict)
        assert len(params) > 0

    def test_device_property(self) -> None:
        from vauban._model_torch import TorchCausalLMWrapper

        wrapper = TorchCausalLMWrapper(_make_mock_hf_model())
        assert wrapper.device is not None

    def test_make_cache(self) -> None:
        from vauban._model_torch import TorchCausalLMWrapper

        transformers = pytest.importorskip("transformers")  # noqa: F841
        wrapper = TorchCausalLMWrapper(_make_mock_hf_model(n_layers=2))
        cache = wrapper.make_cache()
        assert len(cache) == 2

    def test_load_weights(self) -> None:
        from vauban._model_torch import TorchCausalLMWrapper

        wrapper = TorchCausalLMWrapper(_make_mock_hf_model())
        # load_weights with strict=False should not error for unknown keys
        wrapper.load_weights([("fake_key", torch.zeros(1))])
