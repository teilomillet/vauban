# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for vauban._arch: architecture auto-detection."""

import pytest

from vauban._arch import (
    TransformerAdapter,
    _find_attr,
    get_inner_model,
    normalize_transformer,
)


class _MockObj:
    """Minimal mock with configurable attributes."""

    def __init__(self, **kwargs: object) -> None:
        for k, v in kwargs.items():
            object.__setattr__(self, k, v)


# ── _find_attr ───────────────────────────────────────────────────────


class TestFindAttr:
    def test_first_candidate(self) -> None:
        obj = _MockObj(model="m", transformer="t")
        assert _find_attr(obj, ("model", "transformer")) == "m"

    def test_second_candidate(self) -> None:
        obj = _MockObj(transformer="t")
        assert _find_attr(obj, ("model", "transformer")) == "t"

    def test_no_match_raises(self) -> None:
        obj = _MockObj()
        with pytest.raises(AttributeError, match="Cannot find"):
            _find_attr(obj, ("model", "transformer"))

    def test_none_value_skipped(self) -> None:
        """Attributes that are None are treated as absent."""
        obj = _MockObj(model=None, transformer="t")
        assert _find_attr(obj, ("model", "transformer")) == "t"


# ── get_inner_model ──────────────────────────────────────────────────


class TestGetInnerModel:
    def test_llama_style(self) -> None:
        inner = _MockObj()
        model = _MockObj(model=inner)
        assert get_inner_model(model) is inner

    def test_gpt2_style(self) -> None:
        inner = _MockObj()
        model = _MockObj(transformer=inner)
        assert get_inner_model(model) is inner

    def test_gpt_neox_style(self) -> None:
        inner = _MockObj()
        model = _MockObj(gpt_neox=inner)
        assert get_inner_model(model) is inner

    def test_no_inner_raises(self) -> None:
        model = _MockObj()
        with pytest.raises(AttributeError):
            get_inner_model(model)

    def test_flat_architecture_returns_model(self) -> None:
        model = _MockObj(layers=["l1", "l2"])
        assert get_inner_model(model) is model

    def test_recursion_limit_raises(self) -> None:
        with pytest.raises(AttributeError, match="Recursion limit reached"):
            get_inner_model(_MockObj(), _depth=4)


# ── normalize_transformer ───────────────────────────────────────────


class TestNormalizeTransformer:
    def test_canonical_returns_as_is(self) -> None:
        """Model with embed_tokens, layers, norm passes through."""
        inner = _MockObj(
            embed_tokens="e", layers=["l1", "l2"], norm="n",
        )
        assert normalize_transformer(inner) is inner

    def test_missing_attr_wraps(self) -> None:
        """Model missing canonical attrs gets wrapped in TransformerAdapter."""
        inner = _MockObj(wte="e", h=["l1"], ln_f="n")
        result = normalize_transformer(inner)
        assert isinstance(result, TransformerAdapter)


# ── TransformerAdapter ───────────────────────────────────────────────


class TestTransformerAdapter:
    def test_llama_style_normalization(self) -> None:
        inner = _MockObj(embed_tokens="e", layers=["l1", "l2"], norm="n")
        adapter = TransformerAdapter(inner)
        assert adapter.embed_tokens == "e"
        assert adapter.layers == ["l1", "l2"]
        assert adapter.norm == "n"

    def test_gpt2_style_normalization(self) -> None:
        inner = _MockObj(wte="e", h=["l1"], ln_f="n")
        adapter = TransformerAdapter(inner)
        assert adapter.embed_tokens == "e"
        assert adapter.layers == ["l1"]
        assert adapter.norm == "n"

    def test_phi_style_normalization(self) -> None:
        inner = _MockObj(embed_tokens="e", layers=["l1"], model_norm="n")
        adapter = TransformerAdapter(inner)
        assert adapter.embed_tokens == "e"
        assert adapter.norm == "n"

    def test_gpt_neox_style_normalization(self) -> None:
        inner = _MockObj(
            embed="e", blocks=["l1", "l2", "l3"], final_layernorm="n",
        )
        adapter = TransformerAdapter(inner)
        assert adapter.embed_tokens == "e"
        assert adapter.layers == ["l1", "l2", "l3"]
        assert adapter.norm == "n"

    def test_proxy_unknown_attr(self) -> None:
        inner = _MockObj(
            embed_tokens="e", layers=[], norm="n", custom_attr="custom",
        )
        adapter = TransformerAdapter(inner)
        assert adapter.custom_attr == "custom"

    def test_proxy_missing_attr_raises(self) -> None:
        inner = _MockObj(embed_tokens="e", layers=[], norm="n")
        adapter = TransformerAdapter(inner)
        with pytest.raises(AttributeError):
            _ = adapter.nonexistent

    def test_missing_embed_raises(self) -> None:
        inner = _MockObj(layers=[], norm="n")
        with pytest.raises(AttributeError, match="Cannot find"):
            TransformerAdapter(inner)

    def test_missing_layers_raises(self) -> None:
        inner = _MockObj(embed_tokens="e", norm="n")
        with pytest.raises(AttributeError, match="Cannot find"):
            TransformerAdapter(inner)

    def test_missing_norm_raises(self) -> None:
        inner = _MockObj(embed_tokens="e", layers=[])
        with pytest.raises(AttributeError, match="Cannot find"):
            TransformerAdapter(inner)
