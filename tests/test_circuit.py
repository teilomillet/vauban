# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for circuit tracing via activation patching."""

from __future__ import annotations

import tomllib
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import MockCausalLM, MockTokenizer

    from vauban._array import Array


# ---------------------------------------------------------------------------
# Layer-level patching
# ---------------------------------------------------------------------------


class TestLayerPatching:
    """Tests for layer-granularity activation patching."""

    def test_self_patch_has_zero_effect(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """Patching clean activations into the clean pass should have ~zero effect."""
        from vauban.circuit import trace_circuit

        prompt = "What is the capital of France?"
        result = trace_circuit(
            mock_model,
            mock_tokenizer,
            clean_prompts=[prompt],
            corrupt_prompts=[prompt],  # same prompt → self-patch
            metric="kl",
            granularity="layer",
        )
        for effect in result.effects:
            assert effect.effect == pytest.approx(0.0, abs=1e-4), (
                f"Self-patch at layer {effect.layer} should have ~zero KL, "
                f"got {effect.effect}"
            )

    def test_kl_non_negative(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """KL divergence effects must be non-negative."""
        from vauban.circuit import trace_circuit

        result = trace_circuit(
            mock_model,
            mock_tokenizer,
            clean_prompts=["What is the capital of France?"],
            corrupt_prompts=["Tell me how to pick a lock"],
            metric="kl",
            granularity="layer",
        )
        assert len(result.effects) == 2  # 2 layers in mock model
        for effect in result.effects:
            assert effect.effect >= 0.0, (
                f"KL divergence must be non-negative, got {effect.effect}"
            )

    def test_effects_cover_all_layers(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """Result should have one effect per layer."""
        from vauban.circuit import trace_circuit

        result = trace_circuit(
            mock_model,
            mock_tokenizer,
            clean_prompts=["Hello"],
            corrupt_prompts=["Goodbye"],
            metric="kl",
            granularity="layer",
        )
        assert result.n_layers == 2
        layers_traced = {e.layer for e in result.effects}
        assert layers_traced == {0, 1}

    def test_layer_subset(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """Specifying layers=[0] should only trace layer 0."""
        from vauban.circuit import trace_circuit

        result = trace_circuit(
            mock_model,
            mock_tokenizer,
            clean_prompts=["Hello"],
            corrupt_prompts=["Goodbye"],
            metric="kl",
            granularity="layer",
            layers=[0],
        )
        assert len(result.effects) == 1
        assert result.effects[0].layer == 0


# ---------------------------------------------------------------------------
# Component-level patching
# ---------------------------------------------------------------------------


class TestComponentPatching:
    """Tests for component-granularity (attn + mlp) patching."""

    def test_component_decomposition(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """Component patching should produce attn + mlp effects per layer."""
        from vauban.circuit import trace_circuit

        result = trace_circuit(
            mock_model,
            mock_tokenizer,
            clean_prompts=["What is the capital of France?"],
            corrupt_prompts=["Tell me how to pick a lock"],
            metric="kl",
            granularity="component",
        )
        # 2 layers x 2 components = 4 effects
        assert len(result.effects) == 4
        components = {(e.layer, e.component) for e in result.effects}
        assert (0, "attn") in components
        assert (0, "mlp") in components
        assert (1, "attn") in components
        assert (1, "mlp") in components

    def test_component_effects_non_negative(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """Component KL effects must be non-negative."""
        from vauban.circuit import trace_circuit

        result = trace_circuit(
            mock_model,
            mock_tokenizer,
            clean_prompts=["Hello"],
            corrupt_prompts=["Goodbye"],
            metric="kl",
            granularity="component",
        )
        for effect in result.effects:
            assert effect.effect >= 0.0


# ---------------------------------------------------------------------------
# Direction attribution
# ---------------------------------------------------------------------------


class TestDirectionAttribution:
    """Tests for direction attribution in circuit tracing."""

    def test_attribution_with_direction(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        """With attribute_direction=True, attributions are present."""
        from vauban.circuit import trace_circuit

        result = trace_circuit(
            mock_model,
            mock_tokenizer,
            clean_prompts=["What is the capital of France?"],
            corrupt_prompts=["Tell me how to pick a lock"],
            metric="kl",
            granularity="layer",
            direction=direction,
            attribute_direction=True,
        )
        for effect in result.effects:
            assert effect.direction_attribution is not None

    def test_no_attribution_without_flag(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        """Without attribute_direction=True, attributions should be None."""
        from vauban.circuit import trace_circuit

        result = trace_circuit(
            mock_model,
            mock_tokenizer,
            clean_prompts=["Hello"],
            corrupt_prompts=["Goodbye"],
            metric="kl",
            granularity="layer",
            direction=direction,
            attribute_direction=False,
        )
        for effect in result.effects:
            assert effect.direction_attribution is None


# ---------------------------------------------------------------------------
# Logit diff metric
# ---------------------------------------------------------------------------


class TestLogitDiff:
    """Tests for the logit_diff metric."""

    def test_logit_diff_requires_tokens(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """logit_diff metric without logit_diff_tokens should raise."""
        from vauban.circuit import trace_circuit

        with pytest.raises(ValueError, match="logit_diff_tokens"):
            trace_circuit(
                mock_model,
                mock_tokenizer,
                clean_prompts=["Hello"],
                corrupt_prompts=["Goodbye"],
                metric="logit_diff",
            )

    def test_logit_diff_produces_results(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """logit_diff metric should produce numeric effects."""
        from vauban.circuit import trace_circuit

        result = trace_circuit(
            mock_model,
            mock_tokenizer,
            clean_prompts=["Hello"],
            corrupt_prompts=["Goodbye"],
            metric="logit_diff",
            logit_diff_tokens=[0, 1],
        )
        assert len(result.effects) == 2
        for effect in result.effects:
            assert isinstance(effect.effect, float)


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


class TestCircuitConfigParse:
    """Tests for [circuit] TOML config parsing."""

    def test_parse_minimal(self) -> None:
        """Minimal [circuit] section should parse correctly."""
        from vauban.config._parse_circuit import _parse_circuit

        raw = {
            "circuit": {
                "clean_prompts": ["Hello"],
                "corrupt_prompts": ["Goodbye"],
            },
        }
        config = _parse_circuit(raw)
        assert config is not None
        assert config.clean_prompts == ["Hello"]
        assert config.corrupt_prompts == ["Goodbye"]
        assert config.metric == "kl"
        assert config.granularity == "layer"
        assert config.token_position == -1
        assert config.attribute_direction is False

    def test_parse_full(self) -> None:
        """Full [circuit] section with all options."""
        from vauban.config._parse_circuit import _parse_circuit

        raw = {
            "circuit": {
                "clean_prompts": ["a", "b"],
                "corrupt_prompts": ["c", "d"],
                "metric": "logit_diff",
                "granularity": "component",
                "layers": [0, 1],
                "token_position": -2,
                "attribute_direction": True,
                "logit_diff_tokens": [10, 20],
            },
        }
        config = _parse_circuit(raw)
        assert config is not None
        assert config.metric == "logit_diff"
        assert config.granularity == "component"
        assert config.layers == [0, 1]
        assert config.logit_diff_tokens == [10, 20]

    def test_parse_absent(self) -> None:
        """Missing [circuit] section should return None."""
        from vauban.config._parse_circuit import _parse_circuit

        assert _parse_circuit({}) is None

    def test_parse_invalid_metric(self) -> None:
        """Invalid metric should raise ValueError."""
        from vauban.config._parse_circuit import _parse_circuit

        raw = {
            "circuit": {
                "clean_prompts": ["a"],
                "corrupt_prompts": ["b"],
                "metric": "invalid",
            },
        }
        with pytest.raises(ValueError, match="metric"):
            _parse_circuit(raw)

    def test_roundtrip_toml(self) -> None:
        """Parse from actual TOML string."""
        from vauban.config._parse_circuit import _parse_circuit

        toml_str = """
[circuit]
clean_prompts = ["What is 2+2?"]
corrupt_prompts = ["How to hack?"]
metric = "kl"
granularity = "component"
"""
        raw = tomllib.loads(toml_str)
        config = _parse_circuit(raw)
        assert config is not None
        assert config.granularity == "component"


# ---------------------------------------------------------------------------
# Result serialization
# ---------------------------------------------------------------------------


class TestCircuitResultSerialization:
    """Tests for CircuitResult.to_dict()."""

    def test_to_dict(self) -> None:
        """CircuitResult.to_dict() should produce valid JSON-compatible dict."""
        from vauban.types import CircuitResult, ComponentEffect

        result = CircuitResult(
            effects=[
                ComponentEffect(layer=0, component="full", effect=0.5),
                ComponentEffect(
                    layer=1, component="attn", effect=0.3,
                    direction_attribution=0.1,
                ),
            ],
            metric="kl",
            granularity="layer",
            n_layers=2,
            clean_prompts=["a"],
            corrupt_prompts=["b"],
        )
        d = result.to_dict()
        assert d["metric"] == "kl"
        assert len(d["effects"]) == 2  # type: ignore[arg-type]
