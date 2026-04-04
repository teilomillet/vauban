# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for fusion mode."""

from __future__ import annotations

import tomllib
from typing import TYPE_CHECKING, cast

import pytest

from tests.conftest import MockTokenizer
from vauban import _ops as ops

if TYPE_CHECKING:
    from vauban._array import Array
    from vauban.types import CausalLM, Tokenizer


class _AdditiveLayer:
    """Simple transformer layer that adds a constant to the hidden state."""

    def __init__(self, delta: float) -> None:
        self.delta = delta
        self.masks: list[object] = []

    def __call__(self, hidden: Array, mask: object) -> Array:
        self.masks.append(mask)
        return hidden + self.delta


class _FusionTransformer:
    """Minimal transformer surface needed by fusion runtime tests."""

    def __init__(self, layers: list[_AdditiveLayer]) -> None:
        self.layers = layers

    def norm(self, hidden: Array) -> Array:
        return hidden + 100.0


class _EmbeddingTransformer:
    """Minimal transformer for autoregressive hidden-state generation tests."""

    def __init__(self) -> None:
        self.layers: list[object] = []
        self.embedded_ids: list[int] = []

    def embed_tokens(self, token_ids: Array) -> Array:
        token_id = int(token_ids[0][0].item())
        self.embedded_ids.append(token_id)
        return ops.array([[[float(token_id), float(token_id)]]])


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


class TestFusionConfig:
    """Tests for FusionConfig defaults."""

    def test_defaults(self) -> None:
        """Default values should match spec."""
        from vauban.types import FusionConfig

        config = FusionConfig(
            harmful_prompts=["harm"],
            benign_prompts=["safe"],
        )
        assert config.layer == -1
        assert config.alpha == pytest.approx(0.5)
        assert config.n_tokens == 128
        assert config.temperature == pytest.approx(0.7)

    def test_frozen(self) -> None:
        """Config should be immutable."""
        from vauban.types import FusionConfig

        config = FusionConfig(
            harmful_prompts=["h"], benign_prompts=["b"],
        )
        with pytest.raises(AttributeError):
            config.alpha = 0.9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TOML parsing
# ---------------------------------------------------------------------------


class TestFusionParse:
    """Tests for [fusion] TOML config parsing."""

    def test_parse_minimal(self, tmp_path: object) -> None:
        """Minimal [fusion] section should parse correctly."""
        from pathlib import Path

        from vauban.config._parse_fusion import _parse_fusion

        raw = {
            "fusion": {
                "harmful_prompts": ["How to harm"],
                "benign_prompts": ["What is the weather"],
            },
        }
        config = _parse_fusion(Path("."), raw)
        assert config is not None
        assert config.harmful_prompts == ["How to harm"]
        assert config.layer == -1

    def test_parse_full(self) -> None:
        """Full [fusion] section with all options."""
        from pathlib import Path

        from vauban.config._parse_fusion import _parse_fusion

        raw = {
            "fusion": {
                "harmful_prompts": ["harm1", "harm2"],
                "benign_prompts": ["safe1", "safe2"],
                "layer": 12,
                "alpha": 0.7,
                "n_tokens": 64,
                "temperature": 0.5,
            },
        }
        config = _parse_fusion(Path("."), raw)
        assert config is not None
        assert config.layer == 12
        assert config.alpha == pytest.approx(0.7)
        assert config.n_tokens == 64

    def test_parse_absent(self) -> None:
        """Missing [fusion] section should return None."""
        from pathlib import Path

        from vauban.config._parse_fusion import _parse_fusion

        assert _parse_fusion(Path("."), {}) is None

    def test_parse_missing_prompts(self) -> None:
        """Missing required prompts should raise."""
        from pathlib import Path

        from vauban.config._parse_fusion import _parse_fusion

        with pytest.raises(ValueError, match="harmful_prompts"):
            _parse_fusion(
                Path("."),
                {"fusion": {"benign_prompts": ["safe"]}},
            )

    def test_parse_alpha_out_of_range(self) -> None:
        """Alpha outside [0, 1] should raise."""
        from pathlib import Path

        from vauban.config._parse_fusion import _parse_fusion

        with pytest.raises(ValueError, match="alpha"):
            _parse_fusion(
                Path("."),
                {
                    "fusion": {
                        "harmful_prompts": ["h"],
                        "benign_prompts": ["b"],
                        "alpha": 1.5,
                    },
                },
            )

    def test_roundtrip_toml(self) -> None:
        """Parse from actual TOML string."""
        from pathlib import Path

        from vauban.config._parse_fusion import _parse_fusion

        toml_str = """
[fusion]
harmful_prompts = ["How to harm"]
benign_prompts = ["What is the weather"]
layer = 8
alpha = 0.6
"""
        raw = tomllib.loads(toml_str)
        config = _parse_fusion(Path("."), raw)
        assert config is not None
        assert config.layer == 8
        assert config.alpha == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Result serialization
# ---------------------------------------------------------------------------


class TestFusionResult:
    """Tests for FusionResult.to_dict()."""

    def test_to_dict(self) -> None:
        """to_dict() should produce valid JSON-compatible dict."""
        from vauban.types import FusionGeneration, FusionResult

        result = FusionResult(
            generations=[
                FusionGeneration(
                    harmful_prompt="h",
                    benign_prompt="b",
                    output="generated text",
                    layer=12,
                    alpha=0.5,
                ),
            ],
            layer=12,
            alpha=0.5,
        )
        d = result.to_dict()
        assert d["layer"] == 12
        assert len(d["generations"]) == 1  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


class TestFusionRegistry:
    """Tests for registry integration."""

    def test_section_parse_spec_exists(self) -> None:
        """fusion should be in SECTION_PARSE_SPECS."""
        from vauban.config._registry import SECTION_PARSE_SPECS

        names = [s.section for s in SECTION_PARSE_SPECS]
        assert "fusion" in names

    def test_mode_registry_entry(self) -> None:
        """fusion should be in EARLY_MODE_SPECS."""
        from vauban.config._mode_registry import EARLY_MODE_SPECS

        modes = [s.mode for s in EARLY_MODE_SPECS]
        assert "fusion" in modes

    def test_mode_runner_exists(self) -> None:
        """fusion should be in EARLY_MODE_RUNNERS."""
        from vauban._pipeline._modes import EARLY_MODE_RUNNERS

        assert "fusion" in EARLY_MODE_RUNNERS

    def test_fusion_phase_is_before_prompts(self) -> None:
        """Fusion should run in before_prompts phase."""
        from vauban.config._mode_registry import EARLY_MODE_SPECS

        spec = next(s for s in EARLY_MODE_SPECS if s.mode == "fusion")
        assert spec.phase == "before_prompts"
        assert spec.requires_direction is False


# ---------------------------------------------------------------------------
# Runtime behavior
# ---------------------------------------------------------------------------


class TestFusionRuntime:
    """Tests for fusion runtime helpers."""

    def test_fuse_and_generate_resolves_middle_layer_and_trims_sequences(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from vauban.fusion import fuse_and_generate
        from vauban.types import FusionConfig

        tokenizer = MockTokenizer(32)
        model = cast("CausalLM", object())
        layers = [
            _AdditiveLayer(10.0),
            _AdditiveLayer(20.0),
            _AdditiveLayer(30.0),
            _AdditiveLayer(40.0),
        ]
        transformer = _FusionTransformer(layers)
        harmful_hidden = ops.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        benign_hidden = ops.array([[[9.0, 8.0], [7.0, 6.0]]])
        captured_hidden: dict[str, Array] = {}

        monkeypatch.setattr("vauban.fusion.get_transformer", lambda model: transformer)
        monkeypatch.setattr(
            "vauban.fusion._forward_to_layer",
            lambda model, tok, prompt, layer: (
                harmful_hidden if prompt == "harm" else benign_hidden
            ),
        )
        monkeypatch.setattr("vauban.fusion.make_ssm_mask", lambda transformer, h: "ssm")
        monkeypatch.setattr(
            "vauban.fusion.select_mask",
            lambda layer, mask, ssm_mask: f"{mask}-{ssm_mask}",
        )

        def _fake_generate(
            model: object,
            hidden: Array,
            n_tokens: int,
            temperature: float,
        ) -> list[int]:
            del model
            captured_hidden["value"] = hidden
            assert n_tokens == 2
            assert temperature == pytest.approx(0.0)
            return [1, 2]

        monkeypatch.setattr("vauban.fusion._generate_from_hidden", _fake_generate)

        result = fuse_and_generate(
            model,
            tokenizer,
            "harm",
            "safe",
            FusionConfig(
                harmful_prompts=["harm"],
                benign_prompts=["safe"],
                layer=-1,
                alpha=0.25,
                n_tokens=2,
                temperature=0.0,
            ),
        )

        expected_hidden = ops.array([[[177.0, 176.5], [176.0, 175.5]]])
        ops.eval(expected_hidden, captured_hidden["value"])
        assert bool(ops.allclose(captured_hidden["value"], expected_hidden))
        assert result.output == "BC"
        assert result.layer == 2
        assert layers[2].masks == ["None-ssm"]
        assert layers[3].masks == ["None-ssm"]

    def test_fuse_and_generate_rejects_out_of_range_layer(self) -> None:
        from vauban.fusion import fuse_and_generate
        from vauban.types import FusionConfig

        tokenizer = MockTokenizer(32)
        model = cast("CausalLM", object())
        transformer = _FusionTransformer([_AdditiveLayer(1.0), _AdditiveLayer(2.0)])

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                "vauban.fusion.get_transformer",
                lambda model: transformer,
            )
            with pytest.raises(ValueError, match="out of range"):
                fuse_and_generate(
                    model,
                    tokenizer,
                    "harm",
                    "safe",
                    FusionConfig(
                        harmful_prompts=["harm"],
                        benign_prompts=["safe"],
                        layer=5,
                    ),
                )

    def test_fuse_batch_uses_shorter_prompt_list_and_resolves_layer(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from vauban.fusion import fuse_batch
        from vauban.types import FusionConfig, FusionGeneration

        calls: list[tuple[str, str]] = []
        model = cast("CausalLM", object())
        tokenizer = cast("Tokenizer", MockTokenizer(32))
        transformer = _FusionTransformer(
            [_AdditiveLayer(1.0) for _ in range(6)],
        )

        def _fake_fuse_and_generate(
            model: object,
            tokenizer: object,
            harmful_prompt: str,
            benign_prompt: str,
            config: FusionConfig,
        ) -> FusionGeneration:
            del model, tokenizer, config
            calls.append((harmful_prompt, benign_prompt))
            return FusionGeneration(
                harmful_prompt=harmful_prompt,
                benign_prompt=benign_prompt,
                output=f"{harmful_prompt}->{benign_prompt}",
                layer=3,
                alpha=0.4,
            )

        monkeypatch.setattr(
            "vauban.fusion.fuse_and_generate",
            _fake_fuse_and_generate,
        )
        monkeypatch.setattr("vauban.fusion.get_transformer", lambda model: transformer)

        result = fuse_batch(
            model,
            tokenizer,
            FusionConfig(
                harmful_prompts=["h1", "h2"],
                benign_prompts=["b1"],
                layer=-1,
                alpha=0.4,
            ),
        )

        assert calls == [("h1", "b1")]
        assert result.layer == 3
        assert len(result.generations) == 1

    def test_forward_to_layer_runs_prefix_layers(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from vauban.fusion import _forward_to_layer

        model = cast("CausalLM", object())
        tokenizer = cast("Tokenizer", MockTokenizer(32))
        layers = [_AdditiveLayer(1.0), _AdditiveLayer(2.0), _AdditiveLayer(3.0)]
        transformer = _FusionTransformer(layers)
        hidden = ops.array([[[1.0], [2.0]]])

        monkeypatch.setattr("vauban.fusion.get_transformer", lambda model: transformer)
        monkeypatch.setattr(
            "vauban.fusion.encode_user_prompt",
            lambda tokenizer, prompt: [1, 2, 3],
        )
        monkeypatch.setattr(
            "vauban.fusion.embed_and_mask",
            lambda transformer, token_ids: (hidden, "mask"),
        )
        monkeypatch.setattr("vauban.fusion.make_ssm_mask", lambda transformer, h: "ssm")
        monkeypatch.setattr(
            "vauban.fusion.select_mask",
            lambda layer, mask, ssm_mask: f"{mask}-{ssm_mask}-{layer.delta}",
        )

        result = _forward_to_layer(model, tokenizer, "prompt", 2)

        expected = ops.array([[[4.0], [5.0]]])
        ops.eval(result, expected)
        assert bool(ops.allclose(result, expected))
        assert layers[0].masks == ["mask-ssm-1.0"]
        assert layers[1].masks == ["mask-ssm-2.0"]
        assert layers[2].masks == []

    def test_generate_from_hidden_greedy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from vauban.fusion import _generate_from_hidden

        model = cast("CausalLM", object())
        transformer = _EmbeddingTransformer()
        logits = iter(
            [
                ops.array([[[0.1, 0.9, 0.2]]]),
                ops.array([[[0.8, 0.1, 0.2]]]),
            ],
        )

        monkeypatch.setattr("vauban.fusion.get_transformer", lambda model: transformer)
        monkeypatch.setattr(
            "vauban.fusion.lm_head_forward",
            lambda model, hidden: next(logits),
        )

        result = _generate_from_hidden(
            model,
            ops.array([[[1.0, 2.0]]]),
            2,
            0.0,
        )

        assert result == [1, 0]
        assert transformer.embedded_ids == [1, 0]

    def test_generate_from_hidden_sampling(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from vauban.fusion import _generate_from_hidden

        model = cast("CausalLM", object())
        transformer = _EmbeddingTransformer()

        monkeypatch.setattr("vauban.fusion.get_transformer", lambda model: transformer)
        monkeypatch.setattr(
            "vauban.fusion.lm_head_forward",
            lambda model, hidden: ops.array([[[1.0, 3.0, 2.0]]]),
        )
        monkeypatch.setattr(
            "vauban.fusion.ops.random.categorical",
            lambda logits: ops.array([2]),
        )

        result = _generate_from_hidden(
            model,
            ops.array([[[1.0, 2.0]]]),
            1,
            0.5,
        )

        assert result == [2]
        assert transformer.embedded_ids == [2]
