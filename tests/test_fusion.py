# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for fusion mode."""

from __future__ import annotations

import tomllib

import pytest

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
