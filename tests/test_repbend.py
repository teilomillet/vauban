"""Tests for repbend mode."""

from __future__ import annotations

import tomllib

import pytest

from vauban import _ops as ops

# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


class TestRepBendConfig:
    """Tests for RepBendConfig defaults."""

    def test_defaults(self) -> None:
        """Default values should match spec."""
        from vauban.types import RepBendConfig

        config = RepBendConfig(layers=[0, 1])
        assert config.n_epochs == 3
        assert config.learning_rate == pytest.approx(1e-5)
        assert config.batch_size == 8
        assert config.separation_coeff == pytest.approx(1.0)
        assert config.token_position == -1

    def test_frozen(self) -> None:
        """Config should be immutable."""
        from vauban.types import RepBendConfig

        config = RepBendConfig(layers=[0])
        with pytest.raises(AttributeError):
            config.n_epochs = 10  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TOML parsing
# ---------------------------------------------------------------------------


class TestRepBendParse:
    """Tests for [repbend] TOML config parsing."""

    def test_parse_minimal(self) -> None:
        """Minimal [repbend] section should parse correctly."""
        from vauban.config._parse_repbend import _parse_repbend

        raw = {"repbend": {"layers": [5, 10]}}
        config = _parse_repbend(raw)
        assert config is not None
        assert config.layers == [5, 10]
        assert config.n_epochs == 3

    def test_parse_full(self) -> None:
        """Full [repbend] section with all options."""
        from vauban.config._parse_repbend import _parse_repbend

        raw = {
            "repbend": {
                "layers": [8, 12],
                "n_epochs": 5,
                "learning_rate": 0.0001,
                "batch_size": 4,
                "separation_coeff": 2.0,
                "token_position": -2,
            },
        }
        config = _parse_repbend(raw)
        assert config is not None
        assert config.n_epochs == 5
        assert config.separation_coeff == pytest.approx(2.0)

    def test_parse_absent(self) -> None:
        """Missing [repbend] section should return None."""
        from vauban.config._parse_repbend import _parse_repbend

        assert _parse_repbend({}) is None

    def test_parse_missing_layers(self) -> None:
        """Missing required 'layers' should raise."""
        from vauban.config._parse_repbend import _parse_repbend

        with pytest.raises(ValueError, match="layers"):
            _parse_repbend({"repbend": {}})

    def test_parse_empty_layers(self) -> None:
        """Empty layers list should raise."""
        from vauban.config._parse_repbend import _parse_repbend

        with pytest.raises(ValueError, match="non-empty"):
            _parse_repbend({"repbend": {"layers": []}})

    def test_parse_negative_layer(self) -> None:
        """Negative layer index should raise."""
        from vauban.config._parse_repbend import _parse_repbend

        with pytest.raises(ValueError, match=">= 0"):
            _parse_repbend({"repbend": {"layers": [-1]}})

    def test_parse_zero_separation_coeff(self) -> None:
        """Zero separation_coeff should raise."""
        from vauban.config._parse_repbend import _parse_repbend

        with pytest.raises(ValueError, match="separation_coeff"):
            _parse_repbend(
                {"repbend": {"layers": [0], "separation_coeff": 0.0}},
            )

    def test_roundtrip_toml(self) -> None:
        """Parse from actual TOML string."""
        from vauban.config._parse_repbend import _parse_repbend

        toml_str = """
[repbend]
layers = [8, 12, 16]
n_epochs = 5
separation_coeff = 1.5
"""
        raw = tomllib.loads(toml_str)
        config = _parse_repbend(raw)
        assert config is not None
        assert config.layers == [8, 12, 16]
        assert config.separation_coeff == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# Result serialization
# ---------------------------------------------------------------------------


class TestRepBendResult:
    """Tests for RepBendResult.to_dict()."""

    def test_to_dict(self) -> None:
        """to_dict() should produce valid JSON-compatible dict."""
        from vauban.types import RepBendResult

        result = RepBendResult(
            initial_separation=0.1,
            final_separation=0.8,
            loss_history=[0.9, 0.5, 0.2],
            layers=[5, 10],
            model_path="test",
        )
        d = result.to_dict()
        assert d["initial_separation"] == pytest.approx(0.1)
        assert d["final_separation"] == pytest.approx(0.8)
        assert len(d["loss_history"]) == 3  # type: ignore[arg-type]
        assert d["layers"] == [5, 10]


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


class TestRepBendRegistry:
    """Tests for registry integration."""

    def test_section_parse_spec_exists(self) -> None:
        """repbend should be in SECTION_PARSE_SPECS."""
        from vauban.config._registry import SECTION_PARSE_SPECS

        names = [s.section for s in SECTION_PARSE_SPECS]
        assert "repbend" in names

    def test_mode_registry_entry(self) -> None:
        """repbend should be in EARLY_MODE_SPECS."""
        from vauban.config._mode_registry import EARLY_MODE_SPECS

        modes = [s.mode for s in EARLY_MODE_SPECS]
        assert "repbend" in modes

    def test_mode_runner_exists(self) -> None:
        """repbend should be in EARLY_MODE_RUNNERS."""
        from vauban._pipeline._modes import EARLY_MODE_RUNNERS

        assert "repbend" in EARLY_MODE_RUNNERS

    def test_repbend_requires_direction(self) -> None:
        """RepBend should require direction (defense dual of abliteration)."""
        from vauban.config._mode_registry import EARLY_MODE_SPECS

        spec = next(s for s in EARLY_MODE_SPECS if s.mode == "repbend")
        assert spec.phase == "after_measure"
        assert spec.requires_direction is True


# ---------------------------------------------------------------------------
# Cosine similarity helper
# ---------------------------------------------------------------------------


class TestCosineHelper:
    """Tests for the cosine similarity helper."""

    def test_identical_vectors(self) -> None:
        """Cosine similarity of identical vectors should be ~1."""
        from vauban.repbend import _cosine_similarity

        a = ops.array([1.0, 0.0, 0.0])
        b = ops.array([1.0, 0.0, 0.0])
        ops.eval(a, b)
        sim = _cosine_similarity(a, b)
        ops.eval(sim)
        assert float(sim.item()) == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors(self) -> None:
        """Cosine similarity of orthogonal vectors should be ~0."""
        from vauban.repbend import _cosine_similarity

        a = ops.array([1.0, 0.0])
        b = ops.array([0.0, 1.0])
        ops.eval(a, b)
        sim = _cosine_similarity(a, b)
        ops.eval(sim)
        assert float(sim.item()) == pytest.approx(0.0, abs=1e-5)
