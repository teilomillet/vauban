"""Tests for linear probe mode."""

from __future__ import annotations

import tomllib

import pytest

from vauban import _ops as ops

# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


class TestLinearProbeConfig:
    """Tests for LinearProbeConfig defaults."""

    def test_defaults(self) -> None:
        """Default values should match spec."""
        from vauban.types import LinearProbeConfig

        config = LinearProbeConfig(layers=[0, 1])
        assert config.n_epochs == 20
        assert config.learning_rate == pytest.approx(1e-2)
        assert config.batch_size == 32
        assert config.token_position == -1
        assert config.regularization == pytest.approx(1e-4)

    def test_frozen(self) -> None:
        """Config should be immutable."""
        from vauban.types import LinearProbeConfig

        config = LinearProbeConfig(layers=[0])
        with pytest.raises(AttributeError):
            config.n_epochs = 10  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TOML parsing
# ---------------------------------------------------------------------------


class TestLinearProbeParse:
    """Tests for [linear_probe] TOML config parsing."""

    def test_parse_minimal(self) -> None:
        """Minimal [linear_probe] section should parse correctly."""
        from vauban.config._parse_linear_probe import _parse_linear_probe

        raw = {"linear_probe": {"layers": [0, 5, 10]}}
        config = _parse_linear_probe(raw)
        assert config is not None
        assert config.layers == [0, 5, 10]
        assert config.n_epochs == 20

    def test_parse_full(self) -> None:
        """Full [linear_probe] section with all options."""
        from vauban.config._parse_linear_probe import _parse_linear_probe

        raw = {
            "linear_probe": {
                "layers": [8, 12],
                "n_epochs": 10,
                "learning_rate": 0.05,
                "batch_size": 16,
                "token_position": -2,
                "regularization": 0.001,
            },
        }
        config = _parse_linear_probe(raw)
        assert config is not None
        assert config.n_epochs == 10
        assert config.batch_size == 16
        assert config.regularization == pytest.approx(0.001)

    def test_parse_absent(self) -> None:
        """Missing [linear_probe] section should return None."""
        from vauban.config._parse_linear_probe import _parse_linear_probe

        assert _parse_linear_probe({}) is None

    def test_parse_missing_layers(self) -> None:
        """Missing required 'layers' should raise."""
        from vauban.config._parse_linear_probe import _parse_linear_probe

        with pytest.raises(ValueError, match="layers"):
            _parse_linear_probe({"linear_probe": {}})

    def test_parse_empty_layers(self) -> None:
        """Empty layers list should raise."""
        from vauban.config._parse_linear_probe import _parse_linear_probe

        with pytest.raises(ValueError, match="non-empty"):
            _parse_linear_probe({"linear_probe": {"layers": []}})

    def test_parse_negative_layer(self) -> None:
        """Negative layer index should raise."""
        from vauban.config._parse_linear_probe import _parse_linear_probe

        with pytest.raises(ValueError, match=">= 0"):
            _parse_linear_probe({"linear_probe": {"layers": [-1]}})

    def test_roundtrip_toml(self) -> None:
        """Parse from actual TOML string."""
        from vauban.config._parse_linear_probe import _parse_linear_probe

        toml_str = """
[linear_probe]
layers = [8, 12, 16]
n_epochs = 10
learning_rate = 0.05
"""
        raw = tomllib.loads(toml_str)
        config = _parse_linear_probe(raw)
        assert config is not None
        assert config.layers == [8, 12, 16]
        assert config.n_epochs == 10


# ---------------------------------------------------------------------------
# Result serialization
# ---------------------------------------------------------------------------


class TestLinearProbeResult:
    """Tests for LinearProbeResult.to_dict()."""

    def test_to_dict(self) -> None:
        """to_dict() should produce valid JSON-compatible dict."""
        from vauban.types import LinearProbeLayerResult, LinearProbeResult

        result = LinearProbeResult(
            layers=[
                LinearProbeLayerResult(
                    layer=5, accuracy=0.85,
                    loss=0.3, loss_history=[0.7, 0.5, 0.3],
                ),
            ],
            d_model=768,
            model_path="test",
        )
        d = result.to_dict()
        assert d["d_model"] == 768
        assert len(d["layers"]) == 1  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


class TestLinearProbeRegistry:
    """Tests for registry integration."""

    def test_section_parse_spec_exists(self) -> None:
        """linear_probe should be in SECTION_PARSE_SPECS."""
        from vauban.config._registry import SECTION_PARSE_SPECS

        names = [s.section for s in SECTION_PARSE_SPECS]
        assert "linear_probe" in names

    def test_mode_registry_entry(self) -> None:
        """linear_probe should be in EARLY_MODE_SPECS."""
        from vauban.config._mode_registry import EARLY_MODE_SPECS

        modes = [s.mode for s in EARLY_MODE_SPECS]
        assert "linear_probe" in modes

    def test_mode_runner_exists(self) -> None:
        """linear_probe should be in EARLY_MODE_RUNNERS."""
        from vauban._pipeline._modes import EARLY_MODE_RUNNERS

        assert "linear_probe" in EARLY_MODE_RUNNERS


# ---------------------------------------------------------------------------
# Core logic (lightweight, no model)
# ---------------------------------------------------------------------------


class TestTrainSingleProbe:
    """Tests for the single-probe training loop."""

    def test_basic_training(self) -> None:
        """Probe should train on separable data and achieve decent accuracy."""
        from vauban.linear_probe import _train_single_probe

        # Create linearly separable data
        n_per_class = 16
        d = 8
        harmful = ops.random.normal((n_per_class, d)) + 2.0
        harmless = ops.random.normal((n_per_class, d)) - 2.0
        ops.eval(harmful, harmless)

        activations = ops.concatenate([harmful, harmless], axis=0)
        labels = ops.array(
            [1.0] * n_per_class + [0.0] * n_per_class,
        )
        ops.eval(activations, labels)

        accuracy, final_loss, loss_history = _train_single_probe(
            activations,
            labels,
            n_epochs=10,
            learning_rate=0.1,
            batch_size=8,
            regularization=1e-4,
        )
        assert accuracy >= 0.5, f"Accuracy too low: {accuracy}"
        assert len(loss_history) == 10
        assert final_loss < loss_history[0] * 2.0
