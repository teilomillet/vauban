"""Tests for LoRA loading and analysis features."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from vauban import _ops as ops

if TYPE_CHECKING:
    from pathlib import Path

    from vauban._array import Array


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

D_MODEL = 8
IN_FEATURES = 16


def _make_direction() -> Array:
    """Create a unit direction vector."""
    d = ops.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return d / ops.linalg.norm(d)


def _save_synthetic_adapter(tmp_path: Path, rank: int = 1) -> Path:
    """Create a synthetic mlx-format adapter on disk."""
    adapter_dir = tmp_path / "test_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    ops.random.seed(42)
    tensors: dict[str, Array] = {}
    for layer_idx in range(2):
        key_a = f"layers.{layer_idx}.self_attn.o_proj.lora_a"
        key_b = f"layers.{layer_idx}.self_attn.o_proj.lora_b"
        tensors[key_a] = ops.random.normal((rank, IN_FEATURES))
        tensors[key_b] = ops.random.normal((D_MODEL, rank))

    ops.save_safetensors(str(adapter_dir / "adapters.safetensors"), tensors)

    config = {
        "lora_rank": rank,
        "lora_alpha": float(rank),
        "model_path": "test-model",
        "num_lora_layers": 2,
    }
    (adapter_dir / "adapter_config.json").write_text(json.dumps(config))

    return adapter_dir


def _save_rank1_adapter(tmp_path: Path) -> Path:
    """Create a rank-1 adapter with known structure for analysis tests."""
    adapter_dir = tmp_path / "rank1_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Rank-1: lora_a is (1, IN_FEATURES), lora_b is (D_MODEL, 1)
    lora_a = ops.ones((1, IN_FEATURES))
    lora_b = ops.ones((D_MODEL, 1))

    tensors: dict[str, Array] = {
        "layers.0.self_attn.o_proj.lora_a": lora_a,
        "layers.0.self_attn.o_proj.lora_b": lora_b,
    }

    ops.save_safetensors(str(adapter_dir / "adapters.safetensors"), tensors)

    config = {
        "lora_rank": 1,
        "lora_alpha": 1.0,
        "model_path": "test-model",
        "num_lora_layers": 1,
    }
    (adapter_dir / "adapter_config.json").write_text(json.dumps(config))

    return adapter_dir


# ---------------------------------------------------------------------------
# Config dataclass tests
# ---------------------------------------------------------------------------


class TestLoraLoadConfig:
    """Tests for LoraLoadConfig dataclass."""

    def test_defaults(self) -> None:
        """All fields should default to None."""
        from vauban.types import LoraLoadConfig

        config = LoraLoadConfig()
        assert config.adapter_path is None
        assert config.adapter_paths is None
        assert config.weights is None

    def test_frozen(self) -> None:
        """Config should be immutable."""
        from vauban.types import LoraLoadConfig

        config = LoraLoadConfig()
        with pytest.raises(AttributeError):
            config.adapter_path = "foo"  # type: ignore[misc]

    def test_single_path(self) -> None:
        """Single adapter_path should be accepted."""
        from vauban.types import LoraLoadConfig

        config = LoraLoadConfig(adapter_path="output/adapter")
        assert config.adapter_path == "output/adapter"

    def test_multi_paths(self) -> None:
        """Multiple adapter_paths with weights should be accepted."""
        from vauban.types import LoraLoadConfig

        config = LoraLoadConfig(
            adapter_paths=["a/", "b/"],
            weights=[1.0, 0.5],
        )
        assert config.adapter_paths == ["a/", "b/"]
        assert config.weights == [1.0, 0.5]


class TestLoraAnalysisConfig:
    """Tests for LoraAnalysisConfig dataclass."""

    def test_defaults(self) -> None:
        """Default values should match spec."""
        from vauban.types import LoraAnalysisConfig

        config = LoraAnalysisConfig()
        assert config.variance_threshold == 0.99
        assert config.align_with_direction is True
        assert config.adapter_path is None
        assert config.adapter_paths is None

    def test_frozen(self) -> None:
        """Config should be immutable."""
        from vauban.types import LoraAnalysisConfig

        config = LoraAnalysisConfig()
        with pytest.raises(AttributeError):
            config.variance_threshold = 0.5  # type: ignore[misc]


class TestLoraLayerAnalysis:
    """Tests for LoraLayerAnalysis dataclass."""

    def test_to_dict(self) -> None:
        """to_dict should produce a JSON-serializable dict."""
        from vauban.types import LoraLayerAnalysis

        layer = LoraLayerAnalysis(
            key="layers.0.self_attn.o_proj",
            frobenius_norm=1.5,
            singular_values=[1.0, 0.5, 0.1],
            effective_rank=1.8,
            variance_cutoff=2,
            direction_alignment=0.95,
        )
        d = layer.to_dict()
        assert d["key"] == "layers.0.self_attn.o_proj"
        assert d["direction_alignment"] == 0.95
        json.dumps(d)  # Must be serializable

    def test_to_dict_no_alignment(self) -> None:
        """to_dict should omit direction_alignment when None."""
        from vauban.types import LoraLayerAnalysis

        layer = LoraLayerAnalysis(
            key="test",
            frobenius_norm=1.0,
            singular_values=[1.0],
            effective_rank=1.0,
            variance_cutoff=1,
        )
        d = layer.to_dict()
        assert "direction_alignment" not in d


class TestLoraAnalysisResult:
    """Tests for LoraAnalysisResult dataclass."""

    def test_to_dict(self) -> None:
        """to_dict should produce a JSON-serializable dict."""
        from vauban.types import LoraAnalysisResult, LoraLayerAnalysis

        result = LoraAnalysisResult(
            adapter_path="/tmp/adapter",
            layers=[
                LoraLayerAnalysis(
                    key="test",
                    frobenius_norm=1.0,
                    singular_values=[1.0],
                    effective_rank=1.0,
                    variance_cutoff=1,
                ),
            ],
            total_params=100,
            mean_effective_rank=1.0,
            norm_profile=[1.0],
        )
        d = result.to_dict()
        assert d["total_params"] == 100
        assert len(d["layers"]) == 1  # type: ignore[arg-type]
        json.dumps(d)  # Must be serializable


class TestPipelineConfigIntegration:
    """Tests for new fields on PipelineConfig."""

    def test_lora_load_field(self) -> None:
        """PipelineConfig should accept lora_load."""
        from pathlib import Path

        from vauban.types import LoraLoadConfig, PipelineConfig

        config = PipelineConfig(
            model_path="test",
            harmful_path=Path("h.jsonl"),
            harmless_path=Path("s.jsonl"),
            lora_load=LoraLoadConfig(adapter_path="adapter/"),
        )
        assert config.lora_load is not None
        assert config.lora_load.adapter_path == "adapter/"

    def test_lora_analysis_field(self) -> None:
        """PipelineConfig should accept lora_analysis."""
        from pathlib import Path

        from vauban.types import LoraAnalysisConfig, PipelineConfig

        config = PipelineConfig(
            model_path="test",
            harmful_path=Path("h.jsonl"),
            harmless_path=Path("s.jsonl"),
            lora_analysis=LoraAnalysisConfig(adapter_path="adapter/"),
        )
        assert config.lora_analysis is not None
        assert config.lora_analysis.adapter_path == "adapter/"


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestParseLoraLoad:
    """Tests for [lora] TOML config parsing."""

    def test_parse_single_path(self) -> None:
        """Single adapter_path should parse correctly."""
        from vauban.config._parse_lora_load import _parse_lora_load

        raw = {"lora": {"adapter_path": "output/adapter"}}
        config = _parse_lora_load(raw)
        assert config is not None
        assert config.adapter_path == "output/adapter"
        assert config.adapter_paths is None

    def test_parse_multi_paths(self) -> None:
        """Multiple adapter_paths should parse correctly."""
        from vauban.config._parse_lora_load import _parse_lora_load

        raw = {
            "lora": {
                "adapter_paths": ["a/", "b/"],
                "weights": [1.0, 0.5],
            },
        }
        config = _parse_lora_load(raw)
        assert config is not None
        assert config.adapter_paths == ["a/", "b/"]
        assert config.weights == [1.0, 0.5]

    def test_parse_absent(self) -> None:
        """Absent section should return None."""
        from vauban.config._parse_lora_load import _parse_lora_load

        assert _parse_lora_load({}) is None

    def test_parse_mutual_exclusivity(self) -> None:
        """Both adapter_path and adapter_paths should raise."""
        from vauban.config._parse_lora_load import _parse_lora_load

        raw = {
            "lora": {
                "adapter_path": "a/",
                "adapter_paths": ["b/"],
            },
        }
        with pytest.raises(ValueError, match="mutually exclusive"):
            _parse_lora_load(raw)

    def test_parse_neither_path(self) -> None:
        """Neither adapter_path nor adapter_paths should raise."""
        from vauban.config._parse_lora_load import _parse_lora_load

        raw = {"lora": {}}
        with pytest.raises(ValueError, match="requires either"):
            _parse_lora_load(raw)

    def test_parse_weights_mismatch(self) -> None:
        """Weights length mismatch should raise."""
        from vauban.config._parse_lora_load import _parse_lora_load

        raw = {
            "lora": {
                "adapter_paths": ["a/", "b/"],
                "weights": [1.0],
            },
        }
        with pytest.raises(ValueError, match="must match"):
            _parse_lora_load(raw)

    def test_parse_weights_without_paths(self) -> None:
        """Weights without adapter_paths should raise."""
        from vauban.config._parse_lora_load import _parse_lora_load

        raw = {
            "lora": {
                "adapter_path": "a/",
                "weights": [1.0],
            },
        }
        with pytest.raises(ValueError, match="requires adapter_paths"):
            _parse_lora_load(raw)

    def test_parse_not_a_table(self) -> None:
        """Non-table [lora] should raise TypeError."""
        from vauban.config._parse_lora_load import _parse_lora_load

        with pytest.raises(TypeError, match="must be a table"):
            _parse_lora_load({"lora": "string"})


class TestParseLoraAnalysis:
    """Tests for [lora_analysis] TOML config parsing."""

    def test_parse_single_path(self) -> None:
        """Single adapter_path with defaults should parse."""
        from vauban.config._parse_lora_analysis import _parse_lora_analysis

        raw = {"lora_analysis": {"adapter_path": "output/adapter"}}
        config = _parse_lora_analysis(raw)
        assert config is not None
        assert config.adapter_path == "output/adapter"
        assert config.variance_threshold == 0.99
        assert config.align_with_direction is True

    def test_parse_custom_values(self) -> None:
        """Custom variance_threshold and align_with_direction should work."""
        from vauban.config._parse_lora_analysis import _parse_lora_analysis

        raw = {
            "lora_analysis": {
                "adapter_path": "a/",
                "variance_threshold": 0.95,
                "align_with_direction": False,
            },
        }
        config = _parse_lora_analysis(raw)
        assert config is not None
        assert config.variance_threshold == 0.95
        assert config.align_with_direction is False

    def test_parse_absent(self) -> None:
        """Absent section should return None."""
        from vauban.config._parse_lora_analysis import _parse_lora_analysis

        assert _parse_lora_analysis({}) is None

    def test_parse_mutual_exclusivity(self) -> None:
        """Both adapter_path and adapter_paths should raise."""
        from vauban.config._parse_lora_analysis import _parse_lora_analysis

        raw = {
            "lora_analysis": {
                "adapter_path": "a/",
                "adapter_paths": ["b/"],
            },
        }
        with pytest.raises(ValueError, match="mutually exclusive"):
            _parse_lora_analysis(raw)

    def test_parse_neither_path(self) -> None:
        """Neither adapter_path nor adapter_paths should raise."""
        from vauban.config._parse_lora_analysis import _parse_lora_analysis

        raw = {"lora_analysis": {}}
        with pytest.raises(ValueError, match="requires either"):
            _parse_lora_analysis(raw)


# ---------------------------------------------------------------------------
# analyze_adapter tests
# ---------------------------------------------------------------------------


class TestAnalyzeAdapter:
    """Tests for analyze_adapter function."""

    def test_rank1_effective_rank(self, tmp_path: Path) -> None:
        """Rank-1 adapter should have effective_rank close to 1.0."""
        from vauban.lora import analyze_adapter

        adapter_dir = _save_rank1_adapter(tmp_path)
        result = analyze_adapter(str(adapter_dir))

        assert len(result.layers) == 1
        layer = result.layers[0]
        assert layer.effective_rank == pytest.approx(1.0, abs=0.01)

    def test_frobenius_norm(self, tmp_path: Path) -> None:
        """Frobenius norm should be computed correctly for rank-1."""
        from vauban.lora import analyze_adapter

        adapter_dir = _save_rank1_adapter(tmp_path)
        result = analyze_adapter(str(adapter_dir))

        layer = result.layers[0]
        # B @ A = ones(8,1) @ ones(1,16) = ones(8,16)
        # Frobenius norm = sqrt(8*16) = sqrt(128)
        import math

        expected_norm = math.sqrt(D_MODEL * IN_FEATURES)
        assert layer.frobenius_norm == pytest.approx(expected_norm, rel=0.01)

    def test_direction_alignment(self, tmp_path: Path) -> None:
        """Direction alignment should be computed when direction is given."""
        from vauban.lora import analyze_adapter

        adapter_dir = _save_rank1_adapter(tmp_path)
        direction = _make_direction()
        result = analyze_adapter(str(adapter_dir), direction=direction)

        layer = result.layers[0]
        assert layer.direction_alignment is not None
        assert 0.0 <= layer.direction_alignment <= 1.0

    def test_no_direction(self, tmp_path: Path) -> None:
        """Without direction, alignment should be None."""
        from vauban.lora import analyze_adapter

        adapter_dir = _save_rank1_adapter(tmp_path)
        result = analyze_adapter(str(adapter_dir))

        layer = result.layers[0]
        assert layer.direction_alignment is None

    def test_multi_layer_adapter(self, tmp_path: Path) -> None:
        """Multi-layer adapter should have correct layer count."""
        from vauban.lora import analyze_adapter

        adapter_dir = _save_synthetic_adapter(tmp_path, rank=2)
        result = analyze_adapter(str(adapter_dir))

        assert len(result.layers) == 2
        assert result.total_params > 0
        assert len(result.norm_profile) == 2
        assert result.mean_effective_rank > 0.0

    def test_variance_cutoff(self, tmp_path: Path) -> None:
        """Rank-1 adapter should have variance_cutoff of 1."""
        from vauban.lora import analyze_adapter

        adapter_dir = _save_rank1_adapter(tmp_path)
        result = analyze_adapter(str(adapter_dir), variance_threshold=0.99)

        layer = result.layers[0]
        assert layer.variance_cutoff == 1

    def test_result_to_dict(self, tmp_path: Path) -> None:
        """Result should be JSON-serializable."""
        from vauban.lora import analyze_adapter

        adapter_dir = _save_rank1_adapter(tmp_path)
        result = analyze_adapter(str(adapter_dir))

        d = result.to_dict()
        json.dumps(d)  # Must not raise


# ---------------------------------------------------------------------------
# Mode registry tests
# ---------------------------------------------------------------------------


class TestModeRegistry:
    """Tests for lora_analysis mode registration."""

    def test_has_lora_analysis_predicate(self) -> None:
        """Predicate should return True when lora_analysis is set."""
        from pathlib import Path

        from vauban.config._mode_registry import _has_lora_analysis
        from vauban.types import LoraAnalysisConfig, PipelineConfig

        config = PipelineConfig(
            model_path="test",
            harmful_path=Path("h.jsonl"),
            harmless_path=Path("s.jsonl"),
            lora_analysis=LoraAnalysisConfig(adapter_path="a/"),
        )
        assert _has_lora_analysis(config) is True

    def test_has_lora_analysis_none(self) -> None:
        """Predicate should return False when lora_analysis is None."""
        from pathlib import Path

        from vauban.config._mode_registry import _has_lora_analysis
        from vauban.types import PipelineConfig

        config = PipelineConfig(
            model_path="test",
            harmful_path=Path("h.jsonl"),
            harmless_path=Path("s.jsonl"),
        )
        assert _has_lora_analysis(config) is False

    def test_early_mode_spec_registered(self) -> None:
        """lora_analysis should be in EARLY_MODE_SPECS."""
        from vauban.config._mode_registry import EARLY_MODE_SPECS

        modes = [spec.mode for spec in EARLY_MODE_SPECS]
        assert "lora_analysis" in modes

    def test_mode_runner_registered(self) -> None:
        """lora_analysis should be in EARLY_MODE_RUNNERS."""
        from vauban._pipeline._modes import EARLY_MODE_RUNNERS

        assert "lora_analysis" in EARLY_MODE_RUNNERS
