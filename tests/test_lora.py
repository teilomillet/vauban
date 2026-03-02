"""Tests for the LoRA export module."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import pytest

from vauban import _ops as ops

if TYPE_CHECKING:
    from pathlib import Path

    from vauban._array import Array


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


class TestLoraExportConfig:
    """Tests for LoraExportConfig defaults."""

    def test_defaults(self) -> None:
        """Default values should match spec."""
        from vauban.types import LoraExportConfig

        config = LoraExportConfig()
        assert config.format == "mlx"
        assert config.polarity == "remove"

    def test_frozen(self) -> None:
        """Config should be immutable."""
        from vauban.types import LoraExportConfig

        config = LoraExportConfig()
        with pytest.raises(AttributeError):
            config.format = "peft"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TOML parsing
# ---------------------------------------------------------------------------


class TestLoraExportParse:
    """Tests for [lora_export] TOML config parsing."""

    def test_parse_minimal(self) -> None:
        """Minimal [lora_export] section should parse with defaults."""
        from vauban.config._parse_lora_export import _parse_lora_export

        raw = {"lora_export": {}}
        config = _parse_lora_export(raw)
        assert config is not None
        assert config.format == "mlx"
        assert config.polarity == "remove"

    def test_parse_peft_add(self) -> None:
        """Custom format and polarity should be respected."""
        from vauban.config._parse_lora_export import _parse_lora_export

        raw = {"lora_export": {"format": "peft", "polarity": "add"}}
        config = _parse_lora_export(raw)
        assert config is not None
        assert config.format == "peft"
        assert config.polarity == "add"

    def test_parse_absent(self) -> None:
        """Absent section should return None."""
        from vauban.config._parse_lora_export import _parse_lora_export

        assert _parse_lora_export({}) is None

    def test_parse_invalid_format(self) -> None:
        """Invalid format should raise ValueError."""
        from vauban.config._parse_lora_export import _parse_lora_export

        raw = {"lora_export": {"format": "gguf"}}
        with pytest.raises(ValueError, match="must be one of"):
            _parse_lora_export(raw)

    def test_parse_invalid_polarity(self) -> None:
        """Invalid polarity should raise ValueError."""
        from vauban.config._parse_lora_export import _parse_lora_export

        raw = {"lora_export": {"polarity": "flip"}}
        with pytest.raises(ValueError, match="must be one of"):
            _parse_lora_export(raw)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

D_MODEL = 8
IN_FEATURES = 16


def _make_direction() -> Array:
    """Create a unit direction vector."""
    d = ops.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return d / ops.linalg.norm(d)


def _make_weight() -> Array:
    """Create a synthetic weight matrix."""
    ops.random.seed(42)
    return ops.random.normal((D_MODEL, IN_FEATURES))


def _make_flat_weights() -> dict[str, Array]:
    """Create a minimal flat weight dict with target keys."""
    ops.random.seed(42)
    return {
        "model.layers.0.self_attn.o_proj.weight": ops.random.normal(
            (D_MODEL, IN_FEATURES),
        ),
        "model.layers.0.self_attn.q_proj.weight": ops.random.normal(
            (D_MODEL, IN_FEATURES),
        ),
        "model.layers.0.mlp.down_proj.weight": ops.random.normal(
            (D_MODEL, IN_FEATURES),
        ),
        "model.layers.1.self_attn.o_proj.weight": ops.random.normal(
            (D_MODEL, IN_FEATURES),
        ),
        "model.layers.1.mlp.down_proj.weight": ops.random.normal(
            (D_MODEL, IN_FEATURES),
        ),
    }


# ---------------------------------------------------------------------------
# Core conversion tests
# ---------------------------------------------------------------------------


class TestDirectionToLora:
    """Tests for direction_to_lora."""

    def test_shapes(self) -> None:
        """Output shapes should be (1, in_features) and (d_model, 1)."""
        from vauban.lora import direction_to_lora

        d = _make_direction()
        w = _make_weight()
        lora_a, lora_b = direction_to_lora(d, w)
        assert lora_a.shape == (1, IN_FEATURES)
        assert lora_b.shape == (D_MODEL, 1)

    def test_remove_polarity(self) -> None:
        """Remove polarity should negate B."""
        from vauban.lora import direction_to_lora

        d = _make_direction()
        w = _make_weight()
        a_rem, b_rem = direction_to_lora(d, w, polarity="remove")
        a_add, b_add = direction_to_lora(d, w, polarity="add")
        # A should be the same, B should be negated
        assert ops.allclose(a_rem, a_add)
        assert ops.allclose(b_rem, -b_add)

    def test_invalid_polarity(self) -> None:
        """Invalid polarity should raise ValueError."""
        from vauban.lora import direction_to_lora

        d = _make_direction()
        w = _make_weight()
        with pytest.raises(ValueError, match="polarity"):
            direction_to_lora(d, w, polarity="flip")


class TestSubspaceToLora:
    """Tests for subspace_to_lora."""

    def test_shapes(self) -> None:
        """Output shapes should be (k, in_features) and (d_model, k)."""
        from vauban.lora import subspace_to_lora

        k = 3
        basis = ops.eye(D_MODEL)[:k]  # (k, d_model)
        w = _make_weight()
        lora_a, lora_b = subspace_to_lora(basis, w)
        assert lora_a.shape == (k, IN_FEATURES)
        assert lora_b.shape == (D_MODEL, k)


# ---------------------------------------------------------------------------
# Reconstruction test: LoRA delta == cut delta
# ---------------------------------------------------------------------------


class TestReconstruction:
    """Verify that LoRA reconstruction matches the cut operation."""

    def test_rank1_reconstruction(self) -> None:
        """W + (B @ A) * (alpha / rank) should equal cut(W, d, alpha)."""
        from vauban.cut import _orthogonalize_matrix
        from vauban.lora import direction_to_lora

        d = _make_direction()
        w = _make_weight()
        alpha = 1.5

        # cut result
        w_cut = _orthogonalize_matrix(w, d, alpha)

        # LoRA reconstruction
        lora_a, lora_b = direction_to_lora(d, w, polarity="remove")
        rank = 1
        # Bake alpha into the scaling: delta_W = (B @ A) * (alpha / rank)
        delta_w = ops.matmul(lora_b, lora_a) * (alpha / rank)
        w_lora = w + delta_w

        assert ops.allclose(w_lora, w_cut, atol=1e-5)

    def test_add_polarity_reconstruction(self) -> None:
        """Add polarity: W + delta should amplify the direction."""
        from vauban.cut import _orthogonalize_matrix
        from vauban.lora import direction_to_lora

        d = _make_direction()
        w = _make_weight()
        alpha = 1.0

        # With remove, cut subtracts the direction
        w_removed = _orthogonalize_matrix(w, d, alpha)

        # With add, the sign flips: W + alpha * outer(d, d@W)
        lora_a, lora_b = direction_to_lora(d, w, polarity="add")
        rank = 1
        delta_w = ops.matmul(lora_b, lora_a) * (alpha / rank)
        w_added = w + delta_w

        # w_removed subtracts, w_added adds → difference = 2 * alpha * outer(d, d@W)
        diff = w_added - w_removed
        proj = d @ w
        expected_diff = 2 * alpha * ops.outer(d, proj)
        assert ops.allclose(diff, expected_diff, atol=1e-5)


# ---------------------------------------------------------------------------
# Build weights
# ---------------------------------------------------------------------------


class TestBuildLoraWeights:
    """Tests for build_lora_weights."""

    def test_correct_keys(self) -> None:
        """Should produce matrices for target output projection keys only."""
        from vauban.lora import build_lora_weights

        d = _make_direction()
        flat = _make_flat_weights()
        matrices = build_lora_weights(d, flat, [0], alpha=1.0)
        keys = [m.key for m in matrices]
        # Layer 0 has o_proj and down_proj as target keys
        assert "model.layers.0.self_attn.o_proj.weight" in keys
        assert "model.layers.0.mlp.down_proj.weight" in keys
        # Non-target keys should not be included
        assert "model.layers.0.self_attn.q_proj.weight" not in keys
        assert len(matrices) == 2

    def test_skip_3d_weights(self) -> None:
        """3D weights (MoE experts) should be skipped."""
        from vauban.lora import build_lora_weights

        d = _make_direction()
        flat = dict(_make_flat_weights())
        # Add a 3D weight (MoE expert)
        flat["model.layers.0.mlp.experts.down_proj.weight"] = ops.random.normal(
            (4, D_MODEL, IN_FEATURES),
        )
        matrices = build_lora_weights(d, flat, [0], alpha=1.0)
        keys = [m.key for m in matrices]
        assert "model.layers.0.mlp.experts.down_proj.weight" not in keys

    def test_layer_weights(self) -> None:
        """Per-layer alpha multipliers should be applied."""
        from vauban.lora import build_lora_weights

        d = _make_direction()
        flat = _make_flat_weights()
        # Layer 0 with weight 2.0, layer 1 with weight 0.5
        m1 = build_lora_weights(
            d, flat, [0, 1], alpha=1.0, layer_weights=[2.0, 0.5],
        )
        m2 = build_lora_weights(
            d, flat, [0, 1], alpha=2.0, layer_weights=[1.0, 0.25],
        )
        # Both should produce the same effective alpha per layer
        for a, b in zip(m1, m2, strict=True):
            assert ops.allclose(a.lora_b, b.lora_b, atol=1e-5)


# ---------------------------------------------------------------------------
# Save adapter formats
# ---------------------------------------------------------------------------


class TestSaveAdapterMlx:
    """Tests for mlx-lm format adapter saving."""

    def test_files_created(self, tmp_path: Path) -> None:
        """Should create adapters.safetensors and adapter_config.json."""
        from vauban.lora import build_lora_weights, save_adapter_mlx

        d = _make_direction()
        flat = _make_flat_weights()
        matrices = build_lora_weights(d, flat, [0], alpha=1.0)
        output_dir = tmp_path / "adapter_mlx"
        save_adapter_mlx(matrices, output_dir, rank=1, model_path="test")

        assert (output_dir / "adapters.safetensors").exists()
        assert (output_dir / "adapter_config.json").exists()

        config = json.loads((output_dir / "adapter_config.json").read_text())
        assert config["lora_rank"] == 1
        # Alpha baked into matrices → lora_alpha = rank (scaling = 1.0)
        assert config["lora_alpha"] == 1.0
        assert config["model_path"] == "test"

    def test_key_format(self) -> None:
        """mlx-lm keys should strip model. prefix and .weight suffix."""
        from vauban.lora import _mlx_lora_key

        a_key, b_key = _mlx_lora_key("model.layers.0.self_attn.o_proj.weight")
        assert a_key == "layers.0.self_attn.o_proj.lora_a"
        assert b_key == "layers.0.self_attn.o_proj.lora_b"


class TestSaveAdapterPeft:
    """Tests for PEFT format adapter saving."""

    def test_files_created(self, tmp_path: Path) -> None:
        """Should create adapter_model.safetensors and adapter_config.json."""
        from vauban.lora import build_lora_weights, save_adapter_peft

        d = _make_direction()
        flat = _make_flat_weights()
        matrices = build_lora_weights(d, flat, [0], alpha=1.0)
        output_dir = tmp_path / "adapter_peft"
        save_adapter_peft(matrices, output_dir, rank=1, model_path="test")

        assert (output_dir / "adapter_model.safetensors").exists()
        assert (output_dir / "adapter_config.json").exists()

        config = json.loads((output_dir / "adapter_config.json").read_text())
        assert config["peft_type"] == "LORA"
        assert config["r"] == 1
        assert config["lora_alpha"] == 1.0
        assert config["fan_in_fan_out"] is False
        assert config["task_type"] == "CAUSAL_LM"

    def test_key_format(self) -> None:
        """PEFT keys should prepend base_model.model. and use .lora_A.weight."""
        from vauban.lora import _peft_lora_key

        a_key, b_key = _peft_lora_key("model.layers.0.self_attn.o_proj.weight")
        assert a_key == "base_model.model.model.layers.0.self_attn.o_proj.lora_A.weight"
        assert b_key == "base_model.model.model.layers.0.self_attn.o_proj.lora_B.weight"


# ---------------------------------------------------------------------------
# Merge adapters
# ---------------------------------------------------------------------------


class TestMergeAdapters:
    """Tests for adapter merging (task arithmetic)."""

    def test_weighted_sum(self, tmp_path: Path) -> None:
        """Merging two adapters with weights should produce weighted sum."""
        from vauban.lora import build_lora_weights, merge_adapters, save_adapter_mlx

        d = _make_direction()
        flat = _make_flat_weights()
        matrices = build_lora_weights(d, flat, [0], alpha=1.0)

        # Save two copies
        dir_a = tmp_path / "adapter_a"
        dir_b = tmp_path / "adapter_b"
        save_adapter_mlx(matrices, dir_a, rank=1, model_path="test")
        save_adapter_mlx(matrices, dir_b, rank=1, model_path="test")

        # Merge with weights 0.3 and 0.7
        merged = merge_adapters([dir_a, dir_b], [0.3, 0.7])

        # Each key should be 0.3 * v + 0.7 * v = 1.0 * v
        st_path = str(dir_a / "adapters.safetensors")
        original = cast("dict[str, Array]", ops.load(st_path))
        for key in original:
            assert ops.allclose(merged[key], original[key], atol=1e-5)

    def test_mismatched_lengths(self, tmp_path: Path) -> None:
        """Mismatched path/weight lengths should raise ValueError."""
        from vauban.lora import merge_adapters

        with pytest.raises(ValueError, match="must match"):
            merge_adapters([tmp_path], [0.5, 0.5])


# ---------------------------------------------------------------------------
# Biprojected / ortho direction tests
# ---------------------------------------------------------------------------


class TestBiprojectedDirection:
    """Tests for biprojected orthogonalization in LoRA export context."""

    def test_orthogonalized_direction_is_orthogonal(self) -> None:
        """Gram-Schmidt should produce a direction orthogonal to harmless."""
        from vauban.cut import _biprojected_direction

        refusal = ops.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        refusal = refusal / ops.linalg.norm(refusal)
        harmless = ops.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        harmless = harmless / ops.linalg.norm(harmless)

        ortho = _biprojected_direction(refusal, harmless)

        # Result should be orthogonal to harmless
        dot = float(ops.sum(ortho * harmless))
        assert abs(dot) < 1e-5

        # Result should be unit-length
        norm = float(ops.linalg.norm(ortho))
        assert abs(norm - 1.0) < 1e-5

    def test_orthogonalized_lora_reconstruction(self) -> None:
        """LoRA from orthogonalized direction should match orthogonalized cut."""
        from vauban.cut import _biprojected_direction, _orthogonalize_matrix
        from vauban.lora import direction_to_lora

        refusal = ops.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        refusal = refusal / ops.linalg.norm(refusal)
        harmless = ops.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        harmless = harmless / ops.linalg.norm(harmless)

        ortho = _biprojected_direction(refusal, harmless)
        w = _make_weight()
        alpha = 1.0

        # Cut result with orthogonalized direction
        w_cut = _orthogonalize_matrix(w, ortho, alpha)

        # LoRA reconstruction with orthogonalized direction
        lora_a, lora_b = direction_to_lora(ortho, w, polarity="remove")
        delta_w = ops.matmul(lora_b, lora_a) * (alpha / 1)
        w_lora = w + delta_w

        assert ops.allclose(w_lora, w_cut, atol=1e-5)


class TestNormPreserveWarning:
    """Test that norm_preserve logs a warning via the real mode runner."""

    def test_norm_preserve_warning(self, tmp_path: Path) -> None:
        """norm_preserve=True should print a warning to stderr."""
        from io import StringIO
        from unittest.mock import patch

        from tests.conftest import (
            D_MODEL,
            NUM_HEADS,
            NUM_LAYERS,
            VOCAB_SIZE,
            MockCausalLM,
            MockTokenizer,
            make_direction_result,
            make_pipeline_config,
        )
        from vauban._pipeline._context import EarlyModeContext
        from vauban._pipeline._mode_lora_export import _run_lora_export_mode
        from vauban.types import CutConfig, LoraExportConfig

        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        dr = make_direction_result(d_model=D_MODEL)
        config = make_pipeline_config(
            tmp_path,
            cut=CutConfig(norm_preserve=True),
            lora_export=LoraExportConfig(),
            verbose=False,
        )
        ctx = EarlyModeContext(
            config_path="test.toml",
            config=config,
            model=model,
            tokenizer=MockTokenizer(VOCAB_SIZE),
            t0=0.0,
            direction_result=dr,
        )

        captured = StringIO()
        with (
            patch("sys.stderr", captured),
            patch(
                "vauban._pipeline._mode_lora_export.finish_mode_run",
            ),
            patch(
                "vauban._pipeline._mode_lora_export.write_mode_report",
                return_value=tmp_path / "report.json",
            ),
        ):
            _run_lora_export_mode(ctx)

        assert "norm_preserve is incompatible with LoRA export" in captured.getvalue()


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


class TestLoraExportResult:
    """Tests for LoraExportResult."""

    def test_to_dict(self) -> None:
        """to_dict should produce JSON-serializable output."""
        from vauban.types import LoraExportResult

        result = LoraExportResult(
            output_path="/tmp/adapter",
            format="mlx",
            polarity="remove",
            rank=1,
            n_weights=4,
            target_layers=[10, 11, 12],
        )
        d = result.to_dict()
        assert d["rank"] == 1
        assert d["n_weights"] == 4
        # Should be JSON-serializable
        json.dumps(d)
