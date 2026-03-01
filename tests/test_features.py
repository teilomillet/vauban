"""Tests for sparse autoencoder feature decomposition."""

from __future__ import annotations

import tempfile
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from vauban import _ops as ops

if TYPE_CHECKING:
    from vauban._array import Array

D_MODEL = 16
D_SAE = 32


# ---------------------------------------------------------------------------
# SparseAutoencoder unit tests
# ---------------------------------------------------------------------------


class TestSparseAutoencoder:
    """Tests for the SparseAutoencoder class."""

    def test_encode_shape(self) -> None:
        """Encode should produce (n, d_sae) from (n, d_model)."""
        from vauban.features import SparseAutoencoder

        sae = SparseAutoencoder(D_MODEL, D_SAE)
        x = ops.random.normal((4, D_MODEL))
        ops.eval(x)
        codes = sae.encode(x)
        ops.eval(codes)
        assert codes.shape == (4, D_SAE)

    def test_encode_nonnegative(self) -> None:
        """Encoded values should be non-negative (ReLU)."""
        from vauban.features import SparseAutoencoder

        sae = SparseAutoencoder(D_MODEL, D_SAE)
        x = ops.random.normal((8, D_MODEL))
        ops.eval(x)
        codes = sae.encode(x)
        ops.eval(codes)
        # Verify all values non-negative (ReLU output)
        flat = codes.reshape(-1)
        min_val = float(flat[ops.argmax(-flat)].item())
        assert min_val >= 0.0

    def test_decode_shape(self) -> None:
        """Decode should produce (..., d_model) from (..., d_sae)."""
        from vauban.features import SparseAutoencoder

        sae = SparseAutoencoder(D_MODEL, D_SAE)
        codes = ops.random.normal((4, D_SAE))
        ops.eval(codes)
        reconstructed = sae.decode(codes)
        ops.eval(reconstructed)
        assert reconstructed.shape == (4, D_MODEL)

    def test_forward_roundtrip(self) -> None:
        """Forward should produce both reconstructed and codes."""
        from vauban.features import SparseAutoencoder

        sae = SparseAutoencoder(D_MODEL, D_SAE)
        x = ops.random.normal((4, D_MODEL))
        ops.eval(x)
        reconstructed, codes = sae.forward(x)
        ops.eval(reconstructed, codes)
        assert reconstructed.shape == x.shape
        assert codes.shape == (4, D_SAE)

    def test_parameters_and_set(self) -> None:
        """parameters() and set_parameters() should roundtrip."""
        from vauban.features import SparseAutoencoder

        sae = SparseAutoencoder(D_MODEL, D_SAE)
        params = sae.parameters()
        assert len(params) == 4
        # Modify and set back
        sae.set_parameters(params)
        new_params = sae.parameters()
        for p, np_ in zip(params, new_params, strict=True):
            ops.eval(p, np_)
            assert ops.array_equal(p, np_)


# ---------------------------------------------------------------------------
# SAE training
# ---------------------------------------------------------------------------


class TestSAETraining:
    """Tests for SAE training loop."""

    def test_loss_decreases(self) -> None:
        """Training loss should generally decrease over epochs."""
        from vauban.features import SparseAutoencoder, train_sae

        sae = SparseAutoencoder(D_MODEL, D_SAE)
        activations = ops.random.normal((32, D_MODEL))
        ops.eval(activations)

        result = train_sae(
            sae,
            activations,
            l1_coeff=1e-3,
            n_epochs=3,
            learning_rate=1e-2,
            batch_size=16,
        )
        assert len(result.loss_history) > 0
        # First loss should be higher than last (on average)
        first_loss = result.loss_history[0]
        last_loss = result.final_loss
        assert last_loss <= first_loss * 1.5, (
            f"Loss should not increase dramatically: "
            f"first={first_loss:.4f}, last={last_loss:.4f}"
        )

    def test_dead_feature_counting(self) -> None:
        """Dead features should be counted correctly."""
        from vauban.features import SparseAutoencoder, train_sae

        sae = SparseAutoencoder(D_MODEL, D_SAE)
        activations = ops.random.normal((16, D_MODEL))
        ops.eval(activations)

        result = train_sae(
            sae,
            activations,
            n_epochs=1,
            batch_size=16,
        )
        assert result.n_dead_features >= 0
        assert result.n_active_features >= 0
        assert result.n_dead_features + result.n_active_features == D_SAE


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------


class TestSAESaveLoad:
    """Tests for SAE safetensors save/load roundtrip."""

    def test_save_load_roundtrip(self) -> None:
        """SAE parameters should survive save/load roundtrip."""
        from vauban.features import SparseAutoencoder, load_sae, save_sae

        sae = SparseAutoencoder(D_MODEL, D_SAE)
        x = ops.random.normal((4, D_MODEL))
        ops.eval(x)
        orig_codes = sae.encode(x)
        ops.eval(orig_codes)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_sae.safetensors"
            save_sae(sae, path)
            assert path.exists()

            loaded = load_sae(path, D_MODEL, D_SAE)
            loaded_codes = loaded.encode(x)
            ops.eval(loaded_codes)

            # Parameters should match
            for p_orig, p_loaded in zip(
                sae.parameters(), loaded.parameters(), strict=True,
            ):
                ops.eval(p_orig, p_loaded)
                assert ops.allclose(p_orig, p_loaded), (
                    "Loaded parameters should match original"
                )


# ---------------------------------------------------------------------------
# Direction alignment
# ---------------------------------------------------------------------------


class TestDirectionAlignment:
    """Tests for cross-lens direction alignment."""

    def test_alignment_shape(self) -> None:
        """Alignment should return one value per feature."""
        from vauban.features import SparseAutoencoder, feature_direction_alignment

        sae = SparseAutoencoder(D_MODEL, D_SAE)
        direction = ops.random.normal((D_MODEL,))
        direction = direction / ops.linalg.norm(direction)
        ops.eval(direction)

        alignment = feature_direction_alignment(sae, direction)
        assert len(alignment) == D_SAE

    def test_alignment_range(self) -> None:
        """Cosine similarities should be in [-1, 1]."""
        from vauban.features import SparseAutoencoder, feature_direction_alignment

        sae = SparseAutoencoder(D_MODEL, D_SAE)
        direction = ops.random.normal((D_MODEL,))
        direction = direction / ops.linalg.norm(direction)
        ops.eval(direction)

        alignment = feature_direction_alignment(sae, direction)
        for cos_sim in alignment:
            assert -1.1 <= cos_sim <= 1.1, (
                f"Cosine similarity out of range: {cos_sim}"
            )

    def test_aligned_feature(self) -> None:
        """A feature perfectly aligned with the direction should have cos_sim ~1."""
        from vauban.features import SparseAutoencoder, feature_direction_alignment

        sae = SparseAutoencoder(D_MODEL, D_SAE)
        direction = ops.random.normal((D_MODEL,))
        direction = direction / ops.linalg.norm(direction)
        ops.eval(direction)

        # Force first decoder row to match direction
        new_w_dec = sae.w_dec
        # Overwrite row 0
        row_0 = direction * 2.0  # scaled version
        ops.eval(row_0)
        rows: list[Array] = [row_0]
        for i in range(1, D_SAE):
            rows.append(new_w_dec[i])
        sae.w_dec = ops.stack(rows)
        ops.eval(sae.w_dec)

        alignment = feature_direction_alignment(sae, direction)
        assert alignment[0] == pytest.approx(1.0, abs=0.05), (
            f"Aligned feature should have cos_sim ~1, got {alignment[0]}"
        )


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


class TestFeaturesConfigParse:
    """Tests for [features] TOML config parsing."""

    def test_parse_minimal(self, tmp_path: Path) -> None:
        """Minimal [features] section should parse correctly."""
        from vauban.config._parse_features import _parse_features

        raw = {
            "features": {
                "prompts_path": "prompts.jsonl",
                "layers": [0, 1],
            },
        }
        config = _parse_features(tmp_path, raw)
        assert config is not None
        assert config.layers == [0, 1]
        assert config.d_sae == 2048
        assert config.l1_coeff == pytest.approx(1e-3)
        assert config.n_epochs == 5

    def test_parse_full(self, tmp_path: Path) -> None:
        """Full [features] section with all options."""
        from vauban.config._parse_features import _parse_features

        raw = {
            "features": {
                "prompts_path": "/abs/path/prompts.jsonl",
                "layers": [5, 10, 15],
                "d_sae": 4096,
                "l1_coeff": 0.01,
                "n_epochs": 10,
                "learning_rate": 0.0005,
                "batch_size": 64,
                "token_position": -2,
                "dead_feature_threshold": 1e-5,
            },
        }
        config = _parse_features(tmp_path, raw)
        assert config is not None
        assert config.d_sae == 4096
        assert config.batch_size == 64
        assert str(config.prompts_path) == "/abs/path/prompts.jsonl"

    def test_parse_absent(self, tmp_path: Path) -> None:
        """Missing [features] section should return None."""
        from vauban.config._parse_features import _parse_features

        assert _parse_features(tmp_path, {}) is None

    def test_parse_missing_required(self, tmp_path: Path) -> None:
        """Missing required field should raise."""
        from vauban.config._parse_features import _parse_features

        raw = {
            "features": {
                "layers": [0],
                # missing prompts_path
            },
        }
        with pytest.raises(ValueError, match="prompts_path"):
            _parse_features(tmp_path, raw)

    def test_roundtrip_toml(self, tmp_path: Path) -> None:
        """Parse from actual TOML string."""
        from vauban.config._parse_features import _parse_features

        toml_str = """
[features]
prompts_path = "data/prompts.jsonl"
layers = [0, 5, 10]
d_sae = 1024
l1_coeff = 0.005
"""
        raw = tomllib.loads(toml_str)
        config = _parse_features(tmp_path, raw)
        assert config is not None
        assert config.d_sae == 1024
        assert config.l1_coeff == pytest.approx(0.005)


# ---------------------------------------------------------------------------
# Result serialization
# ---------------------------------------------------------------------------


class TestFeaturesResultSerialization:
    """Tests for FeaturesResult.to_dict()."""

    def test_to_dict(self) -> None:
        """FeaturesResult.to_dict() should produce valid JSON-compatible dict."""
        from vauban.types import FeaturesResult, SAELayerResult

        result = FeaturesResult(
            layers=[
                SAELayerResult(
                    layer=0, final_loss=0.5,
                    loss_history=[1.0, 0.7, 0.5],
                    n_dead_features=10, n_active_features=2038,
                ),
            ],
            d_model=768,
            d_sae=2048,
            model_path="test",
            direction_alignment=[[0.1, 0.2]],
        )
        d = result.to_dict()
        assert d["d_model"] == 768
        assert d["d_sae"] == 2048
        assert len(d["layers"]) == 1  # type: ignore[arg-type]
        assert d["direction_alignment"] is not None
