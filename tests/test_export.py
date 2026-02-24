"""Tests for vauban.export: model directory export."""

import json
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx

from vauban.export import export_model


class TestExportModel:
    def test_copies_non_safetensors_files(self, tmp_path: Path) -> None:
        """Config/tokenizer files are copied, safetensors files are not."""
        # Create a fake source model directory
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "config.json").write_text(json.dumps({"model_type": "test"}))
        (source_dir / "tokenizer.json").write_text(json.dumps({"version": "1.0"}))
        (source_dir / "generation_config.json").write_text(json.dumps({}))
        # Write a dummy safetensors file that should NOT be copied
        dummy_weights = {"dummy.weight": mx.zeros((4, 4))}
        mx.save_safetensors(str(source_dir / "model.safetensors"), dummy_weights)

        output_dir = tmp_path / "output"
        test_weights = {"test.weight": mx.ones((4, 4))}

        with patch(
            "vauban.export._resolve_source_dir", return_value=source_dir,
        ):
            result = export_model("fake-model", test_weights, output_dir)

        assert result == output_dir

        # Non-safetensors files should be copied
        assert (output_dir / "config.json").exists()
        assert (output_dir / "tokenizer.json").exists()
        assert (output_dir / "generation_config.json").exists()

        # A model.safetensors should exist (our new weights, not the original)
        assert (output_dir / "model.safetensors").exists()

        # Verify the weights are our new ones, not the originals
        loaded = mx.load(str(output_dir / "model.safetensors"))
        assert "test.weight" in loaded
        assert "dummy.weight" not in loaded

    def test_source_safetensors_not_copied(self, tmp_path: Path) -> None:
        """Safetensors files from source are explicitly excluded."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "config.json").write_text("{}")
        # Multiple safetensors shards
        mx.save_safetensors(
            str(source_dir / "model-00001-of-00002.safetensors"),
            {"a": mx.zeros((2, 2))},
        )
        mx.save_safetensors(
            str(source_dir / "model-00002-of-00002.safetensors"),
            {"b": mx.zeros((2, 2))},
        )

        output_dir = tmp_path / "output"

        with patch(
            "vauban.export._resolve_source_dir", return_value=source_dir,
        ):
            export_model("fake-model", {"w": mx.ones((2, 2))}, output_dir)

        # Only our model.safetensors should exist, not the shards
        assert (output_dir / "model.safetensors").exists()
        assert not (output_dir / "model-00001-of-00002.safetensors").exists()
        assert not (output_dir / "model-00002-of-00002.safetensors").exists()
