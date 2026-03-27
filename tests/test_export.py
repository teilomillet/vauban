# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for vauban.export: model directory export."""

import json
from pathlib import Path
from unittest.mock import patch

from vauban import _ops as ops
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
        dummy_weights = {"dummy.weight": ops.zeros((4, 4))}
        ops.save_safetensors(str(source_dir / "model.safetensors"), dummy_weights)

        output_dir = tmp_path / "output"
        test_weights = {"test.weight": ops.ones((4, 4))}

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
        loaded = ops.load(str(output_dir / "model.safetensors"))
        assert "test.weight" in loaded
        assert "dummy.weight" not in loaded

    def test_source_safetensors_not_copied(self, tmp_path: Path) -> None:
        """Safetensors files from source are explicitly excluded."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "config.json").write_text("{}")
        # Multiple safetensors shards
        ops.save_safetensors(
            str(source_dir / "model-00001-of-00002.safetensors"),
            {"a": ops.zeros((2, 2))},
        )
        ops.save_safetensors(
            str(source_dir / "model-00002-of-00002.safetensors"),
            {"b": ops.zeros((2, 2))},
        )

        output_dir = tmp_path / "output"

        with patch(
            "vauban.export._resolve_source_dir", return_value=source_dir,
        ):
            export_model("fake-model", {"w": ops.ones((2, 2))}, output_dir)

        # Only our model.safetensors should exist, not the shards
        assert (output_dir / "model.safetensors").exists()
        assert not (output_dir / "model-00001-of-00002.safetensors").exists()
        assert not (output_dir / "model-00002-of-00002.safetensors").exists()
