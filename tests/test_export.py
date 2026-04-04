# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for vauban.export: model directory export."""

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

import vauban.export as export_module
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


class TestResolveSourceDir:
    def test_returns_local_directory_as_is(self, tmp_path: Path) -> None:
        """Local model paths should bypass any download helper."""
        assert export_module._resolve_source_dir(str(tmp_path)) == tmp_path

    def test_uses_mlx_download_for_remote_models(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """MLX backends should resolve remote models through mlx-lm."""
        monkeypatch.setattr(export_module, "_BACKEND", "mlx")
        monkeypatch.setattr("mlx_lm.utils._download", lambda model_path: tmp_path)

        assert export_module._resolve_source_dir("mlx-community/test") == tmp_path

    def test_uses_huggingface_download_for_torch_backend(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Torch backends should resolve remote models through HF Hub."""
        monkeypatch.setattr(export_module, "_BACKEND", "torch")
        monkeypatch.setattr(
            "huggingface_hub.snapshot_download",
            lambda model_path: str(tmp_path),
        )

        assert export_module._resolve_source_dir("hf/test-model") == tmp_path


class TestExportBackendBranches:
    def test_torch_save_weights_branch(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Reloading under the torch backend should bind torch safetensors."""
        module = export_module
        backend_module = importlib.import_module("vauban._backend")
        original_backend = backend_module.get_backend()
        calls: list[tuple[dict[str, object], str]] = []

        fake_safetensors = ModuleType("safetensors.torch")
        fake_torch = ModuleType("torch")
        fake_torch.Tensor = object  # type: ignore[attr-defined]

        def _fake_save_file(weights: dict[str, object], path: str) -> None:
            calls.append((weights, path))

        fake_safetensors.save_file = _fake_save_file  # type: ignore[attr-defined]

        try:
            monkeypatch.setattr(backend_module, "get_backend", lambda: "torch")
            with patch.dict(
                sys.modules,
                {
                    "safetensors.torch": fake_safetensors,
                    "torch": fake_torch,
                },
            ):
                reloaded = importlib.reload(module)
                target = tmp_path / "model.safetensors"
                weights = {"w": ops.ones((1, 1))}
                reloaded._save_weights(target, weights)
        finally:
            monkeypatch.setattr(
                backend_module,
                "get_backend",
                lambda: original_backend,
            )
            importlib.reload(module)

        assert len(calls) == 1
        assert "w" in calls[0][0]
        assert calls[0][1] == str(tmp_path / "model.safetensors")

    def test_unknown_backend_raises_during_reload(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Unsupported backends should fail fast at import time."""
        module = export_module
        backend_module = importlib.import_module("vauban._backend")
        original_backend = backend_module.get_backend()

        try:
            monkeypatch.setattr(backend_module, "get_backend", lambda: "bogus")
            with pytest.raises(ValueError, match="Unknown backend"):
                importlib.reload(module)
        finally:
            monkeypatch.setattr(
                backend_module,
                "get_backend",
                lambda: original_backend,
            )
            importlib.reload(module)
