# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for measure-only runs that should stop before cut/export."""

import json
from pathlib import Path

import pytest

import vauban
from tests.conftest import (
    D_MODEL,
    NUM_HEADS,
    NUM_LAYERS,
    VOCAB_SIZE,
    MockCausalLM,
    MockTokenizer,
)
from vauban import _ops as ops


class TestRunMeasureOnly:
    def test_diff_measure_only_skips_cut_and_export(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Diff measure-only runs should write reports and return early."""
        aligned_model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        base_model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(aligned_model.parameters())
        ops.eval(base_model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        models = {
            "aligned-model": (aligned_model, tokenizer),
            "base-model": (base_model, tokenizer),
        }

        def _fake_load_model(model_path: str) -> tuple[MockCausalLM, MockTokenizer]:
            return models[model_path]

        def _unexpected_export(
            model_path: str,
            weights: dict[str, object],
            output_dir: str | Path,
        ) -> Path:
            del model_path, weights, output_dir
            raise AssertionError("measure_only should not export a model")

        monkeypatch.setattr("vauban._model_io.load_model", _fake_load_model)
        monkeypatch.setattr("vauban.dataset.resolve_prompts", lambda _path: ["a", "b"])
        monkeypatch.setattr("vauban.export.export_model", _unexpected_export)
        monkeypatch.setattr("vauban.resolve_prompts", lambda _path: ["a", "b"])

        config_path = tmp_path / "measure_only.toml"
        output_dir = tmp_path / "out"
        config_path.write_text(
            '[model]\npath = "aligned-model"\n'
            '[data]\nharmful = "harmful.jsonl"\nharmless = "harmless.jsonl"\n'
            '[measure]\nmode = "diff"\ndiff_model = "base-model"\nmeasure_only = true\n'
            f'[output]\ndir = "{output_dir}"\n'
        )

        vauban.run(config_path)

        diff_report = output_dir / "diff_report.json"
        assert diff_report.exists()
        report = json.loads(diff_report.read_text())
        assert report["source_model"] == "base-model"
        assert report["target_model"] == "aligned-model"

        log_path = output_dir / "experiment_log.jsonl"
        assert log_path.exists()
        entries = [json.loads(line) for line in log_path.read_text().splitlines()]
        assert entries[-1]["pipeline_mode"] == "measure"
        assert entries[-1]["reports"] == ["diff_report.json"]
        assert not (output_dir / "model.safetensors").exists()
