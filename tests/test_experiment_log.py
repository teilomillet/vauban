# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for write_experiment_log in vauban._pipeline._context."""

import json
from pathlib import Path

from vauban._pipeline._context import write_experiment_log as _write_experiment_log
from vauban.types import (
    CutConfig,
    EvalConfig,
    MeasureConfig,
    PipelineConfig,
)


def _make_config(output_dir: Path) -> PipelineConfig:
    """Build a minimal PipelineConfig pointing at tmp output dir."""
    return PipelineConfig(
        model_path="test-model",
        harmful_path=Path("harmful.jsonl"),
        harmless_path=Path("harmless.jsonl"),
        cut=CutConfig(),
        measure=MeasureConfig(),
        eval=EvalConfig(),
        output_dir=output_dir,
    )


class TestWriteExperimentLog:
    def test_creates_jsonl_file(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        _write_experiment_log(
            "test.toml", config, "default",
            ["eval_report.json"], {"refusal_rate_modified": 0.1}, 42.5,
        )
        log_path = tmp_path / "experiment_log.jsonl"
        assert log_path.exists()
        entry = json.loads(log_path.read_text().strip())
        assert entry["model_path"] == "test-model"
        assert entry["pipeline_mode"] == "default"
        assert entry["reports"] == ["eval_report.json"]
        assert entry["metrics"]["refusal_rate_modified"] == 0.1
        assert entry["elapsed_seconds"] == 42.5
        assert "timestamp" in entry

    def test_appends_multiple_entries(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        _write_experiment_log(
            "a.toml", config, "probe", [], {}, 1.0,
        )
        _write_experiment_log(
            "b.toml", config, "steer", ["steer_report.json"], {}, 2.0,
        )
        log_path = tmp_path / "experiment_log.jsonl"
        lines = [
            line for line in log_path.read_text().splitlines() if line.strip()
        ]
        assert len(lines) == 2
        entry_a = json.loads(lines[0])
        entry_b = json.loads(lines[1])
        assert entry_a["pipeline_mode"] == "probe"
        assert entry_b["pipeline_mode"] == "steer"

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        config = _make_config(nested)
        _write_experiment_log(
            "x.toml", config, "depth", [], {}, 0.5,
        )
        assert (nested / "experiment_log.jsonl").exists()

    def test_never_raises(self, tmp_path: Path) -> None:
        """Should silently swallow errors (e.g. read-only path)."""
        config = _make_config(Path("/nonexistent/readonly/dir"))
        # Should not raise
        _write_experiment_log(
            "test.toml", config, "default", [], {}, 0.0,
        )
