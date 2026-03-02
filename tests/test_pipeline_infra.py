"""Tests for pipeline infrastructure: RunState, EarlyModeContext, log()."""

from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.conftest import make_direction_result, make_pipeline_config
from vauban._pipeline._context import EarlyModeContext, log, write_experiment_log
from vauban._pipeline._run_state import RunState

# ===================================================================
# RunState
# ===================================================================


class TestRunStateConstruction:
    """RunState can be constructed with minimal args."""

    def test_minimal_construction(self, tmp_path: Path) -> None:
        config = make_pipeline_config(tmp_path)
        state = RunState(
            config_path="test.toml",
            config=config,
            model=object(),
            tokenizer=object(),
            t0=100.0,
            verbose=False,
        )
        assert state.config_path == "test.toml"
        assert state.direction_result is None
        assert state.harmful is None
        assert state.harmless is None
        assert state.cosine_scores == []
        assert state.report_files == []
        assert state.metrics == {}

    def test_full_construction(self, tmp_path: Path) -> None:
        config = make_pipeline_config(tmp_path)
        dr = make_direction_result(cosine_scores=[0.5, 0.6])
        state = RunState(
            config_path="test.toml",
            config=config,
            model=object(),
            tokenizer=object(),
            t0=100.0,
            verbose=True,
            harmful=["bad prompt"],
            harmless=["good prompt"],
            direction_result=dr,
        )
        assert state.harmful == ["bad prompt"]
        assert state.harmless == ["good prompt"]
        assert state.direction_result is dr


class TestRunStateElapsed:
    """elapsed() computes wall-clock time using monotonic."""

    def test_elapsed_returns_positive(self, tmp_path: Path) -> None:
        config = make_pipeline_config(tmp_path)
        state = RunState(
            config_path="test.toml",
            config=config,
            model=object(),
            tokenizer=object(),
            t0=100.0,
            verbose=False,
        )
        with patch("vauban._pipeline._run_state.time") as mock_time:
            mock_time.monotonic.return_value = 105.5
            assert state.elapsed() == pytest.approx(5.5)

    def test_elapsed_zero_when_just_started(self, tmp_path: Path) -> None:
        config = make_pipeline_config(tmp_path)
        state = RunState(
            config_path="test.toml",
            config=config,
            model=object(),
            tokenizer=object(),
            t0=200.0,
            verbose=False,
        )
        with patch("vauban._pipeline._run_state.time") as mock_time:
            mock_time.monotonic.return_value = 200.0
            assert state.elapsed() == pytest.approx(0.0)


class TestRunStateEarlyModeContext:
    """early_mode_context() propagates all relevant fields."""

    def test_propagates_all_fields(self, tmp_path: Path) -> None:
        config = make_pipeline_config(tmp_path)
        dr = make_direction_result(cosine_scores=[0.5, 0.6])
        state = RunState(
            config_path="test.toml",
            config=config,
            model=object(),
            tokenizer=object(),
            t0=42.0,
            verbose=True,
            harmful=["h1"],
            harmless=["hl1"],
            direction_result=dr,
        )
        ctx = state.early_mode_context()
        assert isinstance(ctx, EarlyModeContext)
        assert ctx.config_path == "test.toml"
        assert ctx.config is config
        assert ctx.model is state.model
        assert ctx.tokenizer is state.tokenizer
        assert ctx.t0 == 42.0
        assert ctx.harmful == ["h1"]
        assert ctx.harmless == ["hl1"]
        assert ctx.direction_result is dr

    def test_propagates_none_optionals(self, tmp_path: Path) -> None:
        config = make_pipeline_config(tmp_path)
        state = RunState(
            config_path="test.toml",
            config=config,
            model=object(),
            tokenizer=object(),
            t0=0.0,
            verbose=False,
        )
        ctx = state.early_mode_context()
        assert ctx.harmful is None
        assert ctx.harmless is None
        assert ctx.direction_result is None


# ===================================================================
# log()
# ===================================================================


class TestLog:
    """log() verbose/quiet/elapsed formatting."""

    def test_verbose_prints_to_stderr(self) -> None:
        buf = io.StringIO()
        with patch.object(sys, "stderr", buf):
            log("hello world", verbose=True)
        assert "[vauban]" in buf.getvalue()
        assert "hello world" in buf.getvalue()

    def test_quiet_suppresses_output(self) -> None:
        buf = io.StringIO()
        with patch.object(sys, "stderr", buf):
            log("hello world", verbose=False)
        assert buf.getvalue() == ""

    def test_elapsed_formatting(self) -> None:
        buf = io.StringIO()
        with patch.object(sys, "stderr", buf):
            log("msg", verbose=True, elapsed=3.14)
        output = buf.getvalue()
        assert "[vauban +3.1s]" in output
        assert "msg" in output

    def test_elapsed_none_omits_time(self) -> None:
        buf = io.StringIO()
        with patch.object(sys, "stderr", buf):
            log("msg", verbose=True, elapsed=None)
        output = buf.getvalue()
        assert "[vauban]" in output
        assert "+0." not in output


# ===================================================================
# write_experiment_log()
# ===================================================================


class TestWriteExperimentLog:
    """write_experiment_log writes JSONL entries."""

    def test_writes_jsonl_entry(self, tmp_path: Path) -> None:
        import json

        config = make_pipeline_config(tmp_path)
        write_experiment_log(
            config_path="test.toml",
            config=config,
            mode="probe",
            reports=["probe_report.json"],
            metrics={"n_prompts": 5.0},
            elapsed=1.23,
        )
        log_path = tmp_path / "experiment_log.jsonl"
        assert log_path.exists()
        entry = json.loads(log_path.read_text().strip())
        assert entry["pipeline_mode"] == "probe"
        assert entry["reports"] == ["probe_report.json"]
        assert entry["metrics"]["n_prompts"] == 5.0
        assert entry["elapsed_seconds"] == 1.23

    def test_error_handling_does_not_crash(self, tmp_path: Path) -> None:
        """I/O errors are caught and printed, not raised."""
        config = make_pipeline_config(tmp_path)
        buf = io.StringIO()
        with (
            patch.object(sys, "stderr", buf),
            patch.object(
                Path, "open", side_effect=OSError("disk full"),
            ),
        ):
            # Should not raise
            write_experiment_log(
                config_path="test.toml",
                config=config,
                mode="test",
                reports=[],
                metrics={},
                elapsed=0.0,
            )
        assert "warning" in buf.getvalue()
        assert "disk full" in buf.getvalue()
