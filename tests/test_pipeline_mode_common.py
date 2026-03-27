# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for pipeline mode_common: ModeReport, write_mode_report, finish_mode_run."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from tests.conftest import make_early_mode_context
from vauban._pipeline._mode_common import ModeReport, finish_mode_run, write_mode_report

if TYPE_CHECKING:
    from pathlib import Path

# ===================================================================
# ModeReport
# ===================================================================


class TestModeReport:
    """ModeReport is a frozen dataclass."""

    def test_construction(self) -> None:
        report = ModeReport(filename="test.json", payload={"key": "value"})
        assert report.filename == "test.json"
        assert report.payload == {"key": "value"}

    def test_frozen(self) -> None:
        report = ModeReport(filename="test.json", payload={})
        with pytest.raises(AttributeError):
            report.filename = "other.json"  # type: ignore[misc]

    def test_slots(self) -> None:
        report = ModeReport(filename="test.json", payload={})
        assert not hasattr(report, "__dict__")


# ===================================================================
# write_mode_report
# ===================================================================


class TestWriteModeReport:
    """write_mode_report creates JSON in output_dir."""

    def test_creates_json_file(self, tmp_path: Path) -> None:
        report = ModeReport(
            filename="my_report.json",
            payload={"status": "ok", "count": 42},
        )
        result_path = write_mode_report(tmp_path, report)
        assert result_path == tmp_path / "my_report.json"
        assert result_path.exists()
        data = json.loads(result_path.read_text())
        assert data == {"status": "ok", "count": 42}

    def test_creates_nested_directory(self, tmp_path: Path) -> None:
        nested = tmp_path / "sub" / "dir"
        report = ModeReport(filename="report.json", payload=[1, 2, 3])
        result_path = write_mode_report(nested, report)
        assert result_path.exists()
        assert json.loads(result_path.read_text()) == [1, 2, 3]

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        path = tmp_path / "report.json"
        path.write_text('{"old": true}')
        report = ModeReport(filename="report.json", payload={"new": True})
        write_mode_report(tmp_path, report)
        data = json.loads(path.read_text())
        assert data == {"new": True}


# ===================================================================
# finish_mode_run
# ===================================================================


class TestFinishModeRun:
    """finish_mode_run converts metadata and writes experiment log."""

    def test_numeric_metadata_converted(self, tmp_path: Path) -> None:
        context = make_early_mode_context(tmp_path, t0=0.0)
        with patch("vauban._pipeline._mode_common.write_experiment_log") as mock_log:
            finish_mode_run(
                context,
                "test_mode",
                ["report.json"],
                {"count": 5, "rate": 0.95},
            )
            mock_log.assert_called_once()
            args = mock_log.call_args[0]
            metrics = args[4]
            assert metrics["count"] == 5.0
            assert metrics["rate"] == 0.95

    def test_bool_metadata_converted_to_float(self, tmp_path: Path) -> None:
        context = make_early_mode_context(tmp_path, t0=0.0)
        with patch("vauban._pipeline._mode_common.write_experiment_log") as mock_log:
            finish_mode_run(
                context,
                "test_mode",
                ["report.json"],
                {"flag": True},
            )
            mock_log.assert_called_once()
            args = mock_log.call_args[0]
            metrics = args[4]
            assert metrics["flag"] == 1.0

    def test_int_metadata_converted_to_float(self, tmp_path: Path) -> None:
        context = make_early_mode_context(tmp_path, t0=0.0)
        with patch("vauban._pipeline._mode_common.write_experiment_log") as mock_log:
            finish_mode_run(
                context,
                "test_mode",
                ["report.json"],
                {"n": 42},
            )
            mock_log.assert_called_once()
            args = mock_log.call_args[0]
            metrics = args[4]
            assert metrics["n"] == 42.0

    def test_string_metadata_raises_type_error(self, tmp_path: Path) -> None:
        context = make_early_mode_context(tmp_path, t0=0.0)
        with pytest.raises(TypeError, match="must be numeric"):
            finish_mode_run(
                context,
                "test_mode",
                ["report.json"],
                {"name": "not a number"},  # type: ignore[dict-item]
            )

    def test_empty_metadata(self, tmp_path: Path) -> None:
        context = make_early_mode_context(tmp_path, t0=0.0)
        with patch("vauban._pipeline._mode_common.write_experiment_log") as mock_log:
            finish_mode_run(context, "test_mode", ["report.json"], {})
            mock_log.assert_called_once()
            args = mock_log.call_args[0]
            assert args[4] == {}
