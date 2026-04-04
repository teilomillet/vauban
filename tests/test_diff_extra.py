# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Extra coverage tests for vauban._diff internals."""

import json
from pathlib import Path

import pytest

from vauban._diff import (
    MetricDelta,
    ReportDiff,
    _extract_depth_mean_dtr,
    _extract_depth_mean_settling,
    _extract_metrics,
    _extract_optimize_best_refusal,
    _extract_value,
    _get_nested_float,
    diff_reports,
    format_diff,
    format_diff_markdown,
    known_report_filenames,
)


def test_known_report_filenames_are_sorted() -> None:
    names = known_report_filenames()
    assert names == tuple(sorted(names))
    assert "eval_report.json" in names
    assert "optimize_report.json" in names


def test_extract_value_handles_bool_and_non_dict_path() -> None:
    data: dict[str, object] = {
        "summary": {"hardened": True, "confidence": False},
        "not_a_table": 3.0,
    }

    assert _extract_value(data, ["summary", "hardened"]) == 1.0
    assert _extract_value(data, ["summary", "confidence"]) == 0.0
    assert _extract_value(data, ["not_a_table", "child"]) is None
    assert _extract_value(data, ["summary"]) is None


def test_get_nested_float_handles_bool_and_non_numeric() -> None:
    row: dict[str, object] = {
        "enabled": True,
        "disabled": False,
        "name": "demo",
    }

    assert _get_nested_float(row, "enabled") == 1.0
    assert _get_nested_float(row, "disabled") == 0.0
    assert _get_nested_float(row, "name") is None
    assert _get_nested_float(row, "missing") is None


def test_depth_extractors_skip_invalid_entries() -> None:
    data: dict[str, object] = {
        "dtr_results": [
            "ignore-me",
            {"deep_thinking_ratio": "bad"},
            {"deep_thinking_ratio": 0.2, "mean_settling_depth": 4.0},
            {"deep_thinking_ratio": 0.4, "mean_settling_depth": 6.0},
        ],
    }

    assert _extract_depth_mean_dtr(data) == pytest.approx(0.3)
    assert _extract_depth_mean_settling(data) == pytest.approx(5.0)


def test_depth_extractors_return_none_without_numeric_values() -> None:
    assert _extract_depth_mean_dtr({"dtr_results": []}) is None
    assert _extract_depth_mean_settling({"dtr_results": ["bad"]}) is None
    assert _extract_depth_mean_dtr({}) is None
    assert _extract_depth_mean_settling({}) is None
    assert _extract_depth_mean_dtr({
        "dtr_results": [{"mean_settling_depth": 3.0}],
    }) is None


def test_extract_optimize_best_refusal_handles_missing_shape() -> None:
    assert _extract_optimize_best_refusal({"best_refusal": "bad"}) is None
    assert _extract_optimize_best_refusal(
        {"best_refusal": {"other": 0.5}},
    ) is None
    assert _extract_optimize_best_refusal(
        {"best_refusal": {"refusal_rate": 0.25}},
    ) == pytest.approx(0.25)


def test_extract_metrics_for_optimize_report() -> None:
    data_a: dict[str, object] = {"best_refusal": {"refusal_rate": 0.8}}
    data_b: dict[str, object] = {"best_refusal": {"refusal_rate": 0.3}}

    metrics = _extract_metrics("optimize_report.json", data_a, data_b)

    assert len(metrics) == 1
    metric = metrics[0]
    assert metric.name == "best_refusal_rate"
    assert metric.delta == pytest.approx(-0.5)
    assert metric.lower_is_better is True


def test_diff_reports_raises_when_second_directory_is_missing(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()

    with pytest.raises(FileNotFoundError, match="Directory not found"):
        diff_reports(existing, tmp_path / "missing")


def test_diff_reports_handles_optimize_reports(tmp_path: Path) -> None:
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()

    (dir_a / "optimize_report.json").write_text(json.dumps({
        "best_refusal": {"refusal_rate": 0.9},
    }))
    (dir_b / "optimize_report.json").write_text(json.dumps({
        "best_refusal": {"refusal_rate": 0.2},
    }))

    results = diff_reports(dir_a, dir_b)

    assert len(results) == 1
    assert results[0].filename == "optimize_report.json"
    assert results[0].metrics[0].name == "best_refusal_rate"


def test_formatters_cover_quality_variants() -> None:
    reports = [
        ReportDiff(
            filename="detect_report.json",
            metrics=[
                MetricDelta("confidence", 0.2, 0.8, 0.6, False),
                MetricDelta("steady", 1.0, 1.0, 0.0, False),
                MetricDelta("zero_base", 0.0, 1.0, 1.0, True),
            ],
        ),
    ]

    plain = format_diff(Path("a"), Path("b"), reports)
    markdown = format_diff_markdown(Path("a"), Path("b"), reports)

    assert "confidence" in plain
    assert "better" in plain
    assert "steady" in plain
    assert "worse" not in plain.split("steady", maxsplit=1)[1].splitlines()[0]
    assert "—" in markdown
    assert "| confidence | 0.200 | 0.800 | +0.600 | +300.0% | better |" in markdown
