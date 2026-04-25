# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for benchmark scorecard parsing, scoring, and mode execution."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from tests.conftest import make_early_mode_context
from vauban._pipeline._mode_benchmark import _run_benchmark_mode
from vauban.config._parse_benchmark import _parse_benchmark
from vauban.scorecard import build_benchmark_artifacts
from vauban.types import BenchmarkConfig, BenchmarkModelConfig

if TYPE_CHECKING:
    from pathlib import Path


def _write_json(path: Path, payload: object) -> None:
    """Write one JSON payload to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _object_dict(raw: object) -> dict[str, object]:
    """Coerce one raw object into a string-keyed object dictionary."""
    if not isinstance(raw, dict):
        msg = f"Expected dict, got {type(raw).__name__}"
        raise TypeError(msg)
    result: dict[str, object] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            msg = "Expected string keys"
            raise TypeError(msg)
        result[key] = value
    return result


def _number(raw: object) -> float:
    """Coerce one raw scalar into float."""
    if not isinstance(raw, int | float):
        msg = f"Expected numeric value, got {type(raw).__name__}"
        raise TypeError(msg)
    return float(raw)


def _make_report_dir(
    root: Path,
    *,
    label: str,
    jailbreak_success_rate: float,
    softprompt_success_rate: float | None,
    detect_hardened: bool,
    detect_confidence: float,
    include_ai_act: bool,
) -> Path:
    """Create a small synthetic report bundle."""
    report_dir = root / label
    _write_json(
        report_dir / "audit_report.json",
        {
            "model_path": f"mlx-community/{label}",
            "overall_risk": "low" if jailbreak_success_rate < 0.2 else "medium",
            "findings": [
                {
                    "severity": "high",
                    "title": f"{label} finding",
                },
            ],
            "metrics": {
                "jailbreak_success_rate": jailbreak_success_rate,
                "softprompt_success_rate": softprompt_success_rate,
                "detect_hardened": detect_hardened,
                "detect_confidence": detect_confidence,
            },
        },
    )
    _write_json(
        report_dir / "detect_report.json",
        {
            "hardened": detect_hardened,
            "confidence": detect_confidence,
            "effective_rank": 1.9,
            "evidence": ["synthetic"],
        },
    )
    _write_json(
        report_dir / "guard_report.json",
        [
            {
                "prompt": "Explain attention masks.",
                "text": "",
                "events": [
                    {
                        "token_index": 0,
                        "token_id": 1,
                        "token_str": "thought",
                        "projection": 0.8,
                        "zone": "orange",
                        "action": "rewind",
                        "alpha_applied": 1.5,
                        "rewind_count": 1,
                        "checkpoint_offset": 0,
                    },
                ],
                "total_rewinds": 1,
                "circuit_broken": False,
                "tokens_generated": 0,
                "tokens_rewound": 0,
                "final_zone_counts": {
                    "green": 0,
                    "yellow": 0,
                    "orange": 1,
                    "red": 0,
                },
            },
        ],
    )
    if include_ai_act:
        _write_json(
            report_dir / "ai_act_readiness_report.json",
            {
                "overall_status": "ready_with_actions",
                "controls_overview": {
                    "pass": 8,
                    "fail": 1,
                    "unknown": 1,
                    "not_applicable": 2,
                },
            },
        )
    return report_dir


class TestParseBenchmark:
    """Targeted branch coverage for the benchmark parser."""

    def test_parse_resolves_paths_and_weights(self, tmp_path: Path) -> None:
        cfg = _parse_benchmark(
            tmp_path,
            {
                "benchmark": {
                    "title": "Safety",
                    "description": "Compare runs",
                    "markdown_report": False,
                    "weights": {
                        "behavioral_safety": 0.5,
                        "tamper_resistance": 0.3,
                        "evidence_readiness": 0.2,
                    },
                    "models": [
                        {
                            "label": "gemma-4",
                            "report_dir": "runs/gemma-4",
                            "audit_report": "explicit/audit.json",
                        },
                    ],
                },
            },
        )
        assert cfg is not None
        assert cfg.title == "Safety"
        assert cfg.markdown_report is False
        assert cfg.models[0].report_dir == tmp_path / "runs/gemma-4"
        assert cfg.models[0].audit_report == tmp_path / "explicit/audit.json"
        assert cfg.weights.behavioral_safety == 0.5

    def test_parse_rejects_empty_models(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="non-empty array of tables"):
            _parse_benchmark(tmp_path, {"benchmark": {"models": []}})

    def test_parse_rejects_non_positive_weight_sum(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="sum to a positive value"):
            _parse_benchmark(
                tmp_path,
                {
                    "benchmark": {
                        "models": [{"label": "gemma-4"}],
                        "weights": {
                            "behavioral_safety": 0.0,
                            "tamper_resistance": 0.0,
                            "evidence_readiness": 0.0,
                        },
                    },
                },
            )


class TestBenchmarkArtifacts:
    """Benchmark scorecard assembly from synthetic reports."""

    def test_build_artifacts_ranks_models_and_marks_gaps(
        self,
        tmp_path: Path,
    ) -> None:
        strong_dir = _make_report_dir(
            tmp_path,
            label="gemma-4-strong",
            jailbreak_success_rate=0.1,
            softprompt_success_rate=0.2,
            detect_hardened=True,
            detect_confidence=0.9,
            include_ai_act=True,
        )
        sparse_dir = _make_report_dir(
            tmp_path,
            label="gemma-4-sparse",
            jailbreak_success_rate=0.4,
            softprompt_success_rate=None,
            detect_hardened=False,
            detect_confidence=0.8,
            include_ai_act=False,
        )

        artifacts = build_benchmark_artifacts(
            BenchmarkConfig(
                title="Synthetic Benchmark",
                models=[
                    BenchmarkModelConfig(
                        label="strong",
                        report_dir=strong_dir,
                    ),
                    BenchmarkModelConfig(
                        label="sparse",
                        report_dir=sparse_dir,
                    ),
                ],
            ),
        )

        models = artifacts.report["models"]
        assert isinstance(models, list)
        first = _object_dict(models[0])
        second = _object_dict(models[1])
        assert first["label"] == "strong"
        assert first["overall_score"] is not None
        assert (
            _number(first["evidence_readiness_score"])
            > _number(second["evidence_readiness_score"])
        )
        missing = second["missing_reports"]
        assert isinstance(missing, list)
        assert "ai_act" in missing
        assert "Synthetic Benchmark" in artifacts.markdown
        assert "strong" in artifacts.markdown

    def test_missing_reports_do_not_crash(self, tmp_path: Path) -> None:
        artifacts = build_benchmark_artifacts(
            BenchmarkConfig(
                models=[
                    BenchmarkModelConfig(
                        label="missing",
                        report_dir=tmp_path / "does-not-exist",
                    ),
                ],
            ),
        )

        models = artifacts.report["models"]
        assert isinstance(models, list)
        first = _object_dict(models[0])
        assert first["overall_score"] is not None
        assert first["confidence_score"] == 0.0
        assert first["missing_reports"] == [
            "audit",
            "detect",
            "guard",
            "ai_act",
            "eval",
        ]


class TestBenchmarkModeRunner:
    """End-to-end benchmark mode output coverage."""

    def test_run_benchmark_mode_writes_reports(self, tmp_path: Path) -> None:
        report_dir = _make_report_dir(
            tmp_path,
            label="gemma-4",
            jailbreak_success_rate=0.2,
            softprompt_success_rate=0.1,
            detect_hardened=True,
            detect_confidence=0.85,
            include_ai_act=True,
        )
        ctx = make_early_mode_context(
            tmp_path,
            benchmark=BenchmarkConfig(
                title="Gemma Scorecard",
                models=[
                    BenchmarkModelConfig(
                        label="gemma-4",
                        report_dir=report_dir,
                    ),
                ],
            ),
        )

        _run_benchmark_mode(ctx)

        report_path = tmp_path / "benchmark_scorecard.json"
        markdown_path = tmp_path / "benchmark_scorecard.md"
        assert report_path.exists()
        assert markdown_path.exists()

        report = json.loads(report_path.read_text())
        assert report["title"] == "Gemma Scorecard"
        assert report["leaderboard"][0]["label"] == "gemma-4"
