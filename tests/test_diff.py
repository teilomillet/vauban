"""Tests for vauban diff — run comparison."""

import json
import sys
from pathlib import Path

import pytest

from vauban._diff import MetricDelta, ReportDiff, diff_reports, format_diff


class TestDiffReports:
    def test_eval_reports_correct_deltas(self, tmp_path: Path) -> None:
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()

        report_a = {
            "refusal_rate_modified": 0.85,
            "perplexity_modified": 4.2,
            "kl_divergence": 0.023,
            "num_prompts": 20,
            "refusal_rate_original": 0.95,
            "perplexity_original": 4.1,
        }
        report_b = {
            "refusal_rate_modified": 0.12,
            "perplexity_modified": 4.35,
            "kl_divergence": 0.031,
            "num_prompts": 20,
            "refusal_rate_original": 0.95,
            "perplexity_original": 4.1,
        }
        (dir_a / "eval_report.json").write_text(json.dumps(report_a))
        (dir_b / "eval_report.json").write_text(json.dumps(report_b))

        results = diff_reports(dir_a, dir_b)
        assert len(results) == 1
        assert results[0].filename == "eval_report.json"

        metrics = {m.name: m for m in results[0].metrics}
        assert "refusal_rate_modified" in metrics
        rr = metrics["refusal_rate_modified"]
        assert rr.delta == pytest.approx(-0.73, abs=0.001)
        assert rr.lower_is_better is True

        ppl = metrics["perplexity_modified"]
        assert ppl.delta == pytest.approx(0.15, abs=0.001)

    def test_no_shared_reports_returns_empty(self, tmp_path: Path) -> None:
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        # Only a has eval report
        report = {"refusal_rate_modified": 0.5}
        (dir_a / "eval_report.json").write_text(json.dumps(report))

        results = diff_reports(dir_a, dir_b)
        assert results == []

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            diff_reports(tmp_path / "nope", tmp_path / "nope2")

    def test_softprompt_reports(self, tmp_path: Path) -> None:
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()

        (dir_a / "softprompt_report.json").write_text(json.dumps({
            "success_rate": 0.3,
            "final_loss": 2.5,
        }))
        (dir_b / "softprompt_report.json").write_text(json.dumps({
            "success_rate": 0.7,
            "final_loss": 1.1,
        }))

        results = diff_reports(dir_a, dir_b)
        assert len(results) == 1
        metrics = {m.name: m for m in results[0].metrics}
        assert metrics["success_rate"].delta == pytest.approx(0.4)
        assert metrics["final_loss"].delta == pytest.approx(-1.4)

    def test_depth_reports(self, tmp_path: Path) -> None:
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()

        (dir_a / "depth_report.json").write_text(json.dumps({
            "dtr_results": [
                {"deep_thinking_ratio": 0.4, "mean_settling_depth": 10.0},
                {"deep_thinking_ratio": 0.6, "mean_settling_depth": 12.0},
            ],
        }))
        (dir_b / "depth_report.json").write_text(json.dumps({
            "dtr_results": [
                {"deep_thinking_ratio": 0.7, "mean_settling_depth": 8.0},
                {"deep_thinking_ratio": 0.9, "mean_settling_depth": 6.0},
            ],
        }))

        results = diff_reports(dir_a, dir_b)
        assert len(results) == 1
        metrics = {m.name: m for m in results[0].metrics}
        assert "mean_dtr" in metrics
        assert metrics["mean_dtr"].delta == pytest.approx(0.3)
        assert "mean_settling_depth" in metrics


class TestFormatDiff:
    def test_format_with_metrics(self, tmp_path: Path) -> None:
        reports = [
            ReportDiff(
                filename="eval_report.json",
                metrics=[
                    MetricDelta("refusal_rate", 0.85, 0.12, -0.73, True),
                    MetricDelta("perplexity", 4.2, 4.35, 0.15, True),
                ],
            ),
        ]
        output = format_diff(Path("a"), Path("b"), reports)
        assert "DIFF a vs b" in output
        assert "eval_report.json" in output
        assert "refusal_rate" in output
        assert "better" in output
        assert "worse" in output

    def test_format_empty(self) -> None:
        output = format_diff(Path("a"), Path("b"), [])
        assert "No shared reports" in output


class TestDiffCli:
    def test_diff_cli(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()

        (dir_a / "eval_report.json").write_text(json.dumps({
            "refusal_rate_modified": 0.8,
            "perplexity_modified": 4.0,
            "kl_divergence": 0.02,
        }))
        (dir_b / "eval_report.json").write_text(json.dumps({
            "refusal_rate_modified": 0.1,
            "perplexity_modified": 4.1,
            "kl_divergence": 0.03,
        }))

        monkeypatch.setattr(
            sys, "argv", ["vauban", "diff", str(dir_a), str(dir_b)],
        )
        from vauban.__main__ import main

        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0
        captured = capsys.readouterr()
        assert "DIFF" in captured.out
        assert "eval_report.json" in captured.out

    def test_diff_cli_missing_dir(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            ["vauban", "diff", str(tmp_path / "nope"), str(tmp_path / "nope2")],
        )
        from vauban.__main__ import main

        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert "not found" in captured.err

    def test_diff_cli_wrong_args(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["vauban", "diff", "only_one"])
        from vauban.__main__ import main

        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert "expected 2" in captured.err
