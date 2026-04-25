# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Standalone benchmark scorecard runner."""

from __future__ import annotations

import time

from vauban._pipeline._context import EarlyModeContext, log
from vauban._pipeline._mode_common import (
    ModeReport,
    finish_mode_run,
    write_mode_report,
)


def _run_benchmark_mode(context: EarlyModeContext) -> None:
    """Run standalone [benchmark] mode and write its scorecard bundle."""
    config = context.config
    benchmark_cfg = config.benchmark
    if benchmark_cfg is None:
        msg = "[benchmark] section is required for benchmark mode"
        raise ValueError(msg)

    log(
        (
            "Benchmark scorecard"
            f" — title={benchmark_cfg.title!r},"
            f" models={len(benchmark_cfg.models)}"
        ),
        verbose=config.verbose,
        elapsed=time.monotonic() - context.t0,
    )

    from vauban.scorecard import build_benchmark_artifacts

    artifacts = build_benchmark_artifacts(benchmark_cfg)

    report_path = write_mode_report(
        config.output_dir,
        ModeReport(
            filename="benchmark_scorecard.json",
            payload=artifacts.report,
        ),
    )
    report_files = [str(report_path)]
    if benchmark_cfg.markdown_report:
        markdown_path = config.output_dir / "benchmark_scorecard.md"
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(artifacts.markdown)
        report_files.append(str(markdown_path))

    models_raw = artifacts.report.get("models")
    if not isinstance(models_raw, list):
        msg = "benchmark report is missing models"
        raise TypeError(msg)

    finish_mode_run(
        context,
        "benchmark",
        report_files,
        {
            "n_models": len(models_raw),
            "n_ranked": sum(
                1
                for item in models_raw
                if (item_dict := _object_dict(item)) is not None
                and item_dict.get("overall_score") is not None
            ),
        },
    )


def _object_dict(raw: object) -> dict[str, object] | None:
    """Convert one raw object into a string-keyed dictionary."""
    if not isinstance(raw, dict):
        return None
    result: dict[str, object] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            return None
        result[key] = value
    return result
