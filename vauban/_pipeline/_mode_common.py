# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for early-return pipeline modes."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vauban._pipeline._context import EarlyModeContext, write_experiment_log

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class ModeReport:
    """A JSON report artifact produced by an early mode."""

    filename: str
    payload: object


def write_mode_report(output_dir: Path, report: ModeReport) -> Path:
    """Write a JSON report to the mode output directory."""
    report_path = output_dir / report.filename
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.payload, indent=2))
    return report_path


def finish_mode_run(
    context: EarlyModeContext,
    mode_name: str,
    report_files: list[str],
    metadata: dict[str, object],
) -> None:
    """Append the standard experiment log entry for an early mode."""
    metrics: dict[str, float] = {}
    for key, value in metadata.items():
        if isinstance(value, bool | int | float):
            metrics[key] = float(value)
        else:
            msg = (
                f"mode metadata {key!r} must be numeric for experiment logging,"
                f" got {type(value).__name__}"
            )
            raise TypeError(msg)

    write_experiment_log(
        context.config_path,
        context.config,
        mode_name,
        report_files,
        metrics,
        time.monotonic() - context.t0,
    )
