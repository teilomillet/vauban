# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Shared pipeline context and logging utilities."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vauban.types import (
        DirectionResult,
        PipelineConfig,
    )


@dataclass(slots=True)
class EarlyModeContext:
    """Shared runtime context passed to early-mode handlers."""

    config_path: str | Path
    config: PipelineConfig
    model: object
    tokenizer: object
    t0: float
    harmful: list[str] | None = None
    harmless: list[str] | None = None
    direction_result: DirectionResult | None = None


def log(msg: str, *, verbose: bool = True, elapsed: float | None = None) -> None:
    """Print a one-line status message to stderr."""
    if not verbose:
        return
    prefix = f"[vauban {elapsed:+.1f}s]" if elapsed is not None else "[vauban]"
    print(f"{prefix} {msg}", file=sys.stderr, flush=True)


def write_experiment_log(
    config_path: str | Path,
    config: PipelineConfig,
    mode: str,
    reports: list[str],
    metrics: dict[str, float],
    elapsed: float,
) -> None:
    """Append an experiment entry to output_dir/experiment_log.jsonl.

    Best-effort: never crashes the pipeline on I/O errors.
    """
    import datetime

    try:
        entry: dict[str, object] = {
            "timestamp": datetime.datetime.now(
                tz=datetime.UTC,
            ).isoformat(timespec="seconds"),
            "config_path": str(Path(config_path).resolve()),
            "model_path": config.model_path,
            "pipeline_mode": mode,
            "output_dir": str(config.output_dir),
            "reports": reports,
            "metrics": metrics,
            "elapsed_seconds": round(elapsed, 2),
        }
        if config.meta is not None:
            entry["meta"] = {
                "id": config.meta.id,
                "title": config.meta.title,
                "status": config.meta.status,
                "parents": config.meta.parents,
                "tags": config.meta.tags,
            }
        log_path = config.output_dir / "experiment_log.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as exc:
        print(
            f"[vauban] warning: failed to write experiment log: {exc}",
            file=sys.stderr,
            flush=True,
        )
