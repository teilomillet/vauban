# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Flywheel early-mode runner."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from typing import cast as _cast

from vauban._pipeline._context import EarlyModeContext, log
from vauban._pipeline._mode_common import ModeReport, finish_mode_run, write_mode_report
from vauban._serializers import _flywheel_to_dict

if TYPE_CHECKING:
    from vauban.types import CausalLM, Tokenizer


def _run_flywheel_mode(context: EarlyModeContext) -> None:
    """Run [flywheel] early-return mode and write its report."""
    config = context.config
    if config.flywheel is None:
        msg = "flywheel config is required for flywheel mode"
        raise ValueError(msg)
    v = config.verbose
    model = _cast("CausalLM", context.model)
    tokenizer = _cast("Tokenizer", context.tokenizer)

    log(
        "Running flywheel co-evolution",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )

    direction = None
    layer_index = 0
    if context.direction_result is not None:
        direction = context.direction_result.direction
        layer_index = context.direction_result.layer_index

    from vauban.flywheel import run_flywheel

    result = run_flywheel(
        model, tokenizer, config.flywheel,
        direction, layer_index, config.output_dir,
        objective=config.objective,
        verbose=v, t0=context.t0,
    )

    metrics: dict[str, object] = {
        "n_cycles": float(len(result.cycles)),
        "converged": float(result.converged),
        "total_evasions": float(result.total_evasions),
    }
    if result.objective_assessment is not None:
        metrics["objective_passed"] = float(result.objective_assessment.passed)
        metrics["objective_safety_passed"] = float(
            result.objective_assessment.safety_passed,
        )
        metrics["objective_utility_passed"] = float(
            result.objective_assessment.utility_passed,
        )

    report_path = write_mode_report(
        config.output_dir,
        ModeReport("flywheel_report.json", _flywheel_to_dict(result)),
    )

    log(
        f"Done — flywheel report: {report_path}",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    finish_mode_run(
        context,
        "flywheel",
        ["flywheel_report.json", "flywheel_failures.jsonl"],
        metrics,
    )
