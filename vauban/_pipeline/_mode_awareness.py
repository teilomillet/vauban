# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Awareness (steering detection) early-mode runner."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from typing import cast as _cast

from vauban._pipeline._context import EarlyModeContext, log
from vauban._pipeline._mode_common import ModeReport, finish_mode_run, write_mode_report
from vauban._serializers import _awareness_result_to_dict

if TYPE_CHECKING:
    from vauban.types import CausalLM, Tokenizer


def _run_awareness_mode(context: EarlyModeContext) -> None:
    """Run [awareness] early-return mode and write its report."""
    config = context.config
    if config.awareness is None:
        msg = "awareness config is required for awareness mode"
        raise ValueError(msg)
    if context.direction_result is None:
        msg = "direction_result is required for awareness mode but was not computed"
        raise ValueError(msg)

    v = config.verbose
    model = _cast("CausalLM", context.model)
    tokenizer = _cast("Tokenizer", context.tokenizer)

    from vauban.awareness import awareness_calibrate, awareness_detect

    log(
        "Running awareness detection",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )

    # Calibrate once on the clean prompt
    baseline = awareness_calibrate(
        model, tokenizer,
        config.awareness.calibration_prompt,
        context.direction_result.direction,
        config.awareness,
    )

    log(
        f"Calibrated: {len(baseline.layers)} layers profiled",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )

    # Detect per prompt
    awareness_results = [
        awareness_detect(
            model, tokenizer, prompt,
            context.direction_result.direction,
            config.awareness,
            baseline,
        )
        for prompt in config.awareness.prompts
    ]

    n_steered = sum(1 for r in awareness_results if r.steered)
    mean_confidence = (
        sum(r.confidence for r in awareness_results) / len(awareness_results)
        if awareness_results
        else 0.0
    )

    report_path = write_mode_report(
        config.output_dir,
        ModeReport(
            "awareness_report.json",
            [_awareness_result_to_dict(r) for r in awareness_results],
        ),
    )
    log(
        f"Done — awareness report written to {report_path}",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    finish_mode_run(
        context,
        "awareness",
        ["awareness_report.json"],
        {
            "n_prompts": len(awareness_results),
            "n_steered": n_steered,
            "mean_confidence": mean_confidence,
        },
    )
