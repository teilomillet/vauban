# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""SSS (Sensitivity-Scaled Steering) early-mode runner."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from typing import cast as _cast

from vauban._pipeline._context import EarlyModeContext, log
from vauban._pipeline._mode_common import ModeReport, finish_mode_run, write_mode_report
from vauban._serializers import _sss_to_dict

if TYPE_CHECKING:
    from vauban.types import CausalLM, Tokenizer


def _run_sss_mode(context: EarlyModeContext) -> None:
    """Run [sss] early-return mode and write its report."""
    config = context.config
    if config.sss is None:
        msg = "sss config is required for sss mode"
        raise ValueError(msg)
    if context.direction_result is None:
        msg = "direction_result is required for sss mode but was not computed"
        raise ValueError(msg)

    v = config.verbose
    model = _cast("CausalLM", context.model)
    tokenizer = _cast("Tokenizer", context.tokenizer)

    from vauban.sss import _sss_calibrate, sss_generate

    log(
        "Running SSS generation",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )

    # Calibrate once, reuse for all prompts

    profile = _sss_calibrate(
        model, tokenizer,
        config.sss.calibration_prompt,
        context.direction_result.direction,
        n_power_iterations=config.sss.n_power_iterations,
        fd_epsilon=config.sss.fd_epsilon,
        valley_window=config.sss.valley_window,
        top_k_valleys=config.sss.top_k_valleys,
    )

    log(
        f"Calibrated: valley layers = {profile.valley_layers}",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )

    sss_results = [
        sss_generate(
            model, tokenizer, prompt,
            context.direction_result.direction,
            config.sss,
            profile=profile,
        )
        for prompt in config.sss.prompts
    ]

    report_path = write_mode_report(
        config.output_dir,
        ModeReport(
            "sss_report.json",
            [_sss_to_dict(result) for result in sss_results],
        ),
    )
    log(
        f"Done — SSS report written to {report_path}",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    finish_mode_run(
        context,
        "sss",
        ["sss_report.json"],
        {"n_prompts": len(sss_results)},
    )
