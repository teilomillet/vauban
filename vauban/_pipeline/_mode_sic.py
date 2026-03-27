# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""SIC early-mode runner."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from typing import cast as _cast

from vauban._pipeline._context import EarlyModeContext, log
from vauban._pipeline._mode_common import ModeReport, finish_mode_run, write_mode_report
from vauban._serializers import _sic_to_dict

if TYPE_CHECKING:
    from vauban.types import CausalLM, Tokenizer


def _run_sic_mode(context: EarlyModeContext) -> None:
    """Run [sic] early-return mode and write its report."""
    config = context.config
    if config.sic is None:
        msg = "sic config is required for sic mode"
        raise ValueError(msg)
    if context.harmful is None:
        msg = "harmful prompts are required for sic mode but were not loaded"
        raise ValueError(msg)
    if context.harmless is None:
        msg = "harmless prompts are required for sic mode but were not loaded"
        raise ValueError(msg)
    v = config.verbose
    model = _cast("CausalLM", context.model)
    tokenizer = _cast("Tokenizer", context.tokenizer)

    from vauban.measure import load_prompts
    from vauban.sic import sic as sic_sanitize

    log(
        f"Running SIC sanitization (mode={config.sic.mode})",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    if context.direction_result is not None:
        direction_vec = context.direction_result.direction
        layer_idx = context.direction_result.layer_index
    else:
        direction_vec = None
        layer_idx = 0
    if config.eval.prompts_path is not None:
        sic_prompts: list[str] = load_prompts(config.eval.prompts_path)
    else:
        sic_prompts = context.harmful[:config.eval.num_prompts]

    cal_prompts: list[str] | None = None
    if config.sic.calibrate:
        cal_prompts = (
            context.harmless
            if config.sic.calibrate_prompts == "harmless"
            else context.harmful
        )

    sic_result = sic_sanitize(
        model,
        tokenizer,
        sic_prompts,
        config.sic,
        direction_vec,
        layer_idx,
        cal_prompts,
    )
    report_path = write_mode_report(
        config.output_dir,
        ModeReport("sic_report.json", _sic_to_dict(sic_result)),
    )
    log(
        f"Done — SIC report written to {report_path}",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    finish_mode_run(
        context,
        "sic",
        ["sic_report.json"],
        {"n_prompts": len(sic_prompts)},
    )
