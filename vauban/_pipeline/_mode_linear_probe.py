# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Linear probe early-mode runner."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from typing import cast as _cast

from vauban._pipeline._context import EarlyModeContext, log
from vauban._pipeline._mode_common import ModeReport, finish_mode_run, write_mode_report

if TYPE_CHECKING:
    from vauban.types import CausalLM, Tokenizer


def _run_linear_probe_mode(context: EarlyModeContext) -> None:
    """Run [linear_probe] early-return mode: train per-layer probes and report."""
    config = context.config
    if config.linear_probe is None:
        msg = "linear_probe config is required for linear_probe mode"
        raise ValueError(msg)
    v = config.verbose
    model = _cast("CausalLM", context.model)
    tokenizer = _cast("Tokenizer", context.tokenizer)

    log(
        "Training linear probes",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )

    from vauban.dataset import resolve_prompts
    from vauban.linear_probe import train_probe

    harmful = context.harmful
    harmless = context.harmless
    if harmful is None:
        harmful = resolve_prompts(config.harmful_path)
    if harmless is None:
        harmless = resolve_prompts(config.harmless_path)

    result = train_probe(
        model,
        tokenizer,
        harmful,
        harmless,
        config.linear_probe,
    )

    report_path = write_mode_report(
        config.output_dir,
        ModeReport("linear_probe_report.json", result.to_dict()),
    )
    log(
        f"Done — linear probe report written to {report_path}",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    finish_mode_run(
        context,
        "linear_probe",
        ["linear_probe_report.json"],
        {"n_layers": len(result.layers)},
    )
