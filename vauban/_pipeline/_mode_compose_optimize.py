# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Composition-optimization early-mode runner."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from typing import cast as _cast

from vauban._pipeline._context import EarlyModeContext, log
from vauban._pipeline._mode_common import ModeReport, finish_mode_run, write_mode_report

if TYPE_CHECKING:
    from vauban.types import CausalLM, Tokenizer


def _run_compose_optimize_mode(context: EarlyModeContext) -> None:
    """Run [compose_optimize] early-return mode and write its report."""
    from vauban.measure import load_prompts
    from vauban.optimize import optimize_composition

    config = context.config
    if config.compose_optimize is None:
        msg = "compose_optimize config is required for compose_optimize mode"
        raise ValueError(msg)
    if context.harmful is None:
        msg = (
            "harmful prompts are required for compose_optimize mode but were not"
            " loaded"
        )
        raise ValueError(msg)
    v = config.verbose
    model = _cast("CausalLM", context.model)
    tokenizer = _cast("Tokenizer", context.tokenizer)

    log(
        f"Running composition optimization ({config.compose_optimize.n_trials} trials)",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    if config.eval.prompts_path is not None:
        eval_prompts = load_prompts(config.eval.prompts_path)
    else:
        eval_prompts = context.harmful[:config.eval.num_prompts]

    result = optimize_composition(
        model,
        tokenizer,
        eval_prompts,
        config.compose_optimize,
    )
    report = {
        "n_trials": result.n_trials,
        "bank_entries": result.bank_entries,
        "best_refusal": (
            {
                "weights": result.best_refusal.weights,
                "refusal_rate": result.best_refusal.refusal_rate,
                "perplexity": result.best_refusal.perplexity,
            }
            if result.best_refusal is not None
            else None
        ),
        "best_balanced": (
            {
                "weights": result.best_balanced.weights,
                "refusal_rate": result.best_balanced.refusal_rate,
                "perplexity": result.best_balanced.perplexity,
            }
            if result.best_balanced is not None
            else None
        ),
    }
    report_path = write_mode_report(
        config.output_dir,
        ModeReport("compose_optimize_report.json", report),
    )
    log(
        f"Done — compose_optimize report written to {report_path}",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    finish_mode_run(
        context,
        "compose_optimize",
        ["compose_optimize_report.json"],
        {"n_trials": float(config.compose_optimize.n_trials)},
    )
