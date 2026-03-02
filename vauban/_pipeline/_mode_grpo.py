"""GRPO early-mode runner."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from typing import cast as _cast

from vauban._pipeline._context import EarlyModeContext, log
from vauban._pipeline._mode_common import ModeReport, finish_mode_run, write_mode_report

if TYPE_CHECKING:
    from vauban.types import CausalLM, Tokenizer


def _run_grpo_mode(context: EarlyModeContext) -> None:
    """Run [grpo] early-return mode: RL-based safety alignment inversion."""
    config = context.config
    if config.grpo is None:
        msg = "grpo config is required for grpo mode"
        raise ValueError(msg)
    v = config.verbose
    model = _cast("CausalLM", context.model)
    tokenizer = _cast("Tokenizer", context.tokenizer)

    log(
        "Running GRPO alignment inversion",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )

    from vauban.dataset import resolve_prompts
    from vauban.grpo import grpo

    harmful = context.harmful
    if harmful is None:
        harmful = resolve_prompts(config.harmful_path)

    # Optionally limit prompt pool
    pool_size = config.grpo.prompt_pool_size
    if pool_size is not None:
        harmful = harmful[:pool_size]

    result = grpo(
        model,
        tokenizer,
        harmful,
        config.grpo,
    )

    report_path = write_mode_report(
        config.output_dir,
        ModeReport("grpo_report.json", result.to_dict()),
    )
    log(
        f"Done — grpo report written to {report_path}",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    finish_mode_run(
        context,
        "grpo",
        ["grpo_report.json"],
        {
            "initial_refusal_rate": result.initial_refusal_rate,
            "final_refusal_rate": result.final_refusal_rate,
        },
    )
