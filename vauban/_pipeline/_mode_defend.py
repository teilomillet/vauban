"""Defense-stack early-mode runner."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from typing import cast as _cast

from vauban._pipeline._context import EarlyModeContext, log
from vauban._pipeline._mode_common import ModeReport, finish_mode_run, write_mode_report
from vauban._serializers import _defend_to_dict

if TYPE_CHECKING:
    from vauban.types import CausalLM, Tokenizer


def _run_defend_mode(context: EarlyModeContext) -> None:
    """Run [defend] early-return mode and write its report."""
    from vauban.defend import defend_content
    from vauban.measure import load_prompts

    config = context.config
    if config.defend is None:
        msg = "defend config is required for defend mode"
        raise ValueError(msg)
    if context.harmful is None:
        msg = "harmful prompts are required for defend mode but were not loaded"
        raise ValueError(msg)
    v = config.verbose
    model = _cast("CausalLM", context.model)
    tokenizer = _cast("Tokenizer", context.tokenizer)

    log(
        "Running defense stack evaluation",
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
        defend_prompts: list[str] = load_prompts(config.eval.prompts_path)
    else:
        defend_prompts = context.harmful[:config.eval.num_prompts]

    defend_results = [
        defend_content(
            model,
            tokenizer,
            prompt,
            direction_vec,
            config.defend,
            layer_idx,
        )
        for prompt in defend_prompts
    ]

    total_blocked = sum(1 for result in defend_results if result.blocked)
    block_rate = total_blocked / len(defend_results) if defend_results else 0.0

    report = {
        "total_prompts": len(defend_results),
        "total_blocked": total_blocked,
        "block_rate": block_rate,
        "results": [_defend_to_dict(result) for result in defend_results],
    }
    report_path = write_mode_report(
        config.output_dir,
        ModeReport("defend_report.json", report),
    )
    log(
        f"Done — defend report written to {report_path}",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    finish_mode_run(
        context,
        "defend",
        ["defend_report.json"],
        {"block_rate": block_rate},
    )
