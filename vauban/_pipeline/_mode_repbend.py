"""RepBend early-mode runner."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from typing import cast as _cast

from vauban._pipeline._context import EarlyModeContext, log
from vauban._pipeline._mode_common import ModeReport, finish_mode_run, write_mode_report

if TYPE_CHECKING:
    from vauban.types import CausalLM, Tokenizer


def _run_repbend_mode(context: EarlyModeContext) -> None:
    """Run [repbend] early-return mode: contrastive fine-tuning and report."""
    config = context.config
    if config.repbend is None:
        msg = "repbend config is required for repbend mode"
        raise ValueError(msg)
    v = config.verbose
    model = _cast("CausalLM", context.model)
    tokenizer = _cast("Tokenizer", context.tokenizer)

    log(
        "Running RepBend contrastive fine-tuning",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )

    from vauban.dataset import resolve_prompts
    from vauban.repbend import repbend

    harmful = context.harmful
    harmless = context.harmless
    if harmful is None:
        harmful = resolve_prompts(config.harmful_path)
    if harmless is None:
        harmless = resolve_prompts(config.harmless_path)

    result = repbend(
        model,
        tokenizer,
        harmful,
        harmless,
        config.repbend,
    )

    report_path = write_mode_report(
        config.output_dir,
        ModeReport("repbend_report.json", result.to_dict()),
    )
    log(
        f"Done — repbend report written to {report_path}",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    finish_mode_run(
        context,
        "repbend",
        ["repbend_report.json"],
        {"n_layers": len(result.layers)},
    )
