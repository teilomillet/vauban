"""Probe early-mode runner."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from typing import cast as _cast

from vauban._pipeline._context import EarlyModeContext, log
from vauban._pipeline._mode_common import ModeReport, finish_mode_run, write_mode_report
from vauban._serializers import _probe_to_dict

if TYPE_CHECKING:
    from vauban.types import CausalLM, Tokenizer


def _run_probe_mode(context: EarlyModeContext) -> None:
    """Run [probe] early-return mode and write its report."""
    config = context.config
    if config.probe is None:
        msg = "probe config is required for probe mode"
        raise ValueError(msg)
    if context.direction_result is None:
        msg = "direction_result is required for probe mode but was not computed"
        raise ValueError(msg)
    v = config.verbose
    model = _cast("CausalLM", context.model)
    tokenizer = _cast("Tokenizer", context.tokenizer)

    from vauban.probe import probe

    log(
        "Running probe inspection",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    probe_results = [
        probe(model, tokenizer, prompt, context.direction_result.direction)
        for prompt in config.probe.prompts
    ]
    report_path = write_mode_report(
        config.output_dir,
        ModeReport(
            "probe_report.json",
            [_probe_to_dict(result) for result in probe_results],
        ),
    )
    log(
        f"Done — probe report written to {report_path}",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    finish_mode_run(context, "probe", ["probe_report.json"], {})
