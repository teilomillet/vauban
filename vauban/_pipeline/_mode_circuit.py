# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Circuit early-mode runner."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from typing import cast as _cast

from vauban._pipeline._context import EarlyModeContext, log
from vauban._pipeline._mode_common import ModeReport, finish_mode_run, write_mode_report

if TYPE_CHECKING:
    from vauban._array import Array
    from vauban.types import CausalLM, Tokenizer


def _run_circuit_mode(context: EarlyModeContext) -> None:
    """Run [circuit] early-return mode: activation patching and report."""
    config = context.config
    if config.circuit is None:
        msg = "circuit config is required for circuit mode"
        raise ValueError(msg)
    v = config.verbose
    model = _cast("CausalLM", context.model)
    tokenizer = _cast("Tokenizer", context.tokenizer)

    log(
        "Tracing circuit via activation patching",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )

    from vauban.circuit import trace_circuit

    direction: Array | None = None
    if context.direction_result is not None and config.circuit.attribute_direction:
        direction = context.direction_result.direction

    result = trace_circuit(
        model,
        tokenizer,
        config.circuit.clean_prompts,
        config.circuit.corrupt_prompts,
        metric=config.circuit.metric,
        granularity=config.circuit.granularity,
        layers=config.circuit.layers,
        token_position=config.circuit.token_position,
        direction=direction,
        attribute_direction=config.circuit.attribute_direction,
        logit_diff_tokens=config.circuit.logit_diff_tokens,
    )

    report_path = write_mode_report(
        config.output_dir,
        ModeReport("circuit_report.json", result.to_dict()),
    )
    log(
        f"Done — circuit report written to {report_path}",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    finish_mode_run(
        context,
        "circuit",
        ["circuit_report.json"],
        {"n_effects": len(result.effects)},
    )
