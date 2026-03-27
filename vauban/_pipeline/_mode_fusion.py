# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Fusion early-mode runner."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from typing import cast as _cast

from vauban._pipeline._context import EarlyModeContext, log
from vauban._pipeline._mode_common import ModeReport, finish_mode_run, write_mode_report

if TYPE_CHECKING:
    from vauban.types import CausalLM, Tokenizer


def _run_fusion_mode(context: EarlyModeContext) -> None:
    """Run [fusion] early-return mode: latent state blending and generation."""
    config = context.config
    if config.fusion is None:
        msg = "fusion config is required for fusion mode"
        raise ValueError(msg)
    v = config.verbose
    model = _cast("CausalLM", context.model)
    tokenizer = _cast("Tokenizer", context.tokenizer)

    log(
        "Running latent fusion",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )

    from vauban.fusion import fuse_batch

    result = fuse_batch(model, tokenizer, config.fusion)

    report_path = write_mode_report(
        config.output_dir,
        ModeReport("fusion_report.json", result.to_dict()),
    )
    log(
        f"Done — fusion report written to {report_path}",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    finish_mode_run(
        context,
        "fusion",
        ["fusion_report.json"],
        {"n_generations": len(result.generations)},
    )
