# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Standalone tokenizer/control-plane audit runner."""

from __future__ import annotations

import time

from vauban._pipeline._context import EarlyModeContext, log
from vauban._pipeline._mode_common import (
    ModeReport,
    finish_mode_run,
    write_mode_report,
)


def _run_token_audit_mode(context: EarlyModeContext) -> None:
    """Run standalone [token_audit] mode and write its redacted report."""
    config = context.config
    token_audit_cfg = config.token_audit
    if token_audit_cfg is None:
        msg = "[token_audit] section is required for token audit mode"
        raise ValueError(msg)
    if not config.model_path:
        msg = "[token_audit] mode requires [model].path"
        raise ValueError(msg)

    log(
        (
            "Token audit"
            f" — model={config.model_path!r},"
            f" max_token_id={token_audit_cfg.max_token_id!r},"
            " redacted_output=true"
        ),
        verbose=config.verbose,
        elapsed=time.monotonic() - context.t0,
    )

    from vauban._model_io import load_model
    from vauban.token_audit import run_token_audit

    model, tokenizer = load_model(config.model_path)
    result = run_token_audit(
        model,
        tokenizer,
        token_audit_cfg,
        model_path=config.model_path,
    )

    report_path = write_mode_report(
        config.output_dir,
        ModeReport(
            filename="token_audit_report.json",
            payload=result.to_dict(),
        ),
    )

    finish_mode_run(
        context,
        "token_audit",
        [str(report_path)],
        {
            "scanned_token_count": result.scanned_token_count,
            "vocab_size": result.vocab_size,
            "template_like_count": result.category_counts["template_like"],
            "duplicate_surface_count": result.duplicate_surface_count or 0,
            "declared_special_token_count": result.declared_special_token_count or 0,
        },
    )
