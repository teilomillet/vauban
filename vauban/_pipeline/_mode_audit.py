# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Audit early-mode runner."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from typing import cast as _cast

from vauban._pipeline._context import EarlyModeContext, log
from vauban._pipeline._mode_common import ModeReport, finish_mode_run, write_mode_report

if TYPE_CHECKING:
    from vauban.types import CausalLM, Tokenizer


def _run_audit_mode(context: EarlyModeContext) -> None:
    """Run [audit] early-return mode and write its report."""
    from vauban.audit import audit_result_to_dict, audit_result_to_markdown, run_audit

    config = context.config
    if config.audit is None:
        msg = "audit config is required for audit mode"
        raise ValueError(msg)
    v = config.verbose
    model = _cast("CausalLM", context.model)
    tokenizer = _cast("Tokenizer", context.tokenizer)

    log(
        f"Running red-team audit ({config.audit.thoroughness})",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )

    # Load prompts — audit runs at before_prompts phase so context
    # may not have them yet.
    from vauban.dataset import resolve_prompts

    harmful = context.harmful
    if harmful is None:
        harmful = resolve_prompts(config.harmful_path)
    harmless = context.harmless
    if harmless is None:
        harmless = resolve_prompts(config.harmless_path)

    def _log_fn(msg: str) -> None:
        log(msg, verbose=v, elapsed=time.monotonic() - context.t0)

    result = run_audit(
        model,
        tokenizer,
        harmful,
        harmless,
        config.audit,
        config.model_path,
        direction_result=context.direction_result,
        log_fn=_log_fn,
    )

    # Write JSON report
    report_path = write_mode_report(
        config.output_dir,
        ModeReport("audit_report.json", audit_result_to_dict(result)),
    )

    # Write Markdown summary
    md_path = config.output_dir / "audit_report.md"
    md_path.write_text(audit_result_to_markdown(result))

    # Write PDF if requested
    report_files = ["audit_report.json", "audit_report.md"]
    if config.audit.pdf_report:
        try:
            from vauban.audit_pdf import render_audit_report_pdf

            pdf_bytes = render_audit_report_pdf(result)
            pdf_path = config.output_dir / config.audit.pdf_report_filename
            pdf_path.write_bytes(pdf_bytes)
            report_files.append(config.audit.pdf_report_filename)
            log(
                f"PDF report written to {pdf_path}",
                verbose=v,
                elapsed=time.monotonic() - context.t0,
            )
        except ImportError:
            log(
                "reportlab not installed — skipping PDF generation",
                verbose=v,
                elapsed=time.monotonic() - context.t0,
            )

    log(
        f"Done — audit report written to {report_path}"
        f" (risk: {result.overall_risk.upper()},"
        f" {len(result.findings)} findings)",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )

    finish_mode_run(
        context,
        "audit",
        report_files,
        {
            "findings": len(result.findings),
            "jailbreak_success_rate": result.jailbreak_success_rate,
        },
    )
