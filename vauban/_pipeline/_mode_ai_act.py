"""Standalone AI Act deployer-readiness report runner."""

from __future__ import annotations

import time
from typing import cast

from vauban._pipeline._context import EarlyModeContext, log
from vauban._pipeline._mode_common import (
    ModeReport,
    finish_mode_run,
    write_mode_report,
)


def _run_ai_act_mode(context: EarlyModeContext) -> None:
    """Run standalone [ai_act] mode and write its report bundle."""
    config = context.config
    ai_act_cfg = config.ai_act
    if ai_act_cfg is None:
        msg = "[ai_act] section is required for AI Act readiness mode"
        raise ValueError(msg)

    log(
        (
            "AI Act readiness report"
            f" — role={ai_act_cfg.role},"
            f" system={ai_act_cfg.system_name!r}"
        ),
        verbose=config.verbose,
        elapsed=time.monotonic() - context.t0,
    )

    from vauban.ai_act import generate_deployer_readiness_artifacts

    artifacts = generate_deployer_readiness_artifacts(ai_act_cfg)

    report_path = write_mode_report(
        config.output_dir,
        ModeReport(
            filename="ai_act_readiness_report.json",
            payload=artifacts.report,
        ),
    )
    ledger_path = write_mode_report(
        config.output_dir,
        ModeReport(
            filename="ai_act_coverage_ledger.json",
            payload=artifacts.coverage_ledger,
        ),
    )
    library_path = write_mode_report(
        config.output_dir,
        ModeReport(
            filename="ai_act_control_library_v1.json",
            payload=artifacts.control_library,
        ),
    )
    controls_matrix_path = write_mode_report(
        config.output_dir,
        ModeReport(
            filename="ai_act_controls_matrix.json",
            payload=artifacts.controls_matrix,
        ),
    )
    annex_iii_classification_path = write_mode_report(
        config.output_dir,
        ModeReport(
            filename="ai_act_annex_iii_classification.json",
            payload=artifacts.annex_iii_classification,
        ),
    )
    risk_register_path = write_mode_report(
        config.output_dir,
        ModeReport(
            filename="ai_act_risk_register.json",
            payload=artifacts.risk_register,
        ),
    )
    fria_prep_path = write_mode_report(
        config.output_dir,
        ModeReport(
            filename="ai_act_fria_prep.json",
            payload=artifacts.fria_prep,
        ),
    )
    evidence_manifest_path = write_mode_report(
        config.output_dir,
        ModeReport(
            filename="ai_act_evidence_manifest.json",
            payload=artifacts.evidence_manifest,
        ),
    )
    integrity_path = write_mode_report(
        config.output_dir,
        ModeReport(
            filename="ai_act_integrity.json",
            payload=artifacts.integrity,
        ),
    )
    executive_summary_path = config.output_dir / "ai_act_executive_summary.md"
    executive_summary_path.parent.mkdir(parents=True, exist_ok=True)
    executive_summary_path.write_text(artifacts.executive_summary_markdown)
    auditor_appendix_path = config.output_dir / "ai_act_auditor_appendix.md"
    auditor_appendix_path.parent.mkdir(parents=True, exist_ok=True)
    auditor_appendix_path.write_text(artifacts.auditor_appendix_markdown)
    remediation_path = config.output_dir / "ai_act_remediation_plan.md"
    remediation_path.parent.mkdir(parents=True, exist_ok=True)
    remediation_path.write_text(artifacts.remediation_markdown)
    fria_prep_markdown_path = config.output_dir / "ai_act_fria_prep.md"
    fria_prep_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    fria_prep_markdown_path.write_text(artifacts.fria_prep_markdown)
    pdf_report_path: str | None = None
    if (
        artifacts.pdf_report_bytes is not None
        and artifacts.pdf_report_filename is not None
    ):
        pdf_output_path = config.output_dir / artifacts.pdf_report_filename
        pdf_output_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_output_path.write_bytes(artifacts.pdf_report_bytes)
        pdf_report_path = str(pdf_output_path)

    log(
        f"AI Act readiness bundle written to {config.output_dir}",
        verbose=config.verbose,
        elapsed=time.monotonic() - context.t0,
    )

    controls_overview = artifacts.report.get("controls_overview")
    if not isinstance(controls_overview, dict):
        msg = "ai_act report is missing controls_overview"
        raise TypeError(msg)
    controls_overview_dict = cast("dict[str, object]", controls_overview)
    report_files = [
        str(report_path),
        str(ledger_path),
        str(library_path),
        str(controls_matrix_path),
        str(annex_iii_classification_path),
        str(risk_register_path),
        str(fria_prep_path),
        str(evidence_manifest_path),
        str(integrity_path),
        str(executive_summary_path),
        str(auditor_appendix_path),
        str(remediation_path),
        str(fria_prep_markdown_path),
    ]
    if pdf_report_path is not None:
        report_files.append(pdf_report_path)

    finish_mode_run(
        context,
        "ai_act",
        report_files,
        {
            "n_pass": _metric_count(controls_overview_dict, "pass"),
            "n_fail": _metric_count(controls_overview_dict, "fail"),
            "n_unknown": _metric_count(controls_overview_dict, "unknown"),
            "n_not_applicable": _metric_count(
                controls_overview_dict,
                "not_applicable",
            ),
        },
    )


def _metric_count(overview: dict[str, object], key: str) -> int:
    """Extract one integer metric from the controls overview."""
    raw = overview.get(key, 0)
    if isinstance(raw, int):
        return raw
    msg = f"controls_overview[{key!r}] must be an int, got {type(raw).__name__}"
    raise TypeError(msg)
