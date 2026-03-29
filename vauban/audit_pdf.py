# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""PDF rendering for red-team audit reports.

Reuses the ReportLab-based rendering pattern from ``ai_act_pdf.py``
with audit-specific sections: executive summary, attack resistance,
defense posture, refusal coverage, and prioritised recommendations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vauban.types import AuditResult


def render_audit_report_pdf(result: AuditResult) -> bytes:
    """Render an audit report as a deterministic PDF.

    Args:
        result: The completed audit result.

    Returns:
        PDF file contents as bytes.
    """
    import datetime
    import io

    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "AuditTitle",
        parent=styles["Title"],
        fontSize=22,
        spaceAfter=6 * mm,
    )
    h1_style = ParagraphStyle(
        "AuditH1",
        parent=styles["Heading1"],
        fontSize=16,
        spaceAfter=4 * mm,
        spaceBefore=6 * mm,
    )
    h2_style = ParagraphStyle(
        "AuditH2",
        parent=styles["Heading2"],
        fontSize=13,
        spaceAfter=3 * mm,
        spaceBefore=4 * mm,
    )
    body_style = ParagraphStyle(
        "AuditBody",
        parent=styles["BodyText"],
        fontSize=10,
        spaceAfter=2 * mm,
    )
    bullet_style = ParagraphStyle(
        "AuditBullet",
        parent=body_style,
        leftIndent=10 * mm,
        bulletIndent=4 * mm,
        spaceAfter=1.5 * mm,
    )

    story: list[object] = []

    # -- Title page --
    story.append(Paragraph("Red-Team Audit Report", title_style))
    story.append(Spacer(1, 4 * mm))

    now = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d %H:%M UTC")
    meta_data = [
        ["Company", result.company_name],
        ["System", result.system_name],
        ["Model", result.model_path],
        ["Thoroughness", result.thoroughness.capitalize()],
        ["Overall Risk", result.overall_risk.upper()],
        ["Generated", now],
    ]
    meta_table = Table(meta_data, colWidths=[35 * mm, 120 * mm])
    meta_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2 * mm),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 8 * mm))

    # -- Executive summary --
    story.append(Paragraph("Executive Summary", h1_style))
    _risk_color = {
        "critical": "red", "high": "orange",
        "medium": "goldenrod", "low": "green",
    }
    color = _risk_color.get(result.overall_risk, "black")
    story.append(Paragraph(
        f'Overall risk assessment: <font color="{color}">'
        f"<b>{result.overall_risk.upper()}</b></font>",
        body_style,
    ))

    critical_count = sum(1 for f in result.findings if f.severity == "critical")
    high_count = sum(1 for f in result.findings if f.severity == "high")
    if critical_count:
        story.append(Paragraph(
            f"\u2022 {critical_count} critical finding(s) require immediate attention",
            bullet_style,
        ))
    if high_count:
        story.append(Paragraph(
            f"\u2022 {high_count} high-severity finding(s)"
            f" should be addressed before deployment",
            bullet_style,
        ))
    story.append(Paragraph(
        f"\u2022 {len(result.findings)} total findings across"
        f" {len({f.category for f in result.findings})} categories",
        bullet_style,
    ))
    story.append(Spacer(1, 4 * mm))

    # -- Key metrics --
    story.append(Paragraph("Key Metrics", h1_style))
    metrics_data = [["Metric", "Value"]]
    metrics_data.append([
        "Jailbreak bypass rate",
        f"{result.jailbreak_success_rate:.0%} ({result.jailbreak_total} tested)",
    ])
    if result.softprompt_success_rate is not None:
        metrics_data.append([
            "Soft prompt attack success",
            f"{result.softprompt_success_rate:.0%}",
        ])
    if result.bijection_success_rate is not None:
        metrics_data.append([
            "Bijection cipher bypass",
            f"{result.bijection_success_rate:.0%}",
        ])
    if result.surface_refusal_rate is not None:
        metrics_data.append([
            "Refusal coverage",
            f"{result.surface_refusal_rate:.0%}",
        ])
    if result.detect_hardened is not None:
        metrics_data.append([
            "Defense hardening",
            f"{'Yes' if result.detect_hardened else 'No'}"
            f" ({result.detect_confidence:.0%} confidence)",
        ])
    if result.guard_circuit_break_rate is not None:
        metrics_data.append([
            "Guard circuit break rate",
            f"{result.guard_circuit_break_rate:.0%}",
        ])

    metrics_table = Table(metrics_data, colWidths=[60 * mm, 95 * mm])
    metrics_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 0), (-1, 0), "#E8E8E8"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2 * mm),
        ("GRID", (0, 0), (-1, -1), 0.5, "#CCCCCC"),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 6 * mm))

    # -- Detailed findings --
    story.append(Paragraph("Detailed Findings", h1_style))
    for i, finding in enumerate(result.findings, 1):
        sev_upper = finding.severity.upper()
        story.append(Paragraph(
            f"{i}. [{sev_upper}] {finding.title}", h2_style,
        ))
        story.append(Paragraph(finding.description, body_style))
        story.append(Paragraph(
            f"<b>Remediation:</b> {finding.remediation}", body_style,
        ))
        story.append(Paragraph(
            f"<i>Evidence: {finding.evidence}</i>", body_style,
        ))
        story.append(Spacer(1, 2 * mm))

    # -- Recommendations --
    story.append(Paragraph("Prioritised Recommendations", h1_style))
    critical_findings = [
        f for f in result.findings
        if f.severity in ("critical", "high")
    ]
    if critical_findings:
        for i, f in enumerate(critical_findings, 1):
            story.append(Paragraph(
                f"\u2022 <b>P{i}:</b> {f.remediation}", bullet_style,
            ))
    else:
        story.append(Paragraph(
            "No critical or high-severity findings. Model safety posture is adequate.",
            body_style,
        ))

    story.append(Spacer(1, 10 * mm))
    story.append(Paragraph(
        "<i>Generated by Vauban — vauban.dev</i>",
        ParagraphStyle("Footer", parent=body_style, fontSize=8),
    ))

    doc.build(story)
    return buf.getvalue()
