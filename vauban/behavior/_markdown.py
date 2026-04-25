# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Markdown rendering for typed model behavior change reports."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vauban.behavior._primitives import BehaviorReport, JsonValue


def render_behavior_report_markdown(report: BehaviorReport) -> str:
    """Render a model behavior change report as deterministic Markdown."""
    lines: list[str] = [
        f"# {_md_text(report.title)}",
        "",
        "Model Behavior Change Report",
        "",
        "## Summary",
        "",
        (
            "- Model A:"
            f" {_model_summary(report.baseline.label, report.baseline.model_path)}"
        ),
        (
            "- Model B:"
            f" {_model_summary(report.candidate.label, report.candidate.model_path)}"
        ),
        f"- Suite: {_md_text(report.suite.name)}",
        f"- Categories: {', '.join(_md_text(c) for c in report.suite.categories)}",
        f"- Metrics: {', '.join(_md_text(m) for m in report.suite.metrics)}",
        "",
    ]

    _append_target_change(lines, report)
    _append_transformation(lines, report)
    _append_access(lines, report)
    _append_evidence(lines, report)
    _append_claims(lines, report)
    _append_reproduction_targets(lines, report)
    _append_reproduction_results(lines, report)
    _append_findings(lines, report)
    _append_metric_delta_table(lines, report)
    _append_activation_findings(lines, report)
    _append_examples(lines, report)
    _append_recommendation(lines, report)
    _append_limitations(lines, report)
    _append_reproducibility(lines, report)
    return "\n".join(lines).rstrip() + "\n"


def _append_target_change(lines: list[str], report: BehaviorReport) -> None:
    """Append the target change section when provided."""
    if report.target_change is None:
        return
    lines.extend([
        "## Target Change",
        "",
        f"- {_md_text(report.target_change)}",
        "",
    ])


def _append_transformation(lines: list[str], report: BehaviorReport) -> None:
    """Append the audited transformation section when provided."""
    if report.transformation is None:
        return
    transformation = report.transformation
    lines.extend(["## Transformation", ""])
    lines.append(f"- Kind: `{transformation.kind}`")
    lines.append(f"- Summary: {_md_text(transformation.summary)}")
    if transformation.before is not None:
        lines.append(f"- Before: `{_md_text(transformation.before)}`")
    if transformation.after is not None:
        lines.append(f"- After: `{_md_text(transformation.after)}`")
    if transformation.method is not None:
        lines.append(f"- Method: {_md_text(transformation.method)}")
    if transformation.source_ref is not None:
        lines.append(f"- Source: {_md_text(transformation.source_ref)}")
    for note in transformation.notes:
        lines.append(f"- Note: {_md_text(note)}")
    lines.append("")


def _append_access(lines: list[str], report: BehaviorReport) -> None:
    """Append access level and claim-strength boundaries."""
    if report.access is None:
        return
    access = report.access
    lines.extend(["## Access And Claim Strength", ""])
    lines.append(f"- Access level: `{access.level}`")
    lines.append(f"- Maximum claim strength: `{access.claim_strength}`")
    if access.available_evidence:
        joined = ", ".join(_md_text(item) for item in access.available_evidence)
        lines.append(f"- Available evidence: {joined}")
    if access.missing_evidence:
        joined = ", ".join(_md_text(item) for item in access.missing_evidence)
        lines.append(f"- Missing evidence: {joined}")
    for note in access.notes:
        lines.append(f"- Note: {_md_text(note)}")
    lines.append("")


def _append_evidence(lines: list[str], report: BehaviorReport) -> None:
    """Append evidence registry entries."""
    if not report.evidence:
        return
    lines.extend(["## Evidence", ""])
    lines.extend([
        "| ID | Kind | Reference | Description |",
        "| -- | ---- | --------- | ----------- |",
    ])
    for evidence in report.evidence:
        lines.append(
            "| "
            f"{_md_text(evidence.evidence_id)} | "
            f"{evidence.kind} | "
            f"{_md_text(evidence.path_or_url or '')} | "
            f"{_md_text(evidence.description or '')} |",
        )
    lines.append("")


def _append_claims(lines: list[str], report: BehaviorReport) -> None:
    """Append access-aware report claims."""
    if not report.claims:
        return
    lines.extend(["## Claims", ""])
    lines.extend([
        "| ID | Status | Strength | Access | Statement | Evidence |",
        "| -- | ------ | -------- | ------ | --------- | -------- |",
    ])
    for claim in report.claims:
        evidence = ", ".join(_md_text(item) for item in claim.evidence)
        lines.append(
            "| "
            f"{_md_text(claim.claim_id)} | "
            f"{claim.status} | "
            f"{claim.strength} | "
            f"{claim.access_level} | "
            f"{_md_text(claim.statement)} | "
            f"{evidence} |",
        )
    lines.append("")


def _append_reproduction_targets(
    lines: list[str],
    report: BehaviorReport,
) -> None:
    """Append reproduction targets and Vauban-native extensions."""
    if not report.reproduction_targets:
        return
    lines.extend(["## Reproduction Targets", ""])
    lines.extend([
        "| ID | Status | Source | Original Claim | Vauban Extension |",
        "| -- | ------ | ------ | -------------- | ---------------- |",
    ])
    for target in report.reproduction_targets:
        source = target.source_url or target.title
        lines.append(
            "| "
            f"{_md_text(target.target_id)} | "
            f"{target.status} | "
            f"{_md_text(source)} | "
            f"{_md_text(target.original_claim)} | "
            f"{_md_text(target.planned_extension)} |",
        )
    lines.append("")


def _append_reproduction_results(
    lines: list[str],
    report: BehaviorReport,
) -> None:
    """Append observed reproduction outcomes."""
    if not report.reproduction_results:
        return
    lines.extend(["## Reproduction Results", ""])
    lines.extend([
        "| Target | Status | Summary | Evidence |",
        "| ------ | ------ | ------- | -------- |",
    ])
    for result in report.reproduction_results:
        evidence = ", ".join(_md_text(item) for item in result.evidence)
        lines.append(
            "| "
            f"{_md_text(result.target_id)} | "
            f"{result.status} | "
            f"{_md_text(result.summary)} | "
            f"{evidence} |",
        )
    lines.append("")

    for result in report.reproduction_results:
        lines.append(f"### {_md_text(result.target_id)}")
        _append_named_items(lines, "Replicated", result.replicated_claims)
        _append_named_items(lines, "Not replicated", result.failed_claims)
        _append_named_items(lines, "Extensions", result.extensions)
        _append_named_items(lines, "Limitations", result.limitations)
        lines.append("")


def _append_named_items(
    lines: list[str],
    label: str,
    items: tuple[str, ...],
) -> None:
    """Append a compact list of named reproduction-result items."""
    if not items:
        return
    joined = "; ".join(_md_text(item) for item in items)
    lines.append(f"- {label}: {joined}")


def _append_findings(lines: list[str], report: BehaviorReport) -> None:
    """Append high-level behavior-change findings."""
    if not report.findings:
        return
    lines.extend(["## Findings", ""])
    for finding in report.findings:
        lines.append(f"- {_md_text(finding)}")
    lines.append("")


def _append_metric_delta_table(
    lines: list[str],
    report: BehaviorReport,
) -> None:
    """Append the behavioral deltas table."""
    lines.extend(["## Behavioral Deltas", ""])
    if not report.metric_deltas:
        lines.extend(["No metric deltas recorded.", ""])
        return

    lines.extend([
        "| Metric | Category | A | B | Delta | Quality |",
        "| ------ | -------- | -: | -: | ----: | ------- |",
    ])
    for delta in report.metric_deltas:
        category = delta.category or "all"
        lines.append(
            "| "
            f"{_md_text(delta.name)} | "
            f"{_md_text(category)} | "
            f"{_format_float(delta.value_baseline)} | "
            f"{_format_float(delta.value_candidate)} | "
            f"{_format_signed_float(delta.delta)} | "
            f"{delta.quality} |",
        )
    lines.append("")


def _append_activation_findings(
    lines: list[str],
    report: BehaviorReport,
) -> None:
    """Append activation-space finding bullets."""
    lines.extend(["## Activation Findings", ""])
    if not report.activation_findings:
        lines.extend(["No activation findings recorded.", ""])
        return

    for finding in report.activation_findings:
        layer_text = (
            ", ".join(str(layer) for layer in finding.layers)
            if finding.layers
            else "not specified"
        )
        score_text = (
            f"; score={_format_float(finding.score)}"
            if finding.score is not None
            else ""
        )
        lines.append(
            f"- **{_md_text(finding.name)}** ({finding.severity};"
            f" layers={layer_text}{score_text}): {_md_text(finding.summary)}",
        )
    lines.append("")


def _append_examples(lines: list[str], report: BehaviorReport) -> None:
    """Append representative examples."""
    lines.extend(["## Representative Examples", ""])
    if not report.examples:
        lines.extend(["No representative examples recorded.", ""])
        return

    lines.extend([
        "| ID | Category | Redaction | Prompt | Note |",
        "| -- | -------- | --------- | ------ | ---- |",
    ])
    for example in report.examples:
        lines.append(
            "| "
            f"{_md_text(example.example_id)} | "
            f"{_md_text(example.category)} | "
            f"{example.redaction} | "
            f"{_md_text(example.prompt)} | "
            f"{_md_text(example.note or '')} |",
        )
    lines.append("")


def _append_recommendation(lines: list[str], report: BehaviorReport) -> None:
    """Append report recommendation when provided."""
    if report.recommendation is None:
        return
    lines.extend([
        "## Recommendation",
        "",
        f"- {_md_text(report.recommendation)}",
        "",
    ])


def _append_limitations(lines: list[str], report: BehaviorReport) -> None:
    """Append report limitations."""
    lines.extend(["## Limitations", ""])
    if not report.limitations:
        lines.extend(["No limitations recorded.", ""])
        return

    for limitation in report.limitations:
        lines.append(f"- {_md_text(limitation)}")
    lines.append("")


def _append_reproducibility(
    lines: list[str],
    report: BehaviorReport,
) -> None:
    """Append reproducibility details."""
    lines.extend(["## Reproducibility", ""])
    if report.reproducibility is None:
        lines.extend(["No reproducibility details recorded.", ""])
        return

    data = report.reproducibility.to_dict()
    for key in (
        "command",
        "config_path",
        "code_revision",
        "output_dir",
        "seed",
    ):
        value = data.get(key)
        if value is not None:
            lines.append(f"- {key}: `{_md_text(str(value))}`")
    data_refs = _json_string_list(data.get("data_refs"))
    if data_refs:
        lines.append(f"- data_refs: {', '.join(_md_text(ref) for ref in data_refs)}")
    notes = _json_string_list(data.get("notes"))
    for note in notes:
        lines.append(f"- note: {_md_text(note)}")
    lines.append("")


def _model_summary(label: str, model_path: str) -> str:
    """Return a compact model summary for Markdown."""
    return f"{_md_text(label)} (`{_md_text(model_path)}`)"


def _format_float(value: float) -> str:
    """Format one float consistently for report tables."""
    return f"{value:.3f}"


def _format_signed_float(value: float) -> str:
    """Format one signed float consistently for report tables."""
    sign = "+" if value >= 0.0 else ""
    return f"{sign}{value:.3f}"


def _json_string_list(value: JsonValue | None) -> list[str]:
    """Coerce one JSON value to a list of strings if possible."""
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if isinstance(item, str):
            result.append(item)
    return result


def _md_text(text: str) -> str:
    """Escape minimal Markdown table-sensitive characters."""
    return text.replace("|", "\\|").replace("\n", "<br>")
