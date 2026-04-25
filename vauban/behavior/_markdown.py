# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Markdown rendering for typed Vauban Behavior Reports."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vauban.behavior._primitives import BehaviorReport, JsonValue


def render_behavior_report_markdown(report: BehaviorReport) -> str:
    """Render a Vauban Behavior Report as deterministic Markdown."""
    lines: list[str] = [
        f"# {_md_text(report.title)}",
        "",
        "Vauban Behavior Report",
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

    _append_metric_delta_table(lines, report)
    _append_activation_findings(lines, report)
    _append_examples(lines, report)
    _append_limitations(lines, report)
    _append_reproducibility(lines, report)
    return "\n".join(lines).rstrip() + "\n"


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
