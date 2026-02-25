"""Compare JSON reports from two vauban output directories."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MetricDelta:
    """A single metric comparison between two runs."""

    name: str
    value_a: float
    value_b: float
    delta: float
    lower_is_better: bool


@dataclass(frozen=True, slots=True)
class ReportDiff:
    """All metric deltas for a single report file."""

    filename: str
    metrics: list[MetricDelta]


# Maps report filename -> list of (json_path, metric_name, lower_is_better)
_EXTRACTORS: dict[str, list[tuple[list[str], str, bool]]] = {
    "eval_report.json": [
        (["refusal_rate_modified"], "refusal_rate_modified", True),
        (["perplexity_modified"], "perplexity_modified", True),
        (["kl_divergence"], "kl_divergence", True),
    ],
    "surface_report.json": [
        (["summary", "refusal_rate_delta"], "refusal_rate_delta", True),
        (["summary", "threshold_delta"], "threshold_delta", False),
    ],
    "depth_report.json": [
        # Computed from dtr_results: mean DTR across prompts
    ],
    "softprompt_report.json": [
        (["success_rate"], "success_rate", False),
        (["final_loss"], "final_loss", True),
    ],
    "detect_report.json": [
        (["hardened"], "hardened", False),
        (["confidence"], "confidence", False),
    ],
    "optimize_report.json": [
        # best_refusal.refusal_rate
    ],
}


def _extract_value(data: dict[str, object], path: list[str]) -> float | None:
    """Walk a nested dict by key path, return float or None."""
    current: object = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)  # type: ignore[union-attr]
    if isinstance(current, int | float):
        return float(current)
    if isinstance(current, bool):
        return 1.0 if current else 0.0
    return None


def _get_nested_float(
    data: dict[str, object], key: str,
) -> float | None:
    """Extract a numeric value from a dict, returning float or None."""
    val = data.get(key)
    if isinstance(val, bool):
        return 1.0 if val else 0.0
    if isinstance(val, int | float):
        return float(val)
    return None


def _extract_depth_mean_dtr(data: dict[str, object]) -> float | None:
    """Compute mean DTR across all depth prompts."""
    results = data.get("dtr_results")
    if not isinstance(results, list) or not results:
        return None
    dtrs: list[float] = []
    for r in results:
        if not isinstance(r, dict):
            continue
        row: dict[str, object] = r  # type: ignore[assignment]
        val = _get_nested_float(row, "deep_thinking_ratio")
        if val is not None:
            dtrs.append(val)
    if not dtrs:
        return None
    return sum(dtrs) / len(dtrs)


def _extract_depth_mean_settling(data: dict[str, object]) -> float | None:
    """Compute mean settling depth across all depth prompts."""
    results = data.get("dtr_results")
    if not isinstance(results, list) or not results:
        return None
    depths: list[float] = []
    for r in results:
        if not isinstance(r, dict):
            continue
        row: dict[str, object] = r  # type: ignore[assignment]
        val = _get_nested_float(row, "mean_settling_depth")
        if val is not None:
            depths.append(val)
    if not depths:
        return None
    return sum(depths) / len(depths)


def _extract_optimize_best_refusal(data: dict[str, object]) -> float | None:
    """Extract best refusal rate from optimize report."""
    best = data.get("best_refusal")
    if not isinstance(best, dict):
        return None
    row: dict[str, object] = best  # type: ignore[assignment]
    return _get_nested_float(row, "refusal_rate")


def _extract_metrics(
    filename: str,
    data_a: dict[str, object],
    data_b: dict[str, object],
) -> list[MetricDelta]:
    """Extract all comparable metrics from a pair of reports."""
    metrics: list[MetricDelta] = []

    # Standard path-based extractors
    extractors = _EXTRACTORS.get(filename, [])
    for path, name, lower_is_better in extractors:
        va = _extract_value(data_a, path)
        vb = _extract_value(data_b, path)
        if va is not None and vb is not None:
            metrics.append(MetricDelta(
                name=name,
                value_a=va,
                value_b=vb,
                delta=vb - va,
                lower_is_better=lower_is_better,
            ))

    # Special extractors for depth
    if filename == "depth_report.json":
        va_dtr = _extract_depth_mean_dtr(data_a)
        vb_dtr = _extract_depth_mean_dtr(data_b)
        if va_dtr is not None and vb_dtr is not None:
            metrics.append(MetricDelta(
                name="mean_dtr",
                value_a=va_dtr,
                value_b=vb_dtr,
                delta=vb_dtr - va_dtr,
                lower_is_better=False,
            ))
        va_sd = _extract_depth_mean_settling(data_a)
        vb_sd = _extract_depth_mean_settling(data_b)
        if va_sd is not None and vb_sd is not None:
            metrics.append(MetricDelta(
                name="mean_settling_depth",
                value_a=va_sd,
                value_b=vb_sd,
                delta=vb_sd - va_sd,
                lower_is_better=False,
            ))

    # Special extractor for optimize
    if filename == "optimize_report.json":
        va_br = _extract_optimize_best_refusal(data_a)
        vb_br = _extract_optimize_best_refusal(data_b)
        if va_br is not None and vb_br is not None:
            metrics.append(MetricDelta(
                name="best_refusal_rate",
                value_a=va_br,
                value_b=vb_br,
                delta=vb_br - va_br,
                lower_is_better=True,
            ))

    return metrics


def diff_reports(dir_a: Path, dir_b: Path) -> list[ReportDiff]:
    """Compare JSON reports from two output directories.

    Args:
        dir_a: First output directory.
        dir_b: Second output directory.

    Returns:
        List of ReportDiff for each shared report file.

    Raises:
        FileNotFoundError: If either directory does not exist.
    """
    if not dir_a.is_dir():
        msg = f"Directory not found: {dir_a}"
        raise FileNotFoundError(msg)
    if not dir_b.is_dir():
        msg = f"Directory not found: {dir_b}"
        raise FileNotFoundError(msg)

    results: list[ReportDiff] = []
    for filename in sorted(_EXTRACTORS):
        file_a = dir_a / filename
        file_b = dir_b / filename
        if not file_a.exists() or not file_b.exists():
            continue

        data_a: dict[str, object] = json.loads(file_a.read_text())
        data_b: dict[str, object] = json.loads(file_b.read_text())

        metrics = _extract_metrics(filename, data_a, data_b)
        if metrics:
            results.append(ReportDiff(filename=filename, metrics=metrics))

    return results


def format_diff(
    dir_a: Path,
    dir_b: Path,
    reports: list[ReportDiff],
) -> str:
    """Format diff reports as a human-readable string.

    Args:
        dir_a: First directory (for header).
        dir_b: Second directory (for header).
        reports: ReportDiff list from diff_reports().

    Returns:
        Formatted multi-line string.
    """
    if not reports:
        return f"DIFF {dir_a} vs {dir_b}\n\nNo shared reports found.\n"

    lines: list[str] = [f"DIFF {dir_a} vs {dir_b}\n"]

    for report in reports:
        lines.append(f"{report.filename}")
        for m in report.metrics:
            sign = "+" if m.delta >= 0 else ""
            if m.lower_is_better:
                quality = "better" if m.delta < 0 else "worse"
            else:
                quality = "better" if m.delta > 0 else "worse"

            # Skip quality label for zero delta
            quality_str = "" if m.delta == 0.0 else f" {quality}"

            lines.append(
                f"  {m.name:<30s}"
                f" {m.value_a:.3f} → {m.value_b:.3f}"
                f"  ({sign}{m.delta:.3f}){quality_str}",
            )
        lines.append("")

    return "\n".join(lines) + "\n"
