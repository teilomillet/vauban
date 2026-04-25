# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Parse the [benchmark] section of a TOML config."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from vauban.config._parse_helpers import SectionReader, require_toml_table
from vauban.types import (
    BenchmarkConfig,
    BenchmarkModelConfig,
    BenchmarkWeightsConfig,
)

if TYPE_CHECKING:
    from vauban.config._types import TomlDict


def _resolve_optional_path(base_dir: Path, raw: str | None) -> Path | None:
    """Resolve an optional path relative to *base_dir*."""
    if raw is None:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path
    return base_dir / path


def _parse_benchmark(base_dir: Path, raw: TomlDict) -> BenchmarkConfig | None:
    """Parse the optional [benchmark] section into a BenchmarkConfig."""
    sec = raw.get("benchmark")
    if sec is None:
        return None
    reader = SectionReader(
        "[benchmark]",
        require_toml_table("[benchmark]", sec),
    )

    title = reader.string("title", default="Safety Benchmark")
    description = reader.string("description", default="")
    markdown_report = reader.boolean("markdown_report", default=True)

    models_raw = reader.data.get("models")
    if not isinstance(models_raw, list) or not models_raw:
        msg = "[benchmark].models must be a non-empty array of tables"
        raise ValueError(msg)
    models = [
        _parse_benchmark_model(base_dir, item, index)
        for index, item in enumerate(models_raw)
    ]
    labels = [model.label for model in models]
    if len(labels) != len(set(labels)):
        msg = "[benchmark].models labels must be unique"
        raise ValueError(msg)

    weights = _parse_benchmark_weights(reader.data.get("weights"))

    return BenchmarkConfig(
        title=title,
        description=description,
        models=models,
        weights=weights,
        markdown_report=markdown_report,
    )


def _parse_benchmark_model(
    base_dir: Path,
    raw: object,
    index: int,
) -> BenchmarkModelConfig:
    """Parse one [[benchmark.models]] entry."""
    section = f"[benchmark.models[{index}]]"
    reader = SectionReader(section, require_toml_table(section, raw))

    label = reader.string("label")
    if not label.strip():
        msg = f"{section}.label must not be empty"
        raise ValueError(msg)

    report_dir = _resolve_optional_path(base_dir, reader.optional_string("report_dir"))
    model_path = reader.string("model_path", default="")
    audit_report = _resolve_optional_path(
        base_dir,
        reader.optional_string("audit_report"),
    )
    detect_report = _resolve_optional_path(
        base_dir,
        reader.optional_string("detect_report"),
    )
    guard_report = _resolve_optional_path(
        base_dir,
        reader.optional_string("guard_report"),
    )
    ai_act_report = _resolve_optional_path(
        base_dir,
        reader.optional_string("ai_act_report"),
    )
    eval_report = _resolve_optional_path(
        base_dir,
        reader.optional_string("eval_report"),
    )
    notes = reader.string("notes", default="")

    return BenchmarkModelConfig(
        label=label,
        report_dir=report_dir,
        model_path=model_path,
        audit_report=audit_report,
        detect_report=detect_report,
        guard_report=guard_report,
        ai_act_report=ai_act_report,
        eval_report=eval_report,
        notes=notes,
    )


def _parse_benchmark_weights(raw: object) -> BenchmarkWeightsConfig:
    """Parse the optional [benchmark.weights] table."""
    if raw is None:
        return BenchmarkWeightsConfig()
    reader = SectionReader(
        "[benchmark.weights]",
        require_toml_table("[benchmark.weights]", raw),
    )
    behavioral_safety = reader.number("behavioral_safety", default=0.6)
    tamper_resistance = reader.number("tamper_resistance", default=0.2)
    evidence_readiness = reader.number("evidence_readiness", default=0.2)

    for key, value in (
        ("behavioral_safety", behavioral_safety),
        ("tamper_resistance", tamper_resistance),
        ("evidence_readiness", evidence_readiness),
    ):
        if value < 0.0:
            msg = f"[benchmark.weights].{key} must be >= 0.0, got {value}"
            raise ValueError(msg)

    total = behavioral_safety + tamper_resistance + evidence_readiness
    if total <= 0.0:
        msg = "[benchmark.weights] must sum to a positive value"
        raise ValueError(msg)

    return BenchmarkWeightsConfig(
        behavioral_safety=behavioral_safety,
        tamper_resistance=tamper_resistance,
        evidence_readiness=evidence_readiness,
    )
