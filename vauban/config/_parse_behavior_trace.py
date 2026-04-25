# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Parse the [behavior_trace] section of a TOML config."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from vauban.behavior import (
    DEFAULT_BEHAVIOR_METRIC_SPECS,
    DEFAULT_BEHAVIOR_SCORERS,
    BehaviorMetricSpec,
    BehaviorPrompt,
    behavior_metric_specs_for_scorers,
    load_behavior_suite_toml,
    parse_behavior_metric_specs,
    parse_behavior_prompts,
    validate_behavior_scorer_names,
)
from vauban.config._parse_helpers import SectionReader, require_toml_table
from vauban.evaluate import DEFAULT_REFUSAL_PHRASES
from vauban.types import (
    BehaviorDiffMetricConfig,
    BehaviorTraceConfig,
    BehaviorTracePromptConfig,
)

if TYPE_CHECKING:
    from vauban.config._types import TomlDict


def _parse_behavior_trace(
    base_dir: Path,
    raw: TomlDict,
) -> BehaviorTraceConfig | None:
    """Parse the optional [behavior_trace] section."""
    sec = raw.get("behavior_trace")
    if sec is None:
        return None
    reader = SectionReader(
        "[behavior_trace]",
        require_toml_table("[behavior_trace]", sec),
    )

    suite_path = _optional_path(base_dir, reader.optional_string("suite"))
    suite = load_behavior_suite_toml(suite_path) if suite_path is not None else None
    raw_scorers = reader.optional_string_list("scorers")
    suite_scorers = suite.scorers if suite is not None else None
    scorer_names = _scorer_names(raw_scorers, suite_scorers=suite_scorers)
    inline_prompts = parse_behavior_prompts(
        reader.data.get("prompts"),
        "[behavior_trace].prompts",
    )
    suite_prompts = suite.prompts if suite is not None else ()
    prompts = [*_prompt_configs(suite_prompts), *_prompt_configs(inline_prompts)]
    if not prompts:
        msg = (
            "[behavior_trace] requires prompts either via suite = \"...\""
            " or [[behavior_trace.prompts]]"
        )
        raise ValueError(msg)
    _reject_duplicate_prompt_ids(prompts)

    metric_specs = _merge_metric_specs(
        _base_metric_specs(
            suite_metrics=suite.metric_specs if suite is not None else None,
            scorer_names=scorer_names,
            trace_overrides_scorers=raw_scorers is not None,
        ),
        parse_behavior_metric_specs(reader.data.get("metrics")),
    )

    max_tokens = reader.integer("max_tokens", default=80)
    if max_tokens < 1:
        msg = "[behavior_trace].max_tokens must be >= 1"
        raise ValueError(msg)

    refusal_phrases = reader.string_list(
        "refusal_phrases",
        default=list(DEFAULT_REFUSAL_PHRASES),
    )
    if not refusal_phrases:
        msg = "[behavior_trace].refusal_phrases must be non-empty"
        raise ValueError(msg)

    return BehaviorTraceConfig(
        model_label=reader.string("model_label", default="model"),
        suite=suite_path,
        suite_name=reader.string(
            "suite_name",
            default=suite.name if suite is not None else "behavior-change-suite",
        ),
        suite_description=reader.string(
            "suite_description",
            default=(
                suite.description
                if suite is not None
                else "Behavior trace collection suite."
            ),
        ),
        suite_version=(
            reader.optional_string("suite_version")
            or (suite.version if suite is not None else None)
        ),
        suite_source=(
            reader.optional_string("suite_source")
            or (suite.source if suite is not None else None)
        ),
        safety_policy=reader.string(
            "safety_policy",
            default=(
                suite.safety_policy
                if suite is not None
                else "safe_or_redacted_prompts"
            ),
        ),
        prompts=prompts,
        metrics=_metric_configs(metric_specs),
        scorers=list(scorer_names),
        max_tokens=max_tokens,
        refusal_phrases=refusal_phrases,
        record_outputs=reader.boolean("record_outputs", default=False),
        output_trace=_optional_path(
            base_dir,
            reader.optional_string("output_trace"),
        ),
        trace_filename=reader.string(
            "trace_filename",
            default="behavior_trace.jsonl",
        ),
        json_filename=reader.string(
            "json_filename",
            default="behavior_trace_report.json",
        ),
    )


def _scorer_names(
    raw_scorers: list[str] | None,
    *,
    suite_scorers: tuple[str, ...] | None,
) -> tuple[str, ...]:
    """Resolve behavior scorer names from trace override, suite, or defaults."""
    if raw_scorers is not None:
        return validate_behavior_scorer_names(
            tuple(raw_scorers),
            field="[behavior_trace].scorers",
        )
    if suite_scorers is not None:
        return validate_behavior_scorer_names(
            suite_scorers,
            field="[behavior_suite].scorers",
        )
    return DEFAULT_BEHAVIOR_SCORERS


def _base_metric_specs(
    *,
    suite_metrics: tuple[BehaviorMetricSpec, ...] | None,
    scorer_names: tuple[str, ...],
    trace_overrides_scorers: bool,
) -> tuple[BehaviorMetricSpec, ...]:
    """Return inherited metric specs for the resolved scorer configuration."""
    if suite_metrics is not None and not trace_overrides_scorers:
        return suite_metrics
    if scorer_names == DEFAULT_BEHAVIOR_SCORERS:
        return DEFAULT_BEHAVIOR_METRIC_SPECS
    return behavior_metric_specs_for_scorers(scorer_names)


def _prompt_configs(
    prompts: tuple[BehaviorPrompt, ...],
) -> list[BehaviorTracePromptConfig]:
    """Convert behavior-suite prompts into trace config prompt records."""
    return [
        BehaviorTracePromptConfig(
            prompt_id=prompt.prompt_id,
            text=prompt.prompt,
            category=prompt.category,
            expected_behavior=prompt.expected_behavior,
            redaction=prompt.redaction,
            tags=list(prompt.tags),
        )
        for prompt in prompts
    ]


def _metric_configs(
    metrics: tuple[BehaviorMetricSpec, ...],
) -> list[BehaviorDiffMetricConfig]:
    """Convert behavior metric specs into TOML config metric records."""
    return [
        BehaviorDiffMetricConfig(
            name=metric.name,
            description=metric.description,
            polarity=metric.polarity,
            unit=metric.unit,
            family=metric.family,
        )
        for metric in metrics
    ]


def _merge_metric_specs(
    base_metrics: tuple[BehaviorMetricSpec, ...],
    override_metrics: tuple[BehaviorMetricSpec, ...],
) -> tuple[BehaviorMetricSpec, ...]:
    """Merge metric declarations by name, with later declarations winning."""
    merged: dict[str, BehaviorMetricSpec] = {}
    order: list[str] = []
    for metric in (*base_metrics, *override_metrics):
        if metric.name not in merged:
            order.append(metric.name)
        merged[metric.name] = metric
    return tuple(merged[name] for name in order)


def _optional_path(base_dir: Path, raw: str | None) -> Path | None:
    """Resolve an optional path string relative to the config directory."""
    if raw is None:
        return None
    path = Path(raw)
    return path if path.is_absolute() else base_dir / path


def _reject_duplicate_prompt_ids(
    prompts: list[BehaviorTracePromptConfig],
) -> None:
    """Reject duplicate prompt IDs before trace collection."""
    seen: set[str] = set()
    for prompt in prompts:
        if prompt.prompt_id in seen:
            msg = f"[behavior_trace] duplicate prompt id: {prompt.prompt_id!r}"
            raise ValueError(msg)
        seen.add(prompt.prompt_id)
