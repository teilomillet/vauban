# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Parse the [intervention_eval] section of a TOML config."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vauban.config._parse_helpers import SectionReader, require_toml_table
from vauban.evaluate import DEFAULT_REFUSAL_PHRASES
from vauban.types import InterventionEvalConfig, InterventionEvalPrompt

if TYPE_CHECKING:
    from vauban.config._types import TomlDict

_INTERVENTION_KIND_CHOICES: tuple[str, ...] = (
    "activation_steering",
    "activation_ablation",
    "activation_addition",
    "weight_projection",
    "weight_arithmetic",
    "prompt_template",
    "sampling",
    "other",
)


def _parse_intervention_eval(raw: TomlDict) -> InterventionEvalConfig | None:
    """Parse the optional [intervention_eval] section."""
    sec = raw.get("intervention_eval")
    if sec is None:
        return None
    reader = SectionReader(
        "[intervention_eval]",
        require_toml_table("[intervention_eval]", sec),
    )
    prompts = _parse_prompts(reader.data)
    if not prompts:
        msg = "[intervention_eval].prompts must contain at least one prompt"
        raise ValueError(msg)

    alphas = _number_list(reader, "alphas", default=[-1.0, 0.0, 1.0])
    baseline_alpha = reader.number("baseline_alpha", default=0.0)
    if baseline_alpha not in alphas:
        msg = "[intervention_eval].baseline_alpha must appear in alphas"
        raise ValueError(msg)

    max_tokens = reader.integer("max_tokens", default=80)
    if max_tokens < 1:
        msg = "[intervention_eval].max_tokens must be >= 1"
        raise ValueError(msg)

    refusal_phrases = reader.string_list(
        "refusal_phrases",
        default=list(DEFAULT_REFUSAL_PHRASES),
    )
    if not refusal_phrases:
        msg = "[intervention_eval].refusal_phrases must be non-empty"
        raise ValueError(msg)

    return InterventionEvalConfig(
        prompts=prompts,
        alphas=alphas,
        baseline_alpha=baseline_alpha,
        layers=reader.optional_int_list("layers"),
        max_tokens=max_tokens,
        target=reader.string("target", default="measured_direction"),
        kind=reader.literal(
            "kind",
            _INTERVENTION_KIND_CHOICES,
            default="activation_steering",
        ),
        behavior_metric=reader.string(
            "behavior_metric",
            default="refusal_style_rate",
        ),
        activation_metric=reader.string(
            "activation_metric",
            default="mean_projection_delta",
        ),
        refusal_phrases=refusal_phrases,
        limitations=reader.string_list("limitations", default=[]),
        record_outputs=reader.boolean("record_outputs", default=False),
        json_filename=reader.string(
            "json_filename",
            default="intervention_eval_report.json",
        ),
        markdown_filename=reader.string(
            "markdown_filename",
            default="intervention_eval_report.md",
        ),
        toml_fragment_filename=reader.string(
            "toml_fragment_filename",
            default="intervention_results.toml",
        ),
    )


def _parse_prompts(data: TomlDict) -> list[InterventionEvalPrompt]:
    """Parse simple prompt strings or prompt tables from section data."""
    prompts: list[InterventionEvalPrompt] = []
    raw_prompts = data.get("prompts")
    if raw_prompts is None:
        return prompts
    if not isinstance(raw_prompts, list):
        msg = "[intervention_eval].prompts must be a list"
        raise TypeError(msg)
    for index, item in enumerate(raw_prompts):
        if isinstance(item, str):
            prompts.append(
                InterventionEvalPrompt(
                    prompt_id=f"prompt-{index + 1:03d}",
                    prompt=item,
                ),
            )
            continue
        table = require_toml_table(f"[intervention_eval].prompts[{index}]", item)
        reader = SectionReader(f"[intervention_eval].prompts[{index}]", table)
        prompt_id = reader.optional_string("id")
        if prompt_id is None:
            prompt_id = reader.optional_string("prompt_id")
        if prompt_id is None:
            msg = f"[intervention_eval].prompts[{index}].id is required"
            raise ValueError(msg)
        prompts.append(
            InterventionEvalPrompt(
                prompt_id=prompt_id,
                prompt=reader.string("text"),
                category=reader.string("category", default="default"),
            ),
        )
    return prompts


def _number_list(
    reader: SectionReader,
    key: str,
    *,
    default: list[float],
) -> list[float]:
    """Read a TOML list of numeric values."""
    raw = reader.data.get(key, default)
    if not isinstance(raw, list):
        msg = f"{reader.section}.{key} must be a list of numbers"
        raise TypeError(msg)
    values: list[float] = []
    for item in raw:
        if not isinstance(item, int | float):
            msg = f"{reader.section}.{key} elements must be numbers"
            raise TypeError(msg)
        values.append(float(item))
    if not values:
        msg = f"{reader.section}.{key} must be non-empty"
        raise ValueError(msg)
    return values
