# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Controlled intervention evaluation early-mode runner."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, cast

from vauban._pipeline._context import EarlyModeContext, log
from vauban._pipeline._mode_common import (
    ModeReport,
    finish_mode_run,
    write_mode_report,
)
from vauban.behavior import (
    InterventionEffect,
    InterventionKind,
    InterventionPolarity,
    InterventionResult,
)
from vauban.types import (
    InterventionConditionSummary,
    InterventionPromptResult,
)

if TYPE_CHECKING:
    from vauban.types import (
        CausalLM,
        InterventionEvalConfig,
        InterventionEvalPrompt,
        SteerResult,
        Tokenizer,
    )


_INTERVENTION_KINDS: tuple[InterventionKind, ...] = (
    "activation_steering",
    "activation_ablation",
    "activation_addition",
    "weight_projection",
    "weight_arithmetic",
    "prompt_template",
    "sampling",
    "other",
)


def _run_intervention_eval_mode(context: EarlyModeContext) -> None:
    """Run [intervention_eval] and write report artifacts."""
    config = context.config
    eval_config = config.intervention_eval
    if eval_config is None:
        msg = "intervention_eval config is required for intervention_eval mode"
        raise ValueError(msg)
    if context.direction_result is None:
        msg = "direction_result is required for intervention_eval mode"
        raise ValueError(msg)

    model = cast("CausalLM", context.model)
    tokenizer = cast("Tokenizer", context.tokenizer)

    from vauban._forward import get_transformer
    from vauban.probe import steer

    n_layers = len(get_transformer(model).layers)
    layers = eval_config.layers or list(range(n_layers))
    log(
        (
            "Running intervention evaluation"
            f" — prompts={len(eval_config.prompts)},"
            f" alphas={len(eval_config.alphas)}"
        ),
        verbose=config.verbose,
        elapsed=time.monotonic() - context.t0,
    )

    prompt_results: list[InterventionPromptResult] = []
    for alpha in eval_config.alphas:
        for prompt in eval_config.prompts:
            steer_result = steer(
                model,
                tokenizer,
                prompt.prompt,
                context.direction_result.direction,
                layers,
                alpha,
                eval_config.max_tokens,
            )
            prompt_results.append(
                _prompt_result(
                    prompt,
                    alpha,
                    steer_result,
                    eval_config,
                ),
            )

    summaries = _summarize_conditions(prompt_results, eval_config.alphas)
    intervention_results = _intervention_results(eval_config, layers, summaries)
    payload = _report_payload(
        eval_config,
        layers,
        prompt_results,
        summaries,
        intervention_results,
    )
    json_path = write_mode_report(
        config.output_dir,
        ModeReport(eval_config.json_filename, payload),
    )
    markdown_path = config.output_dir / eval_config.markdown_filename
    markdown_path.write_text(
        _render_markdown(eval_config, summaries, intervention_results),
        encoding="utf-8",
    )
    toml_path = config.output_dir / eval_config.toml_fragment_filename
    toml_path.write_text(
        _render_toml_fragment(eval_config, intervention_results),
        encoding="utf-8",
    )

    baseline = _summary_for_alpha(summaries, eval_config.baseline_alpha)
    finish_mode_run(
        context,
        "intervention_eval",
        [str(json_path), str(markdown_path), str(toml_path)],
        {
            "n_prompts": len(eval_config.prompts),
            "n_conditions": len(eval_config.alphas),
            "n_prompt_results": len(prompt_results),
            "n_intervention_results": len(intervention_results),
            "baseline_refusal_style_rate": baseline.refusal_style_rate,
        },
    )


def _prompt_result(
    prompt: InterventionEvalPrompt,
    alpha: float,
    steer_result: SteerResult,
    config: InterventionEvalConfig,
) -> InterventionPromptResult:
    """Build one prompt result from a steered generation."""
    output_text = steer_result.text if config.record_outputs else None
    return InterventionPromptResult(
        prompt_id=prompt.prompt_id,
        category=prompt.category,
        alpha=alpha,
        refused=_contains_refusal_phrase(
            steer_result.text,
            config.refusal_phrases,
        ),
        mean_projection_before=_mean(steer_result.projections_before),
        mean_projection_after=_mean(steer_result.projections_after),
        output_text=output_text,
    )


def _contains_refusal_phrase(text: str, phrases: list[str]) -> bool:
    """Return whether generated text contains one configured refusal phrase."""
    lower = text.lower()
    return any(phrase.lower() in lower for phrase in phrases)


def _summarize_conditions(
    results: list[InterventionPromptResult],
    alphas: list[float],
) -> list[InterventionConditionSummary]:
    """Aggregate prompt results by alpha condition."""
    summaries: list[InterventionConditionSummary] = []
    for alpha in alphas:
        condition = [result for result in results if result.alpha == alpha]
        n_prompts = len(condition)
        refusal_rate = (
            sum(1 for result in condition if result.refused) / n_prompts
            if n_prompts
            else 0.0
        )
        mean_before = _mean([
            result.mean_projection_before for result in condition
        ])
        mean_after = _mean([
            result.mean_projection_after for result in condition
        ])
        summaries.append(
            InterventionConditionSummary(
                alpha=alpha,
                n_prompts=n_prompts,
                refusal_style_rate=refusal_rate,
                mean_projection_before=mean_before,
                mean_projection_after=mean_after,
                mean_projection_delta=mean_after - mean_before,
            ),
        )
    return summaries


def _intervention_results(
    config: InterventionEvalConfig,
    layers: list[int],
    summaries: list[InterventionConditionSummary],
) -> list[InterventionResult]:
    """Convert aggregate condition summaries into report primitives."""
    baseline = _summary_for_alpha(summaries, config.baseline_alpha)
    results: list[InterventionResult] = []
    for summary in summaries:
        if summary.alpha == config.baseline_alpha:
            continue
        delta = summary.refusal_style_rate - baseline.refusal_style_rate
        results.append(
            InterventionResult(
                intervention_id=_intervention_id(summary.alpha),
                kind=_intervention_kind(config.kind),
                summary=_summary_sentence(summary.alpha, delta),
                target=config.target,
                effect=_effect_from_delta(delta),
                polarity=_polarity(summary.alpha, config.baseline_alpha),
                layers=tuple(layers),
                strength=summary.alpha,
                baseline_condition=f"alpha={config.baseline_alpha:g}",
                intervention_condition=f"alpha={summary.alpha:g}",
                behavior_metric=config.behavior_metric,
                activation_metric=config.activation_metric,
                evidence=("intervention_eval_report",),
                limitations=tuple(config.limitations),
            ),
        )
    return results


def _summary_for_alpha(
    summaries: list[InterventionConditionSummary],
    alpha: float,
) -> InterventionConditionSummary:
    """Return the summary for one alpha value."""
    for summary in summaries:
        if summary.alpha == alpha:
            return summary
    msg = f"missing intervention summary for alpha={alpha:g}"
    raise ValueError(msg)


def _effect_from_delta(delta: float) -> InterventionEffect:
    """Map a metric delta to an intervention effect label."""
    if delta > 0.0:
        return "increased"
    if delta < 0.0:
        return "decreased"
    return "no_observed_change"


def _polarity(alpha: float, baseline_alpha: float) -> InterventionPolarity:
    """Map alpha relative to the baseline alpha into report polarity."""
    if alpha > baseline_alpha:
        return "positive"
    if alpha < baseline_alpha:
        return "negative"
    return "control"


def _intervention_kind(value: str) -> InterventionKind:
    """Validate and return a behavior-report intervention kind."""
    if value not in _INTERVENTION_KINDS:
        msg = f"intervention kind must be one of {_INTERVENTION_KINDS!r}"
        raise ValueError(msg)
    return cast("InterventionKind", value)


def _intervention_id(alpha: float) -> str:
    """Return a stable ID for one alpha condition."""
    sign = "pos" if alpha >= 0.0 else "neg"
    magnitude = str(abs(alpha)).replace(".", "_")
    return f"{sign}_alpha_{magnitude}"


def _summary_sentence(alpha: float, delta: float) -> str:
    """Return a compact human-readable intervention summary."""
    return (
        f"Alpha {alpha:g} changed refusal-style rate by {delta:+.3f}"
        " relative to the baseline alpha."
    )


def _report_payload(
    config: InterventionEvalConfig,
    layers: list[int],
    prompt_results: list[InterventionPromptResult],
    summaries: list[InterventionConditionSummary],
    intervention_results: list[InterventionResult],
) -> dict[str, object]:
    """Build the JSON report payload."""
    return {
        "report_version": "intervention_eval_v1",
        "target": config.target,
        "kind": config.kind,
        "layers": layers,
        "alphas": config.alphas,
        "baseline_alpha": config.baseline_alpha,
        "behavior_metric": config.behavior_metric,
        "activation_metric": config.activation_metric,
        "record_outputs": config.record_outputs,
        "prompt_results": [result.to_dict() for result in prompt_results],
        "condition_summaries": [summary.to_dict() for summary in summaries],
        "intervention_results": [
            result.to_dict() for result in intervention_results
        ],
    }


def _render_markdown(
    config: InterventionEvalConfig,
    summaries: list[InterventionConditionSummary],
    intervention_results: list[InterventionResult],
) -> str:
    """Render a compact Markdown intervention-evaluation report."""
    lines: list[str] = [
        "# Intervention Evaluation Report",
        "",
        f"- Target: `{config.target}`",
        f"- Kind: `{config.kind}`",
        f"- Baseline alpha: `{config.baseline_alpha:g}`",
        f"- Behavior metric: `{config.behavior_metric}`",
        f"- Activation metric: `{config.activation_metric}`",
        "",
        "## Condition Summaries",
        "",
        (
            "| Alpha | Prompts | Refusal Style Rate | Projection Before |"
            " Projection After | Projection Delta |"
        ),
        (
            "| ----: | ------: | -----------------: | ----------------: |"
            " ---------------: | ---------------: |"
        ),
    ]
    for summary in summaries:
        lines.append(
            "| "
            f"{summary.alpha:.3f} | "
            f"{summary.n_prompts} | "
            f"{summary.refusal_style_rate:.3f} | "
            f"{summary.mean_projection_before:.3f} | "
            f"{summary.mean_projection_after:.3f} | "
            f"{summary.mean_projection_delta:+.3f} |",
        )
    lines.extend(["", "## Behavior Report Fragment", ""])
    if not intervention_results:
        lines.extend(["No non-baseline intervention results recorded.", ""])
        return "\n".join(lines).rstrip() + "\n"
    for result in intervention_results:
        lines.append(
            f"- `{result.intervention_id}`: {result.effect};"
            f" {result.summary}",
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_toml_fragment(
    config: InterventionEvalConfig,
    intervention_results: list[InterventionResult],
) -> str:
    """Render a TOML fragment for a Model Behavior Change Report."""
    lines: list[str] = [
        "[[behavior_report.evidence]]",
        'id = "intervention_eval_report"',
        'kind = "run_report"',
        f"path_or_url = {json.dumps(config.json_filename)}",
        'description = "Controlled intervention evaluation report."',
        "",
    ]
    for result in intervention_results:
        lines.extend([
            "[[behavior_report.intervention_results]]",
            f"id = {json.dumps(result.intervention_id)}",
            f"kind = {json.dumps(result.kind)}",
            f"summary = {json.dumps(result.summary)}",
            f"target = {json.dumps(result.target)}",
            f"effect = {json.dumps(result.effect)}",
            f"polarity = {json.dumps(result.polarity)}",
            f"layers = {_toml_int_list(list(result.layers))}",
        ])
        if result.strength is not None:
            lines.append(f"strength = {result.strength:g}")
        if result.baseline_condition is not None:
            lines.append(
                f"baseline_condition = {json.dumps(result.baseline_condition)}",
            )
        if result.intervention_condition is not None:
            lines.append(
                "intervention_condition = "
                f"{json.dumps(result.intervention_condition)}",
            )
        if result.behavior_metric is not None:
            lines.append(f"behavior_metric = {json.dumps(result.behavior_metric)}")
        if result.activation_metric is not None:
            lines.append(
                f"activation_metric = {json.dumps(result.activation_metric)}",
            )
        lines.append('evidence = ["intervention_eval_report"]')
        if result.limitations:
            lines.append(f"limitations = {_toml_string_list(list(result.limitations))}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _toml_int_list(values: list[int]) -> str:
    """Render a list of integers as TOML."""
    return "[" + ", ".join(str(value) for value in values) + "]"


def _toml_string_list(values: list[str]) -> str:
    """Render a list of strings as TOML."""
    return "[" + ", ".join(json.dumps(value) for value in values) + "]"


def _mean(values: list[float]) -> float:
    """Return the arithmetic mean of a list, or zero for an empty list."""
    if not values:
        return 0.0
    return sum(values) / len(values)
