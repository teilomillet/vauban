# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Behavior trace collection early-mode runner."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, cast

from vauban._pipeline._context import EarlyModeContext, log
from vauban._pipeline._mode_common import (
    ModeReport,
    finish_mode_run,
    write_mode_report,
)
from vauban.behavior import (
    BehaviorObservation,
    BehaviorTrace,
    ExampleRedaction,
    ExpectedBehavior,
    JsonValue,
    is_refusal_text,
    score_behavior_output,
    write_behavior_trace,
)
from vauban.evaluate import _generate

if TYPE_CHECKING:
    from pathlib import Path

    from vauban.types import (
        BehaviorTraceConfig,
        BehaviorTracePromptConfig,
        CausalLM,
        Tokenizer,
    )


def _run_behavior_trace_mode(context: EarlyModeContext) -> None:
    """Run [behavior_trace] and write reusable JSONL trace artifacts."""
    config = context.config
    trace_config = config.behavior_trace
    if trace_config is None:
        msg = "[behavior_trace] section is required for behavior_trace mode"
        raise ValueError(msg)

    model = cast("CausalLM", context.model)
    tokenizer = cast("Tokenizer", context.tokenizer)
    log(
        (
            "Collecting behavior trace"
            f" — suite={trace_config.suite_name!r},"
            f" prompts={len(trace_config.prompts)}"
        ),
        verbose=config.verbose,
        elapsed=time.monotonic() - context.t0,
    )

    observations = _collect_observations(model, tokenizer, trace_config)
    trace_path = _trace_path(context.config.output_dir, trace_config)
    trace = BehaviorTrace(
        trace_id=trace_path.stem,
        model_label=trace_config.model_label,
        model_path=config.model_path,
        suite_name=trace_config.suite_name,
        source_path=str(trace_path),
        observations=tuple(observations),
        metadata={
            "suite_description": trace_config.suite_description,
            "suite_version": trace_config.suite_version,
            "suite_source": trace_config.suite_source,
            "safety_policy": trace_config.safety_policy,
            "record_outputs": trace_config.record_outputs,
            "max_tokens": trace_config.max_tokens,
        },
    )
    written_trace_path = write_behavior_trace(trace_path, trace)
    json_path = write_mode_report(
        config.output_dir,
        ModeReport(
            trace_config.json_filename,
            _report_payload(trace, trace_config, written_trace_path),
        ),
    )

    refused_count = sum(1 for observation in observations if observation.refused)
    finish_mode_run(
        context,
        "behavior_trace",
        [str(written_trace_path), str(json_path)],
        {
            "n_observations": len(observations),
            "n_refusals": refused_count,
            "refusal_rate": (
                refused_count / len(observations)
                if observations
                else 0.0
            ),
        },
    )


def _collect_observations(
    model: CausalLM,
    tokenizer: Tokenizer,
    config: BehaviorTraceConfig,
) -> list[BehaviorObservation]:
    """Generate outputs for configured prompts and convert them to observations."""
    observations: list[BehaviorObservation] = []
    for prompt in config.prompts:
        output = _generate(model, tokenizer, prompt.text, config.max_tokens)
        refused = is_refusal_text(output, config.refusal_phrases)
        expected_behavior = cast("ExpectedBehavior", prompt.expected_behavior)
        observations.append(
            BehaviorObservation(
                observation_id=f"{config.model_label}:{prompt.prompt_id}",
                model_label=config.model_label,
                prompt_id=prompt.prompt_id,
                category=prompt.category,
                prompt=_prompt_for_trace(prompt),
                output_text=_output_for_trace(
                    output,
                    prompt.redaction,
                    record_outputs=config.record_outputs,
                ),
                expected_behavior=expected_behavior,
                refused=refused,
                metrics=score_behavior_output(
                    output,
                    refused=refused,
                    expected_behavior=expected_behavior,
                ),
                redaction=cast("ExampleRedaction", prompt.redaction),
                metadata={
                    "tags": list(prompt.tags),
                    "scoring": "deterministic_v1",
                },
            ),
        )
    return observations


def _prompt_for_trace(prompt: BehaviorTracePromptConfig) -> str | None:
    """Return prompt text according to its redaction policy."""
    if prompt.redaction == "safe":
        return prompt.text
    if prompt.redaction == "redacted":
        return "[redacted prompt]"
    return None


def _output_for_trace(
    output: str,
    redaction: str,
    *,
    record_outputs: bool,
) -> str | None:
    """Return output text only when both config and prompt redaction allow it."""
    if record_outputs and redaction == "safe":
        return output
    return None


def _trace_path(
    output_dir: Path,
    config: BehaviorTraceConfig,
) -> Path:
    """Resolve the trace output path."""
    if config.output_trace is not None:
        return config.output_trace
    return output_dir / config.trace_filename


def _report_payload(
    trace: BehaviorTrace,
    config: BehaviorTraceConfig,
    trace_path: Path,
) -> dict[str, JsonValue]:
    """Build a compact JSON report for the trace collection run."""
    return {
        "report_version": "behavior_trace_v1",
        "trace": trace.summary_dict(),
        "trace_path": str(trace_path),
        "suite": {
            "name": config.suite_name,
            "description": config.suite_description,
            "version": config.suite_version,
            "source": config.suite_source,
            "safety_policy": config.safety_policy,
            "categories": list(trace.categories),
            "metric_names": list(trace.metric_names),
            "metric_specs": [
                {
                    "name": metric.name,
                    "description": metric.description,
                    "polarity": metric.polarity,
                    "unit": metric.unit,
                    "family": metric.family,
                }
                for metric in config.metrics
            ],
        },
        "config": {
            "model_label": config.model_label,
            "max_tokens": config.max_tokens,
            "record_outputs": config.record_outputs,
            "n_prompts": len(config.prompts),
        },
    }
