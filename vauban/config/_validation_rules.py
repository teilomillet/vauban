"""Validation rules for config validation."""

from __future__ import annotations

from pathlib import Path

from vauban._suggestions import (
    check_unknown_keys,
    check_unknown_sections,
    check_value_constraints,
)
from vauban.config._mode_registry import active_early_modes
from vauban.config._validation_files import (
    _load_refusal_phrases,
    _validate_prompt_jsonl_file,
    _validate_prompt_source,
    _validate_surface_jsonl_file,
)
from vauban.config._validation_models import (
    ValidationCollector,
    ValidationContext,
    ValidationRuleSpec,
)
from vauban.config._validation_render import _early_mode_precedence_text
from vauban.surface import default_multilingual_surface_path, default_surface_path


def _rule_unknown_schema(
    context: ValidationContext,
    collector: ValidationCollector,
) -> None:
    """Add unknown section/key/value warnings before parse-level checks."""
    unknown_warnings = check_unknown_sections(context.raw)
    unknown_warnings.extend(check_unknown_keys(context.raw))
    unknown_warnings.extend(check_value_constraints(context.raw))
    for warning in unknown_warnings:
        collector.add("MEDIUM", warning)


def _rule_prompt_sources(
    context: ValidationContext,
    collector: ValidationCollector,
) -> None:
    """Validate data prompt sources and dataset size balance."""
    config = context.config
    harmful_count = _validate_prompt_source(
        config.harmful_path,
        "[data].harmful",
        collector,
        min_recommended=16,
        missing_fix=(
            "set [data].harmful to an existing JSONL path"
            ' or use [data].harmful = "default"'
        ),
    )
    harmless_count = _validate_prompt_source(
        config.harmless_path,
        "[data].harmless",
        collector,
        min_recommended=16,
        missing_fix=(
            "set [data].harmless to an existing JSONL path"
            ' or use [data].harmless = "default"'
        ),
    )
    if config.borderline_path is not None:
        _validate_prompt_source(
            config.borderline_path,
            "[data].borderline",
            collector,
            min_recommended=8,
            missing_fix=(
                "set [data].borderline to an existing JSONL path"
                " or disable [cut].false_refusal_ortho"
            ),
        )

    if (
        harmful_count is not None
        and harmless_count is not None
        and harmful_count > 0
        and harmless_count > 0
    ):
        ratio = (
            harmful_count / harmless_count
            if harmful_count >= harmless_count
            else harmless_count / harmful_count
        )
        if ratio > 4.0:
            collector.add(
                "LOW",
                (
                    "[data] prompt set sizes are highly imbalanced"
                    f" (harmful={harmful_count}, harmless={harmless_count})"
                ),
                fix=(
                    "use similarly sized harmful/harmless datasets"
                    " for more stable direction estimates"
                ),
            )


def _rule_eval_prompts(
    context: ValidationContext,
    collector: ValidationCollector,
) -> None:
    """Validate optional eval prompts path and sample size."""
    config = context.config
    if config.eval.prompts_path is None:
        return

    eval_count = _validate_prompt_jsonl_file(
        config.eval.prompts_path,
        "[eval].prompts",
        collector,
        min_recommended=8,
        missing_fix=(
            "set [eval].prompts to an existing JSONL path"
            " or remove [eval] if you do not want eval reports"
        ),
    )
    if eval_count is not None and eval_count < 3:
        collector.add(
            "MEDIUM",
            (
                f"[eval].prompts has only {eval_count} prompt(s);"
                " evaluation metrics may be noisy"
            ),
            fix="use at least 8-20 prompts for reliable evaluation",
        )


def _rule_refusal_phrases(
    context: ValidationContext,
    collector: ValidationCollector,
) -> None:
    """Validate optional refusal phrase file constraints."""
    config = context.config
    if config.eval.refusal_phrases_path is None:
        return
    refusal_path = config.eval.refusal_phrases_path
    if not refusal_path.exists():
        collector.add(
            "HIGH",
            f"[eval].refusal_phrases file not found: {refusal_path}",
            fix=(
                "set [eval].refusal_phrases to an existing text file,"
                " or remove it to use built-in refusal phrases"
            ),
        )
        return

    try:
        phrases = _load_refusal_phrases(refusal_path)
    except ValueError as exc:
        msg = (
            f"{exc} — fix: add one refusal phrase per line in"
            f" {refusal_path}, or remove [eval].refusal_phrases"
        )
        raise ValueError(msg) from exc
    if len(phrases) < 2:
        collector.add(
            "MEDIUM",
            (
                f"[eval].refusal_phrases has only {len(phrases)}"
                " phrase(s)"
            ),
            fix="add multiple refusal phrases to reduce false negatives",
        )


def _rule_surface_prompts(
    context: ValidationContext,
    collector: ValidationCollector,
) -> None:
    """Validate surface prompt schema and gate compatibility warnings."""
    config = context.config
    if config.surface is None:
        return

    surface_path_raw = config.surface.prompts_path
    if surface_path_raw == "default":
        surface_path = default_surface_path()
    elif surface_path_raw == "default_multilingual":
        surface_path = default_multilingual_surface_path()
    elif isinstance(surface_path_raw, Path):
        surface_path = surface_path_raw
    else:
        surface_path = Path(surface_path_raw)
    surface_count = _validate_surface_jsonl_file(
        surface_path,
        "[surface].prompts",
        collector,
        missing_fix=(
            "set [surface].prompts to an existing JSONL path"
            ' or use [surface].prompts = "default"'
            ' / "default_multilingual"'
        ),
    )
    if surface_count is not None and surface_count < 8:
        collector.add(
            "LOW",
            (
                f"[surface].prompts has only {surface_count} record(s);"
                " category/label aggregates may be unstable"
            ),
            fix="use a broader surface prompt set (16+ recommended)",
        )
    if (
        not config.surface.generate
        and (
            config.surface.max_worst_cell_refusal_after is not None
            or config.surface.max_worst_cell_refusal_delta is not None
        )
    ):
        collector.add(
            "MEDIUM",
            (
                "[surface] refusal-rate gates are set but"
                " [surface].generate = false; refusal labels are not"
                " computed in projection-only mode"
            ),
            fix=(
                "set [surface].generate = true for refusal-rate gates,"
                " or remove max_worst_cell_refusal_* gates"
            ),
        )


def _rule_output_dir(
    context: ValidationContext,
    collector: ValidationCollector,
) -> None:
    """Validate output directory path kind."""
    config = context.config
    if config.output_dir.exists() and not config.output_dir.is_dir():
        collector.add(
            "HIGH",
            (
                f"[output].dir points to a file, not a directory:"
                f" {config.output_dir}"
            ),
            fix="set [output].dir to a directory path",
        )


def _rule_early_mode_conflicts(
    context: ValidationContext,
    collector: ValidationCollector,
) -> None:
    """Warn when multiple early-return modes are active."""
    early_modes = active_early_modes(context.config)
    if len(early_modes) <= 1:
        return
    collector.add(
        "HIGH",
        f"Multiple early-return modes active: {', '.join(early_modes)}"
        " — only the first will run (precedence: "
        f"{_early_mode_precedence_text()})",
        fix=(
            "keep one early-return mode per config,"
            " and split other modes into separate TOML files"
        ),
    )


def _rule_depth_extract_direction(
    context: ValidationContext,
    collector: ValidationCollector,
) -> None:
    """Warn when depth direction extraction has too few prompts."""
    config = context.config
    if config.depth is None or not config.depth.extract_direction:
        return
    effective = (
        config.depth.direction_prompts
        if config.depth.direction_prompts is not None
        else config.depth.prompts
    )
    if len(effective) < 2:
        source = (
            "direction_prompts"
            if config.depth.direction_prompts is not None
            else "prompts"
        )
        collector.add(
            "HIGH",
            f"[depth].extract_direction = true but {source}"
            f" has only {len(effective)} entry — need >= 2",
            fix=(
                "add at least 2 prompts to the selected source,"
                " or set [depth].extract_direction = false"
            ),
        )


def _rule_eval_without_prompts(
    context: ValidationContext,
    collector: ValidationCollector,
) -> None:
    """Warn if [eval] is present but has no prompt source in default flow."""
    config = context.config
    early_modes = active_early_modes(config)
    eval_raw = context.raw.get("eval")
    if (
        isinstance(eval_raw, dict)
        and "prompts" not in eval_raw
        and config.eval.prompts_path is None
        and not early_modes
    ):
        collector.add(
            "LOW",
            (
                "[eval] section is present but [eval].prompts is not set;"
                " eval_report.json will not be produced in default pipeline"
            ),
            fix=(
                'set [eval].prompts = "eval.jsonl"'
                " or remove the [eval] section"
            ),
        )


def _rule_skipped_sections(
    context: ValidationContext,
    collector: ValidationCollector,
) -> None:
    """Warn when early-return modes cause config sections to be ignored."""
    config = context.config
    early_modes = active_early_modes(config)
    if not early_modes:
        return
    active_mode = early_modes[0]
    skipped: list[str] = []

    if config.depth is not None and config.detect is not None:
        skipped.append("[detect]")
    if config.surface is not None:
        skipped.append("[surface]")
    if config.eval.prompts_path is not None:
        skipped.append("[eval]")

    if skipped:
        collector.add(
            "MEDIUM",
            f"{active_mode} early-return will skip:"
            f" {', '.join(skipped)}"
            f" — these sections have no effect in"
            f" {active_mode.strip('[]')} mode",
            fix=(
                "remove skipped sections from this config,"
                " or run them in a separate non-early-return config"
            ),
        )


VALIDATION_RULE_SPECS: tuple[ValidationRuleSpec, ...] = (
    ValidationRuleSpec("unknown_schema", 10, _rule_unknown_schema),
    ValidationRuleSpec("prompt_sources", 20, _rule_prompt_sources),
    ValidationRuleSpec("eval_prompts", 30, _rule_eval_prompts),
    ValidationRuleSpec("refusal_phrases", 40, _rule_refusal_phrases),
    ValidationRuleSpec("surface_prompts", 50, _rule_surface_prompts),
    ValidationRuleSpec("output_dir", 60, _rule_output_dir),
    ValidationRuleSpec("early_mode_conflicts", 70, _rule_early_mode_conflicts),
    ValidationRuleSpec("depth_extract_direction", 80, _rule_depth_extract_direction),
    ValidationRuleSpec("eval_without_prompts", 90, _rule_eval_without_prompts),
    ValidationRuleSpec("skipped_sections", 100, _rule_skipped_sections),
)
