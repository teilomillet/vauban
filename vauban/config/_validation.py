"""Config validation registry and rules."""

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TextIO

from vauban._suggestions import (
    check_unknown_keys,
    check_unknown_sections,
    check_value_constraints,
)
from vauban.config._loader import load_config
from vauban.config._mode_registry import EARLY_MODE_SPECS, active_early_modes
from vauban.config._types import TomlDict
from vauban.surface import default_multilingual_surface_path, default_surface_path
from vauban.types import DatasetRef, PipelineConfig


@dataclass(frozen=True, slots=True)
class ValidationContext:
    """Shared validation context."""

    config_path: Path
    raw: TomlDict
    config: PipelineConfig


class ValidationRule(Protocol):
    """Validation rule callable protocol."""

    def __call__(self, context: ValidationContext, warnings: list[str]) -> None: ...


@dataclass(frozen=True, slots=True)
class ValidationRuleSpec:
    """Typed registration for one validation rule."""

    name: str
    order: int
    rule: ValidationRule


def validate_config(config_path: str | Path) -> list[str]:
    """Validate a TOML config without loading any model."""
    import sys

    config_path_obj = Path(config_path)
    raw = _load_raw_toml(config_path_obj)
    config = load_config(config_path_obj)
    context = ValidationContext(config_path=config_path_obj, raw=raw, config=config)

    warnings: list[str] = []
    for spec in sorted(VALIDATION_RULE_SPECS, key=lambda item: item.order):
        spec.rule(context, warnings)

    _print_summary(sys.stderr, context, warnings)
    return warnings


def _rule_unknown_schema(context: ValidationContext, warnings: list[str]) -> None:
    """Add unknown section/key/value warnings before parse-level checks."""
    unknown_warnings = check_unknown_sections(context.raw)
    unknown_warnings.extend(check_unknown_keys(context.raw))
    unknown_warnings.extend(check_value_constraints(context.raw))
    for warning in unknown_warnings:
        _add_warning(warnings, "MEDIUM", warning)


def _rule_prompt_sources(context: ValidationContext, warnings: list[str]) -> None:
    """Validate data prompt sources and dataset size balance."""
    config = context.config
    harmful_count = _validate_prompt_source(
        config.harmful_path,
        "[data].harmful",
        warnings,
        min_recommended=16,
        missing_fix=(
            "set [data].harmful to an existing JSONL path"
            ' or use [data].harmful = "default"'
        ),
    )
    harmless_count = _validate_prompt_source(
        config.harmless_path,
        "[data].harmless",
        warnings,
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
            warnings,
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
            _add_warning(
                warnings,
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


def _rule_eval_prompts(context: ValidationContext, warnings: list[str]) -> None:
    """Validate optional eval prompts path and sample size."""
    config = context.config
    if config.eval.prompts_path is None:
        return

    eval_count = _validate_prompt_jsonl_file(
        config.eval.prompts_path,
        "[eval].prompts",
        warnings,
        min_recommended=8,
        missing_fix=(
            "set [eval].prompts to an existing JSONL path"
            " or remove [eval] if you do not want eval reports"
        ),
    )
    if eval_count is not None and eval_count < 3:
        _add_warning(
            warnings,
            "MEDIUM",
            (
                f"[eval].prompts has only {eval_count} prompt(s);"
                " evaluation metrics may be noisy"
            ),
            fix="use at least 8-20 prompts for reliable evaluation",
        )


def _rule_refusal_phrases(context: ValidationContext, warnings: list[str]) -> None:
    """Validate optional refusal phrase file constraints."""
    config = context.config
    if config.eval.refusal_phrases_path is None:
        return
    rp = config.eval.refusal_phrases_path
    if not rp.exists():
        _add_warning(
            warnings,
            "HIGH",
            f"[eval].refusal_phrases file not found: {rp}",
            fix=(
                "set [eval].refusal_phrases to an existing text file,"
                " or remove it to use built-in refusal phrases"
            ),
        )
        return

    try:
        phrases = _load_refusal_phrases(rp)
    except ValueError as exc:
        msg = (
            f"{exc} — fix: add one refusal phrase per line in"
            f" {rp}, or remove [eval].refusal_phrases"
        )
        raise ValueError(msg) from exc
    if len(phrases) < 2:
        _add_warning(
            warnings,
            "MEDIUM",
            (
                f"[eval].refusal_phrases has only {len(phrases)}"
                " phrase(s)"
            ),
            fix=(
                "add multiple refusal phrases to reduce false negatives"
            ),
        )


def _rule_surface_prompts(context: ValidationContext, warnings: list[str]) -> None:
    """Validate surface prompt schema and gate compatibility warnings."""
    config = context.config
    if config.surface is None:
        return

    sp_raw = config.surface.prompts_path
    if sp_raw == "default":
        sp = default_surface_path()
    elif sp_raw == "default_multilingual":
        sp = default_multilingual_surface_path()
    elif isinstance(sp_raw, Path):
        sp = sp_raw
    else:
        sp = Path(sp_raw)
    surface_count = _validate_surface_jsonl_file(
        sp,
        "[surface].prompts",
        warnings,
        missing_fix=(
            "set [surface].prompts to an existing JSONL path"
            ' or use [surface].prompts = "default"'
            ' / "default_multilingual"'
        ),
    )
    if surface_count is not None and surface_count < 8:
        _add_warning(
            warnings,
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
        _add_warning(
            warnings,
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


def _rule_output_dir(context: ValidationContext, warnings: list[str]) -> None:
    """Validate output directory path kind."""
    config = context.config
    if config.output_dir.exists() and not config.output_dir.is_dir():
        _add_warning(
            warnings,
            "HIGH",
            (
                f"[output].dir points to a file, not a directory:"
                f" {config.output_dir}"
            ),
            fix="set [output].dir to a directory path",
        )


def _rule_early_mode_conflicts(
    context: ValidationContext,
    warnings: list[str],
) -> None:
    """Warn when multiple early-return modes are active."""
    early_modes = active_early_modes(context.config)
    if len(early_modes) <= 1:
        return
    _add_warning(
        warnings,
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
    warnings: list[str],
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
        src = (
            "direction_prompts"
            if config.depth.direction_prompts is not None
            else "prompts"
        )
        _add_warning(
            warnings,
            "HIGH",
            f"[depth].extract_direction = true but {src}"
            f" has only {len(effective)} entry — need >= 2",
            fix=(
                "add at least 2 prompts to the selected source,"
                " or set [depth].extract_direction = false"
            ),
        )


def _rule_eval_without_prompts(
    context: ValidationContext,
    warnings: list[str],
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
        _add_warning(
            warnings,
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


def _rule_skipped_sections(context: ValidationContext, warnings: list[str]) -> None:
    """Warn when early-return modes cause config sections to be ignored."""
    config = context.config
    early_modes = active_early_modes(config)
    if not early_modes:
        return
    active_mode = early_modes[0]
    skipped: list[str] = []

    # depth skips everything including [detect]; other modes still run detect
    if config.depth is not None and config.detect is not None:
        skipped.append("[detect]")

    # [surface] and [eval] are skipped by ALL early-return modes
    if config.surface is not None:
        skipped.append("[surface]")
    if config.eval.prompts_path is not None:
        skipped.append("[eval]")

    if skipped:
        _add_warning(
            warnings,
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


def _print_summary(
    stderr: TextIO,
    context: ValidationContext,
    warnings: list[str],
) -> None:
    """Print validation summary to stderr with the existing text format."""
    config = context.config
    early_modes = active_early_modes(config)

    mode = "measure → cut → export"
    if config.depth is not None:
        mode = "depth analysis"
    elif config.probe is not None:
        mode = "probe inspection"
    elif config.steer is not None:
        mode = "steer generation"
    elif config.cast is not None:
        mode = "CAST steering"
    elif config.sic is not None:
        mode = "SIC sanitization"
    elif config.optimize is not None:
        mode = "Optuna optimization"
    elif config.softprompt is not None:
        mode = "soft prompt attack"
    elif config.circuit is not None:
        mode = "circuit tracing"
    elif config.features is not None:
        mode = "SAE feature decomposition"
    extras: list[str] = []
    if config.detect is not None and config.depth is None:
        extras.append("detect")
    if config.surface is not None and not early_modes:
        extras.append("surface")
    if config.eval.prompts_path is not None and not early_modes:
        extras.append("eval")
    mode_str = mode
    if extras:
        mode_str += f" + {', '.join(extras)}"

    print(f"Config:   {context.config_path}", file=stderr)
    print(f"Model:    {config.model_path}", file=stderr)
    print(f"Pipeline: {mode_str}", file=stderr)
    print(f"Output:   {config.output_dir}", file=stderr)

    if warnings:
        print(f"\nWarnings ({len(warnings)}):", file=stderr)
        for warning in warnings:
            print(f"  - {warning}", file=stderr)
    else:
        print("\nNo issues found.", file=stderr)


def _load_raw_toml(path: Path) -> TomlDict:
    """Load raw TOML mapping for intent-level validation checks."""
    with path.open("rb") as f:
        raw: TomlDict = tomllib.load(f)
    return raw


def _add_warning(
    warnings: list[str],
    severity: str,
    message: str,
    *,
    fix: str | None = None,
) -> None:
    """Append a structured warning with optional fix guidance."""
    full = f"[{severity}] {message}"
    if fix is not None:
        full += f" — fix: {fix}"
    warnings.append(full)


def _validate_prompt_source(
    source: Path | DatasetRef,
    key: str,
    warnings: list[str],
    *,
    min_recommended: int,
    missing_fix: str,
) -> int | None:
    """Validate a prompt source (local JSONL or HF dataset reference)."""
    if isinstance(source, DatasetRef):
        if source.limit is not None and source.limit < 2:
            _add_warning(
                warnings,
                "MEDIUM",
                (
                    f"{key} uses HF dataset limit={source.limit};"
                    " this is likely too small for stable estimation"
                ),
                fix="increase [data.*].limit or remove it",
            )
        return None
    return _validate_prompt_jsonl_file(
        source,
        key,
        warnings,
        min_recommended=min_recommended,
        missing_fix=missing_fix,
    )


def _validate_prompt_jsonl_file(
    path: Path,
    key: str,
    warnings: list[str],
    *,
    min_recommended: int,
    missing_fix: str,
) -> int | None:
    """Validate JSONL prompt schema for files using {'prompt': str} lines."""
    if not path.exists():
        _add_warning(
            warnings,
            "HIGH",
            f"{key} file not found: {path}",
            fix=missing_fix,
        )
        return None

    valid_count = 0
    seen_non_empty = 0
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            seen_non_empty += 1
            try:
                obj_raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                _add_warning(
                    warnings,
                    "HIGH",
                    (
                        f"{key} line {line_no} is not valid JSON"
                        f" ({exc.msg}) in {path}"
                    ),
                    fix='use JSONL lines like {"prompt": "your text"}',
                )
                return None
            if not isinstance(obj_raw, dict):
                _add_warning(
                    warnings,
                    "HIGH",
                    f"{key} line {line_no} must be a JSON object in {path}",
                    fix='use JSONL lines like {"prompt": "your text"}',
                )
                return None
            prompt_raw = obj_raw.get("prompt")
            if not isinstance(prompt_raw, str) or not prompt_raw.strip():
                _add_warning(
                    warnings,
                    "HIGH",
                    (
                        f"{key} line {line_no} must contain a non-empty"
                        f" string key 'prompt' in {path}"
                    ),
                    fix='ensure each line has {"prompt": "..."}',
                )
                return None
            valid_count += 1

    if seen_non_empty == 0:
        _add_warning(
            warnings,
            "HIGH",
            f"{key} is empty: {path}",
            fix='add JSONL records like {"prompt": "..."}',
        )
        return 0

    if valid_count < min_recommended:
        _add_warning(
            warnings,
            "LOW",
            (
                f"{key} has only {valid_count} prompt(s);"
                " results may be noisy on small sets"
            ),
            fix=f"use at least {min_recommended} prompts",
        )
    return valid_count


def _validate_surface_jsonl_file(
    path: Path,
    key: str,
    warnings: list[str],
    *,
    missing_fix: str,
) -> int | None:
    """Validate JSONL surface schema for prompt/label/category records.

    Optional keys, when present:
    - ``messages``: non-empty list of {role, content}
    - ``style``: non-empty string
    - ``language``: non-empty string
    - ``turn_depth``: integer >= 1
    - ``framing``: non-empty string
    """
    allowed_roles: frozenset[str] = frozenset({"system", "user", "assistant"})
    if not path.exists():
        _add_warning(
            warnings,
            "HIGH",
            f"{key} file not found: {path}",
            fix=missing_fix,
        )
        return None

    valid_count = 0
    seen_non_empty = 0
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            seen_non_empty += 1
            try:
                obj_raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                _add_warning(
                    warnings,
                    "HIGH",
                    (
                        f"{key} line {line_no} is not valid JSON"
                        f" ({exc.msg}) in {path}"
                    ),
                    fix=(
                        'use JSONL lines like {"prompt": "...",'
                        ' "label": "harmful", "category": "weapons",'
                        ' "messages": [{"role": "user", "content": "..."}],'
                        ' "style": "direct", "language": "en",'
                        ' "turn_depth": 1, "framing": "explicit"}'
                    ),
                )
                return None
            if not isinstance(obj_raw, dict):
                _add_warning(
                    warnings,
                    "HIGH",
                    f"{key} line {line_no} must be a JSON object in {path}",
                    fix=(
                        'use JSONL lines like {"prompt": "...",'
                        ' "label": "harmful", "category": "weapons",'
                        ' "messages": [{"role": "user", "content": "..."}],'
                        ' "style": "direct", "language": "en",'
                        ' "turn_depth": 1, "framing": "explicit"}'
                    ),
                )
                return None
            prompt_raw = obj_raw.get("prompt")
            messages_raw = obj_raw.get("messages")
            label_raw = obj_raw.get("label")
            category_raw = obj_raw.get("category")
            if (
                not isinstance(label_raw, str) or not label_raw.strip()
                or not isinstance(category_raw, str) or not category_raw.strip()
            ):
                _add_warning(
                    warnings,
                    "HIGH",
                    (
                        f"{key} line {line_no} must include non-empty string"
                        " keys: label, category"
                    ),
                    fix=(
                        'use JSONL lines like {"prompt": "...",'
                        ' "label": "harmful", "category": "weapons"}'
                    ),
                )
                return None

            has_prompt = isinstance(prompt_raw, str) and bool(prompt_raw.strip())
            if prompt_raw is not None and not has_prompt:
                _add_warning(
                    warnings,
                    "HIGH",
                    (
                        f"{key} line {line_no} has invalid key 'prompt':"
                        " expected non-empty string"
                    ),
                    fix='set "prompt" to a non-empty string',
                )
                return None

            has_messages = False
            if messages_raw is not None:
                if not isinstance(messages_raw, list) or not messages_raw:
                    _add_warning(
                        warnings,
                        "HIGH",
                        (
                            f"{key} line {line_no} has invalid key 'messages':"
                            " expected non-empty list"
                        ),
                        fix=(
                            'set "messages" to a non-empty list like'
                            ' [{"role": "user", "content": "..."}]'
                        ),
                    )
                    return None
                for i, message in enumerate(messages_raw):
                    if not isinstance(message, dict):
                        _add_warning(
                            warnings,
                            "HIGH",
                            (
                                f"{key} line {line_no} has invalid"
                                f" messages[{i}] (expected object)"
                            ),
                            fix=(
                                'use {"role": "user|assistant|system",'
                                ' "content": "..."}'
                            ),
                        )
                        return None
                    role_raw = message.get("role")
                    content_raw = message.get("content")
                    if (
                        not isinstance(role_raw, str)
                        or role_raw not in allowed_roles
                        or not isinstance(content_raw, str)
                        or not content_raw.strip()
                    ):
                        _add_warning(
                            warnings,
                            "HIGH",
                            (
                                f"{key} line {line_no} has invalid messages[{i}]"
                                " (role/content)"
                            ),
                            fix=(
                                'use {"role": "user|assistant|system",'
                                ' "content": "..."}'
                            ),
                        )
                        return None
                has_messages = True

            if not has_prompt and not has_messages:
                _add_warning(
                    warnings,
                    "HIGH",
                    (
                        f"{key} line {line_no} must include either"
                        " non-empty 'prompt' or non-empty 'messages'"
                    ),
                    fix=(
                        'add "prompt": "..." or "messages":'
                        ' [{"role": "user", "content": "..."}]'
                    ),
                )
                return None

            style_raw = obj_raw.get("style")
            language_raw = obj_raw.get("language")
            framing_raw = obj_raw.get("framing")
            turn_depth_raw = obj_raw.get("turn_depth")

            if style_raw is not None and (
                not isinstance(style_raw, str) or not style_raw.strip()
            ):
                _add_warning(
                    warnings,
                    "HIGH",
                    (
                        f"{key} line {line_no} has invalid optional key"
                        " 'style': expected non-empty string"
                    ),
                    fix=(
                        'set "style" to a non-empty string'
                        ' (example: "direct")'
                    ),
                )
                return None

            if language_raw is not None and (
                not isinstance(language_raw, str) or not language_raw.strip()
            ):
                _add_warning(
                    warnings,
                    "HIGH",
                    (
                        f"{key} line {line_no} has invalid optional key"
                        " 'language': expected non-empty string"
                    ),
                    fix=(
                        'set "language" to a non-empty string'
                        ' (example: "en")'
                    ),
                )
                return None

            if framing_raw is not None and (
                not isinstance(framing_raw, str) or not framing_raw.strip()
            ):
                _add_warning(
                    warnings,
                    "HIGH",
                    (
                        f"{key} line {line_no} has invalid optional key"
                        " 'framing': expected non-empty string"
                    ),
                    fix=(
                        'set "framing" to a non-empty string'
                        ' (example: "explicit")'
                    ),
                )
                return None

            if turn_depth_raw is not None and (
                isinstance(turn_depth_raw, bool)
                or not isinstance(turn_depth_raw, int)
                or turn_depth_raw < 1
            ):
                _add_warning(
                    warnings,
                    "HIGH",
                    (
                        f"{key} line {line_no} has invalid optional key"
                        " 'turn_depth': expected integer >= 1"
                    ),
                    fix=(
                        'set "turn_depth" to an integer >= 1'
                        " (example: 1)"
                    ),
                )
                return None
            valid_count += 1

    if seen_non_empty == 0:
        _add_warning(
            warnings,
            "HIGH",
            f"{key} is empty: {path}",
            fix=(
                'add JSONL records with prompt/label/category keys'
            ),
        )
        return 0

    return valid_count


def _load_refusal_phrases(path: Path) -> list[str]:
    """Load refusal phrases from a text file (one per line)."""
    phrases: list[str] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            phrases.append(stripped)
    if not phrases:
        msg = f"Refusal phrases file is empty: {path}"
        raise ValueError(msg)
    return phrases


def _early_mode_precedence_text() -> str:
    """Render precedence text without section brackets."""
    return " > ".join(spec.section.strip("[]") for spec in EARLY_MODE_SPECS)
