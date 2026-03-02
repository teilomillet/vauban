"""Config validation registry and compatibility facade."""

from __future__ import annotations

from pathlib import Path

from vauban.config._loader import load_config
from vauban.config._validation_files import (
    _load_refusal_phrases as _load_refusal_phrases_impl,
)
from vauban.config._validation_files import (
    _validate_prompt_jsonl_file as _validate_prompt_jsonl_file_impl,
)
from vauban.config._validation_models import (
    Severity,
    ValidationCollector,
    ValidationContext,
    ValidationIssue,
    ValidationRule,
    ValidationRuleSpec,
)
from vauban.config._validation_render import _print_summary
from vauban.config._validation_rules import (
    VALIDATION_RULE_SPECS,
)
from vauban.config._validation_rules import (
    _rule_depth_extract_direction as _rule_depth_extract_direction_impl,
)
from vauban.config._validation_rules import (
    _rule_early_mode_conflicts as _rule_early_mode_conflicts_impl,
)
from vauban.config._validation_rules import (
    _rule_eval_without_prompts as _rule_eval_without_prompts_impl,
)
from vauban.config._validation_rules import (
    _rule_output_dir as _rule_output_dir_impl,
)
from vauban.config._validation_rules import (
    _rule_skipped_sections as _rule_skipped_sections_impl,
)


def validate_config(config_path: str | Path) -> list[str]:
    """Validate a TOML config without loading any model."""
    import sys

    from vauban.config._validation_files import _load_raw_toml

    config_path_obj = Path(config_path)
    raw = _load_raw_toml(config_path_obj)
    config = load_config(config_path_obj)
    context = ValidationContext(config_path=config_path_obj, raw=raw, config=config)

    collector = ValidationCollector()
    for spec in sorted(VALIDATION_RULE_SPECS, key=lambda item: item.order):
        spec.rule(context, collector)

    warnings = collector.render()
    _print_summary(sys.stderr, context, warnings)
    return warnings


def _add_warning(
    warnings: list[str],
    severity: Severity,
    message: str,
    *,
    fix: str | None = None,
) -> None:
    """Append one warning using the legacy rendered-string interface."""
    warnings.append(
        ValidationIssue(
            severity=severity,
            message=message,
            fix=fix,
        ).render(),
    )


def _load_refusal_phrases(path: Path) -> list[str]:
    """Compatibility wrapper for loading refusal phrases."""
    return _load_refusal_phrases_impl(path)


def _validate_prompt_jsonl_file(
    path: Path,
    key: str,
    warnings: list[str],
    *,
    min_recommended: int,
    missing_fix: str,
) -> int | None:
    """Compatibility wrapper for prompt JSONL validation."""
    collector = ValidationCollector()
    result = _validate_prompt_jsonl_file_impl(
        path,
        key,
        collector,
        min_recommended=min_recommended,
        missing_fix=missing_fix,
    )
    warnings.extend(collector.render())
    return result


def _rule_output_dir(context: ValidationContext, warnings: list[str]) -> None:
    """Compatibility wrapper for output-dir validation."""
    collector = ValidationCollector()
    _rule_output_dir_impl(context, collector)
    warnings.extend(collector.render())


def _rule_early_mode_conflicts(
    context: ValidationContext,
    warnings: list[str],
) -> None:
    """Compatibility wrapper for early-mode conflict validation."""
    collector = ValidationCollector()
    _rule_early_mode_conflicts_impl(context, collector)
    warnings.extend(collector.render())


def _rule_depth_extract_direction(
    context: ValidationContext,
    warnings: list[str],
) -> None:
    """Compatibility wrapper for depth direction validation."""
    collector = ValidationCollector()
    _rule_depth_extract_direction_impl(context, collector)
    warnings.extend(collector.render())


def _rule_eval_without_prompts(
    context: ValidationContext,
    warnings: list[str],
) -> None:
    """Compatibility wrapper for eval-without-prompts validation."""
    collector = ValidationCollector()
    _rule_eval_without_prompts_impl(context, collector)
    warnings.extend(collector.render())


def _rule_skipped_sections(
    context: ValidationContext,
    warnings: list[str],
) -> None:
    """Compatibility wrapper for skipped-section validation."""
    collector = ValidationCollector()
    _rule_skipped_sections_impl(context, collector)
    warnings.extend(collector.render())


__all__ = [
    "VALIDATION_RULE_SPECS",
    "Severity",
    "ValidationCollector",
    "ValidationContext",
    "ValidationIssue",
    "ValidationRule",
    "ValidationRuleSpec",
    "_add_warning",
    "_load_refusal_phrases",
    "_rule_depth_extract_direction",
    "_rule_early_mode_conflicts",
    "_rule_eval_without_prompts",
    "_rule_output_dir",
    "_rule_skipped_sections",
    "_validate_prompt_jsonl_file",
    "validate_config",
]
