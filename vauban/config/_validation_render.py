"""Rendering helpers for config validation output."""

from __future__ import annotations

from typing import TYPE_CHECKING, TextIO

from vauban.config._mode_registry import EARLY_MODE_SPECS, active_early_modes

if TYPE_CHECKING:
    from vauban.config._validation_models import ValidationContext


def _print_summary(
    stderr: TextIO,
    context: ValidationContext,
    warnings: list[str],
) -> None:
    """Print validation summary to stderr with the existing text format."""
    config = context.config
    early_modes = active_early_modes(config)

    mode_labels: dict[str, str] = {
        "[depth]": "depth analysis",
        "[svf]": "SVF training",
        "[features]": "SAE feature decomposition",
        "[probe]": "probe inspection",
        "[steer]": "steer generation",
        "[cast]": "CAST steering",
        "[sic]": "SIC sanitization",
        "[optimize]": "Optuna optimization",
        "[compose_optimize]": "composition optimization",
        "[softprompt]": "soft prompt attack",
        "[defend]": "defense stack",
        "[circuit]": "circuit tracing",
    }
    first_mode = early_modes[0] if early_modes else None
    match first_mode:
        case str() as mode_key if mode_key in mode_labels:
            mode = mode_labels[mode_key]
        case _:
            mode = "measure → cut → export"
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


def _early_mode_precedence_text() -> str:
    """Render precedence text without section brackets."""
    return " > ".join(spec.section.strip("[]") for spec in EARLY_MODE_SPECS)
