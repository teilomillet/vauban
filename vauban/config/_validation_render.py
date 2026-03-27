# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Rendering helpers for config validation output."""

from __future__ import annotations

from typing import TYPE_CHECKING, TextIO

from vauban.config._mode_registry import (
    EARLY_MODE_SPECS,
    active_early_modes,
    early_mode_label,
)

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
    first_mode = early_modes[0] if early_modes else None
    match first_mode:
        case str() as mode_key if early_mode_label(mode_key) is not None:
            mode = early_mode_label(mode_key) or "measure → cut → export"
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

    model_display = config.model_path or "(none — standalone mode)"
    print(f"Config:   {context.config_path}", file=stderr)
    print(f"Model:    {model_display}", file=stderr)
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
