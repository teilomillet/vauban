# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Surface-mapping phases before and after model cutting."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from vauban._pipeline._context import log
from vauban._pipeline._helpers import surface_gate_failures

if TYPE_CHECKING:
    from vauban._pipeline._run_state import RunState


def prepare_surface_phase(state: RunState) -> None:
    """Resolve prompts and map the pre-cut refusal surface when configured."""
    from vauban.surface import (
        default_full_surface_path,
        default_multilingual_surface_path,
        default_surface_path,
        load_surface_prompts,
        map_surface,
    )

    config = state.config
    if state.direction_result is not None:
        state.surface_direction = state.direction_result.direction
        state.surface_layer = state.direction_result.layer_index
    elif state.subspace_result is not None:
        best = state.subspace_result.best_direction()
        state.surface_direction = best.direction
        state.surface_layer = best.layer_index

    if config.surface is None or state.surface_direction is None:
        return

    log(
        "Mapping refusal surface (before cut)",
        verbose=state.verbose,
        elapsed=state.elapsed(),
    )
    if config.surface.prompts_path == "default":
        prompts_path = default_surface_path()
    elif config.surface.prompts_path == "default_multilingual":
        prompts_path = default_multilingual_surface_path()
    elif config.surface.prompts_path == "default_full":
        prompts_path = default_full_surface_path()
    else:
        prompts_path = config.surface.prompts_path
    state.surface_prompts = load_surface_prompts(prompts_path)
    state.surface_before = map_surface(
        state.model,
        state.tokenizer,
        state.surface_prompts,
        state.surface_direction,
        state.surface_layer,
        generate=config.surface.generate,
        max_tokens=config.surface.max_tokens,
        refusal_phrases=state.refusal_phrases,
        progress=config.surface.progress,
        refusal_mode=config.eval.refusal_mode,
    )

    tc = state.surface_before.taxonomy_coverage
    if tc is not None:
        n_present = len(tc.present)
        n_total = n_present + len(tc.missing)
        missing_str = ""
        if tc.missing:
            names = sorted(tc.missing)
            if len(names) > 5:
                missing_str = (
                    f", missing: {', '.join(names[:5])} "
                    f"(+{len(names) - 5} more)"
                )
            else:
                missing_str = f", missing: {', '.join(names)}"
        log(
            f"Taxonomy coverage: {n_present}/{n_total} categories"
            f" ({tc.coverage_ratio:.1%}){missing_str}",
            verbose=state.verbose,
            elapsed=state.elapsed(),
        )


def finalize_surface_phase(state: RunState) -> None:
    """Map the post-cut surface, compare it, and enforce configured gates."""
    from vauban._serializers import _surface_comparison_to_dict
    from vauban.surface import compare_surfaces, map_surface

    if state.surface_before is None:
        return
    if state.modified_model is None:
        msg = "modified_model is required for surface-after mapping"
        raise ValueError(msg)
    if state.surface_prompts is None:
        msg = "surface_prompts is required for surface-after mapping"
        raise ValueError(msg)
    if state.surface_direction is None:
        msg = "surface_direction is required for surface-after mapping"
        raise ValueError(msg)
    if state.config.surface is None:
        msg = "surface config is required for surface-after mapping"
        raise ValueError(msg)

    log(
        "Mapping refusal surface (after cut)",
        verbose=state.verbose,
        elapsed=state.elapsed(),
    )
    surface_after = map_surface(
        state.modified_model,
        state.tokenizer,
        state.surface_prompts,
        state.surface_direction,
        state.surface_layer,
        generate=state.config.surface.generate,
        max_tokens=state.config.surface.max_tokens,
        refusal_phrases=state.refusal_phrases,
        progress=state.config.surface.progress,
        refusal_mode=state.config.eval.refusal_mode,
    )
    comparison = compare_surfaces(state.surface_before, surface_after)
    report_path = state.config.output_dir / "surface_report.json"
    report_path.write_text(
        json.dumps(_surface_comparison_to_dict(comparison), indent=2),
    )
    gate_failures = surface_gate_failures(state.config.surface, comparison)
    if gate_failures:
        joined = "\n".join(f"- {failure}" for failure in gate_failures)
        msg = (
            "Surface quality gates failed:\n"
            f"{joined}\n"
            "Adjust [surface] gate thresholds or improve model behavior."
        )
        raise RuntimeError(msg)
    state.report_files.append("surface_report.json")
