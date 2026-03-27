# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Pipeline helper functions for reports, gates, and I/O."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from vauban.types import (
        DBDIResult,
        DiffResult,
        DirectionResult,
        PipelineConfig,
        SoftPromptResult,
        SubspaceResult,
        SurfaceComparison,
        SurfaceConfig,
    )


def surface_gate_failures(
    surface: SurfaceConfig,
    comparison: SurfaceComparison,
) -> list[str]:
    """Return human-readable failures for configured surface gate thresholds."""
    failures: list[str] = []

    max_worst_after = surface.max_worst_cell_refusal_after
    if (
        max_worst_after is not None
        and comparison.worst_cell_refusal_rate_after > max_worst_after
    ):
        failures.append(
            (
                "worst-cell refusal rate after cut"
                f" {comparison.worst_cell_refusal_rate_after:.4f}"
                f" exceeds max_worst_cell_refusal_after={max_worst_after:.4f}"
            ),
        )

    max_worst_delta = surface.max_worst_cell_refusal_delta
    if (
        max_worst_delta is not None
        and comparison.worst_cell_refusal_rate_delta > max_worst_delta
    ):
        failures.append(
            (
                "worst-cell refusal-rate delta"
                f" {comparison.worst_cell_refusal_rate_delta:.4f}"
                f" exceeds max_worst_cell_refusal_delta={max_worst_delta:.4f}"
            ),
        )

    min_coverage = surface.min_coverage_score
    if (
        min_coverage is not None
        and comparison.coverage_score_after < min_coverage
    ):
        failures.append(
            (
                "coverage score after cut"
                f" {comparison.coverage_score_after:.4f}"
                f" is below min_coverage_score={min_coverage:.4f}"
            ),
        )

    return failures


def load_refusal_phrases(path: Path) -> list[str]:
    """Load refusal phrases from a text file (one per line)."""
    from vauban.config._validation import (
        _load_refusal_phrases,
    )

    return _load_refusal_phrases(path)


def is_default_data(config: PipelineConfig) -> bool:
    """Check whether data paths are just bundled defaults (not user-provided)."""
    from vauban.measure import default_prompt_paths

    h_default, hl_default = default_prompt_paths()
    return config.harmful_path == h_default and config.harmless_path == hl_default


def write_measure_reports(
    config: PipelineConfig,
    direction_result: DirectionResult | None,
    subspace_result: SubspaceResult | None,
    dbdi_result: DBDIResult | None,
    diff_result: DiffResult | None,
) -> list[str]:
    """Write the JSON reports produced directly by the measure stage."""
    reports: list[str] = []

    if diff_result is not None:
        report_path = config.output_dir / "diff_report.json"
        report: dict[str, object] = {
            "source_model": diff_result.source_model,
            "target_model": diff_result.target_model,
            "best_layer": diff_result.best_layer,
            "d_model": diff_result.d_model,
            "best_layer_singular_values": diff_result.singular_values,
            "explained_variance_per_layer": diff_result.explained_variance,
            "per_layer_singular_values": diff_result.per_layer_singular_values,
        }
    elif subspace_result is not None:
        report_path = config.output_dir / "subspace_report.json"
        report = {
            "layer_index": subspace_result.layer_index,
            "d_model": subspace_result.d_model,
            "model_path": subspace_result.model_path,
            "singular_values": subspace_result.singular_values,
            "explained_variance": subspace_result.explained_variance,
            "layer_types": subspace_result.layer_types,
        }
    elif dbdi_result is not None:
        report_path = config.output_dir / "dbdi_report.json"
        report = {
            "hdd_layer_index": dbdi_result.hdd_layer_index,
            "red_layer_index": dbdi_result.red_layer_index,
            "hdd_cosine_scores": dbdi_result.hdd_cosine_scores,
            "red_cosine_scores": dbdi_result.red_cosine_scores,
            "d_model": dbdi_result.d_model,
            "model_path": dbdi_result.model_path,
            "layer_types": dbdi_result.layer_types,
        }
    elif direction_result is not None:
        report_path = config.output_dir / "direction_report.json"
        report = direction_result.to_dict()
    else:
        return reports

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    reports.append(report_path.name)
    return reports


def write_arena_card(
    path: Path,
    result: SoftPromptResult,
    prompts: list[str],
) -> None:
    """Write a human-readable arena submission card.

    Produces a text file with the optimized suffix (copy-paste ready),
    transfer results, and per-prompt submissions for Gray Swan Arena.
    """
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("ARENA SUBMISSION CARD")
    lines.append("=" * 60)
    lines.append("")

    # Summary
    lines.append(f"Mode:       {result.mode}")
    lines.append(f"Tokens:     {result.n_tokens}")
    lines.append(f"Steps:      {result.n_steps}")
    lines.append(f"Primary ASR: {result.success_rate:.2%}")
    lines.append(f"Final loss:  {result.final_loss:.4f}")
    lines.append("")

    # Copy-paste suffix
    lines.append("-" * 60)
    lines.append("SUFFIX (copy-paste ready)")
    lines.append("-" * 60)
    lines.append(result.token_text or "")
    lines.append("")

    # Transfer results
    if result.transfer_results:
        lines.append("-" * 60)
        lines.append("TRANSFER RESULTS")
        lines.append("-" * 60)
        for tr in result.transfer_results:
            lines.append(f"  {tr.model_id}: {tr.success_rate:.2%}")
        lines.append("")

    # Per-prompt submissions
    lines.append("-" * 60)
    lines.append("PER-PROMPT SUBMISSIONS")
    lines.append("-" * 60)
    suffix = result.token_text or ""
    for i, prompt in enumerate(prompts):
        lines.append(f"\n--- Prompt {i + 1} ---")
        lines.append(f"{prompt} {suffix}")
        if i < len(result.eval_responses):
            preview = result.eval_responses[i][:200]
            lines.append(f"  Response: {preview}")
    lines.append("")

    # GAN round history
    if result.gan_history:
        lines.append("-" * 60)
        lines.append("GAN ROUND HISTORY")
        lines.append("-" * 60)
        for rnd in result.gan_history:
            won = "WON" if rnd.attacker_won else "LOST"
            asr = rnd.attack_result.success_rate
            lines.append(
                f"  Round {rnd.round_index}: {won}"
                f" (ASR={asr:.2%})",
            )
            for tr in rnd.transfer_results:
                lines.append(
                    f"    Transfer {tr.model_id}:"
                    f" {tr.success_rate:.2%}",
                )
        lines.append("")

    lines.append("=" * 60)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
