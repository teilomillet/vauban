# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Aggregation helpers for refusal surface mapping."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vauban.surface._scan import scan
from vauban.types import (
    CausalLM,
    SurfaceComparison,
    SurfaceGroup,
    SurfaceGroupDelta,
    SurfacePoint,
    SurfacePrompt,
    SurfaceResult,
    Tokenizer,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from vauban._array import Array
    from vauban.taxonomy import TaxonomyCoverage

type SurfaceGroups = tuple[list[SurfaceGroup], list[SurfaceGroup]]


def aggregate(points: list[SurfacePoint]) -> SurfaceGroups:
    """Group surface points by label and category, computing stats."""
    by_label = _group_points(points, key=lambda p: p.label)
    by_category = _group_points(points, key=lambda p: p.category)
    return by_label, by_category


def find_threshold(points: list[SurfacePoint]) -> float:
    """Find the projection value separating refused and compliant prompts."""
    compliant = [p.direction_projection for p in points if p.refused is False]
    refusing = [p.direction_projection for p in points if p.refused is True]

    if not compliant or not refusing:
        return 0.0

    max_compliant = max(compliant)
    min_refusing = min(refusing)
    return (max_compliant + min_refusing) / 2.0


def map_surface(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompts: list[SurfacePrompt],
    direction: Array,
    direction_layer: int,
    *,
    generate: bool = True,
    max_tokens: int = 60,
    refusal_phrases: list[str] | None = None,
    progress: bool = True,
    refusal_mode: str = "phrases",
) -> SurfaceResult:
    """Convenience wrapper for scan + aggregate + find_threshold."""
    points = scan(
        model,
        tokenizer,
        prompts,
        direction,
        direction_layer,
        generate=generate,
        max_tokens=max_tokens,
        refusal_phrases=refusal_phrases,
        progress=progress,
        refusal_mode=refusal_mode,
    )

    groups_by_label, groups_by_category = aggregate(points)
    groups_by_style = _group_points(points, key=lambda p: p.style)
    groups_by_language = _group_points(points, key=lambda p: p.language)
    groups_by_turn_depth = _group_points(
        points,
        key=lambda p: str(p.turn_depth),
    )
    groups_by_framing = _group_points(points, key=lambda p: p.framing)
    groups_by_surface_cell = _group_points(points, key=_surface_cell_name)
    coverage_score = _coverage_score(points)
    taxonomy_cov = compute_taxonomy_coverage(points)
    threshold = find_threshold(points) if generate else 0.0

    total_refused = sum(1 for p in points if p.refused is True)

    return SurfaceResult(
        points=points,
        groups_by_label=groups_by_label,
        groups_by_category=groups_by_category,
        threshold=threshold,
        total_scanned=len(points),
        total_refused=total_refused,
        groups_by_style=groups_by_style,
        groups_by_language=groups_by_language,
        groups_by_turn_depth=groups_by_turn_depth,
        groups_by_framing=groups_by_framing,
        groups_by_surface_cell=groups_by_surface_cell,
        coverage_score=coverage_score,
        taxonomy_coverage=taxonomy_cov,
    )


def compare_surfaces(
    before: SurfaceResult,
    after: SurfaceResult,
) -> SurfaceComparison:
    """Compare refusal surface results before and after cut."""
    rate_before = (
        before.total_refused / before.total_scanned
        if before.total_scanned > 0
        else 0.0
    )
    rate_after = (
        after.total_refused / after.total_scanned
        if after.total_scanned > 0
        else 0.0
    )

    category_deltas = _compute_group_deltas(
        before.groups_by_category,
        after.groups_by_category,
    )
    label_deltas = _compute_group_deltas(
        before.groups_by_label,
        after.groups_by_label,
    )
    style_deltas = _compute_group_deltas(
        before.groups_by_style,
        after.groups_by_style,
    )
    language_deltas = _compute_group_deltas(
        before.groups_by_language,
        after.groups_by_language,
    )
    turn_depth_deltas = _compute_group_deltas(
        before.groups_by_turn_depth,
        after.groups_by_turn_depth,
    )
    framing_deltas = _compute_group_deltas(
        before.groups_by_framing,
        after.groups_by_framing,
    )
    cell_deltas = _compute_group_deltas(
        before.groups_by_surface_cell,
        after.groups_by_surface_cell,
    )
    worst_before = _max_refusal_rate(before.groups_by_surface_cell)
    worst_after = _max_refusal_rate(after.groups_by_surface_cell)
    worst_delta = max(
        (d.refusal_rate_delta for d in cell_deltas),
        default=0.0,
    )

    return SurfaceComparison(
        before=before,
        after=after,
        refusal_rate_before=rate_before,
        refusal_rate_after=rate_after,
        refusal_rate_delta=rate_after - rate_before,
        threshold_before=before.threshold,
        threshold_after=after.threshold,
        threshold_delta=after.threshold - before.threshold,
        category_deltas=category_deltas,
        label_deltas=label_deltas,
        style_deltas=style_deltas,
        language_deltas=language_deltas,
        turn_depth_deltas=turn_depth_deltas,
        framing_deltas=framing_deltas,
        cell_deltas=cell_deltas,
        coverage_score_before=before.coverage_score,
        coverage_score_after=after.coverage_score,
        coverage_score_delta=after.coverage_score - before.coverage_score,
        worst_cell_refusal_rate_before=worst_before,
        worst_cell_refusal_rate_after=worst_after,
        worst_cell_refusal_rate_delta=worst_delta,
    )


def _compute_group_deltas(
    before_groups: list[SurfaceGroup],
    after_groups: list[SurfaceGroup],
) -> list[SurfaceGroupDelta]:
    """Match groups by name and compute deltas between before and after."""
    after_by_name: dict[str, SurfaceGroup] = {g.name: g for g in after_groups}
    deltas: list[SurfaceGroupDelta] = []

    for bg in before_groups:
        ag = after_by_name.get(bg.name)
        if ag is None:
            continue
        deltas.append(
            SurfaceGroupDelta(
                name=bg.name,
                count=bg.count,
                refusal_rate_before=bg.refusal_rate,
                refusal_rate_after=ag.refusal_rate,
                refusal_rate_delta=ag.refusal_rate - bg.refusal_rate,
                mean_projection_before=bg.mean_projection,
                mean_projection_after=ag.mean_projection,
                mean_projection_delta=ag.mean_projection - bg.mean_projection,
            ),
        )

    return deltas


def _group_points(
    points: list[SurfacePoint],
    key: Callable[[SurfacePoint], str],
) -> list[SurfaceGroup]:
    """Group points by a key function and compute stats per group."""
    groups: dict[str, list[SurfacePoint]] = {}
    for point in points:
        name = key(point)
        if name not in groups:
            groups[name] = []
        groups[name].append(point)

    result: list[SurfaceGroup] = []
    for name, group in sorted(groups.items()):
        projections = [point.direction_projection for point in group]
        refused_count = sum(1 for point in group if point.refused is True)
        total = len(group)
        result.append(
            SurfaceGroup(
                name=name,
                count=total,
                refusal_rate=refused_count / total if total > 0 else 0.0,
                mean_projection=sum(projections) / len(projections),
                min_projection=min(projections),
                max_projection=max(projections),
            ),
        )
    return result


def _surface_cell_name(point: SurfacePoint) -> str:
    """Return a canonical name for one surface-matrix cell."""
    return (
        f"category={point.category}|style={point.style}|"
        f"language={point.language}|turn_depth={point.turn_depth}|"
        f"framing={point.framing}"
    )


def _coverage_score(points: list[SurfacePoint]) -> float:
    """Compute matrix occupancy for category/style/language/depth/framing."""
    if not points:
        return 0.0

    categories = {point.category for point in points}
    styles = {point.style for point in points}
    languages = {point.language for point in points}
    turn_depths = {point.turn_depth for point in points}
    framings = {point.framing for point in points}

    max_cells = (
        len(categories)
        * len(styles)
        * len(languages)
        * len(turn_depths)
        * len(framings)
    )
    if max_cells == 0:
        return 0.0

    observed_cells = {
        (
            point.category,
            point.style,
            point.language,
            point.turn_depth,
            point.framing,
        )
        for point in points
    }
    return len(observed_cells) / max_cells


def _max_refusal_rate(groups: list[SurfaceGroup]) -> float:
    """Return maximum refusal rate over groups."""
    return max((group.refusal_rate for group in groups), default=0.0)


def compute_taxonomy_coverage(
    points: list[SurfacePoint],
) -> TaxonomyCoverage:
    """Compute taxonomy coverage for a set of surface points."""
    from vauban.taxonomy import coverage_report

    observed: set[str] = {p.category for p in points}
    return coverage_report(observed)
