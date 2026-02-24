"""Refusal surface mapping — scan prompts, measure projections, detect refusals."""

import json
import sys
from collections.abc import Callable
from pathlib import Path

import mlx.core as mx

from vauban.evaluate import DEFAULT_REFUSAL_PHRASES, _generate
from vauban.probe import probe
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


def default_surface_path() -> Path:
    """Return path to the bundled categorized surface prompt file."""
    return Path(__file__).parent / "data" / "surface.jsonl"


def load_surface_prompts(path: str | Path) -> list[SurfacePrompt]:
    """Load surface prompts from a JSONL file.

    Each line must have ``prompt``, ``label``, and ``category`` keys.
    """
    prompts: list[SurfacePrompt] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompts.append(
                SurfacePrompt(
                    prompt=obj["prompt"],
                    label=obj["label"],
                    category=obj["category"],
                ),
            )
    return prompts


def scan(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompts: list[SurfacePrompt],
    direction: mx.array,
    direction_layer: int,
    *,
    generate: bool = True,
    max_tokens: int = 60,
    refusal_phrases: list[str] | None = None,
    progress: bool = True,
) -> list[SurfacePoint]:
    """Scan prompts: probe projections and optionally generate + detect refusal.

    Args:
        model: The causal language model.
        tokenizer: Tokenizer with chat template support.
        prompts: Surface prompts with label/category metadata.
        direction: The refusal direction vector.
        direction_layer: Layer index to read the direction projection from.
        generate: If True, generate a response and detect refusal.
        max_tokens: Maximum tokens to generate per prompt.
        refusal_phrases: Phrases indicating refusal. Defaults to standard set.
        progress: If True, print progress to stderr.

    Returns:
        A list of SurfacePoint results, one per prompt.
    """
    if refusal_phrases is None:
        refusal_phrases = DEFAULT_REFUSAL_PHRASES

    points: list[SurfacePoint] = []
    total = len(prompts)

    for i, sp in enumerate(prompts):
        if progress:
            print(
                f"\r  Scanning {i + 1}/{total}: {sp.prompt[:50]}...",
                end="",
                file=sys.stderr,
                flush=True,
            )

        probe_result = probe(model, tokenizer, sp.prompt, direction)
        direction_proj = probe_result.projections[direction_layer]

        refused: bool | None = None
        response: str | None = None
        if generate:
            response = _generate(model, tokenizer, sp.prompt, max_tokens)  # type: ignore[arg-type]
            lower = response.lower()
            refused = any(
                phrase.lower() in lower for phrase in refusal_phrases
            )

        points.append(
            SurfacePoint(
                prompt=sp.prompt,
                label=sp.label,
                category=sp.category,
                projections=probe_result.projections,
                direction_projection=direction_proj,
                refused=refused,
                response=response,
            ),
        )

    if progress:
        print("", file=sys.stderr)

    return points


type SurfaceGroups = tuple[list[SurfaceGroup], list[SurfaceGroup]]


def aggregate(
    points: list[SurfacePoint],
) -> SurfaceGroups:
    """Group surface points by label and category, computing stats.

    Returns:
        A tuple of (groups_by_label, groups_by_category).
    """
    by_label = _group_points(points, key=lambda p: p.label)
    by_category = _group_points(points, key=lambda p: p.category)
    return by_label, by_category


def find_threshold(points: list[SurfacePoint]) -> float:
    """Find the projection value separating refused and compliant prompts.

    Uses the midpoint between the highest compliant projection and the
    lowest refusing projection. Returns 0.0 if all refuse or none refuse.
    """
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
    direction: mx.array,
    direction_layer: int,
    *,
    generate: bool = True,
    max_tokens: int = 60,
    refusal_phrases: list[str] | None = None,
    progress: bool = True,
) -> SurfaceResult:
    """Convenience: scan + aggregate + find_threshold in one call.

    Args:
        model: The causal language model.
        tokenizer: Tokenizer with chat template support.
        prompts: Surface prompts with label/category metadata.
        direction: The refusal direction vector.
        direction_layer: Layer index to read the direction projection from.
        generate: If True, generate responses and detect refusal.
        max_tokens: Maximum tokens to generate per prompt.
        refusal_phrases: Phrases indicating refusal.
        progress: If True, print progress to stderr.

    Returns:
        A complete SurfaceResult with points, groups, and threshold.
    """
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
    )

    groups_by_label, groups_by_category = aggregate(points)
    threshold = find_threshold(points) if generate else 0.0

    total_refused = sum(1 for p in points if p.refused is True)

    return SurfaceResult(
        points=points,
        groups_by_label=groups_by_label,
        groups_by_category=groups_by_category,
        threshold=threshold,
        total_scanned=len(points),
        total_refused=total_refused,
    )


def compare_surfaces(
    before: SurfaceResult,
    after: SurfaceResult,
) -> SurfaceComparison:
    """Compare refusal surface results before and after cut.

    Computes overall refusal rate deltas, threshold shift, and
    per-group deltas for both categories and labels.

    Args:
        before: Surface result from the original model.
        after: Surface result from the modified model.

    Returns:
        A SurfaceComparison with all deltas computed.
    """
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

    return SurfaceComparison(
        before=before,
        after=after,
        refusal_rate_before=rate_before,
        refusal_rate_after=rate_after,
        refusal_rate_delta=rate_after - rate_before,
        threshold_before=before.threshold,
        threshold_after=after.threshold,
        threshold_delta=after.threshold - before.threshold,
        category_deltas=_compute_group_deltas(
            before.groups_by_category, after.groups_by_category,
        ),
        label_deltas=_compute_group_deltas(
            before.groups_by_label, after.groups_by_label,
        ),
    )


def _compute_group_deltas(
    before_groups: list[SurfaceGroup],
    after_groups: list[SurfaceGroup],
) -> list[SurfaceGroupDelta]:
    """Match groups by name and compute deltas between before and after."""
    after_by_name: dict[str, SurfaceGroup] = {
        g.name: g for g in after_groups
    }
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
    for p in points:
        name = key(p)
        if name not in groups:
            groups[name] = []
        groups[name].append(p)

    result: list[SurfaceGroup] = []
    for name, group in sorted(groups.items()):
        projs = [p.direction_projection for p in group]
        refused_count = sum(1 for p in group if p.refused is True)
        total = len(group)

        result.append(
            SurfaceGroup(
                name=name,
                count=total,
                refusal_rate=refused_count / total if total > 0 else 0.0,
                mean_projection=sum(projs) / len(projs),
                min_projection=min(projs),
                max_projection=max(projs),
            ),
        )

    return result
