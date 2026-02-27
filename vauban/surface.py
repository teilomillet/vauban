"""Refusal surface mapping — scan prompts, measure projections, detect refusals."""

import json
import sys
from collections.abc import Callable
from pathlib import Path

from vauban import _ops as ops
from vauban._array import Array
from vauban._forward import embed_and_mask, extract_logits, force_eval, make_cache
from vauban.evaluate import DEFAULT_REFUSAL_PHRASES, _judge_single
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

DEFAULT_SURFACE_STYLE = "unspecified"
DEFAULT_SURFACE_LANGUAGE = "unspecified"
DEFAULT_SURFACE_FRAMING = "unspecified"
DEFAULT_SURFACE_TURN_DEPTH = 1
_ALLOWED_SURFACE_ROLES: frozenset[str] = frozenset(
    {"system", "user", "assistant"},
)


def default_surface_path() -> Path:
    """Return path to the bundled categorized surface prompt file."""
    return Path(__file__).parent / "data" / "surface.jsonl"


def default_multilingual_surface_path() -> Path:
    """Return path to the bundled multilingual surface prompt file."""
    return Path(__file__).parent / "data" / "surface_multilingual.jsonl"


def load_surface_prompts(path: str | Path) -> list[SurfacePrompt]:
    """Load surface prompts from a JSONL file.

    Each line must have ``label`` and ``category``. It must include either a
    non-empty ``prompt`` string, a non-empty ``messages`` list, or both.
    Optional keys ``style``, ``language``, ``turn_depth``, and ``framing``
    are validated when present and defaulted when missing. For message-only
    records, ``prompt`` is derived from the last user message.
    """
    path_obj = Path(path)
    prompts: list[SurfacePrompt] = []
    with path_obj.open() as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            obj_raw = json.loads(stripped)
            if not isinstance(obj_raw, dict):
                msg = (
                    f"surface prompts line {line_no} must be a JSON object"
                    f" in {path_obj}"
                )
                raise ValueError(msg)
            obj: dict[str, object] = {}
            for raw_key, raw_value in obj_raw.items():
                if not isinstance(raw_key, str):
                    msg = (
                        "surface prompt keys must be strings on line"
                        f" {line_no} in {path_obj}"
                    )
                    raise ValueError(msg)
                obj[raw_key] = raw_value

            label = _require_non_empty_text(
                obj, "label", line_no, path_obj,
            )
            category = _require_non_empty_text(
                obj, "category", line_no, path_obj,
            )
            messages = _optional_messages(
                obj, "messages", line_no, path_obj,
            )
            prompt_raw = obj.get("prompt")
            if prompt_raw is None:
                if messages is None:
                    msg = (
                        "surface prompts line"
                        f" {line_no} must include key 'prompt' or 'messages'"
                        f" in {path_obj}"
                    )
                    raise ValueError(msg)
                prompt = _derive_prompt_from_messages(messages)
            elif not isinstance(prompt_raw, str) or not prompt_raw.strip():
                msg = (
                    f"surface prompts line {line_no} has invalid key 'prompt'"
                    f" in {path_obj}; expected a non-empty string"
                )
                raise ValueError(msg)
            else:
                prompt = prompt_raw

            style = _optional_non_empty_text(
                obj,
                "style",
                DEFAULT_SURFACE_STYLE,
                line_no,
                path_obj,
            )
            language = _optional_non_empty_text(
                obj,
                "language",
                DEFAULT_SURFACE_LANGUAGE,
                line_no,
                path_obj,
            )
            framing = _optional_non_empty_text(
                obj,
                "framing",
                DEFAULT_SURFACE_FRAMING,
                line_no,
                path_obj,
            )
            turn_depth = _optional_turn_depth(
                obj,
                "turn_depth",
                _infer_turn_depth(messages),
                line_no,
                path_obj,
            )

            prompts.append(
                SurfacePrompt(
                    prompt=prompt,
                    label=label,
                    category=category,
                    style=style,
                    language=language,
                    turn_depth=turn_depth,
                    framing=framing,
                    messages=messages,
                ),
            )
    return prompts


def scan(
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
        refusal_mode: "phrases" for substring matching, "judge" for model-based.

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

        messages = _surface_messages(sp)
        projections = _probe_with_messages(
            model,
            tokenizer,
            messages,
            direction,
        )
        direction_proj = projections[direction_layer]

        refused: bool | None = None
        response: str | None = None
        if generate:
            response = _generate_with_messages(
                model,
                tokenizer,
                messages,
                max_tokens,
            )
            if refusal_mode == "judge":
                refused = _judge_single(model, tokenizer, sp.prompt, response)
            else:
                lower = response.lower()
                refused = any(
                    phrase.lower() in lower for phrase in refusal_phrases
                )

        points.append(
            SurfacePoint(
                prompt=sp.prompt,
                label=sp.label,
                category=sp.category,
                projections=projections,
                direction_projection=direction_proj,
                refused=refused,
                response=response,
                style=sp.style,
                language=sp.language,
                turn_depth=sp.turn_depth,
                framing=sp.framing,
                messages=sp.messages,
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
    direction: Array,
    direction_layer: int,
    *,
    generate: bool = True,
    max_tokens: int = 60,
    refusal_phrases: list[str] | None = None,
    progress: bool = True,
    refusal_mode: str = "phrases",
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
        refusal_mode: "phrases" for substring matching, "judge" for model-based.

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
    worst_delta = (
        max((d.refusal_rate_delta for d in cell_deltas), default=0.0)
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


def _require_non_empty_text(
    obj: dict[str, object],
    key: str,
    line_no: int,
    path: Path,
) -> str:
    """Read a required non-empty string key from a parsed JSON object."""
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        msg = (
            f"surface prompts line {line_no} must include non-empty"
            f" string key {key!r} in {path}"
        )
        raise ValueError(msg)
    return value


def _optional_non_empty_text(
    obj: dict[str, object],
    key: str,
    default: str,
    line_no: int,
    path: Path,
) -> str:
    """Read an optional non-empty string key, defaulting when absent."""
    value = obj.get(key)
    if value is None:
        return default
    if not isinstance(value, str) or not value.strip():
        msg = (
            f"surface prompts line {line_no} has invalid optional key"
            f" {key!r} in {path}; expected a non-empty string"
        )
        raise ValueError(msg)
    return value


def _optional_turn_depth(
    obj: dict[str, object],
    key: str,
    default: int,
    line_no: int,
    path: Path,
) -> int:
    """Read optional integer turn depth (>= 1), defaulting when absent."""
    value = obj.get(key)
    if value is None:
        return default
    if not isinstance(value, int) or value < 1:
        msg = (
            f"surface prompts line {line_no} has invalid optional key"
            f" {key!r} in {path}; expected an integer >= 1"
        )
        raise ValueError(msg)
    return value


def _optional_messages(
    obj: dict[str, object],
    key: str,
    line_no: int,
    path: Path,
) -> list[dict[str, str]] | None:
    """Read optional chat messages from one surface prompt record."""
    value = obj.get(key)
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        msg = (
            f"surface prompts line {line_no} has invalid key {key!r}"
            f" in {path}; expected a non-empty list"
        )
        raise ValueError(msg)

    messages: list[dict[str, str]] = []
    for i, item in enumerate(value):
        if not isinstance(item, dict):
            msg = (
                f"surface prompts line {line_no} has invalid {key}[{i}]"
                f" in {path}; expected an object"
            )
            raise ValueError(msg)
        role_raw = item.get("role")
        content_raw = item.get("content")
        if (
            not isinstance(role_raw, str)
            or role_raw not in _ALLOWED_SURFACE_ROLES
            or not isinstance(content_raw, str)
            or not content_raw.strip()
        ):
            msg = (
                f"surface prompts line {line_no} has invalid {key}[{i}]"
                f" in {path}; expected role/content strings with role in"
                f" {sorted(_ALLOWED_SURFACE_ROLES)}"
            )
            raise ValueError(msg)
        messages.append({"role": role_raw, "content": content_raw})

    return messages


def _derive_prompt_from_messages(messages: list[dict[str, str]]) -> str:
    """Derive a display prompt from messages (prefer the last user turn)."""
    for message in reversed(messages):
        if message["role"] == "user":
            return message["content"]
    return messages[-1]["content"]


def _infer_turn_depth(messages: list[dict[str, str]] | None) -> int:
    """Infer turn depth from message history (number of user turns)."""
    if messages is None:
        return DEFAULT_SURFACE_TURN_DEPTH

    user_turns = sum(1 for message in messages if message["role"] == "user")
    return user_turns if user_turns > 0 else DEFAULT_SURFACE_TURN_DEPTH


def _surface_messages(surface_prompt: SurfacePrompt) -> list[dict[str, str]]:
    """Return full chat history for probing/generation."""
    if surface_prompt.messages is not None:
        return [
            {"role": message["role"], "content": message["content"]}
            for message in surface_prompt.messages
        ]
    return [{"role": "user", "content": surface_prompt.prompt}]


def _probe_with_messages(
    model: CausalLM,
    tokenizer: Tokenizer,
    messages: list[dict[str, str]],
    direction: Array,
) -> list[float]:
    """Compute per-layer projections for a full message list."""
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    if not isinstance(text, str):
        msg = "apply_chat_template must return str when tokenize=False"
        raise TypeError(msg)
    token_ids = ops.array(tokenizer.encode(text))[None, :]

    transformer = model.model
    h, mask = embed_and_mask(transformer, token_ids)

    projections: list[float] = []
    for layer in transformer.layers:
        h = layer(h, mask)
        last_token = h[0, -1, :]
        proj = ops.sum(last_token * direction)
        force_eval(proj)
        projections.append(float(proj.item()))
    return projections


def _generate_with_messages(
    model: CausalLM,
    tokenizer: Tokenizer,
    messages: list[dict[str, str]],
    max_tokens: int,
    eos_token_id: int | None = None,
) -> str:
    """Greedy generation over an arbitrary chat message list."""
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    if not isinstance(text, str):
        msg = "apply_chat_template must return str when tokenize=False"
        raise TypeError(msg)
    tokens = tokenizer.encode(text)
    generated: list[int] = []

    if eos_token_id is None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)

    cache = make_cache(model)
    token_ids = ops.array([tokens])
    for _ in range(max_tokens):
        result = model(token_ids, cache=cache)  # type: ignore[call-non-callable]
        logits = extract_logits(result)
        next_token = int(ops.argmax(logits[:, -1, :], axis=-1).item())
        generated.append(next_token)
        if eos_token_id is not None and next_token == eos_token_id:
            break
        token_ids = ops.array([[next_token]])

    return tokenizer.decode(generated)


def _surface_cell_name(point: SurfacePoint) -> str:
    """Return a canonical name for one surface-matrix cell."""
    return (
        f"category={point.category}|style={point.style}|"
        f"language={point.language}|turn_depth={point.turn_depth}|"
        f"framing={point.framing}"
    )


def _coverage_score(points: list[SurfacePoint]) -> float:
    """Compute matrix occupancy for (category, style, language, depth, framing)."""
    if not points:
        return 0.0

    categories = {p.category for p in points}
    styles = {p.style for p in points}
    languages = {p.language for p in points}
    turn_depths = {p.turn_depth for p in points}
    framings = {p.framing for p in points}

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
        (p.category, p.style, p.language, p.turn_depth, p.framing)
        for p in points
    }
    return len(observed_cells) / max_cells


def _max_refusal_rate(groups: list[SurfaceGroup]) -> float:
    """Return maximum refusal rate over groups (0.0 when empty)."""
    return max((group.refusal_rate for group in groups), default=0.0)
