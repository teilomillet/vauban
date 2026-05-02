# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Runtime circuit breaker with tiered response and KV cache rewind.

Monitors generation token-by-token, classifying each token's activation
projection into a zone (green/yellow/orange/red) and taking graduated
action: pass, steer, rewind to a cached checkpoint, or hard stop.

Reference: Gray Swan "Circuit Breakers" (Zou et al., 2024) for the
concept; this implementation operates as a sidecar (no weight modification)
using measured directions from the vauban pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING as _TC

from vauban import _ops as ops
from vauban._forward import (
    embed_and_mask,
    embed_and_mask_with_prefix,
    encode_chat_prompt,
    force_eval,
    get_transformer,
    lm_head_forward,
    make_cache,
    make_ssm_mask,
    select_mask,
)
from vauban.types import (
    GuardConfig,
    GuardEvent,
    GuardResult,
    GuardTierSpec,
    GuardVerdict,
    GuardZone,
)

if _TC:  # pragma: no cover
    from vauban._array import Array
    from vauban.types import CausalLM, LayerCache, Tokenizer


# ---------------------------------------------------------------------------
# Zone classification
# ---------------------------------------------------------------------------


def _classify_zone(
    projection: float,
    tiers: list[GuardTierSpec],
) -> tuple[GuardZone, float]:
    """Resolve zone and alpha from the tier list.

    Walks tiers (ascending threshold) and returns the zone/alpha of the
    highest tier where ``projection >= threshold``.  Falls back to the
    first tier when no tier matches.
    """
    zone: GuardZone = tiers[0].zone
    alpha = tiers[0].alpha
    for tier in tiers:
        if projection >= tier.threshold:
            zone = tier.zone
            alpha = tier.alpha
        else:
            break
    return zone, alpha


# ---------------------------------------------------------------------------
# KV cache checkpoint helpers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _CacheCheckpoint:
    """Snapshot of KV cache state for rewind."""

    layer_states: list[tuple[Array | None, Array | None, int]]
    token_count: int
    generated_ids: list[int]
    next_token_ids: Array  # token IDs to feed after restore


def _snapshot_cache(
    cache: list[LayerCache],
    generated: list[int],
    next_token_ids: Array,
) -> _CacheCheckpoint:
    """Deep-copy current cache state for later restoration.

    Slices the valid region of each layer's K/V arrays.  MLX's
    copy-on-write semantics ensure the snapshot stays valid after
    subsequent in-place cache mutations.
    """
    states: list[tuple[Array | None, Array | None, int]] = []
    for lc in cache:
        offset = lc.offset
        keys = getattr(lc, "keys", None)
        values = getattr(lc, "values", None)
        k = keys[..., :offset, :] if keys is not None and offset > 0 else None
        v = values[..., :offset, :] if values is not None and offset > 0 else None
        states.append((k, v, offset))
    return _CacheCheckpoint(
        layer_states=states,
        token_count=len(generated),
        generated_ids=list(generated),
        next_token_ids=next_token_ids,
    )


def _restore_cache(
    model: CausalLM,
    checkpoint: _CacheCheckpoint,
) -> list[LayerCache]:
    """Create a fresh cache and populate it from a checkpoint snapshot.

    Returns a new cache list — the caller must replace its reference.
    """
    new_cache = make_cache(model)
    for lc, (k, v, offset) in zip(new_cache, checkpoint.layer_states, strict=True):
        if k is not None:
            lc.keys = k
        if v is not None:
            lc.values = v
        lc.offset = offset
    return new_cache


# ---------------------------------------------------------------------------
# Forward step with zone classification
# ---------------------------------------------------------------------------


def _guard_forward(
    model: CausalLM,
    token_ids: Array,
    direction: Array,
    guard_layers: list[int],
    tiers: list[GuardTierSpec],
    cache: list[LayerCache],
    *,
    condition_direction: Array | None = None,
) -> tuple[Array, float, GuardZone, float]:
    """Run one forward step with zone-classified conditional steering.

    Returns:
        ``(logits, mean_projection, zone, alpha_applied)``
    """
    transformer = get_transformer(model)
    h, mask = embed_and_mask(transformer, token_ids)

    direction = ops.to_device_like(direction, h)
    detect_dir = condition_direction if condition_direction is not None else direction
    detect_dir = ops.to_device_like(detect_dir, h)
    guard_layer_set = set(guard_layers)
    projections: list[float] = []
    ssm_mask = make_ssm_mask(transformer, h)

    # Accumulate the max detection projection across guard layers
    max_detect_value = float("-inf")

    for i, layer in enumerate(transformer.layers):
        h = layer(h, select_mask(layer, mask, ssm_mask), cache=cache[i])

        if i not in guard_layer_set:
            continue

        last_token = h[0, -1, :]

        # Detection projection
        detect_projection = ops.sum(last_token * detect_dir)
        force_eval(detect_projection)
        detect_value = float(detect_projection.item())
        if detect_value > max_detect_value:
            max_detect_value = detect_value

        # Steer-direction projection
        steer_projection = ops.sum(last_token * direction)
        force_eval(steer_projection)
        projections.append(float(steer_projection.item()))

    # Classify zone from the max detection projection across layers
    zone, alpha = _classify_zone(max_detect_value, tiers)

    # Apply steering if zone warrants it (yellow or above)
    if zone != "green" and alpha > 0.0:
        # Re-walk guard layers and apply correction (same pattern as CAST)
        # The steering was NOT applied above — we need a second pass
        # through the last guard layer.  Instead, we modify the hidden
        # state in-place at the output of the last transformer layer.
        # Since we already completed the full forward, we correct h
        # (the post-norm input to lm_head) by projecting out the
        # direction component scaled by alpha.
        mean_proj = sum(projections) / len(projections) if projections else 0.0
        correction = alpha * mean_proj * direction
        h_list = [h[0, j, :] for j in range(h.shape[1])]
        h_list[-1] = h_list[-1] - correction
        h = ops.stack(h_list)[None, :, :]

    h = transformer.norm(h)
    logits = lm_head_forward(model, h)
    force_eval(logits)

    mean_projection = sum(projections) / len(projections) if projections else 0.0
    return logits, mean_projection, zone, alpha


# ---------------------------------------------------------------------------
# Guard generate — main entry point
# ---------------------------------------------------------------------------


def guard_generate(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompt: str,
    direction: Array,
    layers: list[int],
    config: GuardConfig,
    condition_direction: Array | None = None,
    defensive_embeds: Array | None = None,
) -> GuardResult:
    """Generate text with runtime circuit breaker monitoring.

    Each generated token is classified into a zone based on its
    activation projection onto the measured direction.  Graduated
    actions are taken per zone: pass, steer, rewind (restore KV cache
    to last green checkpoint), or circuit break (hard stop).

    Args:
        model: The causal language model.
        tokenizer: Tokenizer with chat template support.
        prompt: The user prompt to generate from.
        direction: Measured refusal direction for projection/steering.
        layers: Layer indices to monitor.
        config: Guard configuration (tiers, rewind limits, etc.).
        condition_direction: Separate detection direction (AdaSteer).
        defensive_embeds: Pre-computed defensive embeddings for rewind
            injection.  Shape ``(1, n_tokens, d_model)``.

    Returns:
        GuardResult with full event log and generation stats.
    """
    messages = [{"role": "user", "content": prompt}]
    token_ids = encode_chat_prompt(tokenizer, messages)
    cache = make_cache(model)

    generated: list[int] = []
    events: list[GuardEvent] = []
    rewind_count = 0
    green_streak = 0
    tokens_rewound = 0
    circuit_broken = False
    zone_counts: dict[str, int] = {"green": 0, "yellow": 0, "orange": 0, "red": 0}

    # Initial prefill — run all prompt tokens through the model
    _prefill_logits, _, _, _ = _guard_forward(
        model, token_ids, direction, layers, config.tiers, cache,
        condition_direction=condition_direction,
    )

    # First token from prefill logits
    next_token = ops.argmax(_prefill_logits[:, -1, :], axis=-1)
    token_ids = next_token[:, None]

    # Take the initial checkpoint after prefill
    checkpoint = _snapshot_cache(cache, generated, token_ids)

    step = 0
    while step < config.max_tokens:
        logits, projection, zone, alpha = _guard_forward(
            model, token_ids, direction, layers, config.tiers, cache,
            condition_direction=condition_direction,
        )

        next_tok = ops.argmax(logits[:, -1, :], axis=-1)
        token_id = int(next_tok.item())
        token_str = tokenizer.decode([token_id])

        zone_counts[zone] += 1

        if zone == "green":
            action: str = "pass"
            generated.append(token_id)
            green_streak += 1
            if green_streak >= config.checkpoint_interval:
                checkpoint = _snapshot_cache(
                    cache, generated, next_tok[:, None],
                )
                green_streak = 0

        elif zone == "yellow":
            action = "steer"
            generated.append(token_id)
            green_streak = 0

        elif zone == "orange":
            rewind_count += 1
            if rewind_count > config.max_rewinds:
                action = "break"
                circuit_broken = True
                events.append(GuardEvent(
                    token_index=step,
                    token_id=token_id,
                    token_str=token_str,
                    projection=projection,
                    zone=zone,
                    action="break",
                    alpha_applied=alpha,
                    rewind_count=rewind_count,
                    checkpoint_offset=checkpoint.token_count,
                ))
                break

            action = "rewind"
            tokens_rewound += len(generated) - checkpoint.token_count

            if defensive_embeds is not None:
                # Full re-prefill with defensive prefix
                cache = _rewind_with_defense(
                    model, tokenizer, messages, defensive_embeds,
                    layers, direction, config.tiers,
                    condition_direction,
                )
                generated = []
                # Get logits from defense-prefilled cache
                def_logits, _, _, _ = _guard_forward(
                    model, token_ids, direction, layers,
                    config.tiers, cache,
                    condition_direction=condition_direction,
                )
                def_tok = ops.argmax(def_logits[:, -1, :], axis=-1)
                checkpoint = _snapshot_cache(
                    cache, generated, def_tok[:, None],
                )
                token_ids = def_tok[:, None]
            else:
                cache = _restore_cache(model, checkpoint)
                generated = list(checkpoint.generated_ids)
                token_ids = checkpoint.next_token_ids

            green_streak = 0
            events.append(GuardEvent(
                token_index=step,
                token_id=token_id,
                token_str=token_str,
                projection=projection,
                zone=zone,
                action="rewind",
                alpha_applied=alpha,
                rewind_count=rewind_count,
                checkpoint_offset=checkpoint.token_count,
            ))
            step += 1
            continue

        else:  # red
            action = "break"
            circuit_broken = True
            events.append(GuardEvent(
                token_index=step,
                token_id=token_id,
                token_str=token_str,
                projection=projection,
                zone=zone,
                action="break",
                alpha_applied=alpha,
                rewind_count=rewind_count,
                checkpoint_offset=checkpoint.token_count,
            ))
            break

        events.append(GuardEvent(
            token_index=step,
            token_id=token_id,
            token_str=token_str,
            projection=projection,
            zone=zone,
            action=action,
            alpha_applied=alpha,
            rewind_count=rewind_count,
            checkpoint_offset=checkpoint.token_count,
        ))

        token_ids = next_tok[:, None]
        step += 1

    return GuardResult(
        prompt=prompt,
        text=tokenizer.decode(generated),
        events=events,
        total_rewinds=rewind_count,
        circuit_broken=circuit_broken,
        tokens_generated=len(generated),
        tokens_rewound=tokens_rewound,
        final_zone_counts=zone_counts,
    )


# ---------------------------------------------------------------------------
# Defensive rewind — full re-prefill with defensive prefix
# ---------------------------------------------------------------------------


def _rewind_with_defense(
    model: CausalLM,
    tokenizer: Tokenizer,
    messages: list[dict[str, str]],
    defensive_embeds: Array,
    layers: list[int],
    direction: Array,
    tiers: list[GuardTierSpec],
    condition_direction: Array | None,
) -> list[LayerCache]:
    """Create fresh cache with defensive embeddings prepended to prompt.

    Returns a populated cache ready for autoregressive continuation.
    The defensive prefix changes the sequence length, so all KV positions
    are recomputed from scratch.
    """
    token_ids = encode_chat_prompt(tokenizer, messages)
    new_cache = make_cache(model)
    transformer = get_transformer(model)
    h, mask = embed_and_mask_with_prefix(transformer, defensive_embeds, token_ids)
    ssm_mask = make_ssm_mask(transformer, h)

    for i, layer in enumerate(transformer.layers):
        h = layer(h, select_mask(layer, mask, ssm_mask), cache=new_cache[i])

    force_eval(h)
    return new_cache


# ---------------------------------------------------------------------------
# Calibration — set tier thresholds from clean prompt projections
# ---------------------------------------------------------------------------


def calibrate_guard_thresholds(
    model: CausalLM,
    tokenizer: Tokenizer,
    clean_prompts: list[str],
    direction: Array,
    layers: list[int],
    *,
    percentiles: tuple[float, float, float] = (0.90, 0.95, 0.99),
) -> list[GuardTierSpec]:
    """Auto-calibrate guard tier thresholds from clean prompt projections.

    Runs each clean prompt through the model, collects per-token
    projections at the guard layers, and sets tier thresholds at the
    given percentiles of the resulting distribution.

    Args:
        model: The causal language model.
        tokenizer: Tokenizer with chat template support.
        clean_prompts: Harmless prompts for calibration.
        direction: Measured refusal direction.
        layers: Layer indices to monitor.
        percentiles: Tuple of three percentile values for yellow, orange,
            red thresholds (default: 90th, 95th, 99th).

    Returns:
        A 4-tier list (green + three calibrated tiers).
    """
    all_projections: list[float] = []
    guard_layer_set = set(layers)

    for prompt_text in clean_prompts:
        messages = [{"role": "user", "content": prompt_text}]
        token_ids = encode_chat_prompt(tokenizer, messages)
        transformer = get_transformer(model)
        h, mask = embed_and_mask(transformer, token_ids)
        prompt_direction = ops.to_device_like(direction, h)
        ssm_mask = make_ssm_mask(transformer, h)
        cache = make_cache(model)

        for i, layer in enumerate(transformer.layers):
            h = layer(h, select_mask(layer, mask, ssm_mask), cache=cache[i])
            if i in guard_layer_set:
                # Collect projections for all token positions
                for pos in range(h.shape[1]):
                    tok = h[0, pos, :]
                    proj = ops.sum(tok * prompt_direction)
                    force_eval(proj)
                    all_projections.append(float(proj.item()))

    if not all_projections:
        from vauban.types import _DEFAULT_GUARD_TIERS
        return list(_DEFAULT_GUARD_TIERS)

    sorted_projs = sorted(all_projections)
    n = len(sorted_projs)

    def _percentile(p: float) -> float:
        idx = min(int(p * n), n - 1)
        return sorted_projs[idx]

    p_yellow = _percentile(percentiles[0])
    p_orange = _percentile(percentiles[1])
    p_red = _percentile(percentiles[2])

    green = 0.0
    yellow = max(green, p_yellow)
    orange = max(yellow, p_orange)
    red = max(orange, p_red)

    return [
        GuardTierSpec(threshold=green, zone="green", alpha=0.0),
        GuardTierSpec(threshold=yellow, zone="yellow", alpha=0.5),
        GuardTierSpec(threshold=orange, zone="orange", alpha=1.5),
        GuardTierSpec(threshold=red, zone="red", alpha=3.0),
    ]


# ---------------------------------------------------------------------------
# GuardSession — stateful integration API for inference engines
# ---------------------------------------------------------------------------


class GuardSession:
    """Stateful circuit breaker session for inference engine integration.

    This is the production integration API.  The inference engine owns the
    model and KV cache; ``GuardSession`` owns only the monitoring state.
    Each call to :meth:`check` takes the current hidden-state activation
    and returns a :class:`GuardVerdict` telling the caller what to do.

    Usage::

        from vauban.guard import GuardSession

        # At startup (once per model):
        session = GuardSession.from_file("direction.npy")

        # Per generation request:
        session.reset()
        for step, hidden_state in enumerate(generate_tokens()):
            verdict = session.check(hidden_state)
            if verdict.action == "rewind":
                kv_cache.rollback_to(session.checkpoint_offset)
            elif verdict.action == "break":
                return partial_response()

    The ``check`` method performs one dot product — no model forward pass,
    no heavy computation.  Overhead is ~microseconds per token.
    """

    def __init__(
        self,
        direction: Array,
        tiers: list[GuardTierSpec] | None = None,
        *,
        max_rewinds: int = 3,
        checkpoint_interval: int = 1,
        condition_direction: Array | None = None,
        extra_directions: list[Array] | None = None,
    ) -> None:
        from vauban.types import _DEFAULT_GUARD_TIERS

        self._direction = direction
        self._tiers = tiers if tiers is not None else list(_DEFAULT_GUARD_TIERS)
        self._condition_direction = condition_direction
        self._extra_directions = extra_directions or []
        self._max_rewinds = max_rewinds
        self._checkpoint_interval = checkpoint_interval

        # Per-generation state
        self._step = 0
        self._rewind_count = 0
        self._green_streak = 0
        self._checkpoint_offset = 0
        self._circuit_broken = False
        self._events: list[GuardVerdict] = []

    # -- Factories ----------------------------------------------------------

    @classmethod
    def from_file(
        cls,
        direction_path: str,
        tiers_path: str | None = None,
        *,
        extra_direction_paths: list[str] | None = None,
        **kwargs: object,
    ) -> GuardSession:
        """Load direction(s) from .npy and optional tiers from JSON.

        Args:
            direction_path: Path to the primary direction ``.npy``.
            tiers_path: Optional path to a JSON file with tier specs.
                Each entry must have ``threshold``, ``zone``, ``alpha``.
            extra_direction_paths: Additional direction ``.npy`` files
                (e.g. an encoding-aware direction measured from
                cipher-wrapped prompts).  The guard projects onto all
                directions and classifies using the max projection.
            **kwargs: Forwarded to ``__init__`` (max_rewinds, etc.).
        """
        import json

        import numpy as np

        direction = ops.array(np.load(direction_path))

        tiers: list[GuardTierSpec] | None = None
        if tiers_path is not None:
            with open(tiers_path) as f:
                raw = json.load(f)
            tiers = [
                GuardTierSpec(
                    threshold=float(t["threshold"]),
                    zone=t["zone"],
                    alpha=float(t["alpha"]),
                )
                for t in raw
            ]

        extra: list[Array] = []
        if extra_direction_paths:
            extra = [
                ops.array(np.load(p)) for p in extra_direction_paths
            ]

        return cls(direction, tiers, extra_directions=extra, **kwargs)  # type: ignore[arg-type]

    # -- Core API -----------------------------------------------------------

    def check(self, activation: Array) -> GuardVerdict:
        """Evaluate one token's activation and return a verdict.

        Args:
            activation: Hidden-state vector at the monitored layer.
                Shape ``(d_model,)`` — a single position from a single
                layer during autoregressive generation.

        Returns:
            A :class:`GuardVerdict` with the zone, recommended action,
            projection value, and current session state.
        """
        if self._circuit_broken:
            verdict = GuardVerdict(
                zone="red",
                action="break",
                projection=0.0,
                alpha=0.0,
                step=self._step,
                rewind_count=self._rewind_count,
                checkpoint_offset=self._checkpoint_offset,
            )
            self._events.append(verdict)
            return verdict

        # Detection projection — max across all monitored directions
        detect_dir = (
            self._condition_direction
            if self._condition_direction is not None
            else self._direction
        )
        detect_dir = ops.to_device_like(detect_dir, activation)
        projection = float(ops.sum(activation * detect_dir).item())
        for extra_dir in self._extra_directions:
            extra_dir = ops.to_device_like(extra_dir, activation)
            extra_proj = float(ops.sum(activation * extra_dir).item())
            if extra_proj > projection:
                projection = extra_proj

        # Classify
        zone, alpha = _classify_zone(projection, self._tiers)

        # Determine action
        action: str
        if zone == "green":
            action = "pass"
            self._green_streak += 1
            if self._green_streak >= self._checkpoint_interval:
                self._checkpoint_offset = self._step
                self._green_streak = 0

        elif zone == "yellow":
            action = "steer"
            self._green_streak = 0

        elif zone == "orange":
            self._rewind_count += 1
            self._green_streak = 0
            if self._rewind_count > self._max_rewinds:
                action = "break"
                self._circuit_broken = True
            else:
                action = "rewind"

        else:  # red
            action = "break"
            self._circuit_broken = True

        verdict = GuardVerdict(
            zone=zone,
            action=action,
            projection=projection,
            alpha=alpha,
            step=self._step,
            rewind_count=self._rewind_count,
            checkpoint_offset=self._checkpoint_offset,
        )
        self._events.append(verdict)
        self._step += 1
        return verdict

    def rewind(self) -> int:
        """Acknowledge a rewind and return the offset to roll back to.

        Call this after receiving ``action="rewind"`` from :meth:`check`
        and before resuming generation.  Resets the step counter to the
        checkpoint offset.

        Returns:
            The token offset the inference engine should restore its
            KV cache to.
        """
        offset = self._checkpoint_offset
        self._step = offset
        self._green_streak = 0
        return offset

    def reset(self) -> None:
        """Reset session state for a new generation request."""
        self._step = 0
        self._rewind_count = 0
        self._green_streak = 0
        self._checkpoint_offset = 0
        self._circuit_broken = False
        self._events = []

    # -- Properties ---------------------------------------------------------

    @property
    def direction(self) -> Array:
        """The measured direction vector."""
        return self._direction

    @property
    def tiers(self) -> list[GuardTierSpec]:
        """The active tier configuration."""
        return self._tiers

    @property
    def rewind_count(self) -> int:
        """Total rewinds in the current generation."""
        return self._rewind_count

    @property
    def circuit_broken(self) -> bool:
        """Whether the circuit breaker has tripped."""
        return self._circuit_broken

    @property
    def checkpoint_offset(self) -> int:
        """Token offset of the last green checkpoint."""
        return self._checkpoint_offset

    @property
    def events(self) -> list[GuardVerdict]:
        """Full audit log of verdicts for the current generation."""
        return list(self._events)

    @property
    def step(self) -> int:
        """Current token step."""
        return self._step
