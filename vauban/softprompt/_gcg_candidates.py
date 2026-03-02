"""Candidate and beam helpers for GCG."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from vauban import _ops as ops
from vauban._forward import force_eval
from vauban.softprompt._search import (
    _sample_prompt_ids,
    _select_prompt_ids,
    _select_worst_k_prompt_ids,
)

if TYPE_CHECKING:
    from vauban._array import Array
    from vauban.softprompt._gcg_objective import GCGSharedState
    from vauban.types import SoftPromptConfig


def _allowed_indices_from_mask(
    vocab_mask: Array | None,
    vocab_size: int,
) -> list[int]:
    """Build the list of token IDs allowed for initialization."""
    if vocab_mask is None:
        return list(range(vocab_size))
    force_eval(vocab_mask)
    return [index for index in range(vocab_size) if bool(vocab_mask[index].item())]


def _initialize_restart_tokens(
    config: SoftPromptConfig,
    allowed_indices: list[int],
    restart_idx: int,
) -> list[int]:
    """Initialize the token IDs for a single restart."""
    if config.init_tokens is not None and restart_idx == 0:
        initial = list(config.init_tokens)
        if len(initial) < config.n_tokens:
            initial.extend(
                random.choice(allowed_indices)
                for _ in range(config.n_tokens - len(initial))
            )
        return initial[: config.n_tokens]
    return [random.choice(allowed_indices) for _ in range(config.n_tokens)]


def _initialize_beam(
    current_ids: list[int],
    beam_width: int,
    n_tokens: int,
    allowed_indices: list[int],
) -> tuple[list[list[int]], list[float]]:
    """Initialize the tracked beam for a restart."""
    beam: list[list[int]] = [list(current_ids)]
    beam_losses = [float("inf")]
    for _ in range(max(beam_width - 1, 0)):
        beam.append([random.choice(allowed_indices) for _ in range(n_tokens)])
        beam_losses.append(float("inf"))
    return beam, beam_losses


def _select_step_prompts(
    config: SoftPromptConfig,
    state: GCGSharedState,
    all_prompt_ids: list[Array],
    soft_embeds: Array,
    step: int,
) -> list[Array]:
    """Select the prompt subset used for the current optimization step."""
    if config.prompt_strategy == "worst_k":
        return _select_worst_k_prompt_ids(
            state.model,
            soft_embeds,
            all_prompt_ids,
            state.target_ids,
            state.n_tokens,
            config.worst_k,
            state.direction,
            state.direction_weight,
            state.direction_mode,
            state.direction_layers,
            state.eos_token_id,
            state.eos_loss_mode,
            state.eos_loss_weight,
            state.ref_model,
            state.kl_ref_weight,
            loss_mode=state.loss_mode,
            refusal_ids=state.refusal_ids,
            defense_aware_weight=state.defense.weight,
            sic_layer=state.defense.sic_layer,
            sic_threshold=state.defense.sic_threshold,
            cast_layers=state.defense.cast_layers,
            cast_threshold=state.defense.cast_threshold,
            perplexity_weight=state.perplexity_weight,
            token_position=state.token_position,
        )
    if config.prompt_strategy == "sample":
        return _sample_prompt_ids(all_prompt_ids, config.worst_k)
    return _select_prompt_ids(all_prompt_ids, step, config.prompt_strategy)


def _score_token_candidates(
    grad: Array,
    embed_matrix: Array,
    vocab_mask: Array | None,
) -> Array:
    """Project gradients into token space and apply optional masking."""
    scores = -ops.matmul(grad[0], embed_matrix.T)
    if vocab_mask is None:
        return scores
    return ops.where(vocab_mask, scores, ops.array(-1e10))


def _top_candidate_indices(
    scores: Array,
    top_k: int,
    vocab_size: int,
) -> tuple[Array, int]:
    """Return the top token IDs per position."""
    effective_k = min(top_k, vocab_size)
    sorted_indices = ops.argsort(-scores, axis=-1)
    top_indices = sorted_indices[:, :effective_k]
    force_eval(top_indices)
    return top_indices, effective_k


def _sample_greedy_candidates(
    current_ids: list[int],
    top_indices: Array,
    batch_size: int,
    n_tokens: int,
    effective_k: int,
) -> list[list[int]]:
    """Sample single-token mutations from the current candidate."""
    candidates: list[list[int]] = []
    for _ in range(batch_size):
        pos = random.randint(0, n_tokens - 1)
        tok_idx = random.randint(0, effective_k - 1)
        candidate = list(current_ids)
        candidate[pos] = int(top_indices[pos, tok_idx].item())
        candidates.append(candidate)
    return candidates


def _sample_beam_candidates(
    beam: list[list[int]],
    top_indices: Array,
    batch_size: int,
    beam_width: int,
    n_tokens: int,
    effective_k: int,
) -> list[list[int]]:
    """Sample candidate mutations distributed across the active beam."""
    candidates_per_member = max(1, batch_size // beam_width)
    candidates: list[list[int]] = []
    for member in beam:
        for _ in range(candidates_per_member):
            pos = random.randint(0, n_tokens - 1)
            tok_idx = random.randint(0, effective_k - 1)
            candidate = list(member)
            candidate[pos] = int(top_indices[pos, tok_idx].item())
            candidates.append(candidate)
    return candidates


def _update_beam(
    beam: list[list[int]],
    beam_losses: list[float],
    candidates: list[list[int]],
    candidate_losses: list[float],
    beam_width: int,
) -> tuple[list[list[int]], list[float]]:
    """Merge and deduplicate beam candidates, keeping the top entries."""
    pool: list[tuple[list[int], float]] = list(
        zip(candidates, candidate_losses, strict=True),
    )
    for idx, member in enumerate(beam):
        pool.append((member, beam_losses[idx]))
    seen: set[tuple[int, ...]] = set()
    unique: list[tuple[list[int], float]] = []
    for ids, loss in pool:
        key = tuple(ids)
        if key not in seen:
            seen.add(key)
            unique.append((ids, loss))
    unique.sort(key=lambda item: item[1])
    trimmed = unique[:beam_width]
    return (
        [ids for ids, _ in trimmed],
        [loss for _, loss in trimmed],
    )
