"""AmpleGCG: generator-based adversarial suffix search.

Three-phase approach:
1. **Collect**: Run GCG internally, harvest suffixes below loss threshold
2. **Train**: Fit a 2-layer MLP (prompt embedding → suffix token logits)
3. **Generate**: Sample N candidates from MLP, filter by loss, return best

Produces hundreds of adversarial suffix candidates per query. Near-100% ASR
when collection phase yields sufficient training data.

Reference: arxiv.org/abs/2404.07921
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vauban import _ops as ops
from vauban._forward import force_eval, get_transformer
from vauban.softprompt._amplecgc_collect import collect_gcg_suffixes
from vauban.softprompt._amplecgc_train import train_amplecgc_generator
from vauban.softprompt._encoding import _pre_encode_prompts
from vauban.softprompt._gcg_objective import (
    _build_gcg_shared_state,
    _evaluate_candidate_loss,
)
from vauban.softprompt._generation import _evaluate_attack
from vauban.softprompt._runtime import (
    _compute_accessibility_score,
    _encode_targets,
)
from vauban.softprompt._search import _compute_per_prompt_losses
from vauban.types import (
    CausalLM,
    EnvironmentConfig,
    SoftPromptConfig,
    SoftPromptResult,
    Tokenizer,
)

if TYPE_CHECKING:
    from vauban._array import Array


def _amplecgc_attack(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompts: list[str],
    config: SoftPromptConfig,
    direction: Array | None,
    ref_model: CausalLM | None = None,
    all_prompt_ids_override: list[Array] | None = None,
    transfer_models: list[tuple[str, CausalLM, Tokenizer]] | None = None,
    infix_map: dict[int, int] | None = None,
    environment_config: EnvironmentConfig | None = None,
) -> SoftPromptResult:
    """AmpleGCG: generator-based adversarial suffix search.

    Phase 1: Collect suffixes via GCG restarts.
    Phase 2: Train a generator MLP on collected suffixes.
    Phase 3: Sample candidates from the generator and select the best.

    Args:
        model: The causal language model.
        tokenizer: Tokenizer with encode/decode support.
        prompts: Attack prompts to optimize against.
        config: SoftPromptConfig with amplecgc_* fields.
        direction: Optional refusal direction for direction-guided mode.
        ref_model: Optional reference model for KL collision loss.
        all_prompt_ids_override: Pre-encoded prompt IDs.
        transfer_models: Optional transfer models for post-hoc scoring.
        infix_map: Per-prompt infix split positions.
        environment_config: Optional environment config for rollout scoring.
    """
    transformer = get_transformer(model)
    embed_dim = transformer.embed_tokens.weight.shape[1]

    # Phase 1: Collect suffixes via GCG
    collected = collect_gcg_suffixes(
        model, tokenizer, prompts, config, direction, ref_model,
    )

    if not collected:
        # Fallback: if collection yielded nothing, return a minimal result
        return SoftPromptResult(
            mode="amplecgc",
            success_rate=0.0,
            final_loss=float("inf"),
            loss_history=[],
            n_steps=0,
            n_tokens=config.n_tokens,
            embeddings=None,
            token_ids=[],
            token_text="",
            eval_responses=[],
            accessibility_score=0.0,
            per_prompt_losses=[],
            early_stopped=False,
        )

    vocab_size = transformer.embed_tokens.weight.shape[0]

    # Phase 2: Train generator MLP
    generator = train_amplecgc_generator(
        collected,
        embed_dim=embed_dim,
        n_tokens=config.n_tokens,
        vocab_size=vocab_size,
        hidden_dim=config.amplecgc_hidden_dim,
        train_steps=config.amplecgc_train_steps,
        learning_rate=config.amplecgc_train_lr,
    )

    # Phase 3: Generate candidates and select best
    target_ids = _encode_targets(
        tokenizer, config.target_prefixes, config.target_repeat_count,
    )
    force_eval(target_ids)

    if all_prompt_ids_override is not None:
        all_prompt_ids = all_prompt_ids_override
    else:
        effective_prompts = prompts if prompts else ["Hello"]
        all_prompt_ids = _pre_encode_prompts(
            tokenizer, effective_prompts, config.system_prompt,
        )

    objective_state = _build_gcg_shared_state(
        model, tokenizer, config, target_ids, direction,
        ref_model=ref_model, infix_map=infix_map,
    )

    # Generate candidate prompt embedding (mean of first prompt)
    first_prompt_ids = all_prompt_ids[0]
    prompt_embeds = transformer.embed_tokens(first_prompt_ids[None, :])
    force_eval(prompt_embeds)
    mean_prompt_embed = ops.mean(prompt_embeds[0], axis=0)
    force_eval(mean_prompt_embed)

    # Sample candidates from generator
    candidates = generator.sample(
        mean_prompt_embed,
        config.amplecgc_n_candidates,
        temperature=config.amplecgc_sample_temperature,
    )

    # Also include collected suffixes as candidates
    for suffix_ids, _ in collected:
        candidates.append(suffix_ids)

    # Score all candidates
    candidate_losses: list[float] = []
    for candidate in candidates:
        loss = _evaluate_candidate_loss(
            objective_state, candidate, all_prompt_ids,
        )
        candidate_losses.append(loss)

    # Select best candidate
    best_idx = candidate_losses.index(min(candidate_losses))
    best_ids = candidates[best_idx]
    best_loss = candidate_losses[best_idx]

    # Build final embeddings
    final_token_array = ops.array(best_ids)[None, :]
    final_embeds = transformer.embed_tokens(final_token_array)
    force_eval(final_embeds)

    # Compute per-prompt losses
    per_prompt_losses = _compute_per_prompt_losses(
        model, final_embeds, all_prompt_ids, target_ids,
        config.n_tokens, direction, config.direction_weight,
        config.direction_mode, objective_state.direction_layers,
        objective_state.eos_token_id, config.eos_loss_mode,
        config.eos_loss_weight,
        ref_model, config.kl_ref_weight,
        loss_mode=config.loss_mode,
        refusal_ids=objective_state.refusal_ids,
        defense_aware_weight=objective_state.defense.weight,
        sic_layer=objective_state.defense.sic_layer,
        sic_threshold=objective_state.defense.sic_threshold,
        cast_layers=objective_state.defense.cast_layers,
        cast_threshold=objective_state.defense.cast_threshold,
        perplexity_weight=objective_state.perplexity_weight,
        token_position=objective_state.token_position,
        suffix_token_ids=final_token_array,
    )

    accessibility_score = _compute_accessibility_score(best_loss)
    token_text = tokenizer.decode(best_ids)

    success_rate, responses = _evaluate_attack(
        model, tokenizer, prompts, final_embeds, config,
    )

    # Loss history: collection losses + generation losses
    loss_history = [loss for _, loss in collected] + candidate_losses

    return SoftPromptResult(
        mode="amplecgc",
        success_rate=success_rate,
        final_loss=best_loss,
        loss_history=loss_history,
        n_steps=len(collected) + len(candidates),
        n_tokens=config.n_tokens,
        embeddings=None,
        token_ids=best_ids,
        token_text=token_text,
        eval_responses=responses,
        accessibility_score=accessibility_score,
        per_prompt_losses=per_prompt_losses,
        early_stopped=False,
    )
