"""EGD (Exponentiated Gradient Descent) on the probability simplex."""

import random

from vauban import _ops as ops
from vauban._array import Array
from vauban._forward import force_eval
from vauban.softprompt._attack_init import prepare_attack
from vauban.softprompt._constraints import _build_vocab_mask
from vauban.softprompt._gcg_objective import _compute_average_objective_loss
from vauban.softprompt._generation import _evaluate_attack
from vauban.softprompt._runtime import (
    _build_peaked_probs,
    _compute_accessibility_score,
    _compute_learning_rate,
    _compute_temperature,
    _resolve_init_ids,
    _score_transfer_loss,
)
from vauban.softprompt._search import (
    _compute_per_prompt_losses,
    _sample_prompt_ids,
    _select_prompt_ids,
    _select_worst_k_prompt_ids,
    _split_into_batches,
)
from vauban.types import (
    CausalLM,
    EnvironmentConfig,
    SoftPromptConfig,
    SoftPromptResult,
    Tokenizer,
)


def _egd_attack(
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
    """EGD (Exponentiated Gradient Descent) on the probability simplex.

    Relaxes discrete GCG into continuous optimization over per-position
    token distributions. Each position maintains a probability vector
    over the vocabulary; soft embeddings are computed as weighted sums
    of the embedding matrix rows. After optimization, tokens are
    extracted via argmax.

    Reference: arxiv.org/abs/2508.14853

    Args:
        model: The causal language model.
        tokenizer: Tokenizer with encode/decode support.
        prompts: Attack prompts to optimize against.
        config: Soft prompt configuration.
        direction: Optional refusal direction for direction-guided mode.
    """
    init = prepare_attack(
        model, tokenizer, prompts, config, direction,
        ref_model=ref_model,
        all_prompt_ids_override=all_prompt_ids_override,
        transfer_models=transfer_models,
        infix_map=infix_map,
    )
    transformer = init.transformer
    vocab_size = init.vocab_size
    embed_matrix = init.embed_matrix
    target_ids = init.target_ids
    all_prompt_ids = init.all_prompt_ids
    objective_state = init.objective_state
    transfer_data = init.transfer_data
    direction_layers_set = objective_state.direction_layers
    refusal_ids = objective_state.refusal_ids

    # Pre-compute vocab mask
    vocab_mask = _build_vocab_mask(
        tokenizer, vocab_size, config.token_constraint,
        embed_matrix=init.embed_matrix,
    )

    # Seed RNG for reproducible random init
    if config.seed is not None:
        random.seed(config.seed)
        ops.random.seed(config.seed)

    # Initialize p on the simplex: warm-start or random
    init_ids = _resolve_init_ids(
        config.init_tokens, config.n_tokens, vocab_mask, vocab_size,
    )
    p = _build_peaked_probs(init_ids, vocab_size)
    # Apply vocab mask to initial distribution
    if vocab_mask is not None:
        force_eval(vocab_mask)
        p = p * vocab_mask
        row_sums = ops.sum(p, axis=-1, keepdims=True)
        p = p / (row_sums + 1e-30)
    force_eval(p)

    loss_history: list[float] = []
    best_loss = float("inf")
    best_p = p
    steps_without_improvement = 0
    actual_steps = 0
    early_stopped = False

    for step in range(config.n_steps):
        actual_steps = step + 1

        # Compute current soft embeddings for worst-k selection
        current_soft_embeds = (p @ embed_matrix)[None, :]

        # Select prompts for this step
        if config.prompt_strategy == "worst_k":
            selected_ids = _select_worst_k_prompt_ids(
                model, current_soft_embeds, all_prompt_ids, target_ids,
                config.n_tokens, config.worst_k,
                direction, config.direction_weight,
                config.direction_mode, direction_layers_set,
                objective_state.eos_token_id,
                config.eos_loss_mode, config.eos_loss_weight,
                ref_model, config.kl_ref_weight,
                loss_mode=config.loss_mode, refusal_ids=refusal_ids,
                defense_aware_weight=objective_state.defense.weight,
                sic_layer=objective_state.defense.sic_layer,
                sic_threshold=objective_state.defense.sic_threshold,
                cast_layers=objective_state.defense.cast_layers,
                cast_threshold=objective_state.defense.cast_threshold,
                perplexity_weight=objective_state.perplexity_weight,
                token_position=objective_state.token_position,
            )
        elif config.prompt_strategy == "sample":
            selected_ids = _sample_prompt_ids(
                all_prompt_ids, config.worst_k,
            )
        else:
            selected_ids = _select_prompt_ids(
                all_prompt_ids, step, config.prompt_strategy,
            )

        # Gradient accumulation over mini-batches
        batches = _split_into_batches(selected_ids, config.grad_accum_steps)
        total_loss = 0.0
        accum_grad = ops.zeros_like(p)

        for batch in batches:
            def loss_fn(
                probs: Array,
                _sel: list[Array] = batch,
                _ew: float = config.entropy_weight,
            ) -> Array:
                soft_embeds = (probs @ embed_matrix)[None, :]
                # For perplexity: use argmax of probs as token IDs
                # (gradient flows through soft_embeds, not argmax)
                _suf_ids: Array | None = None
                if objective_state.perplexity_weight > 0.0:
                    _suf_ids = ops.argmax(probs, axis=-1)[None, :]
                obj_loss = _compute_average_objective_loss(
                    objective_state,
                    soft_embeds,
                    _sel,
                    suffix_token_ids=_suf_ids,
                )
                if _ew > 0.0:
                    entropy = -ops.sum(
                        probs * ops.log(probs + 1e-30), axis=-1,
                    )
                    obj_loss = obj_loss - _ew * ops.mean(entropy)
                return obj_loss

            batch_loss, batch_grad = ops.value_and_grad(loss_fn)(p)
            force_eval(batch_loss, batch_grad)
            total_loss += float(batch_loss.item())
            accum_grad = accum_grad + batch_grad

        current_loss = total_loss / len(batches)
        grad = accum_grad / len(batches)
        loss_history.append(current_loss)

        # Track best
        if current_loss < best_loss:
            best_loss = current_loss
            best_p = p
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1

        # Early stopping
        if config.patience > 0 and steps_without_improvement >= config.patience:
            early_stopped = True
            break

        # Compute LR for this step
        lr = _compute_learning_rate(
            config.learning_rate, step, config.n_steps, config.lr_schedule,
        )

        # EGD update: p = p * exp(-lr * grad), then row-normalize
        p = p * ops.exp(-lr * grad)
        # Row-normalize to simplex
        row_sums = ops.sum(p, axis=-1, keepdims=True)
        p = p / (row_sums + 1e-30)

        # Apply vocab mask after update
        if vocab_mask is not None:
            p = p * vocab_mask
            row_sums = ops.sum(p, axis=-1, keepdims=True)
            p = p / (row_sums + 1e-30)

        # Temperature sharpening: p = softmax(log(p) / temperature)
        temp = _compute_temperature(
            config.egd_temperature, step, config.n_steps,
            config.temperature_schedule,
        )
        if temp != 1.0:
            log_p = ops.log(p + 1e-30)
            p = ops.softmax(log_p / temp, axis=-1)

        force_eval(p)

    # Use best p for evaluation
    p = best_p

    # Extract token IDs via argmax
    token_ids_array = ops.argmax(p, axis=-1)
    force_eval(token_ids_array)
    raw_ids = token_ids_array.tolist()
    token_ids: list[int] = (
        [int(raw_ids)]
        if not isinstance(raw_ids, list)
        else [int(t) for t in raw_ids]
    )

    # Build final embeddings from best tokens
    final_token_array = ops.array(token_ids)[None, :]
    final_embeds = transformer.embed_tokens(final_token_array)
    force_eval(final_embeds)

    # Compute per-prompt losses and accessibility score
    per_prompt_losses = _compute_per_prompt_losses(
        model, final_embeds, all_prompt_ids, target_ids,
        config.n_tokens, direction, config.direction_weight,
        config.direction_mode, direction_layers_set,
        objective_state.eos_token_id, config.eos_loss_mode, config.eos_loss_weight,
        ref_model, config.kl_ref_weight,
        loss_mode=config.loss_mode, refusal_ids=refusal_ids,
        defense_aware_weight=objective_state.defense.weight,
        sic_layer=objective_state.defense.sic_layer,
        sic_threshold=objective_state.defense.sic_threshold,
        cast_layers=objective_state.defense.cast_layers,
        cast_threshold=objective_state.defense.cast_threshold,
        perplexity_weight=objective_state.perplexity_weight,
        token_position=objective_state.token_position,
        suffix_token_ids=final_token_array,
    )
    final_loss = best_loss

    token_text = tokenizer.decode(token_ids)

    # Post-hoc transfer scoring: EGD optimises over the continuous simplex,
    # so transfer loss cannot steer gradient steps (unlike GCG re-ranking).
    # It is added here to surface transferability in the accessibility score.
    if transfer_data:
        transfer_avg = _score_transfer_loss(token_text, transfer_data)
        final_loss += config.transfer_loss_weight * transfer_avg

    # Post-hoc environment rollout scoring (same pattern as transfer)
    if environment_config is not None:
        from vauban.environment import run_agent_loop

        env_result = run_agent_loop(
            model, tokenizer, environment_config, token_text,
        )
        final_loss -= env_result.reward * 10.0

    accessibility_score = _compute_accessibility_score(final_loss)

    success_rate, responses = _evaluate_attack(
        model, tokenizer, prompts, final_embeds, config,
    )

    return SoftPromptResult(
        mode="egd",
        success_rate=success_rate,
        final_loss=final_loss,
        loss_history=loss_history,
        n_steps=actual_steps,
        n_tokens=config.n_tokens,
        embeddings=None,
        token_ids=token_ids,
        token_text=token_text,
        eval_responses=responses,
        accessibility_score=accessibility_score,
        per_prompt_losses=per_prompt_losses,
        early_stopped=early_stopped,
    )
