"""COLD-Attack: Constrained decoding via Langevin dynamics.

Energy-based optimization over logit space with Gaussian noise for
exploration. Produces more natural adversarial prompts than GCG/EGD
by combining gradient descent with stochastic Langevin noise and
fluency energy constraints.

Reference: arxiv.org/abs/2402.08679
"""

import random

from vauban import _ops as ops
from vauban._array import Array
from vauban._forward import force_eval
from vauban.softprompt._attack_init import prepare_attack
from vauban.softprompt._constraints import _build_vocab_mask
from vauban.softprompt._gcg_objective import _compute_average_objective_loss
from vauban.softprompt._generation import _evaluate_attack
from vauban.softprompt._runtime import (
    _build_one_hot,
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


def _cold_attack(
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
    """COLD-Attack: Langevin dynamics in logit space.

    Maintains per-position logit vectors z (unconstrained) and converts
    to probabilities via softmax(z / temperature). Updates follow:

        z <- z - lr * grad(energy) + sqrt(2 * lr) * noise_scale * N(0, 1)

    The energy function combines the shared attack objective with a
    fluency regularizer (perplexity_weight).

    Reference: arxiv.org/abs/2402.08679

    Args:
        model: The causal language model.
        tokenizer: Tokenizer with encode/decode support.
        prompts: Attack prompts to optimize against.
        config: Soft prompt configuration with mode="cold".
        direction: Optional refusal direction for direction-guided mode.
        ref_model: Optional reference model for KL collision loss.
        all_prompt_ids_override: Pre-encoded prompt IDs (injection/infix).
        transfer_models: Optional transfer models for post-hoc scoring.
        infix_map: Per-prompt infix split positions.
        environment_config: Optional environment config for rollout scoring.
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
    )

    # COLD parameters
    temperature = config.cold_temperature
    noise_scale = config.cold_noise_scale

    # Seed RNG for reproducible random init
    if config.seed is not None:
        random.seed(config.seed)
        ops.random.seed(config.seed)

    # Initialize logits z: warm-start or random (uniform softmax is degenerate)
    init_ids = _resolve_init_ids(
        config.init_tokens, config.n_tokens, vocab_mask, vocab_size,
    )
    z = _build_one_hot(init_ids, vocab_size) * 5.0

    # Apply vocab mask in logit space: set masked positions to large negative
    mask_penalty: Array | None = None
    if vocab_mask is not None:
        force_eval(vocab_mask)
        neg_inf = ops.array(-1e9)
        # Invert mask: 0 -> -inf, 1 -> 0
        mask_penalty = (1.0 - vocab_mask) * neg_inf
        z = z + mask_penalty

    force_eval(z)

    loss_history: list[float] = []
    best_loss = float("inf")
    best_z = z
    steps_without_improvement = 0
    actual_steps = 0
    early_stopped = False

    for step in range(config.n_steps):
        actual_steps = step + 1

        # Compute annealed temperature for this step
        temp = _compute_temperature(
            temperature, step, config.n_steps,
            config.temperature_schedule,
        )

        # Convert logits to probabilities via temperature-scaled softmax
        p = ops.softmax(z / temp, axis=-1)
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
        accum_grad = ops.zeros_like(z)

        for batch in batches:
            def loss_fn(
                logits: Array,
                _sel: list[Array] = batch,
                _temp: float = temp,
                _ew: float = config.entropy_weight,
            ) -> Array:
                probs = ops.softmax(logits / _temp, axis=-1)
                soft_embeds = (probs @ embed_matrix)[None, :]
                # For perplexity: use argmax of probs as token IDs
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

            batch_loss, batch_grad = ops.value_and_grad(loss_fn)(z)
            force_eval(batch_loss, batch_grad)
            total_loss += float(batch_loss.item())
            accum_grad = accum_grad + batch_grad

        current_loss = total_loss / len(batches)
        grad = accum_grad / len(batches)
        loss_history.append(current_loss)

        # Track best
        if current_loss < best_loss:
            best_loss = current_loss
            best_z = z
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

        # Langevin dynamics update: gradient descent + calibrated noise
        noise = ops.random.normal(z.shape) * ops.sqrt(ops.array(2.0 * lr))
        z = z - lr * grad + noise * noise_scale

        # Re-apply vocab mask penalty after update
        if mask_penalty is not None:
            z = z + mask_penalty

        force_eval(z)

    # Use best z for evaluation
    z = best_z

    # Extract token IDs via argmax of softmax(z / temperature)
    p = ops.softmax(z / temperature, axis=-1)
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

    # Post-hoc transfer scoring
    if transfer_data:
        transfer_avg = _score_transfer_loss(token_text, transfer_data)
        final_loss += config.transfer_loss_weight * transfer_avg

    # Post-hoc environment rollout scoring
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
        mode="cold",
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
