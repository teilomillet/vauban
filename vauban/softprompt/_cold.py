"""COLD-Attack: Constrained decoding via Langevin dynamics.

Energy-based optimization over logit space with Gaussian noise for
exploration. Produces more natural adversarial prompts than GCG/EGD
by combining gradient descent with stochastic Langevin noise and
fluency energy constraints.

Reference: arxiv.org/abs/2402.08679
"""

from vauban import _ops as ops
from vauban._array import Array
from vauban._forward import force_eval
from vauban.softprompt._constraints import _build_vocab_mask
from vauban.softprompt._encoding import _pre_encode_prompts
from vauban.softprompt._gcg_objective import (
    _build_gcg_shared_state,
    _compute_average_objective_loss,
)
from vauban.softprompt._generation import _evaluate_attack
from vauban.softprompt._runtime import (
    _compute_accessibility_score,
    _compute_learning_rate,
    _encode_targets,
    _prepare_transfer_data,
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
    transformer = model.model
    vocab_size = transformer.embed_tokens.weight.shape[0]
    embed_matrix = transformer.embed_tokens.weight

    target_ids = _encode_targets(
        tokenizer, config.target_prefixes, config.target_repeat_count,
    )
    force_eval(target_ids)

    # Pre-encode all prompts (or use override with history baked in)
    if all_prompt_ids_override is not None:
        all_prompt_ids = all_prompt_ids_override
    else:
        effective_prompts = prompts if prompts else ["Hello"]
        all_prompt_ids = _pre_encode_prompts(
            tokenizer, effective_prompts, config.system_prompt,
        )

    # Pre-encode transfer model data for post-optimization scoring
    transfer_data = _prepare_transfer_data(
        transfer_models, config, prompts,
    )

    objective_state = _build_gcg_shared_state(
        model,
        tokenizer,
        config,
        target_ids,
        direction,
        ref_model=ref_model,
        infix_map=infix_map,
    )
    direction_layers_set = objective_state.direction_layers
    refusal_ids = objective_state.refusal_ids

    # Pre-compute vocab mask
    vocab_mask = _build_vocab_mask(
        tokenizer, vocab_size, config.token_constraint,
    )

    # COLD parameters
    temperature = config.cold_temperature
    noise_scale = config.cold_noise_scale

    # Initialize logits z: warm-start or zeros (uniform softmax)
    if config.init_tokens is not None:
        # Warm-start: high logit mass on init tokens
        init = list(config.init_tokens)
        if len(init) < config.n_tokens:
            init.extend([0] * (config.n_tokens - len(init)))
        init = init[: config.n_tokens]
        warm_ids = ops.array(init)
        one_hot_raw = (
            ops.arange(vocab_size)[None, :] == warm_ids[:, None]
        )
        one_hot: Array = (
            one_hot_raw
            if isinstance(one_hot_raw, Array)
            else ops.array(one_hot_raw)
        )
        # Set warm-start tokens to high logit value
        z = one_hot.astype(ops.float32) * 5.0
    else:
        z = ops.zeros((config.n_tokens, vocab_size))

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

        # Convert logits to probabilities via temperature-scaled softmax
        p = ops.softmax(z / temperature, axis=-1)
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
            ) -> Array:
                probs = ops.softmax(logits / temperature, axis=-1)
                soft_embeds = (probs @ embed_matrix)[None, :]
                # For perplexity: use argmax of probs as token IDs
                _suf_ids: Array | None = None
                if objective_state.perplexity_weight > 0.0:
                    _suf_ids = ops.argmax(probs, axis=-1)[None, :]
                return _compute_average_objective_loss(
                    objective_state,
                    soft_embeds,
                    _sel,
                    suffix_token_ids=_suf_ids,
                )

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
