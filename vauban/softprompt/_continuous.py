"""Continuous soft prompt optimization via Adam."""

from vauban import _ops as ops
from vauban._array import Array
from vauban._forward import force_eval
from vauban.softprompt._attack_init import prepare_attack
from vauban.softprompt._gcg_objective import _compute_average_objective_loss
from vauban.softprompt._generation import _evaluate_attack
from vauban.softprompt._runtime import (
    _compute_accessibility_score,
    _compute_embed_regularization,
    _compute_learning_rate,
)
from vauban.softprompt._search import (
    _compute_per_prompt_losses,
    _sample_prompt_ids,
    _select_prompt_ids,
    _select_worst_k_prompt_ids,
    _split_into_batches,
)
from vauban.types import CausalLM, SoftPromptConfig, SoftPromptResult, Tokenizer


def _continuous_attack(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompts: list[str],
    config: SoftPromptConfig,
    direction: Array | None,
    ref_model: CausalLM | None = None,
    all_prompt_ids_override: list[Array] | None = None,
    init_embeddings: Array | None = None,
) -> SoftPromptResult:
    """Continuous soft prompt optimization via Adam.

    Optimizes learnable embedding vectors prepended to the prompt
    to minimize cross-entropy loss on target prefix tokens.
    Supports multi-prompt optimization, cosine LR schedule,
    embedding norm regularization, and early stopping.

    Args:
        init_embeddings: Optional warm-start embeddings from a previous
            optimization round (e.g. LARGO reflection). When provided,
            these are used instead of random initialization, with small
            Gaussian perturbation scaled by ``config.init_scale``.
    """
    init = prepare_attack(
        model, tokenizer, prompts, config, direction,
        ref_model=ref_model,
        all_prompt_ids_override=all_prompt_ids_override,
        perplexity_weight_override=0.0,
    )
    d_model = init.d_model
    embed_matrix = init.embed_matrix
    target_ids = init.target_ids
    all_prompt_ids = init.all_prompt_ids
    objective_state = init.objective_state
    direction_layers_set = objective_state.direction_layers
    refusal_ids = objective_state.refusal_ids

    if init_embeddings is not None:
        # Warm-start: perturb previous embeddings slightly
        noise = ops.random.normal(init_embeddings.shape) * config.init_scale
        soft_embeds = init_embeddings + noise
    else:
        soft_embeds = (
            ops.random.normal((1, config.n_tokens, d_model)) * config.init_scale
        )
    force_eval(soft_embeds)

    # Adam state: manual tracking since we're optimizing a bare array
    m = ops.zeros_like(soft_embeds)
    v = ops.zeros_like(soft_embeds)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    loss_history: list[float] = []
    best_loss = float("inf")
    best_embeds = soft_embeds
    steps_without_improvement = 0
    actual_steps = 0
    early_stopped = False

    for step in range(config.n_steps):
        actual_steps = step + 1

        # Select prompts for this step
        if config.prompt_strategy == "worst_k":
            selected_ids = _select_worst_k_prompt_ids(
                model, soft_embeds, all_prompt_ids, target_ids,
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
                perplexity_weight=config.perplexity_weight,
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
        accum_grad = ops.zeros_like(soft_embeds)

        for batch in batches:
            def loss_fn(
                embeds: Array,
                _sel: list[Array] = batch,
            ) -> Array:
                avg = _compute_average_objective_loss(
                    objective_state,
                    embeds,
                    _sel,
                )
                if config.embed_reg_weight > 0.0:
                    avg = avg + _compute_embed_regularization(
                        embeds, embed_matrix, config.embed_reg_weight,
                    )
                return avg

            batch_loss, batch_grad = ops.value_and_grad(loss_fn)(soft_embeds)
            force_eval(batch_loss, batch_grad)
            total_loss += float(batch_loss.item())
            accum_grad = accum_grad + batch_grad

        current_loss = total_loss / len(batches)
        grad = accum_grad / len(batches)
        loss_history.append(current_loss)

        # Track best
        if current_loss < best_loss:
            best_loss = current_loss
            best_embeds = soft_embeds
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

        # Manual Adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        m_hat = m / (1 - beta1 ** (step + 1))
        v_hat = v / (1 - beta2 ** (step + 1))
        update = lr * m_hat / (ops.sqrt(v_hat) + eps)
        soft_embeds = soft_embeds - update
        force_eval(soft_embeds, m, v)

    # Use best embeddings for evaluation
    soft_embeds = best_embeds

    # Compute per-prompt losses and accessibility score
    per_prompt_losses = _compute_per_prompt_losses(
        model, soft_embeds, all_prompt_ids, target_ids,
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
        perplexity_weight=config.perplexity_weight,
        token_position=objective_state.token_position,
    )
    final_loss = loss_history[-1] if loss_history else 0.0
    accessibility_score = _compute_accessibility_score(final_loss)

    # Evaluate final embeddings
    success_rate, responses = _evaluate_attack(
        model, tokenizer, prompts, soft_embeds, config,
    )

    return SoftPromptResult(
        mode="continuous",
        success_rate=success_rate,
        final_loss=final_loss,
        loss_history=loss_history,
        n_steps=actual_steps,
        n_tokens=config.n_tokens,
        embeddings=soft_embeds,
        token_ids=None,
        token_text=None,
        eval_responses=responses,
        accessibility_score=accessibility_score,
        per_prompt_losses=per_prompt_losses,
        early_stopped=early_stopped,
    )
