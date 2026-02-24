"""Continuous soft prompt optimization via Adam."""

import mlx.core as mx

from vauban.softprompt._generation import _evaluate_attack
from vauban.softprompt._loss import (
    _compute_defensive_loss,
    _compute_loss,
    _compute_untargeted_loss,
)
from vauban.softprompt._utils import (
    _compute_accessibility_score,
    _compute_embed_regularization,
    _compute_learning_rate,
    _compute_per_prompt_losses,
    _encode_refusal_tokens,
    _encode_targets,
    _pre_encode_prompts,
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
    direction: mx.array | None,
    ref_model: CausalLM | None = None,
) -> SoftPromptResult:
    """Continuous soft prompt optimization via Adam.

    Optimizes learnable embedding vectors prepended to the prompt
    to minimize cross-entropy loss on target prefix tokens.
    Supports multi-prompt optimization, cosine LR schedule,
    embedding norm regularization, and early stopping.
    """
    d_model = model.model.embed_tokens.weight.shape[1]
    embed_matrix = model.model.embed_tokens.weight

    soft_embeds = (
        mx.random.normal((1, config.n_tokens, d_model)) * config.init_scale
    )
    mx.eval(soft_embeds)

    target_ids = _encode_targets(tokenizer, config.target_prefixes)
    mx.eval(target_ids)

    # Pre-encode all prompts
    effective_prompts = prompts if prompts else ["Hello"]
    all_prompt_ids = _pre_encode_prompts(tokenizer, effective_prompts)

    # Pre-compute direction config
    direction_layers_set: set[int] | None = (
        set(config.direction_layers) if config.direction_layers is not None
        else None
    )
    refusal_ids: mx.array | None = None
    if config.loss_mode in ("untargeted", "defensive"):
        refusal_ids = _encode_refusal_tokens(tokenizer)
        mx.eval(refusal_ids)

    # Pre-compute EOS token ID
    eos_token_id: int | None = getattr(tokenizer, "eos_token_id", None)

    # Adam state: manual tracking since we're optimizing a bare array
    m = mx.zeros_like(soft_embeds)
    v = mx.zeros_like(soft_embeds)
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
                eos_token_id, config.eos_loss_mode, config.eos_loss_weight,
                ref_model, config.kl_ref_weight,
                loss_mode=config.loss_mode, refusal_ids=refusal_ids,
            )
        else:
            selected_ids = _select_prompt_ids(
                all_prompt_ids, step, config.prompt_strategy,
            )

        # Gradient accumulation over mini-batches
        batches = _split_into_batches(selected_ids, config.grad_accum_steps)
        total_loss = 0.0
        accum_grad = mx.zeros_like(soft_embeds)

        for batch in batches:
            def loss_fn(
                embeds: mx.array,
                _sel: list[mx.array] = batch,
            ) -> mx.array:
                total = mx.array(0.0)
                for pid in _sel:
                    if (
                        config.loss_mode == "defensive"
                        and refusal_ids is not None
                    ):
                        total = total + _compute_defensive_loss(
                            model, embeds, pid,
                            config.n_tokens, refusal_ids,
                            direction, config.direction_weight,
                            config.direction_mode, direction_layers_set,
                            eos_token_id, config.eos_loss_mode,
                            config.eos_loss_weight,
                            ref_model, config.kl_ref_weight,
                        )
                    elif (
                        config.loss_mode == "untargeted"
                        and refusal_ids is not None
                    ):
                        total = total + _compute_untargeted_loss(
                            model, embeds, pid,
                            config.n_tokens, refusal_ids,
                            direction, config.direction_weight,
                            config.direction_mode, direction_layers_set,
                            eos_token_id, config.eos_loss_mode,
                            config.eos_loss_weight,
                            ref_model, config.kl_ref_weight,
                        )
                    else:
                        total = total + _compute_loss(
                            model, embeds, pid, target_ids,
                            config.n_tokens, direction,
                            config.direction_weight,
                            config.direction_mode, direction_layers_set,
                            eos_token_id, config.eos_loss_mode,
                            config.eos_loss_weight,
                            ref_model, config.kl_ref_weight,
                        )
                avg = total / len(_sel)
                if config.embed_reg_weight > 0.0:
                    avg = avg + _compute_embed_regularization(
                        embeds, embed_matrix, config.embed_reg_weight,
                    )
                return avg

            batch_loss, batch_grad = mx.value_and_grad(loss_fn)(soft_embeds)
            mx.eval(batch_loss, batch_grad)
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
        update = lr * m_hat / (mx.sqrt(v_hat) + eps)
        soft_embeds = soft_embeds - update
        mx.eval(soft_embeds, m, v)

    # Use best embeddings for evaluation
    soft_embeds = best_embeds

    # Compute per-prompt losses and accessibility score
    per_prompt_losses = _compute_per_prompt_losses(
        model, soft_embeds, all_prompt_ids, target_ids,
        config.n_tokens, direction, config.direction_weight,
        config.direction_mode, direction_layers_set,
        eos_token_id, config.eos_loss_mode, config.eos_loss_weight,
        ref_model, config.kl_ref_weight,
        loss_mode=config.loss_mode, refusal_ids=refusal_ids,
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
