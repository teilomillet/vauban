"""EGD (Exponentiated Gradient Descent) on the probability simplex."""

import mlx.core as mx

from vauban._array import Array
from vauban._forward import force_eval
from vauban.softprompt._generation import _evaluate_attack
from vauban.softprompt._loss import (
    _compute_defensive_loss,
    _compute_loss,
    _compute_untargeted_loss,
)
from vauban.softprompt._utils import (
    _build_vocab_mask,
    _compute_accessibility_score,
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


def _egd_attack(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompts: list[str],
    config: SoftPromptConfig,
    direction: Array | None,
    ref_model: CausalLM | None = None,
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
    transformer = model.model
    vocab_size = transformer.embed_tokens.weight.shape[0]
    embed_matrix = transformer.embed_tokens.weight

    target_ids = _encode_targets(tokenizer, config.target_prefixes)
    force_eval(target_ids)

    # Pre-encode all prompts
    effective_prompts = prompts if prompts else ["Hello"]
    all_prompt_ids = _pre_encode_prompts(tokenizer, effective_prompts)

    # Pre-compute direction config
    direction_layers_set: set[int] | None = (
        set(config.direction_layers) if config.direction_layers is not None
        else None
    )
    refusal_ids: Array | None = None
    if config.loss_mode in ("untargeted", "defensive"):
        refusal_ids = _encode_refusal_tokens(tokenizer)
        force_eval(refusal_ids)

    # Pre-compute vocab mask and EOS token ID
    vocab_mask = _build_vocab_mask(
        tokenizer, vocab_size, config.token_constraint,
    )
    eos_token_id: int | None = getattr(tokenizer, "eos_token_id", None)

    # Initialize p uniformly on the simplex: (n_tokens, vocab_size)
    p = mx.ones((config.n_tokens, vocab_size)) / vocab_size
    # Apply vocab mask to initial distribution
    if vocab_mask is not None:
        force_eval(vocab_mask)
        p = p * vocab_mask
        row_sums = mx.sum(p, axis=-1, keepdims=True)
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
        accum_grad = mx.zeros_like(p)

        for batch in batches:
            def loss_fn(
                probs: Array,
                _sel: list[Array] = batch,
            ) -> Array:
                soft_embeds = (probs @ embed_matrix)[None, :]
                total = mx.array(0.0)
                for pid in _sel:
                    if (
                        config.loss_mode == "defensive"
                        and refusal_ids is not None
                    ):
                        total = total + _compute_defensive_loss(
                            model, soft_embeds, pid,
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
                            model, soft_embeds, pid,
                            config.n_tokens, refusal_ids,
                            direction, config.direction_weight,
                            config.direction_mode, direction_layers_set,
                            eos_token_id, config.eos_loss_mode,
                            config.eos_loss_weight,
                            ref_model, config.kl_ref_weight,
                        )
                    else:
                        total = total + _compute_loss(
                            model, soft_embeds, pid, target_ids,
                            config.n_tokens, direction,
                            config.direction_weight,
                            config.direction_mode, direction_layers_set,
                            eos_token_id, config.eos_loss_mode,
                            config.eos_loss_weight,
                            ref_model, config.kl_ref_weight,
                        )
                return total / len(_sel)

            batch_loss, batch_grad = mx.value_and_grad(loss_fn)(p)
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
        p = p * mx.exp(-lr * grad)
        # Row-normalize to simplex
        row_sums = mx.sum(p, axis=-1, keepdims=True)
        p = p / (row_sums + 1e-30)

        # Apply vocab mask after update
        if vocab_mask is not None:
            p = p * vocab_mask
            row_sums = mx.sum(p, axis=-1, keepdims=True)
            p = p / (row_sums + 1e-30)

        # Temperature sharpening: p = softmax(log(p) / temperature)
        if config.egd_temperature != 1.0:
            log_p = mx.log(p + 1e-30)
            p = mx.softmax(log_p / config.egd_temperature, axis=-1)

        force_eval(p)

    # Use best p for evaluation
    p = best_p

    # Extract token IDs via argmax
    token_ids_array = mx.argmax(p, axis=-1)
    force_eval(token_ids_array)
    raw_ids = token_ids_array.tolist()
    token_ids: list[int] = (
        [int(raw_ids)]
        if not isinstance(raw_ids, list)
        else [int(t) for t in raw_ids]
    )

    # Build final embeddings from best tokens
    final_token_array = mx.array(token_ids)[None, :]
    final_embeds = transformer.embed_tokens(final_token_array)
    force_eval(final_embeds)

    # Compute per-prompt losses and accessibility score
    per_prompt_losses = _compute_per_prompt_losses(
        model, final_embeds, all_prompt_ids, target_ids,
        config.n_tokens, direction, config.direction_weight,
        config.direction_mode, direction_layers_set,
        eos_token_id, config.eos_loss_mode, config.eos_loss_weight,
        ref_model, config.kl_ref_weight,
        loss_mode=config.loss_mode, refusal_ids=refusal_ids,
    )
    final_loss = best_loss
    accessibility_score = _compute_accessibility_score(final_loss)

    success_rate, responses = _evaluate_attack(
        model, tokenizer, prompts, final_embeds, config,
    )

    token_text = tokenizer.decode(token_ids)

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
