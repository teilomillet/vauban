"""Soft prompt attack: continuous embedding optimization, GCG, and EGD."""

import math
import random

import mlx.core as mx
import mlx.nn as nn

from vauban.evaluate import DEFAULT_REFUSAL_PHRASES
from vauban.probe import _make_cache
from vauban.types import (
    CausalLM,
    LayerCache,
    SoftPromptConfig,
    SoftPromptResult,
    Tokenizer,
)


def softprompt_attack(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompts: list[str],
    config: SoftPromptConfig,
    direction: mx.array | None = None,
) -> SoftPromptResult:
    """Run a soft prompt attack against a model.

    Optimizes a learnable prefix in embedding space that steers generation
    away from refusal. Supports continuous (gradient-based), GCG
    (discrete token search), and EGD (exponentiated gradient descent) modes.

    Args:
        model: The causal language model to attack.
        tokenizer: Tokenizer with encode/decode support.
        prompts: Attack prompts to optimize against.
        config: Soft prompt configuration.
        direction: Optional refusal direction for direction-guided mode.
    """
    if config.seed is not None:
        mx.random.seed(config.seed)
        random.seed(config.seed)

    if config.mode == "continuous":
        return _continuous_attack(model, tokenizer, prompts, config, direction)
    if config.mode == "gcg":
        return _gcg_attack(model, tokenizer, prompts, config, direction)
    if config.mode == "egd":
        return _egd_attack(model, tokenizer, prompts, config, direction)

    msg = (
        f"Unknown soft prompt mode: {config.mode!r},"
        " must be 'continuous', 'gcg', or 'egd'"
    )
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_learning_rate(
    base_lr: float,
    step: int,
    n_steps: int,
    schedule: str,
) -> float:
    """Compute learning rate for the given step.

    Args:
        base_lr: Base learning rate.
        step: Current step (0-indexed).
        n_steps: Total number of steps.
        schedule: "constant" or "cosine".

    Returns:
        Learning rate for this step.
    """
    if schedule == "cosine" and n_steps > 1:
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * step / (n_steps - 1)))
    return base_lr


def _compute_embed_regularization(
    soft_embeds: mx.array,
    embed_matrix: mx.array,
    weight: float,
) -> mx.array:
    """L2 penalty on embedding norm deviation (Huang et al.).

    Penalizes the soft prompt embeddings when their mean norm
    differs from the mean norm of real token embeddings.

    Args:
        soft_embeds: Soft prompt embeddings, shape (1, n_tokens, d_model).
        embed_matrix: Token embedding matrix, shape (vocab_size, d_model).
        weight: Regularization weight.

    Returns:
        Scalar regularization loss.
    """
    mean_soft_norm = mx.mean(mx.linalg.norm(soft_embeds[0], axis=-1))
    mean_real_norm = mx.mean(mx.linalg.norm(embed_matrix, axis=-1))
    return weight * (mean_soft_norm - mean_real_norm) ** 2


def _pre_encode_prompts(
    tokenizer: Tokenizer,
    prompts: list[str],
) -> list[mx.array]:
    """Pre-tokenize all prompts into token ID arrays.

    Args:
        tokenizer: Tokenizer with encode and chat template support.
        prompts: List of prompt strings.

    Returns:
        List of token ID arrays, each shape (1, seq_len).
    """
    encoded: list[mx.array] = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        if not isinstance(text, str):
            msg = "apply_chat_template must return str when tokenize=False"
            raise TypeError(msg)
        ids = mx.array(tokenizer.encode(text))[None, :]
        encoded.append(ids)
    return encoded


def _select_prompt_ids(
    all_ids: list[mx.array],
    step: int,
    strategy: str,
) -> list[mx.array]:
    """Select prompt ID arrays for this optimization step.

    Args:
        all_ids: All pre-encoded prompt ID arrays.
        step: Current optimization step (0-indexed).
        strategy: "all", "cycle", or "first".

    Returns:
        Subset of prompt ID arrays to use this step.
    """
    if strategy == "first":
        return [all_ids[0]]
    if strategy == "cycle":
        idx = step % len(all_ids)
        return [all_ids[idx]]
    # "all"
    return all_ids


def _compute_accessibility_score(final_loss: float) -> float:
    """Compute accessibility score from final loss (Nordby metric).

    Args:
        final_loss: Final optimization loss value.

    Returns:
        Accessibility score in (0, 1].
    """
    return math.exp(-final_loss)


def _compute_per_prompt_losses(
    model: CausalLM,
    soft_embeds: mx.array,
    all_ids: list[mx.array],
    target_ids: mx.array,
    n_tokens: int,
    direction: mx.array | None,
    direction_weight: float,
    direction_mode: str = "last",
    direction_layers: set[int] | None = None,
) -> list[float]:
    """Compute final loss for each prompt individually.

    Args:
        model: The causal language model.
        soft_embeds: Optimized soft prompt embeddings.
        all_ids: Pre-encoded prompt token ID arrays.
        target_ids: Target token IDs.
        n_tokens: Number of soft prompt tokens.
        direction: Optional refusal direction vector.
        direction_weight: Weight for direction auxiliary loss.
        direction_mode: Direction penalty mode ("last", "raid", "all_positions").
        direction_layers: Layer indices for direction penalty (None = all).

    Returns:
        List of loss values, one per prompt.
    """
    losses: list[float] = []
    for prompt_ids in all_ids:
        loss = _compute_loss(
            model, soft_embeds, prompt_ids, target_ids,
            n_tokens, direction, direction_weight,
            direction_mode, direction_layers,
        )
        mx.eval(loss)
        losses.append(float(loss.item()))
    return losses


def _encode_refusal_tokens(tokenizer: Tokenizer) -> mx.array:
    """Encode refusal phrases into a deduplicated set of first-token IDs.

    Each refusal phrase is tokenized and only the first token is kept,
    as the first token is the strongest signal for refusal detection.

    Args:
        tokenizer: Tokenizer with encode support.

    Returns:
        1-D array of deduplicated refusal token IDs.
    """
    seen: set[int] = set()
    ids: list[int] = []
    for phrase in DEFAULT_REFUSAL_PHRASES:
        tokens = tokenizer.encode(phrase)
        if tokens and tokens[0] not in seen:
            seen.add(tokens[0])
            ids.append(tokens[0])
    return mx.array(ids)


# ---------------------------------------------------------------------------
# Forward pass and loss
# ---------------------------------------------------------------------------


def _forward_with_prefix(
    model: CausalLM,
    soft_embeds: mx.array,
    prompt_token_ids: mx.array,
) -> mx.array:
    """Forward pass with soft prefix prepended to prompt embeddings.

    Args:
        model: The causal language model.
        soft_embeds: Learnable prefix embeddings, shape (1, n_tokens, d_model).
        prompt_token_ids: Token IDs for the prompt, shape (1, seq_len).

    Returns:
        Logits tensor, shape (1, n_tokens + seq_len, vocab_size).
    """
    transformer = model.model
    prompt_embeds = transformer.embed_tokens(prompt_token_ids)
    h = mx.concatenate([soft_embeds, prompt_embeds], axis=1)

    mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
    mask = mask.astype(h.dtype)

    for layer in transformer.layers:
        h = layer(h, mask)

    h = transformer.norm(h)

    if hasattr(model, "lm_head"):
        lm_head: nn.Module = model.lm_head  # type: ignore[attr-defined]
        logits: mx.array = lm_head(h)
    else:
        logits = transformer.embed_tokens.as_linear(h)

    return logits


def _lm_head(model: CausalLM, h: mx.array) -> mx.array:
    """Apply the language model head to hidden states."""
    if hasattr(model, "lm_head"):
        lm_head: nn.Module = model.lm_head  # type: ignore[attr-defined]
        return lm_head(h)
    return model.model.embed_tokens.as_linear(h)


def _compute_loss(
    model: CausalLM,
    soft_embeds: mx.array,
    prompt_token_ids: mx.array,
    target_ids: mx.array,
    n_tokens: int,
    direction: mx.array | None,
    direction_weight: float,
    direction_mode: str = "last",
    direction_layers: set[int] | None = None,
) -> mx.array:
    """Compute cross-entropy loss with teacher forcing.

    Feeds [soft_prefix | prompt | target] through the model and takes
    the cross-entropy loss at positions where the model should predict
    each target token.

    Supports RAID multi-layer direction penalty (Schwinn et al., 2024)
    and all-positions direction penalty modes.

    Args:
        model: The causal language model.
        soft_embeds: Learnable prefix, shape (1, n_tokens, d_model).
        prompt_token_ids: Prompt token IDs, shape (1, seq_len).
        target_ids: Target token IDs to predict, shape (n_target,).
        n_tokens: Number of soft prompt tokens.
        direction: Optional refusal direction vector.
        direction_weight: Weight for direction auxiliary loss.
        direction_mode: "last" (original), "raid" (per-layer at last prompt
            position), or "all_positions" (mean over all positions per layer).
        direction_layers: Set of layer indices for RAID/all_positions penalty.
            None means all layers.

    Returns:
        Scalar loss value.
    """
    transformer = model.model
    prompt_embeds = transformer.embed_tokens(prompt_token_ids)
    target_embeds = transformer.embed_tokens(target_ids[None, :])

    # Teacher forcing: model sees prefix + prompt + target tokens
    h = mx.concatenate([soft_embeds, prompt_embeds, target_embeds], axis=1)

    mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
    mask = mask.astype(h.dtype)

    n_prompt = prompt_token_ids.shape[1]

    # Per-layer direction penalty accumulation (RAID / all_positions)
    direction_penalty = mx.array(0.0)
    n_penalty_layers = 0

    for i, layer in enumerate(transformer.layers):
        h = layer(h, mask)
        if (
            direction is not None
            and direction_weight > 0.0
            and direction_mode != "last"
            and (direction_layers is None or i in direction_layers)
        ):
            if direction_mode == "raid":
                # Project at last prompt position (before targets)
                last_pos = n_tokens + n_prompt - 1
                proj = mx.sum(h[:, last_pos, :] * direction)
            else:  # "all_positions"
                proj = mx.mean(mx.sum(h * direction, axis=-1))
            direction_penalty = direction_penalty + proj
            n_penalty_layers += 1

    h = transformer.norm(h)
    logits = _lm_head(model, h)

    # Logits at position i predict token i+1.
    n_target = target_ids.shape[0]
    start = n_tokens + n_prompt - 1
    target_logits = logits[:, start : start + n_target, :]

    ce_loss = nn.losses.cross_entropy(
        target_logits.reshape(-1, target_logits.shape[-1]),
        target_ids,
        reduction="mean",
    )

    if direction is not None and direction_weight > 0.0:
        if direction_mode == "last":
            # Original single-position behavior
            last_prompt_pos = n_tokens + n_prompt - 1
            last_hidden = h[:, last_prompt_pos, :]
            proj = mx.sum(last_hidden * direction)
            ce_loss = ce_loss + direction_weight * proj
        elif n_penalty_layers > 0:
            ce_loss = ce_loss + direction_weight * (
                direction_penalty / n_penalty_layers
            )

    return ce_loss


def _compute_untargeted_loss(
    model: CausalLM,
    soft_embeds: mx.array,
    prompt_token_ids: mx.array,
    n_tokens: int,
    refusal_ids: mx.array,
    direction: mx.array | None,
    direction_weight: float,
    direction_mode: str = "last",
    direction_layers: set[int] | None = None,
) -> mx.array:
    """Compute untargeted jailbreak loss (UJA, Deng et al. 2024).

    Forward pass with [soft_prefix | prompt] (no target tokens). Gets logits
    at last position, computes softmax, penalizes sum of refusal token
    probabilities: loss = -log(1 - sum(softmax(logits)[refusal_ids]) + eps).

    Args:
        model: The causal language model.
        soft_embeds: Learnable prefix, shape (1, n_tokens, d_model).
        prompt_token_ids: Prompt token IDs, shape (1, seq_len).
        n_tokens: Number of soft prompt tokens.
        refusal_ids: Token IDs of refusal tokens, shape (n_refusal,).
        direction: Optional refusal direction vector.
        direction_weight: Weight for direction auxiliary loss.
        direction_mode: Direction penalty mode.
        direction_layers: Layer indices for direction penalty.

    Returns:
        Scalar loss value.
    """
    transformer = model.model
    prompt_embeds = transformer.embed_tokens(prompt_token_ids)
    h = mx.concatenate([soft_embeds, prompt_embeds], axis=1)

    mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
    mask = mask.astype(h.dtype)

    n_prompt = prompt_token_ids.shape[1]

    # Per-layer direction penalty accumulation
    direction_penalty = mx.array(0.0)
    n_penalty_layers = 0

    for i, layer in enumerate(transformer.layers):
        h = layer(h, mask)
        if (
            direction is not None
            and direction_weight > 0.0
            and direction_mode != "last"
            and (direction_layers is None or i in direction_layers)
        ):
            if direction_mode == "raid":
                last_pos = n_tokens + n_prompt - 1
                proj = mx.sum(h[:, last_pos, :] * direction)
            else:  # "all_positions"
                proj = mx.mean(mx.sum(h * direction, axis=-1))
            direction_penalty = direction_penalty + proj
            n_penalty_layers += 1

    h = transformer.norm(h)
    logits = _lm_head(model, h)

    # Logits at the last position predict the first generated token
    last_logits = logits[:, -1, :]  # (1, vocab_size)
    probs = mx.softmax(last_logits, axis=-1)  # (1, vocab_size)

    # Sum probability mass on refusal tokens
    refusal_probs = probs[0, refusal_ids]  # (n_refusal,)
    refusal_sum = mx.sum(refusal_probs)

    # Minimize: -log(1 - P(refusal) + eps)
    eps = 1e-8
    loss = -mx.log(1.0 - refusal_sum + eps)

    # Direction penalty
    if direction is not None and direction_weight > 0.0:
        if direction_mode == "last":
            last_prompt_pos = n_tokens + n_prompt - 1
            last_hidden = h[:, last_prompt_pos, :]
            proj = mx.sum(last_hidden * direction)
            loss = loss + direction_weight * proj
        elif n_penalty_layers > 0:
            loss = loss + direction_weight * (
                direction_penalty / n_penalty_layers
            )

    return loss


def _encode_targets(
    tokenizer: Tokenizer,
    target_prefixes: list[str],
) -> mx.array:
    """Encode target prefix strings into a flat token ID array.

    Args:
        tokenizer: Tokenizer with encode support.
        target_prefixes: Target strings to encode.

    Returns:
        1-D array of token IDs.
    """
    all_ids: list[int] = []
    for prefix in target_prefixes:
        ids = tokenizer.encode(prefix)
        all_ids.extend(ids)
    return mx.array(all_ids)


# ---------------------------------------------------------------------------
# KV-cached generation
# ---------------------------------------------------------------------------


def _prefill_with_cache(
    model: CausalLM,
    soft_embeds: mx.array,
    prompt_token_ids: mx.array,
    cache: list[LayerCache],
) -> mx.array:
    """Forward pass populating KV cache, returns logits at last position.

    Feeds [soft_prefix | prompt] through the model layer by layer,
    updating the KV cache at each layer so subsequent decode steps
    have full context.

    Returns:
        Logits at the last position, shape (1, 1, vocab_size).
    """
    transformer = model.model
    prompt_embeds = transformer.embed_tokens(prompt_token_ids)
    h = mx.concatenate([soft_embeds, prompt_embeds], axis=1)

    mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
    mask = mask.astype(h.dtype)

    for i, layer in enumerate(transformer.layers):
        h = layer(h, mask, cache=cache[i])

    h = transformer.norm(h)
    logits = _lm_head(model, h)
    return logits[:, -1:, :]


def _decode_step(
    model: CausalLM,
    token_id: int,
    cache: list[LayerCache],
) -> mx.array:
    """Single autoregressive decode step with KV cache.

    Returns:
        Logits for the next token, shape (1, 1, vocab_size).
    """
    transformer = model.model
    h = transformer.embed_tokens(mx.array([[token_id]]))

    for i, layer in enumerate(transformer.layers):
        h = layer(h, None, cache=cache[i])

    h = transformer.norm(h)
    return _lm_head(model, h)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _evaluate_attack(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompts: list[str],
    soft_embeds: mx.array,
    config: SoftPromptConfig,
) -> tuple[float, list[str]]:
    """Evaluate attack success by generating with the optimized prefix.

    Uses KV cache for proper autoregressive generation: prefills the
    cache with [soft_prefix | prompt], then decodes token-by-token.

    Args:
        model: The causal language model.
        tokenizer: Tokenizer for encoding/decoding.
        prompts: Prompts to evaluate on.
        soft_embeds: Optimized soft prompt embeddings.
        config: Soft prompt configuration.

    Returns:
        Tuple of (success_rate, list of generated responses).
    """
    responses: list[str] = []
    successes = 0

    eos_token_id: int | None = getattr(tokenizer, "eos_token_id", None)

    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        if not isinstance(text, str):
            msg = "apply_chat_template must return str when tokenize=False"
            raise TypeError(msg)
        prompt_ids = mx.array(tokenizer.encode(text))[None, :]

        # Prefill: forward [soft_prefix | prompt] through model with cache
        cache = _make_cache(model)
        next_logits = _prefill_with_cache(model, soft_embeds, prompt_ids, cache)
        mx.eval(next_logits)

        # Decode autoregressively using the cache
        generated_ids: list[int] = []
        for _ in range(config.max_gen_tokens):
            next_token = int(mx.argmax(next_logits[:, -1, :], axis=-1).item())
            if eos_token_id is not None and next_token == eos_token_id:
                break
            generated_ids.append(next_token)
            next_logits = _decode_step(model, next_token, cache)
            mx.eval(next_logits)

        response = tokenizer.decode(generated_ids)
        responses.append(response)

        lower = response.lower()
        is_refusal = any(
            phrase.lower() in lower for phrase in DEFAULT_REFUSAL_PHRASES
        )
        if not is_refusal:
            successes += 1

    success_rate = successes / len(prompts) if prompts else 0.0
    return success_rate, responses


# ---------------------------------------------------------------------------
# Continuous attack
# ---------------------------------------------------------------------------


def _continuous_attack(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompts: list[str],
    config: SoftPromptConfig,
    direction: mx.array | None,
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
    if config.loss_mode == "untargeted":
        refusal_ids = _encode_refusal_tokens(tokenizer)
        mx.eval(refusal_ids)

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
        selected_ids = _select_prompt_ids(
            all_prompt_ids, step, config.prompt_strategy,
        )

        # Compute average loss across selected prompts
        def loss_fn(
            embeds: mx.array,
            _sel: list[mx.array] = selected_ids,
        ) -> mx.array:
            total = mx.array(0.0)
            for pid in _sel:
                if config.loss_mode == "untargeted" and refusal_ids is not None:
                    total = total + _compute_untargeted_loss(
                        model, embeds, pid,
                        config.n_tokens, refusal_ids,
                        direction, config.direction_weight,
                        config.direction_mode, direction_layers_set,
                    )
                else:
                    total = total + _compute_loss(
                        model, embeds, pid, target_ids,
                        config.n_tokens, direction, config.direction_weight,
                        config.direction_mode, direction_layers_set,
                    )
            avg = total / len(_sel)
            # Embedding norm regularization (Huang et al.)
            if config.embed_reg_weight > 0.0:
                avg = avg + _compute_embed_regularization(
                    embeds, embed_matrix, config.embed_reg_weight,
                )
            return avg

        loss_and_grad = mx.value_and_grad(loss_fn)
        loss_val, grad = loss_and_grad(soft_embeds)
        mx.eval(loss_val, grad)
        current_loss = float(loss_val.item())
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


# ---------------------------------------------------------------------------
# GCG attack
# ---------------------------------------------------------------------------


def _gcg_attack(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompts: list[str],
    config: SoftPromptConfig,
    direction: mx.array | None,
) -> SoftPromptResult:
    """GCG (Greedy Coordinate Gradient) discrete token search.

    Finds adversarial token sequences by using gradient information
    to guide a discrete search over the vocabulary. Supports
    multi-prompt optimization, multiple restarts, and early stopping.
    """
    transformer = model.model
    vocab_size = transformer.embed_tokens.weight.shape[0]
    embed_matrix = transformer.embed_tokens.weight

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
    if config.loss_mode == "untargeted":
        refusal_ids = _encode_refusal_tokens(tokenizer)
        mx.eval(refusal_ids)

    overall_best_ids: list[int] = []
    overall_best_loss = float("inf")
    all_loss_history: list[float] = []
    total_steps = 0
    early_stopped = False

    for _restart in range(config.n_restarts):
        # Initialize with random tokens
        current_ids = [
            random.randint(0, vocab_size - 1) for _ in range(config.n_tokens)
        ]
        restart_best_ids = list(current_ids)
        restart_best_loss = float("inf")
        steps_without_improvement = 0

        for step in range(config.n_steps):
            total_steps += 1

            # Select prompts for this step
            selected_ids = _select_prompt_ids(
                all_prompt_ids, step, config.prompt_strategy,
            )

            # Get embeddings for current tokens
            token_array = mx.array(current_ids)[None, :]
            soft_embeds = transformer.embed_tokens(token_array)
            mx.eval(soft_embeds)

            # Compute gradient w.r.t. embeddings (averaged across prompts)
            def loss_fn(
                embeds: mx.array,
                _sel: list[mx.array] = selected_ids,
            ) -> mx.array:
                total = mx.array(0.0)
                for pid in _sel:
                    if (
                        config.loss_mode == "untargeted"
                        and refusal_ids is not None
                    ):
                        total = total + _compute_untargeted_loss(
                            model, embeds, pid,
                            config.n_tokens, refusal_ids,
                            direction, config.direction_weight,
                            config.direction_mode, direction_layers_set,
                        )
                    else:
                        total = total + _compute_loss(
                            model, embeds, pid, target_ids,
                            config.n_tokens, direction,
                            config.direction_weight,
                            config.direction_mode, direction_layers_set,
                        )
                return total / len(_sel)

            loss_and_grad = mx.value_and_grad(loss_fn)
            loss_val, grad = loss_and_grad(soft_embeds)
            mx.eval(loss_val, grad)
            current_loss = float(loss_val.item())
            all_loss_history.append(current_loss)

            if current_loss < restart_best_loss:
                restart_best_loss = current_loss
                restart_best_ids = list(current_ids)
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1

            # Early stopping within restart
            if (
                config.patience > 0
                and steps_without_improvement >= config.patience
            ):
                early_stopped = True
                break

            # Score token candidates: -grad @ embed_matrix.T
            # grad shape: (1, n_tokens, d_model)
            scores = -mx.matmul(grad[0], embed_matrix.T)  # (n_tokens, vocab_size)
            mx.eval(scores)

            # Top-k per position
            effective_k = min(config.top_k, vocab_size)
            sorted_indices = mx.argsort(-scores, axis=-1)  # descending order
            top_indices = sorted_indices[:, :effective_k]  # (n_tokens, top_k)
            mx.eval(top_indices)

            # Generate batch_size candidates
            candidates: list[list[int]] = []
            for _ in range(config.batch_size):
                pos = random.randint(0, config.n_tokens - 1)
                tok_idx = random.randint(0, effective_k - 1)
                new_token = int(top_indices[pos, tok_idx].item())
                candidate = list(current_ids)
                candidate[pos] = new_token
                candidates.append(candidate)

            # Evaluate all candidates (averaged across selected prompts)
            candidate_losses: list[float] = []
            for candidate in candidates:
                cand_array = mx.array(candidate)[None, :]
                cand_embeds = transformer.embed_tokens(cand_array)
                cand_total = mx.array(0.0)
                for pid in selected_ids:
                    if (
                        config.loss_mode == "untargeted"
                        and refusal_ids is not None
                    ):
                        cand_total = cand_total + _compute_untargeted_loss(
                            model, cand_embeds, pid,
                            config.n_tokens, refusal_ids,
                            direction, config.direction_weight,
                            config.direction_mode, direction_layers_set,
                        )
                    else:
                        cand_total = cand_total + _compute_loss(
                            model, cand_embeds, pid, target_ids,
                            config.n_tokens, direction,
                            config.direction_weight,
                            config.direction_mode, direction_layers_set,
                        )
                cand_avg = cand_total / len(selected_ids)
                mx.eval(cand_avg)
                candidate_losses.append(float(cand_avg.item()))

            best_candidate_idx = candidate_losses.index(min(candidate_losses))
            if candidate_losses[best_candidate_idx] < current_loss:
                current_ids = candidates[best_candidate_idx]

        # Update overall best across restarts
        if restart_best_loss < overall_best_loss:
            overall_best_loss = restart_best_loss
            overall_best_ids = restart_best_ids

    # Evaluate with best tokens
    current_ids = overall_best_ids
    final_token_array = mx.array(current_ids)[None, :]
    final_embeds = transformer.embed_tokens(final_token_array)
    mx.eval(final_embeds)

    # Compute per-prompt losses and accessibility score
    per_prompt_losses = _compute_per_prompt_losses(
        model, final_embeds, all_prompt_ids, target_ids,
        config.n_tokens, direction, config.direction_weight,
        config.direction_mode, direction_layers_set,
    )
    final_loss = overall_best_loss
    accessibility_score = _compute_accessibility_score(final_loss)

    success_rate, responses = _evaluate_attack(
        model, tokenizer, prompts, final_embeds, config,
    )

    token_text = tokenizer.decode(current_ids)

    return SoftPromptResult(
        mode="gcg",
        success_rate=success_rate,
        final_loss=final_loss,
        loss_history=all_loss_history,
        n_steps=total_steps,
        n_tokens=config.n_tokens,
        embeddings=None,
        token_ids=current_ids,
        token_text=token_text,
        eval_responses=responses,
        accessibility_score=accessibility_score,
        per_prompt_losses=per_prompt_losses,
        early_stopped=early_stopped,
    )


# ---------------------------------------------------------------------------
# EGD attack
# ---------------------------------------------------------------------------


def _egd_attack(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompts: list[str],
    config: SoftPromptConfig,
    direction: mx.array | None,
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
    if config.loss_mode == "untargeted":
        refusal_ids = _encode_refusal_tokens(tokenizer)
        mx.eval(refusal_ids)

    # Initialize p uniformly on the simplex: (n_tokens, vocab_size)
    p = mx.ones((config.n_tokens, vocab_size)) / vocab_size
    mx.eval(p)

    loss_history: list[float] = []
    best_loss = float("inf")
    best_p = p
    steps_without_improvement = 0
    actual_steps = 0
    early_stopped = False

    for step in range(config.n_steps):
        actual_steps = step + 1

        # Select prompts for this step
        selected_ids = _select_prompt_ids(
            all_prompt_ids, step, config.prompt_strategy,
        )

        # Compute soft embeddings from probability distributions
        # p: (n_tokens, vocab_size), embed_matrix: (vocab_size, d_model)
        # soft_embeds: (1, n_tokens, d_model)
        def loss_fn(
            probs: mx.array,
            _sel: list[mx.array] = selected_ids,
        ) -> mx.array:
            soft_embeds = (probs @ embed_matrix)[None, :]
            total = mx.array(0.0)
            for pid in _sel:
                if (
                    config.loss_mode == "untargeted"
                    and refusal_ids is not None
                ):
                    total = total + _compute_untargeted_loss(
                        model, soft_embeds, pid,
                        config.n_tokens, refusal_ids,
                        direction, config.direction_weight,
                        config.direction_mode, direction_layers_set,
                    )
                else:
                    total = total + _compute_loss(
                        model, soft_embeds, pid, target_ids,
                        config.n_tokens, direction, config.direction_weight,
                        config.direction_mode, direction_layers_set,
                    )
            return total / len(_sel)

        loss_and_grad = mx.value_and_grad(loss_fn)
        loss_val, grad = loss_and_grad(p)
        mx.eval(loss_val, grad)
        current_loss = float(loss_val.item())
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

        # Temperature sharpening: p = softmax(log(p) / temperature)
        if config.egd_temperature != 1.0:
            log_p = mx.log(p + 1e-30)
            p = mx.softmax(log_p / config.egd_temperature, axis=-1)

        mx.eval(p)

    # Use best p for evaluation
    p = best_p

    # Extract token IDs via argmax
    token_ids_array = mx.argmax(p, axis=-1)
    mx.eval(token_ids_array)
    raw_ids = token_ids_array.tolist()
    token_ids: list[int] = (
        [int(raw_ids)]
        if not isinstance(raw_ids, list)
        else [int(t) for t in raw_ids]
    )

    # Build final embeddings from best tokens
    final_token_array = mx.array(token_ids)[None, :]
    final_embeds = transformer.embed_tokens(final_token_array)
    mx.eval(final_embeds)

    # Compute per-prompt losses and accessibility score
    per_prompt_losses = _compute_per_prompt_losses(
        model, final_embeds, all_prompt_ids, target_ids,
        config.n_tokens, direction, config.direction_weight,
        config.direction_mode, direction_layers_set,
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
