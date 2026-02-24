"""Soft prompt attack: continuous embedding optimization and GCG."""

import random

import mlx.core as mx
import mlx.nn as nn

from vauban.evaluate import DEFAULT_REFUSAL_PHRASES
from vauban.types import CausalLM, SoftPromptConfig, SoftPromptResult, Tokenizer


def softprompt_attack(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompts: list[str],
    config: SoftPromptConfig,
    direction: mx.array | None = None,
) -> SoftPromptResult:
    """Run a soft prompt attack against a model.

    Optimizes a learnable prefix in embedding space that steers generation
    away from refusal. Supports continuous (gradient-based) and GCG
    (discrete token search) modes.

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

    msg = f"Unknown soft prompt mode: {config.mode!r}, must be 'continuous' or 'gcg'"
    raise ValueError(msg)


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


def _compute_loss(
    model: CausalLM,
    soft_embeds: mx.array,
    prompt_token_ids: mx.array,
    target_ids: mx.array,
    n_tokens: int,
    direction: mx.array | None,
    direction_weight: float,
) -> mx.array:
    """Compute cross-entropy loss on target prefix tokens.

    Args:
        model: The causal language model.
        soft_embeds: Learnable prefix, shape (1, n_tokens, d_model).
        prompt_token_ids: Prompt token IDs, shape (1, seq_len).
        target_ids: Target token IDs to predict, shape (n_target,).
        n_tokens: Number of soft prompt tokens.
        direction: Optional refusal direction vector.
        direction_weight: Weight for direction auxiliary loss.

    Returns:
        Scalar loss value.
    """
    logits = _forward_with_prefix(model, soft_embeds, prompt_token_ids)

    n_target = target_ids.shape[0]
    # Target logits: positions just before each target token
    # After soft prefix + prompt, the model should predict target tokens
    # We take the last n_target logit positions before the end
    total_len = logits.shape[1]
    start = total_len - n_target
    target_logits = logits[:, start:, :]

    ce_loss = nn.losses.cross_entropy(
        target_logits.reshape(-1, target_logits.shape[-1]),
        target_ids,
        reduction="mean",
    )

    if direction is not None and direction_weight > 0.0:
        # Get hidden state before lm_head for direction projection
        transformer = model.model
        prompt_embeds = transformer.embed_tokens(prompt_token_ids)
        h = mx.concatenate([soft_embeds, prompt_embeds], axis=1)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
        mask = mask.astype(h.dtype)
        for layer in transformer.layers:
            h = layer(h, mask)
        h = transformer.norm(h)
        last_hidden = h[:, -1, :]
        proj = mx.sum(last_hidden * direction)
        # Minimize projection = push away from refusal direction
        ce_loss = ce_loss + direction_weight * proj

    return ce_loss


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


def _evaluate_attack(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompts: list[str],
    soft_embeds: mx.array,
    config: SoftPromptConfig,
) -> tuple[float, list[str]]:
    """Evaluate attack success by generating with the optimized prefix.

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

    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        if not isinstance(text, str):
            msg = "apply_chat_template must return str when tokenize=False"
            raise TypeError(msg)
        prompt_ids = mx.array(tokenizer.encode(text))[None, :]

        logits = _forward_with_prefix(model, soft_embeds, prompt_ids)
        # Greedy decode from last position
        generated_ids: list[int] = []
        next_logits = logits[:, -1:, :]

        # Generate tokens autoregressively using the prefix context
        # For simplicity, use the initial forward pass logits for first token,
        # then continue without KV cache (short generations)
        for _ in range(config.max_gen_tokens):
            next_token = int(mx.argmax(next_logits[:, -1, :], axis=-1).item())
            generated_ids.append(next_token)
            # Simple forward for next token
            next_input = mx.array([[next_token]])
            next_embeds = model.model.embed_tokens(next_input)
            # Single-token forward through model
            h = next_embeds
            for layer_mod in model.model.layers:
                h = layer_mod(h, None)
            h = model.model.norm(h)
            if hasattr(model, "lm_head"):
                lm_head: nn.Module = model.lm_head  # type: ignore[attr-defined]
                next_logits = lm_head(h)
            else:
                next_logits = model.model.embed_tokens.as_linear(h)

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
    """
    d_model = model.model.embed_tokens.weight.shape[1]
    soft_embeds = mx.random.normal((1, config.n_tokens, d_model)) * config.init_scale
    mx.eval(soft_embeds)

    target_ids = _encode_targets(tokenizer, config.target_prefixes)
    mx.eval(target_ids)

    # Use first prompt for optimization (or average over a few)
    opt_prompt = prompts[0] if prompts else "Hello"
    messages = [{"role": "user", "content": opt_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    if not isinstance(text, str):
        msg = "apply_chat_template must return str when tokenize=False"
        raise TypeError(msg)
    prompt_ids = mx.array(tokenizer.encode(text))[None, :]

    # Adam state: manual tracking since we're optimizing a bare array
    m = mx.zeros_like(soft_embeds)
    v = mx.zeros_like(soft_embeds)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    loss_history: list[float] = []

    for step in range(config.n_steps):
        def loss_fn(embeds: mx.array) -> mx.array:
            return _compute_loss(
                model, embeds, prompt_ids, target_ids,
                config.n_tokens, direction, config.direction_weight,
            )

        loss_and_grad = mx.value_and_grad(loss_fn)
        loss_val, grad = loss_and_grad(soft_embeds)
        mx.eval(loss_val, grad)
        loss_history.append(float(loss_val.item()))

        # Manual Adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        m_hat = m / (1 - beta1 ** (step + 1))
        v_hat = v / (1 - beta2 ** (step + 1))
        update = config.learning_rate * m_hat / (mx.sqrt(v_hat) + eps)
        soft_embeds = soft_embeds - update
        mx.eval(soft_embeds, m, v)

    # Evaluate final embeddings
    success_rate, responses = _evaluate_attack(
        model, tokenizer, prompts, soft_embeds, config,
    )

    return SoftPromptResult(
        mode="continuous",
        success_rate=success_rate,
        final_loss=loss_history[-1] if loss_history else 0.0,
        loss_history=loss_history,
        n_steps=config.n_steps,
        n_tokens=config.n_tokens,
        embeddings=soft_embeds,
        token_ids=None,
        token_text=None,
        eval_responses=responses,
    )


def _gcg_attack(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompts: list[str],
    config: SoftPromptConfig,
    direction: mx.array | None,
) -> SoftPromptResult:
    """GCG (Greedy Coordinate Gradient) discrete token search.

    Finds adversarial token sequences by using gradient information
    to guide a discrete search over the vocabulary.
    """
    transformer = model.model
    vocab_size = transformer.embed_tokens.weight.shape[0]

    # Initialize with random tokens
    current_ids = [random.randint(0, vocab_size - 1) for _ in range(config.n_tokens)]

    target_ids = _encode_targets(tokenizer, config.target_prefixes)
    mx.eval(target_ids)

    # Use first prompt for optimization
    opt_prompt = prompts[0] if prompts else "Hello"
    messages = [{"role": "user", "content": opt_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    if not isinstance(text, str):
        msg = "apply_chat_template must return str when tokenize=False"
        raise TypeError(msg)
    prompt_ids = mx.array(tokenizer.encode(text))[None, :]

    embed_matrix = transformer.embed_tokens.weight  # (vocab_size, d_model)

    loss_history: list[float] = []
    best_ids = list(current_ids)
    best_loss = float("inf")

    for _step in range(config.n_steps):
        # Get embeddings for current tokens
        token_array = mx.array(current_ids)[None, :]
        soft_embeds = transformer.embed_tokens(token_array)
        mx.eval(soft_embeds)

        # Compute gradient w.r.t. embeddings
        def loss_fn(embeds: mx.array) -> mx.array:
            return _compute_loss(
                model, embeds, prompt_ids, target_ids,
                config.n_tokens, direction, config.direction_weight,
            )

        loss_and_grad = mx.value_and_grad(loss_fn)
        loss_val, grad = loss_and_grad(soft_embeds)
        mx.eval(loss_val, grad)
        current_loss = float(loss_val.item())
        loss_history.append(current_loss)

        if current_loss < best_loss:
            best_loss = current_loss
            best_ids = list(current_ids)

        # Score token candidates: -grad @ embed_matrix.T
        # grad shape: (1, n_tokens, d_model)
        scores = -mx.matmul(grad[0], embed_matrix.T)  # (n_tokens, vocab_size)
        mx.eval(scores)

        # Top-k per position
        effective_k = min(config.top_k, vocab_size)
        # argsort to get top-k indices per position
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

        # Evaluate all candidates, keep the best
        candidate_losses: list[float] = []
        for candidate in candidates:
            cand_array = mx.array(candidate)[None, :]
            cand_embeds = transformer.embed_tokens(cand_array)
            cand_loss = _compute_loss(
                model, cand_embeds, prompt_ids, target_ids,
                config.n_tokens, direction, config.direction_weight,
            )
            mx.eval(cand_loss)
            candidate_losses.append(float(cand_loss.item()))

        best_candidate_idx = candidate_losses.index(min(candidate_losses))
        if candidate_losses[best_candidate_idx] < current_loss:
            current_ids = candidates[best_candidate_idx]

    # Use best overall token IDs
    current_ids = best_ids

    # Evaluate with best tokens
    final_token_array = mx.array(current_ids)[None, :]
    final_embeds = transformer.embed_tokens(final_token_array)
    mx.eval(final_embeds)

    success_rate, responses = _evaluate_attack(
        model, tokenizer, prompts, final_embeds, config,
    )

    token_text = tokenizer.decode(current_ids)

    return SoftPromptResult(
        mode="gcg",
        success_rate=success_rate,
        final_loss=best_loss,
        loss_history=loss_history,
        n_steps=config.n_steps,
        n_tokens=config.n_tokens,
        embeddings=None,
        token_ids=current_ids,
        token_text=token_text,
        eval_responses=responses,
    )
