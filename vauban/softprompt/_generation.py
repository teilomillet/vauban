"""KV-cached generation and attack evaluation."""

import mlx.core as mx
import mlx.nn as nn

from vauban.evaluate import DEFAULT_REFUSAL_PHRASES
from vauban.probe import _make_cache
from vauban.softprompt._loss import _lm_head
from vauban.types import CausalLM, LayerCache, SoftPromptConfig, Tokenizer


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
        # Defensive mode: refusal = success; attack modes: non-refusal = success
        if config.loss_mode == "defensive":
            if is_refusal:
                successes += 1
        elif not is_refusal:
            successes += 1

    success_rate = successes / len(prompts) if prompts else 0.0
    return success_rate, responses
