"""Latent fusion: blend harmful + benign hidden states and generate.

Implements the Latent Fusion Jailbreak (arxiv 2508.10029) — the
prompt-side dual of abliteration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vauban import _ops as ops
from vauban._forward import (
    embed_and_mask,
    encode_user_prompt,
    force_eval,
    get_transformer,
    lm_head_forward,
    make_ssm_mask,
    select_mask,
)
from vauban.types import FusionConfig, FusionGeneration, FusionResult

if TYPE_CHECKING:
    from vauban._array import Array
    from vauban.types import CausalLM, Tokenizer


def fuse_and_generate(
    model: CausalLM,
    tokenizer: Tokenizer,
    harmful_prompt: str,
    benign_prompt: str,
    config: FusionConfig,
) -> FusionGeneration:
    """Fuse hidden states of a harmful+benign pair at a target layer and generate.

    Args:
        model: The causal language model.
        tokenizer: Tokenizer with chat template support.
        harmful_prompt: The harmful prompt.
        benign_prompt: The benign prompt.
        config: Fusion configuration.

    Returns:
        A FusionGeneration with the fused output.
    """
    transformer = get_transformer(model)
    n_layers = len(transformer.layers)

    # Resolve fusion layer (-1 = middle)
    fusion_layer = config.layer
    if fusion_layer < 0:
        fusion_layer = n_layers // 2

    if fusion_layer >= n_layers:
        msg = (
            f"Fusion layer {fusion_layer} out of range"
            f" (model has {n_layers} layers)"
        )
        raise ValueError(msg)

    # Tokenize both prompts via chat template
    h_harm = _forward_to_layer(model, tokenizer, harmful_prompt, fusion_layer)
    h_benign = _forward_to_layer(model, tokenizer, benign_prompt, fusion_layer)

    # Blend: use the shorter sequence length
    min_seq = min(h_harm.shape[1], h_benign.shape[1])
    h_harm_trimmed = h_harm[:, :min_seq, :]
    h_benign_trimmed = h_benign[:, :min_seq, :]

    alpha = config.alpha
    h_fused = alpha * h_harm_trimmed + (1.0 - alpha) * h_benign_trimmed
    force_eval(h_fused)

    # Continue forward through remaining layers
    h = h_fused
    ssm_mask = make_ssm_mask(transformer, h)
    for layer in transformer.layers[fusion_layer:]:
        h = layer(h, select_mask(layer, None, ssm_mask))
    h = transformer.norm(h)
    force_eval(h)

    # Generate tokens autoregressively from the fused state
    output_tokens = _generate_from_hidden(model, h, config.n_tokens, config.temperature)
    output_text = tokenizer.decode(output_tokens)

    return FusionGeneration(
        harmful_prompt=harmful_prompt,
        benign_prompt=benign_prompt,
        output=output_text,
        layer=fusion_layer,
        alpha=config.alpha,
    )


def fuse_batch(
    model: CausalLM,
    tokenizer: Tokenizer,
    config: FusionConfig,
) -> FusionResult:
    """Run fusion for all (harmful, benign) prompt pairs.

    Pairs are formed by zip — the shorter list determines the count.

    Args:
        model: The causal language model.
        tokenizer: Tokenizer with chat template support.
        config: Fusion configuration.

    Returns:
        FusionResult with all generations.
    """
    n_pairs = min(len(config.harmful_prompts), len(config.benign_prompts))
    generations: list[FusionGeneration] = []

    for i in range(n_pairs):
        gen = fuse_and_generate(
            model,
            tokenizer,
            config.harmful_prompts[i],
            config.benign_prompts[i],
            config,
        )
        generations.append(gen)

    # Resolve effective layer for reporting
    fusion_layer = config.layer
    if fusion_layer < 0:
        fusion_layer = len(get_transformer(model).layers) // 2

    return FusionResult(
        generations=generations,
        layer=fusion_layer,
        alpha=config.alpha,
    )


def _forward_to_layer(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompt: str,
    target_layer: int,
) -> Array:
    """Forward a prompt through layers [0..target_layer) and return hidden state."""
    transformer = get_transformer(model)
    token_ids = encode_user_prompt(tokenizer, prompt)
    h, mask = embed_and_mask(transformer, token_ids)

    ssm_mask = make_ssm_mask(transformer, h)
    for layer in transformer.layers[:target_layer]:
        h = layer(h, select_mask(layer, mask, ssm_mask))

    force_eval(h)
    return h


def _generate_from_hidden(
    model: CausalLM,
    hidden: Array,
    n_tokens: int,
    temperature: float,
) -> list[int]:
    """Greedy/sampled generation from a hidden state.

    Args:
        model: CausalLM with a lm_head or equivalent.
        hidden: Final hidden states, shape (1, seq_len, d_model).
        n_tokens: Number of tokens to generate.
        temperature: Sampling temperature (0 = greedy).

    Returns:
        List of generated token IDs.
    """
    tokens: list[int] = []
    # Get logits from last position
    last_h = hidden[:, -1:, :]  # (1, 1, d_model)
    transformer = get_transformer(model)

    for _ in range(n_tokens):
        # Project to vocabulary via the LM head
        logits = lm_head_forward(model, last_h)
        logits = logits[:, -1, :]  # (1, vocab_size)

        if temperature <= 0:
            token_id = int(ops.argmax(logits, axis=-1).item())
        else:
            scaled = logits / temperature
            probs = ops.softmax(scaled, axis=-1)
            force_eval(probs)
            token_id = int(
                ops.random.categorical(ops.log(probs + 1e-10)).item(),
            )

        tokens.append(token_id)

        # Embed the new token for next step
        new_embed = transformer.embed_tokens(
            ops.array([[token_id]]),
        )
        last_h = new_embed

    return tokens
