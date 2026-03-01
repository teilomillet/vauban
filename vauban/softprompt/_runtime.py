"""Runtime helpers for soft prompt attacks."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from vauban import _ops as ops
from vauban._forward import embed_and_mask_with_prefix, force_eval, lm_head_forward
from vauban.evaluate import DEFAULT_REFUSAL_PHRASES
from vauban.softprompt._encoding import _pre_encode_prompts
from vauban.softprompt._loss import _compute_loss

if TYPE_CHECKING:
    from vauban._array import Array
    from vauban.types import CausalLM, SoftPromptConfig, Tokenizer


def _compute_learning_rate(
    base_lr: float,
    step: int,
    n_steps: int,
    schedule: str,
) -> float:
    """Compute learning rate for the given step."""
    if schedule == "cosine" and n_steps > 1:
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * step / (n_steps - 1)))
    return base_lr


def _compute_embed_regularization(
    soft_embeds: Array,
    embed_matrix: Array,
    weight: float,
) -> Array:
    """L2 penalty on embedding norm deviation."""
    mean_soft_norm = ops.mean(ops.linalg.norm(soft_embeds[0], axis=-1))
    mean_real_norm = ops.mean(ops.linalg.norm(embed_matrix, axis=-1))
    return weight * (mean_soft_norm - mean_real_norm) ** 2


def _compute_accessibility_score(final_loss: float) -> float:
    """Compute accessibility score from final loss."""
    return math.exp(-final_loss)


def _project_to_tokens(
    soft_embeds: Array,
    embed_matrix: Array,
) -> list[int]:
    """Project continuous embeddings to nearest discrete tokens."""
    scores = soft_embeds[0] @ embed_matrix.T
    token_ids_array = ops.argmax(scores, axis=-1)
    force_eval(token_ids_array)
    raw = token_ids_array.tolist()
    if not isinstance(raw, list):
        return [int(raw)]
    return [int(token) for token in raw]


def _encode_refusal_tokens(tokenizer: Tokenizer) -> Array:
    """Encode refusal phrases into a deduplicated set of first-token IDs."""
    seen: set[int] = set()
    ids: list[int] = []
    for phrase in DEFAULT_REFUSAL_PHRASES:
        tokens = tokenizer.encode(phrase)
        if tokens and tokens[0] not in seen:
            seen.add(tokens[0])
            ids.append(tokens[0])
    return ops.array(ids)


def _forward_with_prefix(
    model: CausalLM,
    soft_embeds: Array,
    prompt_token_ids: Array,
) -> Array:
    """Forward pass with soft prefix prepended to prompt embeddings."""
    transformer = model.model
    h, mask = embed_and_mask_with_prefix(transformer, soft_embeds, prompt_token_ids)
    for layer in transformer.layers:
        h = layer(h, mask)
    h = transformer.norm(h)
    return lm_head_forward(model, h)


def _encode_targets(
    tokenizer: Tokenizer,
    target_prefixes: list[str],
    repeat_count: int = 0,
) -> Array:
    """Encode target prefix strings into a flat token ID array."""
    all_ids: list[int] = []
    for prefix in target_prefixes:
        all_ids.extend(tokenizer.encode(prefix))
    if repeat_count > 0:
        all_ids = all_ids * repeat_count
    return ops.array(all_ids)


def _prepare_transfer_data(
    transfer_models: list[tuple[str, CausalLM, Tokenizer]] | None,
    config: SoftPromptConfig,
    prompts: list[str],
) -> list[tuple[CausalLM, Tokenizer, list[Array], Array]]:
    """Pre-encode prompts and targets for each transfer model."""
    if not transfer_models or config.transfer_loss_weight <= 0.0:
        return []

    effective_prompts = prompts if prompts else ["Hello"]
    result: list[tuple[CausalLM, Tokenizer, list[Array], Array]] = []
    for _name, model, tokenizer in transfer_models:
        prompt_ids = _pre_encode_prompts(
            tokenizer,
            effective_prompts,
            config.system_prompt,
        )
        target_ids = _encode_targets(
            tokenizer,
            config.target_prefixes,
            config.target_repeat_count,
        )
        force_eval(target_ids)
        result.append((model, tokenizer, prompt_ids, target_ids))
    return result


def _score_transfer_loss(
    token_text: str,
    transfer_data: list[tuple[CausalLM, Tokenizer, list[Array], Array]],
) -> float:
    """Compute mean loss across transfer models for a candidate suffix."""
    total = 0.0
    for model, tokenizer, prompts, targets in transfer_data:
        tokens = tokenizer.encode(token_text)
        n_tokens = len(tokens)
        embeds = model.model.embed_tokens(ops.array(tokens)[None, :])
        loss = ops.array(0.0)
        for prompt_ids in prompts:
            loss = loss + _compute_loss(
                model,
                embeds,
                prompt_ids,
                targets,
                n_tokens,
                None,
                0.0,
            )
        avg_loss = loss / len(prompts)
        force_eval(avg_loss)
        total += float(avg_loss.item())
    return total / len(transfer_data)
