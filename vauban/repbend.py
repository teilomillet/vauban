# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""RepBend: contrastive fine-tuning to separate harmful/safe representations.

Implements the defense dual of abliteration (RepBend, arxiv 2504.01550).
Pushes harmful and safe activation centroids apart via cosine separation loss.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from vauban import _ops as ops
from vauban._forward import force_eval, get_transformer
from vauban.measure._activations import _collect_per_prompt_activations
from vauban.types import RepBendConfig, RepBendResult

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vauban._array import Array
    from vauban.types import CausalLM, Tokenizer


def repbend(
    model: CausalLM,
    tokenizer: Tokenizer,
    harmful_prompts: list[str],
    harmless_prompts: list[str],
    config: RepBendConfig,
) -> RepBendResult:
    """Run RepBend contrastive fine-tuning on target layers.

    Maximizes cosine distance between harmful and harmless activation
    centroids at specified layers.

    Args:
        model: The causal language model.
        tokenizer: Tokenizer with chat template support.
        harmful_prompts: Prompts labelled as harmful.
        harmless_prompts: Prompts labelled as harmless.
        config: RepBend configuration.

    Returns:
        RepBendResult with initial/final separation and loss history.
    """
    # Collect initial activations
    all_prompts = harmful_prompts + harmless_prompts
    n_harmful = len(harmful_prompts)

    per_layer = _collect_per_prompt_activations(
        model,
        tokenizer,
        all_prompts,
        token_position=config.token_position,
    )

    # Compute initial separation across target layers
    initial_sep = _compute_separation(per_layer, config.layers, n_harmful)

    # Train: gradient descent to maximize separation
    loss_history = _train_repbend(
        model,
        tokenizer,
        harmful_prompts,
        harmless_prompts,
        config,
    )

    # Re-collect activations after training
    per_layer_after = _collect_per_prompt_activations(
        model,
        tokenizer,
        all_prompts,
        token_position=config.token_position,
    )
    final_sep = _compute_separation(per_layer_after, config.layers, n_harmful)

    model_path = getattr(model, "_model_path", "unknown")
    if not isinstance(model_path, str):
        model_path = "unknown"

    return RepBendResult(
        initial_separation=initial_sep,
        final_separation=final_sep,
        loss_history=loss_history,
        layers=config.layers,
        model_path=model_path,
    )


def _compute_separation(
    per_layer: list[Array],
    target_layers: list[int],
    n_harmful: int,
) -> float:
    """Compute mean cosine distance between harmful/harmless centroids."""
    separations: list[float] = []

    for layer_idx in target_layers:
        if layer_idx >= len(per_layer):
            log.warning(
                "Layer %d out of range (model has %d layers) — skipping",
                layer_idx,
                len(per_layer),
            )
            continue
        acts = per_layer[layer_idx]  # (n_total, d_model)
        harmful_centroid = ops.mean(acts[:n_harmful], axis=0)
        harmless_centroid = ops.mean(acts[n_harmful:], axis=0)
        cos_sim = _cosine_similarity(harmful_centroid, harmless_centroid)
        force_eval(cos_sim)
        separations.append(1.0 - float(cos_sim.item()))

    return sum(separations) / max(len(separations), 1)


def _cosine_similarity(a: Array, b: Array) -> Array:
    """Compute cosine similarity between two vectors."""
    dot = ops.sum(a * b)
    norm_a = ops.linalg.norm(a)
    norm_b = ops.linalg.norm(b)
    return dot / (norm_a * norm_b + 1e-8)


def _train_repbend(
    model: CausalLM,
    tokenizer: Tokenizer,
    harmful_prompts: list[str],
    harmless_prompts: list[str],
    config: RepBendConfig,
) -> list[float]:
    """Run contrastive fine-tuning loop.

    For each epoch, collect activations and update target layer weights
    to maximize separation between harmful/harmless centroids.

    Returns the loss history.
    """
    n_harmful = len(harmful_prompts)
    all_prompts = harmful_prompts + harmless_prompts
    loss_history: list[float] = []

    # Get trainable parameters from target layers
    transformer = get_transformer(model)

    for _epoch in range(config.n_epochs):
        # Collect activations
        per_layer = _collect_per_prompt_activations(
            model,
            tokenizer,
            all_prompts,
            token_position=config.token_position,
        )

        epoch_loss = 0.0
        n_updates = 0

        for layer_idx in config.layers:
            if layer_idx >= len(per_layer):
                log.warning(
                    "Layer %d out of range (model has %d layers) — skipping",
                    layer_idx,
                    len(per_layer),
                )
                continue

            acts = per_layer[layer_idx]  # (n_total, d_model)
            harmful_centroid = ops.mean(acts[:n_harmful], axis=0)
            harmless_centroid = ops.mean(acts[n_harmful:], axis=0)

            # Monitoring loss: cos_sim (lower = more separated).
            # Actual optimization is via manual outer-product update below.
            cos_sim = _cosine_similarity(harmful_centroid, harmless_centroid)
            loss = config.separation_coeff * cos_sim
            force_eval(loss)
            epoch_loss += float(loss.item())
            n_updates += 1

            # Compute gradient direction: push centroids apart
            diff = harmful_centroid - harmless_centroid
            diff_norm = ops.linalg.norm(diff)
            force_eval(diff_norm)

            if float(diff_norm.item()) < 1e-8:
                continue

            direction = diff / diff_norm
            force_eval(direction)

            # Update the layer's output projection weight
            layer = transformer.layers[layer_idx]
            if not hasattr(layer, "self_attn") or not hasattr(
                layer.self_attn, "o_proj",
            ):
                log.warning(
                    "Layer %d has no self_attn.o_proj — skipping weight update",
                    layer_idx,
                )
                continue
            o_proj = layer.self_attn.o_proj
            if not hasattr(o_proj, "weight"):
                log.warning(
                    "Layer %d o_proj has no weight attribute — skipping",
                    layer_idx,
                )
                continue
            w = o_proj.weight  # (out, in)
            # Project weight update: amplify the separation direction
            update = config.learning_rate * config.separation_coeff * ops.outer(
                direction, direction,
            )
            force_eval(update)
            new_w = w + update
            force_eval(new_w)
            o_proj.weight = new_w

        avg_loss = epoch_loss / max(n_updates, 1)
        loss_history.append(avg_loss)

    return loss_history
