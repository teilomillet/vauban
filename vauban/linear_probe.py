# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Linear probe: per-layer binary classifiers on activations.

Tests whether refused knowledge remains linearly decodable
(Shrivastava & Holtzman 2025).
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, cast

from vauban import _ops as ops
from vauban._forward import force_eval
from vauban.measure._activations import _collect_per_prompt_activations
from vauban.types import (
    LinearProbeConfig,
    LinearProbeLayerResult,
    LinearProbeResult,
)

if TYPE_CHECKING:
    from vauban._array import Array
    from vauban.types import CausalLM, Tokenizer


def train_probe(
    model: CausalLM,
    tokenizer: Tokenizer,
    harmful_prompts: list[str],
    harmless_prompts: list[str],
    config: LinearProbeConfig,
) -> LinearProbeResult:
    """Train per-layer logistic classifiers on harmful vs harmless activations.

    Args:
        model: The causal language model.
        tokenizer: Tokenizer with chat template support.
        harmful_prompts: Prompts labelled as harmful (label=1).
        harmless_prompts: Prompts labelled as harmless (label=0).
        config: Linear probe configuration.

    Returns:
        LinearProbeResult with per-layer accuracy and loss curves.
    """
    if not harmful_prompts:
        msg = "harmful_prompts must be non-empty"
        raise ValueError(msg)
    if not harmless_prompts:
        msg = "harmless_prompts must be non-empty"
        raise ValueError(msg)

    # Collect per-prompt activations
    all_prompts = harmful_prompts + harmless_prompts
    per_layer = _collect_per_prompt_activations(
        model,
        tokenizer,
        all_prompts,
        token_position=config.token_position,
    )

    n_harmful = len(harmful_prompts)
    n_total = len(all_prompts)
    labels = ops.array(
        [1.0] * n_harmful + [0.0] * (n_total - n_harmful),
    )
    labels = ops.to_device_like(labels, per_layer[0])
    force_eval(labels)

    d_model = int(per_layer[0].shape[1])
    layer_results: list[LinearProbeLayerResult] = []

    for layer_idx in config.layers:
        if layer_idx >= len(per_layer):
            msg = (
                f"Layer {layer_idx} out of range"
                f" (model has {len(per_layer)} layers)"
            )
            raise ValueError(msg)

        activations = per_layer[layer_idx]  # (n_total, d_model)
        result = _train_single_probe(
            activations,
            labels,
            n_epochs=config.n_epochs,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            regularization=config.regularization,
        )
        layer_results.append(
            LinearProbeLayerResult(
                layer=layer_idx,
                accuracy=result[0],
                loss=result[1],
                loss_history=result[2],
            ),
        )

    model_path = getattr(model, "_model_path", "unknown")
    if not isinstance(model_path, str):
        model_path = "unknown"

    return LinearProbeResult(
        layers=layer_results,
        d_model=d_model,
        model_path=model_path,
    )


def _train_single_probe(
    activations: Array,
    labels: Array,
    *,
    n_epochs: int,
    learning_rate: float,
    batch_size: int,
    regularization: float,
) -> tuple[float, float, list[float]]:
    """Train a single logistic probe via SGD with L2 regularization.

    Returns (accuracy, final_loss, loss_history).
    """
    n_samples = int(activations.shape[0])
    d_model = int(activations.shape[1])

    # Initialize weights and bias
    w = ops.to_device_like(ops.zeros((d_model,)), activations)
    b = ops.to_device_like(ops.array(0.0), activations)
    force_eval(w, b)

    # Adam state
    m_w = ops.zeros_like(w)
    v_w = ops.zeros_like(w)
    m_b = ops.to_device_like(ops.array(0.0), activations)
    v_b = ops.to_device_like(ops.array(0.0), activations)
    force_eval(m_w, v_w, m_b, v_b)

    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    t = 0

    loss_history: list[float] = []

    for _epoch in range(n_epochs):
        indices = list(range(n_samples))
        random.shuffle(indices)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]
            x_batch = activations[batch_idx]  # (bs, d_model)
            y_batch = labels[batch_idx]       # (bs,)

            # Forward: logits = x @ w + b
            logits = ops.matmul(x_batch, w) + b  # (bs,)

            # Sigmoid
            probs = _sigmoid(logits)  # (bs,)

            # BCE loss + L2
            bce = _bce_loss(probs, y_batch)
            l2 = regularization * ops.sum(w * w)
            loss = bce + l2
            force_eval(loss)
            epoch_loss += float(loss.item())
            n_batches += 1

            # Gradients (manual)
            diff = probs - y_batch  # (bs,)
            bs = float(end - start)
            grad_w = ops.matmul(diff, x_batch) / bs + 2.0 * regularization * w
            grad_b = ops.sum(diff) / bs
            force_eval(grad_w, grad_b)

            # Adam update
            t += 1
            m_w = beta1 * m_w + (1.0 - beta1) * grad_w
            v_w = beta2 * v_w + (1.0 - beta2) * grad_w * grad_w
            m_b = beta1 * m_b + (1.0 - beta1) * grad_b
            v_b = beta2 * v_b + (1.0 - beta2) * grad_b * grad_b

            m_w_hat = m_w / (1.0 - beta1**t)
            v_w_hat = v_w / (1.0 - beta2**t)
            m_b_hat = m_b / (1.0 - beta1**t)
            v_b_hat = v_b / (1.0 - beta2**t)

            w = w - learning_rate * m_w_hat / (ops.sqrt(v_w_hat) + eps)
            b = b - learning_rate * m_b_hat / (ops.sqrt(v_b_hat) + eps)
            force_eval(w, b, m_w, v_w, m_b, v_b)

        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)

    # Compute final accuracy
    all_logits = ops.matmul(activations, w) + b
    preds = (all_logits > 0.0).astype(ops.float32)
    match = cast("Array", preds == labels)
    correct = ops.sum(match.astype(ops.float32))
    force_eval(correct)
    accuracy = float(correct.item()) / n_samples

    return accuracy, loss_history[-1] if loss_history else 0.0, loss_history


def _sigmoid(x: Array) -> Array:
    """Numerically stable sigmoid: 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + ops.exp(-x))


def _bce_loss(probs: Array, targets: Array) -> Array:
    """Binary cross-entropy loss (mean reduction)."""
    eps = 1e-7
    clamped = ops.clip(probs, eps, 1.0 - eps)
    loss = -(targets * ops.log(clamped) + (1.0 - targets) * ops.log(1.0 - clamped))
    return ops.mean(loss)
