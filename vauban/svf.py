# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Steering Vector Fields — context-dependent boundary MLP.

Trains a differentiable boundary f(h) → scalar whose gradient gives
the steering direction at each activation. Context-dependent, multi-layer
coordinated, replaces static vectors.

Reference: Li, Li & Huang (2026) — arxiv.org/abs/2602.01654
"""

from pathlib import Path

from vauban import _ops as ops
from vauban._array import Array
from vauban._forward import force_eval
from vauban.measure._activations import _collect_per_prompt_activations
from vauban.types import CausalLM, SVFResult, Tokenizer

# ---------------------------------------------------------------------------
# SVFBoundary — lightweight MLP boundary classifier
# ---------------------------------------------------------------------------


class SVFBoundary:
    """MLP boundary f(h) → scalar with per-layer FiLM conditioning.

    Architecture:
        1. Shared projection R: (d_model,) → (projection_dim,)
        2. Per-layer FiLM: scale * projected + shift
        3. Hidden layer + ReLU
        4. Output layer → scalar

    The gradient ∇_h f(h) gives the context-dependent steering direction.
    """

    def __init__(
        self,
        d_model: int,
        projection_dim: int,
        hidden_dim: int,
        n_layers: int,
    ) -> None:
        """Initialize SVF boundary parameters.

        Args:
            d_model: Model hidden dimension.
            projection_dim: Shared projection dimension.
            hidden_dim: Hidden layer dimension.
            n_layers: Number of transformer layers (for FiLM conditioning).
        """
        self.d_model = d_model
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Shared projection R: d_model → projection_dim
        scale = (2.0 / (d_model + projection_dim)) ** 0.5
        self.projection = ops.random.normal(
            (d_model, projection_dim),
        ) * scale

        # Per-layer FiLM parameters
        self.film_scale = ops.ones((n_layers, projection_dim))
        self.film_shift = ops.zeros((n_layers, projection_dim))

        # Hidden layer
        h_scale = (2.0 / (projection_dim + hidden_dim)) ** 0.5
        self.hidden_weight = ops.random.normal(
            (projection_dim, hidden_dim),
        ) * h_scale
        self.hidden_bias = ops.zeros((hidden_dim,))

        # Output layer
        o_scale = (2.0 / (hidden_dim + 1)) ** 0.5
        self.output_weight = ops.random.normal(
            (hidden_dim, 1),
        ) * o_scale
        self.output_bias = ops.zeros((1,))

        force_eval(
            self.projection, self.film_scale, self.film_shift,
            self.hidden_weight, self.hidden_bias,
            self.output_weight, self.output_bias,
        )

    def forward(self, h: Array, layer_idx: int) -> Array:
        """Compute boundary score f(h) for a hidden state at a given layer.

        Args:
            h: Hidden state, shape (d_model,).
            layer_idx: Transformer layer index for FiLM conditioning.

        Returns:
            Scalar boundary score (positive = target class, negative = opposite).
        """
        projection = ops.to_device_like(self.projection, h)
        film_scale = ops.to_device_like(self.film_scale, h)
        film_shift = ops.to_device_like(self.film_shift, h)
        hidden_weight = ops.to_device_like(self.hidden_weight, h)
        hidden_bias = ops.to_device_like(self.hidden_bias, h)
        output_weight = ops.to_device_like(self.output_weight, h)
        output_bias = ops.to_device_like(self.output_bias, h)

        # Project into shared space
        projected = ops.matmul(h, projection)  # (projection_dim,)

        # FiLM conditioning
        conditioned = (
            film_scale[layer_idx] * projected + film_shift[layer_idx]
        )

        # Hidden layer with ReLU
        hidden = ops.matmul(conditioned, hidden_weight) + hidden_bias
        hidden = ops.maximum(hidden, ops.array(0.0))

        # Output
        score = ops.matmul(hidden, output_weight) + output_bias
        return score[0]  # scalar

    def forward_batch(self, h: Array, layer_idx: int) -> Array:
        """Compute boundary scores for a batch of hidden states.

        Args:
            h: Hidden states, shape (n, d_model).
            layer_idx: Transformer layer index for FiLM conditioning.

        Returns:
            Scores, shape (n,). Positive = target class, negative = opposite.
        """
        projection = ops.to_device_like(self.projection, h)
        film_scale = ops.to_device_like(self.film_scale, h)
        film_shift = ops.to_device_like(self.film_shift, h)
        hidden_weight = ops.to_device_like(self.hidden_weight, h)
        hidden_bias = ops.to_device_like(self.hidden_bias, h)
        output_weight = ops.to_device_like(self.output_weight, h)
        output_bias = ops.to_device_like(self.output_bias, h)

        projected = ops.matmul(h, projection)  # (n, projection_dim)
        conditioned = (
            film_scale[layer_idx] * projected + film_shift[layer_idx]
        )
        hidden = ops.matmul(conditioned, hidden_weight) + hidden_bias
        hidden = ops.maximum(hidden, ops.array(0.0))
        scores = ops.matmul(hidden, output_weight) + output_bias
        return scores[:, 0]  # (n,)

    def parameters(self) -> list[Array]:
        """Return all trainable parameters as a flat list."""
        return [
            self.projection,
            self.film_scale,
            self.film_shift,
            self.hidden_weight,
            self.hidden_bias,
            self.output_weight,
            self.output_bias,
        ]

    def set_parameters(self, params: list[Array]) -> None:
        """Set all trainable parameters from a flat list."""
        (
            self.projection,
            self.film_scale,
            self.film_shift,
            self.hidden_weight,
            self.hidden_bias,
            self.output_weight,
            self.output_bias,
        ) = params


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _svf_loss(
    params: list[Array],
    boundary: SVFBoundary,
    target_acts: list[Array],
    opposite_acts: list[Array],
    layers: list[int],
) -> Array:
    """Compute binary cross-entropy loss for SVF boundary training.

    Uses batched forward passes per layer for efficiency.

    Args:
        params: Flat parameter list (for value_and_grad).
        boundary: SVFBoundary instance (parameters replaced from params).
        target_acts: Per-layer activations for target class, each (n, d_model).
        opposite_acts: Per-layer activations for opposite class, each (n, d_model).
        layers: Layer indices to train on.

    Returns:
        Scalar BCE loss.
    """
    boundary.set_parameters(params)
    eps = 1e-7
    total_loss: Array | None = None
    count = 0

    for layer_idx in layers:
        # Target class (label=1): want f(h) > 0 → sigmoid(f(h)) → 1
        n_target = target_acts[layer_idx].shape[0]
        if n_target > 0:
            scores = boundary.forward_batch(target_acts[layer_idx], layer_idx)
            sigmoid_scores = 1.0 / (1.0 + ops.exp(-scores))
            layer_loss = -ops.sum(ops.log(sigmoid_scores + eps))
            total_loss = (
                layer_loss
                if total_loss is None
                else total_loss + layer_loss
            )
            count += n_target

        # Opposite class (label=0): want f(h) < 0 → sigmoid(f(h)) → 0
        n_opposite = opposite_acts[layer_idx].shape[0]
        if n_opposite > 0:
            scores = boundary.forward_batch(opposite_acts[layer_idx], layer_idx)
            sigmoid_scores = 1.0 / (1.0 + ops.exp(-scores))
            layer_loss = -ops.sum(ops.log(1.0 - sigmoid_scores + eps))
            total_loss = (
                layer_loss
                if total_loss is None
                else total_loss + layer_loss
            )
            count += n_opposite

    if total_loss is None:
        return ops.array(0.0)
    return total_loss / count


def train_svf_boundary(
    model: CausalLM,
    tokenizer: Tokenizer,
    target_prompts: list[str],
    opposite_prompts: list[str],
    d_model: int,
    n_layers: int,
    projection_dim: int = 16,
    hidden_dim: int = 64,
    n_epochs: int = 10,
    learning_rate: float = 1e-3,
    layers: list[int] | None = None,
) -> tuple[SVFBoundary, SVFResult]:
    """Train an SVF boundary from target/opposite prompt sets.

    Collects per-prompt activations, trains the boundary MLP with BCE
    loss via Adam optimization.

    Args:
        model: The causal language model.
        tokenizer: Tokenizer with chat template support.
        target_prompts: Prompts for the target class (e.g., harmful).
        opposite_prompts: Prompts for the opposite class (e.g., harmless).
        d_model: Model hidden dimension.
        n_layers: Number of transformer layers.
        projection_dim: Shared projection dimension.
        hidden_dim: Hidden layer dimension.
        n_epochs: Number of training epochs.
        learning_rate: Adam learning rate.
        layers: Layer indices to train on (None = all layers).

    Returns:
        Tuple of (trained SVFBoundary, SVFResult).
    """
    train_layers = layers if layers is not None else list(range(n_layers))

    # Collect per-prompt activations
    target_acts = _collect_per_prompt_activations(
        model, tokenizer, target_prompts,
    )
    opposite_acts = _collect_per_prompt_activations(
        model, tokenizer, opposite_prompts,
    )

    # Initialize boundary
    boundary = SVFBoundary(d_model, projection_dim, hidden_dim, n_layers)
    reference = _first_non_empty_activation(target_acts, opposite_acts)
    if reference is not None:
        params_on_device = [
            ops.to_device_like(p, reference)
            for p in boundary.parameters()
        ]
        boundary.set_parameters(params_on_device)

    # Adam state
    params = boundary.parameters()
    m_state = [ops.zeros_like(p) for p in params]
    v_state = [ops.zeros_like(p) for p in params]
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    loss_history: list[float] = []

    for epoch in range(n_epochs):
        def loss_fn(p: list[Array]) -> Array:
            return _svf_loss(p, boundary, target_acts, opposite_acts, train_layers)

        loss_val, grads = ops.value_and_grad(loss_fn)(params)
        force_eval(loss_val, *grads)

        loss_float = float(loss_val.item())
        loss_history.append(loss_float)

        # Adam update
        t = epoch + 1
        for i in range(len(params)):
            m_state[i] = beta1 * m_state[i] + (1 - beta1) * grads[i]
            v_state[i] = beta2 * v_state[i] + (1 - beta2) * grads[i] * grads[i]
            m_hat = m_state[i] / (1 - beta1 ** t)
            v_hat = v_state[i] / (1 - beta2 ** t)
            params[i] = params[i] - learning_rate * m_hat / (ops.sqrt(v_hat) + eps)

        force_eval(*params, *m_state, *v_state)
        boundary.set_parameters(params)

    # Compute final accuracy
    correct = 0
    total = 0
    per_layer_separation: list[float] = []

    for layer_idx in train_layers:
        layer_correct = 0
        layer_total = 0
        for i in range(target_acts[layer_idx].shape[0]):
            h = target_acts[layer_idx][i]
            score = boundary.forward(h, layer_idx)
            force_eval(score)
            if float(score.item()) > 0:
                correct += 1
                layer_correct += 1
            total += 1
            layer_total += 1

        for i in range(opposite_acts[layer_idx].shape[0]):
            h = opposite_acts[layer_idx][i]
            score = boundary.forward(h, layer_idx)
            force_eval(score)
            if float(score.item()) <= 0:
                correct += 1
                layer_correct += 1
            total += 1
            layer_total += 1

        layer_acc = layer_correct / layer_total if layer_total > 0 else 0.0
        per_layer_separation.append(layer_acc)

    final_accuracy = correct / total if total > 0 else 0.0

    result = SVFResult(
        train_loss_history=loss_history,
        final_accuracy=final_accuracy,
        per_layer_separation=per_layer_separation,
        projection_dim=projection_dim,
        hidden_dim=hidden_dim,
        n_layers_trained=len(train_layers),
        model_path="",
    )

    return boundary, result


def _first_non_empty_activation(
    target_acts: list[Array],
    opposite_acts: list[Array],
) -> Array | None:
    """Return a reference activation for device placement."""
    for activations in target_acts + opposite_acts:
        if activations.shape[0] > 0:
            return activations
    return None


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------


def save_svf_boundary(boundary: SVFBoundary, path: str | Path) -> None:
    """Save SVF boundary parameters to safetensors format.

    Args:
        boundary: Trained SVF boundary.
        path: Output file path (.safetensors).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, Array] = {
        "projection": boundary.projection,
        "film_scale": boundary.film_scale,
        "film_shift": boundary.film_shift,
        "hidden_weight": boundary.hidden_weight,
        "hidden_bias": boundary.hidden_bias,
        "output_weight": boundary.output_weight,
        "output_bias": boundary.output_bias,
    }
    ops.save_safetensors(str(path), tensors)


def load_svf_boundary(path: str | Path) -> SVFBoundary:
    """Load SVF boundary parameters from safetensors format.

    Dimensions are inferred from stored tensor shapes — no hardcoded
    hyperparameters needed.

    Args:
        path: Input file path (.safetensors).

    Returns:
        SVFBoundary with loaded parameters.
    """
    tensors = ops.load(str(path))
    if not isinstance(tensors, dict):
        msg = f"Expected dict from safetensors file, got {type(tensors).__name__}"
        raise ValueError(msg)
    projection = tensors["projection"]
    d_model = projection.shape[0]
    projection_dim = projection.shape[1]
    hidden_dim = tensors["hidden_weight"].shape[1]
    n_layers = tensors["film_scale"].shape[0]

    boundary = SVFBoundary(d_model, projection_dim, hidden_dim, n_layers)
    boundary.projection = projection
    boundary.film_scale = tensors["film_scale"]
    boundary.film_shift = tensors["film_shift"]
    boundary.hidden_weight = tensors["hidden_weight"]
    boundary.hidden_bias = tensors["hidden_bias"]
    boundary.output_weight = tensors["output_weight"]
    boundary.output_bias = tensors["output_bias"]
    return boundary


# ---------------------------------------------------------------------------
# Gradient computation for runtime steering
# ---------------------------------------------------------------------------


def svf_gradient(
    boundary: SVFBoundary,
    h: Array,
    layer_idx: int,
) -> tuple[float, Array]:
    """Compute boundary score and gradient direction at a hidden state.

    The gradient ∇_h f(h) gives the context-dependent steering direction,
    normalized to a unit vector.

    Args:
        boundary: Trained SVF boundary.
        h: Hidden state, shape (d_model,).
        layer_idx: Transformer layer index.

    Returns:
        Tuple of (score, normalized_gradient).
        - score: boundary value f(h) (positive = target side)
        - normalized_gradient: unit vector pointing toward target side
    """
    def score_fn(x: Array) -> Array:
        return boundary.forward(x, layer_idx)

    score_val, grad_val = ops.value_and_grad(score_fn)(h)
    force_eval(score_val, grad_val)

    score = float(score_val.item())

    # Normalize gradient to unit vector
    grad_norm = ops.linalg.norm(grad_val)
    force_eval(grad_norm)
    norm_float = float(grad_norm.item())

    if norm_float > 1e-10:
        normalized = grad_val / grad_norm
    else:
        normalized = ops.to_device_like(ops.zeros((boundary.d_model,)), h)
    force_eval(normalized)

    return score, normalized
