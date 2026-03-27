# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Sparse Autoencoder (SAE) for feature decomposition.

Trains per-layer sparse autoencoders on transformer activations to
decompose the residual stream into interpretable features.

Architecture::

    encode: x → ReLU(W_enc @ (x - b_dec) + b_enc)     # d_model → d_sae
    decode: codes → W_dec @ codes + b_dec               # d_sae → d_model
    loss:   MSE(x, x_hat) + l1_coeff * mean(|codes|)

Cross-lens integration: ``feature_direction_alignment()`` computes cosine
similarity between each decoder column and a refusal direction.
"""

from pathlib import Path

from vauban import _ops as ops
from vauban._array import Array
from vauban._forward import force_eval
from vauban.measure._activations import _collect_per_prompt_activations
from vauban.types import CausalLM, FeaturesResult, SAELayerResult, Tokenizer

# ---------------------------------------------------------------------------
# SparseAutoencoder
# ---------------------------------------------------------------------------


class SparseAutoencoder:
    """Sparse autoencoder with tied decoder bias.

    Attributes:
        d_model: Input/output dimension.
        d_sae: Number of SAE features (dictionary size).
    """

    def __init__(self, d_model: int, d_sae: int) -> None:
        """Initialize SAE with Xavier-scaled weights.

        Args:
            d_model: Model hidden dimension.
            d_sae: Dictionary size (number of features).
        """
        self.d_model = d_model
        self.d_sae = d_sae

        enc_scale = (2.0 / (d_model + d_sae)) ** 0.5
        self.w_enc = ops.random.normal((d_model, d_sae)) * enc_scale
        self.b_enc = ops.zeros((d_sae,))
        self.w_dec = ops.random.normal((d_sae, d_model)) * enc_scale
        self.b_dec = ops.zeros((d_model,))

        force_eval(self.w_enc, self.b_enc, self.w_dec, self.b_dec)

    def encode(self, x: Array) -> Array:
        """Encode input to sparse feature activations.

        Args:
            x: Input activations, shape (..., d_model).

        Returns:
            Feature codes, shape (..., d_sae). Non-negative (ReLU).
        """
        return ops.maximum(
            ops.matmul(x - self.b_dec, self.w_enc) + self.b_enc,
            ops.array(0.0),
        )

    def decode(self, codes: Array) -> Array:
        """Decode feature activations back to model space.

        Args:
            codes: Feature codes, shape (..., d_sae).

        Returns:
            Reconstructed activations, shape (..., d_model).
        """
        return ops.matmul(codes, self.w_dec) + self.b_dec

    def forward(self, x: Array) -> tuple[Array, Array]:
        """Full encode-decode pass.

        Args:
            x: Input activations, shape (..., d_model).

        Returns:
            Tuple of (reconstructed, codes).
        """
        codes = self.encode(x)
        reconstructed = self.decode(codes)
        return reconstructed, codes

    def parameters(self) -> list[Array]:
        """Return all trainable parameters as a flat list."""
        return [self.w_enc, self.b_enc, self.w_dec, self.b_dec]

    def set_parameters(self, params: list[Array]) -> None:
        """Set all trainable parameters from a flat list."""
        self.w_enc, self.b_enc, self.w_dec, self.b_dec = params


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def _sae_loss(
    params: list[Array],
    sae: SparseAutoencoder,
    activations: Array,
    l1_coeff: float,
) -> Array:
    """Compute SAE loss: MSE reconstruction + L1 sparsity.

    Args:
        params: Flat parameter list (for value_and_grad).
        sae: SparseAutoencoder instance.
        activations: Batch of activations, shape (n, d_model).
        l1_coeff: L1 sparsity coefficient.

    Returns:
        Scalar loss value.
    """
    sae.set_parameters(params)
    reconstructed, codes = sae.forward(activations)
    mse = ops.mean(ops.sum((activations - reconstructed) ** 2, axis=-1))
    l1 = ops.mean(ops.sum(ops.abs(codes), axis=-1))
    return mse + l1_coeff * l1


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_sae(
    sae: SparseAutoencoder,
    activations: Array,
    *,
    l1_coeff: float = 1e-3,
    n_epochs: int = 5,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    dead_feature_threshold: float = 1e-6,
) -> SAELayerResult:
    """Train a sparse autoencoder on a batch of activations.

    Uses Adam optimizer following the SVF training pattern.

    Args:
        sae: SparseAutoencoder instance to train.
        activations: Training activations, shape (n, d_model).
        l1_coeff: L1 sparsity coefficient.
        n_epochs: Number of training epochs.
        learning_rate: Adam learning rate.
        batch_size: Mini-batch size.
        dead_feature_threshold: Features with max activation below
            this threshold are considered dead.

    Returns:
        SAELayerResult with training metrics.
    """
    n_samples = activations.shape[0]
    params = sae.parameters()
    m_state = [ops.zeros_like(p) for p in params]
    v_state = [ops.zeros_like(p) for p in params]
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    loss_history: list[float] = []
    step = 0

    for _epoch in range(n_epochs):
        # Iterate over mini-batches
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch = activations[start:end]

            def loss_fn(
                p: list[Array], _b: Array = batch,
            ) -> Array:
                return _sae_loss(p, sae, _b, l1_coeff)

            loss_val, grads = ops.value_and_grad(loss_fn)(params)
            force_eval(loss_val, *grads)

            loss_history.append(float(loss_val.item()))

            # Adam update
            step += 1
            for i in range(len(params)):
                m_state[i] = beta1 * m_state[i] + (1 - beta1) * grads[i]
                v_state[i] = beta2 * v_state[i] + (1 - beta2) * grads[i] * grads[i]
                m_hat = m_state[i] / (1 - beta1 ** step)
                v_hat = v_state[i] / (1 - beta2 ** step)
                params[i] = params[i] - learning_rate * m_hat / (ops.sqrt(v_hat) + eps)

            force_eval(*params, *m_state, *v_state)
            sae.set_parameters(params)

    # Count dead features via vectorized max reduction + comparison
    n_dead = 0
    if activations.shape[0] > 0:
        codes = sae.encode(activations)
        max_activations = codes[0]
        for i in range(1, codes.shape[0]):
            max_activations = ops.maximum(max_activations, codes[i])
        force_eval(max_activations)
        n_dead = int(ops.sum(max_activations < dead_feature_threshold).item())
    else:
        n_dead = sae.d_sae  # no data → all features are dead

    return SAELayerResult(
        layer=-1,  # set by caller
        final_loss=loss_history[-1] if loss_history else 0.0,
        loss_history=loss_history,
        n_dead_features=n_dead,
        n_active_features=sae.d_sae - n_dead,
    )


def train_sae_multi_layer(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompts: list[str],
    layers: list[int],
    d_sae: int = 2048,
    *,
    l1_coeff: float = 1e-3,
    n_epochs: int = 5,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    token_position: int = -1,
    dead_feature_threshold: float = 1e-6,
    direction: Array | None = None,
    model_path: str = "",
) -> tuple[dict[int, SparseAutoencoder], FeaturesResult]:
    """Train SAEs on multiple layers and return results.

    Collects per-prompt activations, trains one SAE per layer, and
    optionally computes direction alignment.

    Args:
        model: The causal language model.
        tokenizer: Tokenizer with chat template support.
        prompts: Training prompts.
        layers: Layer indices to train SAEs on.
        d_sae: Dictionary size.
        l1_coeff: L1 sparsity coefficient.
        n_epochs: Training epochs per SAE.
        learning_rate: Adam learning rate.
        batch_size: Mini-batch size.
        token_position: Token index for activation extraction.
        dead_feature_threshold: Dead feature detection threshold.
        direction: Refusal direction for cross-lens alignment.
        model_path: Model identifier for the result metadata.

    Returns:
        Tuple of (dict mapping layer -> trained SAE, FeaturesResult).
    """
    from vauban._forward import get_transformer

    transformer = get_transformer(model)
    d_model = int(transformer.embed_tokens.weight.shape[1])

    # Collect activations
    all_acts = _collect_per_prompt_activations(
        model, tokenizer, prompts, token_position=token_position,
    )

    saes: dict[int, SparseAutoencoder] = {}
    layer_results: list[SAELayerResult] = []
    direction_alignment: list[list[float]] | None = None
    if direction is not None:
        direction_alignment = []

    for layer_idx in layers:
        activations = all_acts[layer_idx]
        sae = SparseAutoencoder(d_model, d_sae)
        result = train_sae(
            sae,
            activations,
            l1_coeff=l1_coeff,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            dead_feature_threshold=dead_feature_threshold,
        )
        # Re-create with correct layer index
        layer_results.append(SAELayerResult(
            layer=layer_idx,
            final_loss=result.final_loss,
            loss_history=result.loss_history,
            n_dead_features=result.n_dead_features,
            n_active_features=result.n_active_features,
        ))
        saes[layer_idx] = sae

        if direction is not None and direction_alignment is not None:
            alignment = feature_direction_alignment(sae, direction)
            direction_alignment.append(alignment)

    return saes, FeaturesResult(
        layers=layer_results,
        d_model=d_model,
        d_sae=d_sae,
        model_path=model_path,
        direction_alignment=direction_alignment,
    )


# ---------------------------------------------------------------------------
# Cross-lens: direction alignment
# ---------------------------------------------------------------------------


def feature_direction_alignment(
    sae: SparseAutoencoder,
    direction: Array,
) -> list[float]:
    """Compute cosine similarity between each SAE feature and a direction.

    Each row of W_dec (shape d_sae x d_model) is a learned feature
    direction. This measures how aligned each feature is with the
    refusal direction via a single vectorized matmul.

    Args:
        sae: Trained SparseAutoencoder.
        direction: Direction vector, shape (d_model,).

    Returns:
        List of cosine similarities, one per feature.
    """
    # w_dec shape: (d_sae, d_model)
    # dot products: (d_sae,)
    dots = ops.matmul(sae.w_dec, direction)
    # per-row norms: (d_sae,)
    feat_norms = ops.sqrt(ops.sum(sae.w_dec * sae.w_dec, axis=-1))
    dir_norm = ops.sqrt(ops.sum(direction * direction))
    cos_sims = dots / (feat_norms * dir_norm + 1e-10)
    force_eval(cos_sims)
    return [float(cos_sims[i].item()) for i in range(sae.d_sae)]


# ---------------------------------------------------------------------------
# Save / Load (safetensors)
# ---------------------------------------------------------------------------


def save_sae(sae: SparseAutoencoder, path: Path) -> None:
    """Save SAE parameters to a safetensors file.

    Args:
        sae: SparseAutoencoder to save.
        path: Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {
        "w_enc": sae.w_enc,
        "b_enc": sae.b_enc,
        "w_dec": sae.w_dec,
        "b_dec": sae.b_dec,
    }
    ops.save_safetensors(str(path), tensors)


def load_sae(path: Path, d_model: int, d_sae: int) -> SparseAutoencoder:
    """Load SAE parameters from a safetensors file.

    Args:
        path: Input file path.
        d_model: Model hidden dimension.
        d_sae: Dictionary size.

    Returns:
        SparseAutoencoder with loaded parameters.
    """
    tensors = ops.load(str(path))
    if not isinstance(tensors, dict):
        msg = f"Expected dict from safetensors file, got {type(tensors).__name__}"
        raise ValueError(msg)
    sae = SparseAutoencoder(d_model, d_sae)
    sae.w_enc = tensors["w_enc"]
    sae.b_enc = tensors["b_enc"]
    sae.w_dec = tensors["w_dec"]
    sae.b_dec = tensors["b_dec"]
    return sae
