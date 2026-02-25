"""Direction transfer testing — test a direction from model A on model B."""

import mlx.core as mx

from vauban.measure._activations import _collect_activations
from vauban.measure._direction import _best_direction, _cosine_separation
from vauban.types import CausalLM, DirectionTransferResult, Tokenizer


def check_direction_transfer(
    transfer_model: CausalLM,
    tokenizer: Tokenizer,
    direction: mx.array,
    harmful_prompts: list[str],
    harmless_prompts: list[str],
    model_id: str,
    clip_quantile: float = 0.0,
) -> DirectionTransferResult:
    """Test whether a refusal direction from one model separates on another.

    Collects activations on the target model, projects onto the transferred
    direction, and compares with the target's native best direction.

    Args:
        transfer_model: The target model to test on.
        tokenizer: Tokenizer with chat template support.
        direction: The direction vector from the source model.
        harmful_prompts: Harmful prompts for activation collection.
        harmless_prompts: Harmless prompts for activation collection.
        model_id: Identifier for the target model (for reporting).
        clip_quantile: Winsorization quantile for activation collection.

    Returns:
        DirectionTransferResult with separation metrics.
    """
    harmful_acts = _collect_activations(
        transfer_model, tokenizer, harmful_prompts, clip_quantile,
    )
    harmless_acts = _collect_activations(
        transfer_model, tokenizer, harmless_prompts, clip_quantile,
    )

    num_layers = len(harmful_acts)

    # Check dimension compatibility
    target_d_model = int(harmful_acts[0].shape[0])
    source_d_model = int(direction.shape[0])
    if target_d_model != source_d_model:
        msg = (
            f"Direction dimension mismatch: source has {source_d_model},"
            f" target model {model_id!r} has {target_d_model}"
        )
        raise ValueError(msg)

    # Compute per-layer cosine separation using the transferred direction
    per_layer_cosines: list[float] = []
    for i in range(num_layers):
        cos_sep = _cosine_separation(harmful_acts[i], harmless_acts[i], direction)
        mx.eval(cos_sep)
        per_layer_cosines.append(float(cos_sep.item()))

    transferred_separation = max(per_layer_cosines) if per_layer_cosines else 0.0

    # Compute the target model's native best direction separation
    _, _, native_cosines = _best_direction(harmful_acts, harmless_acts)
    best_native_separation = max(native_cosines) if native_cosines else 0.0

    # Transfer efficiency: how well the source direction works relative to native
    transfer_efficiency = (
        transferred_separation / best_native_separation
        if best_native_separation > 0.0
        else 0.0
    )

    return DirectionTransferResult(
        model_id=model_id,
        cosine_separation=transferred_separation,
        best_native_separation=best_native_separation,
        transfer_efficiency=transfer_efficiency,
        per_layer_cosines=per_layer_cosines,
    )
