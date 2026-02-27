"""Core measurement functions: measure, measure_dbdi, measure_subspace."""

from typing import TYPE_CHECKING

from vauban import _ops as ops
from vauban._forward import force_eval, svd_stable
from vauban.measure._activations import (
    _collect_activations,
    _collect_per_prompt_activations,
)
from vauban.measure._direction import (
    _best_direction,
    _collect_activations_at_instruction_end,
)
from vauban.measure._layers import detect_layer_types
from vauban.types import (
    CausalLM,
    DBDIResult,
    DirectionResult,
    SubspaceResult,
    Tokenizer,
)

if TYPE_CHECKING:
    from vauban._array import Array


def measure(
    model: CausalLM,
    tokenizer: Tokenizer,
    harmful_prompts: list[str],
    harmless_prompts: list[str],
    clip_quantile: float = 0.0,
) -> DirectionResult:
    """Extract the refusal direction from a model.

    Runs harmful and harmless prompts through the model, collects
    per-layer activations at the last token position, computes the
    difference-in-means, and selects the layer with highest cosine
    separation.

    Args:
        model: The causal language model.
        tokenizer: Tokenizer with chat template support.
        harmful_prompts: Prompts that typically trigger refusal.
        harmless_prompts: Benign prompts for contrast.
        clip_quantile: Winsorization quantile for activation clipping.
            0.0 disables clipping. 0.01 clips the top/bottom 1% of
            activation magnitudes per dimension before averaging.
    """
    harmful_acts = _collect_activations(
        model, tokenizer, harmful_prompts, clip_quantile,
    )
    harmless_acts = _collect_activations(
        model, tokenizer, harmless_prompts, clip_quantile,
    )

    direction, layer_index, cosine_scores = _best_direction(
        harmful_acts, harmless_acts,
    )
    d_model = harmful_acts[0].shape[-1]
    layer_types = detect_layer_types(model)

    return DirectionResult(
        direction=direction,
        layer_index=layer_index,
        cosine_scores=cosine_scores,
        d_model=int(d_model),
        model_path="",
        layer_types=layer_types,
    )


def measure_dbdi(
    model: CausalLM,
    tokenizer: Tokenizer,
    harmful_prompts: list[str],
    harmless_prompts: list[str],
    clip_quantile: float = 0.0,
) -> DBDIResult:
    """Extract DBDI (Decomposed Behavioral Direction Intervention) directions.

    Decomposes the refusal direction into:
    - HDD (harm detection direction): extracted at instruction-final token
    - RED (refusal execution direction): extracted at sequence-final token

    Cutting only RED suppresses refusal while preserving harm awareness.

    Args:
        model: The causal language model.
        tokenizer: Tokenizer with chat template support.
        harmful_prompts: Prompts that typically trigger refusal.
        harmless_prompts: Benign prompts for contrast.
        clip_quantile: Winsorization quantile for activation clipping.
    """
    # RED: sequence-final token (standard position)
    red_harmful = _collect_activations(
        model, tokenizer, harmful_prompts, clip_quantile, token_position=-1,
    )
    red_harmless = _collect_activations(
        model, tokenizer, harmless_prompts, clip_quantile, token_position=-1,
    )

    # HDD: instruction-final token
    hdd_harmful = _collect_activations_at_instruction_end(
        model, tokenizer, harmful_prompts, clip_quantile,
    )
    hdd_harmless = _collect_activations_at_instruction_end(
        model, tokenizer, harmless_prompts, clip_quantile,
    )

    red_dir, red_layer, red_scores = _best_direction(red_harmful, red_harmless)
    hdd_dir, hdd_layer, hdd_scores = _best_direction(hdd_harmful, hdd_harmless)

    d_model = int(red_harmful[0].shape[-1])
    layer_types = detect_layer_types(model)

    return DBDIResult(
        hdd=hdd_dir,
        red=red_dir,
        hdd_layer_index=hdd_layer,
        red_layer_index=red_layer,
        hdd_cosine_scores=hdd_scores,
        red_cosine_scores=red_scores,
        d_model=d_model,
        model_path="",
        layer_types=layer_types,
    )


def measure_subspace(
    model: CausalLM,
    tokenizer: Tokenizer,
    harmful_prompts: list[str],
    harmless_prompts: list[str],
    top_k: int = 5,
    clip_quantile: float = 0.0,
) -> SubspaceResult:
    """Extract the top-k refusal subspace from a model via SVD.

    Builds a per-prompt difference matrix at each layer and computes
    the SVD to find the principal directions of the harmful-harmless
    activation difference. Picks the best layer by explained variance
    in the top-k singular values.

    Args:
        model: The causal language model.
        tokenizer: Tokenizer with chat template support.
        harmful_prompts: Prompts that typically trigger refusal.
        harmless_prompts: Benign prompts for contrast.
        top_k: Number of principal directions to extract.
        clip_quantile: Winsorization quantile for per-prompt activation
            clipping. 0.0 disables clipping.
    """
    harmful_per_prompt = _collect_per_prompt_activations(
        model, tokenizer, harmful_prompts, clip_quantile,
    )
    harmless_per_prompt = _collect_per_prompt_activations(
        model, tokenizer, harmless_prompts, clip_quantile,
    )

    num_layers = len(harmful_per_prompt)
    d_model = int(harmful_per_prompt[0].shape[-1])

    # Ensure we don't request more directions than available
    min_prompts = min(len(harmful_prompts), len(harmless_prompts))
    k = min(top_k, min_prompts, d_model)

    best_layer = 0
    best_variance = -1.0
    per_layer_bases: list[Array] = []
    per_layer_sv: list[list[float]] = []
    per_layer_ev: list[list[float]] = []

    for i in range(num_layers):
        # Build difference matrix: each row is (harmful[j] - harmless[j])
        n = min(harmful_per_prompt[i].shape[0], harmless_per_prompt[i].shape[0])
        diff_matrix = harmful_per_prompt[i][:n] - harmless_per_prompt[i][:n]

        # SVD on CPU for numerical stability
        u, s, vt = svd_stable(diff_matrix)
        force_eval(u, s, vt)

        # Top-k basis vectors (rows of Vt)
        actual_k = min(k, vt.shape[0])
        basis = vt[:actual_k]

        sv = [float(s[j].item()) for j in range(actual_k)]
        total_var = float(ops.sum(s * s).item())
        ev = [
            float((s[j] * s[j]).item()) / (total_var + 1e-10)
            for j in range(actual_k)
        ]

        per_layer_bases.append(basis)
        per_layer_sv.append(sv)
        per_layer_ev.append(ev)

        # Best layer = highest explained variance in top-k
        topk_variance = sum(ev)
        if topk_variance > best_variance:
            best_variance = topk_variance
            best_layer = i

    layer_types = detect_layer_types(model)

    return SubspaceResult(
        basis=per_layer_bases[best_layer],
        singular_values=per_layer_sv[best_layer],
        explained_variance=per_layer_ev[best_layer],
        layer_index=best_layer,
        d_model=d_model,
        model_path="",
        per_layer_bases=per_layer_bases,
        layer_types=layer_types,
    )
