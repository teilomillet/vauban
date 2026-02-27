"""Silhouette scoring for harmful/harmless activation separation."""

from vauban import _ops as ops
from vauban._array import Array
from vauban._forward import force_eval
from vauban.measure._activations import _collect_per_prompt_activations
from vauban.types import CausalLM, Tokenizer


def silhouette_scores(
    model: CausalLM,
    tokenizer: Tokenizer,
    harmful_prompts: list[str],
    harmless_prompts: list[str],
    clip_quantile: float = 0.0,
) -> list[float]:
    """Compute per-layer silhouette scores for harmful/harmless separation.

    The silhouette score measures how well-separated the two groups are:
    ``s = (b - a) / max(a, b)`` where *a* is the mean intra-cluster
    distance and *b* is the mean nearest-cluster distance for each sample.

    Higher scores (closer to 1.0) indicate better separation at that layer.

    Args:
        model: The causal language model.
        tokenizer: Tokenizer with chat template support.
        harmful_prompts: Prompts that typically trigger refusal.
        harmless_prompts: Benign prompts for contrast.
        clip_quantile: Winsorization quantile for activation clipping.

    Returns:
        Per-layer silhouette scores, one per layer.
    """
    harmful_pp = _collect_per_prompt_activations(
        model, tokenizer, harmful_prompts, clip_quantile,
    )
    harmless_pp = _collect_per_prompt_activations(
        model, tokenizer, harmless_prompts, clip_quantile,
    )

    num_layers = len(harmful_pp)
    scores: list[float] = []

    for i in range(num_layers):
        score = _silhouette_score_layer(harmful_pp[i], harmless_pp[i])
        scores.append(score)

    return scores


def _silhouette_score_layer(
    group_a: Array,
    group_b: Array,
) -> float:
    """Silhouette score for two groups at a single layer.

    Args:
        group_a: Activations of shape (n_a, d_model).
        group_b: Activations of shape (n_b, d_model).

    Returns:
        Mean silhouette score in [-1, 1].
    """
    n_a = group_a.shape[0]
    n_b = group_b.shape[0]

    if n_a < 2 or n_b < 2:
        return 0.0

    # Pairwise L2 distances within and between groups
    # Using ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    def _pairwise_dist(x: Array, y: Array) -> Array:
        x_sq = ops.sum(x * x, axis=1, keepdims=True)  # (n, 1)
        y_sq = ops.sum(y * y, axis=1, keepdims=True)  # (m, 1)
        cross = x @ y.T  # (n, m)
        dist = x_sq + y_sq.T - 2.0 * cross
        return ops.maximum(dist, ops.array(0.0))  # clamp negatives from float error

    dist_aa = _pairwise_dist(group_a, group_a)  # (n_a, n_a)
    dist_ab = _pairwise_dist(group_a, group_b)  # (n_a, n_b)
    dist_bb = _pairwise_dist(group_b, group_b)  # (n_b, n_b)
    dist_ba = _pairwise_dist(group_b, group_a)  # (n_b, n_a)

    force_eval(dist_aa, dist_ab, dist_bb, dist_ba)

    total_s = 0.0
    total_n = n_a + n_b

    # Silhouette for group_a samples
    for j in range(n_a):
        # Mean distance to other points in same cluster
        a_val = float(ops.sum(dist_aa[j]).item()) / (n_a - 1)
        # Mean distance to points in other cluster
        b_val = float(ops.sum(dist_ab[j]).item()) / n_b
        denom = max(a_val, b_val)
        total_s += (b_val - a_val) / denom if denom > 0 else 0.0

    # Silhouette for group_b samples
    for j in range(n_b):
        a_val = float(ops.sum(dist_bb[j]).item()) / (n_b - 1)
        b_val = float(ops.sum(dist_ba[j]).item()) / n_a
        denom = max(a_val, b_val)
        total_s += (b_val - a_val) / denom if denom > 0 else 0.0

    return total_s / total_n
