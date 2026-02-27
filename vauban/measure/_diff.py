"""Weight-diff measurement between base and aligned models.

Extracts safety directions by SVD of the weight difference
``W_aligned - W_base`` for ``o_proj`` and ``down_proj`` at each layer.
"""

from vauban import _ops as ops
from vauban._array import Array
from vauban._forward import force_eval, svd_stable
from vauban.types import CausalLM, DiffResult


def measure_diff(
    base_model: CausalLM,
    aligned_model: CausalLM,
    top_k: int = 5,
    source_model_id: str = "",
    target_model_id: str = "",
) -> DiffResult:
    """Extract safety directions from weight differences via SVD.

    For each layer, computes ``W_aligned - W_base`` for ``o_proj.weight``
    and ``down_proj.weight``, runs SVD on each diff independently, then
    selects the top-k left singular vectors (ranked by singular value)
    across both projections as safety directions.

    The best layer is selected by highest explained variance in the top-k
    singular values.

    Args:
        base_model: The base (pre-alignment) model.
        aligned_model: The aligned (instruction-tuned) model.
        top_k: Number of singular directions to keep.
        source_model_id: Identifier for the base model.
        target_model_id: Identifier for the aligned model.
    """
    base_layers = base_model.model.layers
    aligned_layers = aligned_model.model.layers
    n_layers = len(base_layers)

    per_layer_bases: list[Array] = []
    per_layer_singular_values: list[list[float]] = []
    per_layer_explained: list[float] = []
    d_model_detected = 0

    for i in range(n_layers):
        # Collect (singular_value, left_singular_vector) pairs across projs
        sv_vec_pairs: list[tuple[float, Array]] = []
        total_sq_sum = 0.0

        for proj_name in ("o_proj", "down_proj"):
            base_attn = getattr(base_layers[i], "self_attn", None)
            aligned_attn = getattr(aligned_layers[i], "self_attn", None)
            base_mlp = getattr(base_layers[i], "mlp", None)
            aligned_mlp = getattr(aligned_layers[i], "mlp", None)

            base_w: Array | None = None
            aligned_w: Array | None = None

            if proj_name == "o_proj" and base_attn and aligned_attn:
                base_w = _get_weight(base_attn, proj_name)
                aligned_w = _get_weight(aligned_attn, proj_name)
            elif proj_name == "down_proj" and base_mlp and aligned_mlp:
                base_w = _get_weight(base_mlp, proj_name)
                aligned_w = _get_weight(aligned_mlp, proj_name)

            if base_w is None or aligned_w is None:
                continue

            diff = aligned_w - base_w

            # Handle 3D MoE weights: flatten experts into feature dim
            if diff.ndim == 3:
                n_experts, out_dim, in_dim = diff.shape
                diff = diff.reshape(n_experts * out_dim, in_dim)

            # SVD on CPU for numerical stability
            u, s, _vt = svd_stable(diff)
            force_eval(u, s)

            # Track d_model from o_proj (which has shape d_model x d_model)
            if proj_name == "o_proj" and d_model_detected == 0:
                d_model_detected = diff.shape[0]

            sq_sum = float(ops.sum(s * s).item())
            total_sq_sum += sq_sum

            for j in range(min(top_k, s.shape[0])):
                sv_vec_pairs.append((float(s[j].item()), u[:, j]))

        if not sv_vec_pairs:
            per_layer_bases.append(ops.zeros((top_k, 1)))
            per_layer_singular_values.append([0.0] * top_k)
            per_layer_explained.append(0.0)
            continue

        # Sort by singular value descending, take top-k
        sv_vec_pairs.sort(key=lambda x: x[0], reverse=True)
        selected = sv_vec_pairs[:top_k]

        s_list = [sv for sv, _ in selected]
        vectors = [vec for _, vec in selected]

        # Normalize each vector to unit length
        normalized: list[Array] = []
        for vec in vectors:
            norm = float(ops.linalg.norm(vec).item())
            if norm > 1e-8:
                normalized.append(vec / norm)
            else:
                normalized.append(vec)

        # Pad to top_k if needed (vectors may have different sizes from
        # o_proj vs down_proj, so we take only d_model-sized ones for the
        # basis). Filter to the d_model-sized vectors.
        d_model_vecs = [v for v in normalized if v.shape[0] == d_model_detected]
        d_model_svs = [
            s_list[j]
            for j, v in enumerate(normalized)
            if v.shape[0] == d_model_detected
        ]

        if not d_model_vecs:
            # No d_model-sized vectors; use first available
            d_model_vecs = normalized
            d_model_svs = s_list
            if d_model_detected == 0 and d_model_vecs:
                d_model_detected = d_model_vecs[0].shape[0]

        # Pad to top_k
        while len(d_model_svs) < top_k:
            d_model_svs.append(0.0)
        while len(d_model_vecs) < top_k:
            d_model_vecs.append(
                ops.zeros((d_model_detected if d_model_detected > 0 else 1,)),
            )

        basis = ops.stack(d_model_vecs[:top_k])

        topk_sq = sum(sv * sv for sv in d_model_svs[:top_k])
        explained = topk_sq / total_sq_sum if total_sq_sum > 0 else 0.0

        per_layer_bases.append(basis)
        per_layer_singular_values.append(d_model_svs[:top_k])
        per_layer_explained.append(explained)

    # Select best layer by explained variance
    best_layer = 0
    best_explained = 0.0
    for i, ev in enumerate(per_layer_explained):
        if ev > best_explained:
            best_explained = ev
            best_layer = i

    best_basis = per_layer_bases[best_layer]
    best_svs = per_layer_singular_values[best_layer]
    d_model = int(best_basis.shape[1]) if best_basis.ndim == 2 else d_model_detected

    return DiffResult(
        basis=best_basis,
        singular_values=best_svs,
        explained_variance=per_layer_explained,
        best_layer=best_layer,
        d_model=d_model,
        source_model=source_model_id,
        target_model=target_model_id,
        per_layer_bases=per_layer_bases,
        per_layer_singular_values=per_layer_singular_values,
    )


def _get_weight(
    module: object,
    attr_name: str,
) -> Array | None:
    """Safely retrieve a weight matrix from a module's sub-module."""
    sub = getattr(module, attr_name, None)
    if sub is None:
        return None
    w = getattr(sub, "weight", None)
    if w is not None and hasattr(w, "shape"):
        return w
    return None
