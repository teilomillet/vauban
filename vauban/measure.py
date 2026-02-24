"""Measure the refusal direction from a model's activation space."""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from vauban.types import (
    CausalLM,
    DBDIResult,
    DirectionResult,
    SubspaceResult,
    Tokenizer,
)


def load_prompts(path: str | Path) -> list[str]:
    """Load prompts from a JSONL file. Each line must have a 'prompt' key."""
    prompts: list[str] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompts.append(obj["prompt"])
    return prompts


def default_prompt_paths() -> tuple[Path, Path]:
    """Return paths to the bundled harmful and harmless prompt files."""
    data_dir = Path(__file__).parent / "data"
    return data_dir / "harmful.jsonl", data_dir / "harmless.jsonl"


def default_eval_path() -> Path:
    """Return path to the bundled eval prompt file."""
    return Path(__file__).parent / "data" / "eval.jsonl"


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


def _best_direction(
    harmful_acts: list[mx.array],
    harmless_acts: list[mx.array],
) -> tuple[mx.array, int, list[float]]:
    """Find the best refusal direction across layers.

    Computes difference-in-means at each layer, normalizes, and selects
    the layer with highest cosine separation.

    Args:
        harmful_acts: Per-layer mean activations for harmful prompts.
        harmless_acts: Per-layer mean activations for harmless prompts.

    Returns:
        Tuple of (unit direction, best layer index, per-layer cosine scores).
    """
    num_layers = len(harmful_acts)

    best_layer = 0
    best_score = -1.0
    cosine_scores: list[float] = []

    for i in range(num_layers):
        diff = harmful_acts[i] - harmless_acts[i]
        direction = diff / (mx.linalg.norm(diff) + 1e-8)
        score = _cosine_separation(
            harmful_acts[i], harmless_acts[i], direction,
        )
        cosine_scores.append(float(score.item()))
        if cosine_scores[-1] > best_score:
            best_score = cosine_scores[-1]
            best_layer = i

    # Recompute best direction
    best_diff = harmful_acts[best_layer] - harmless_acts[best_layer]
    best_dir = best_diff / (mx.linalg.norm(best_diff) + 1e-8)
    mx.eval(best_dir)

    return best_dir, best_layer, cosine_scores


def find_instruction_boundary(tokenizer: Tokenizer, prompt: str) -> int:
    """Find the instruction-final token position in a chat-templated sequence.

    Tokenizes the chat template with empty content vs. actual content.
    Matches suffix tokens from the end to find the generation prompt
    boundary. The instruction-final token is ``len(full) - suffix_len - 1``.

    Falls back to ``len(full) - 1`` if no suffix is detected.

    Args:
        tokenizer: Tokenizer with chat template support.
        prompt: The user prompt text.

    Returns:
        Token index of the instruction-final position.
    """
    messages_full = [{"role": "user", "content": prompt}]
    messages_empty = [{"role": "user", "content": ""}]

    full_result = tokenizer.apply_chat_template(messages_full, tokenize=True)
    empty_result = tokenizer.apply_chat_template(messages_empty, tokenize=True)

    if isinstance(full_result, str):
        msg = "apply_chat_template must return list[int] when tokenize=True"
        raise TypeError(msg)
    if isinstance(empty_result, str):
        msg = "apply_chat_template must return list[int] when tokenize=True"
        raise TypeError(msg)

    full_ids: list[int] = full_result
    empty_ids: list[int] = empty_result

    suffix_len = _match_suffix(full_ids, empty_ids)
    if suffix_len == 0:
        return len(full_ids) - 1

    return len(full_ids) - suffix_len - 1


def _match_suffix(full: list[int], empty: list[int]) -> int:
    """Count matching tokens from the end of both sequences.

    Args:
        full: Token IDs of the full (content-bearing) template.
        empty: Token IDs of the empty-content template.

    Returns:
        Number of matching suffix tokens.
    """
    count = 0
    i_full = len(full) - 1
    i_empty = len(empty) - 1
    while i_full >= 0 and i_empty >= 0:
        if full[i_full] != empty[i_empty]:
            break
        count += 1
        i_full -= 1
        i_empty -= 1
    return count


def _collect_activations_at_instruction_end(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompts: list[str],
    clip_quantile: float = 0.0,
) -> list[mx.array]:
    """Collect per-layer mean activations at the instruction-final token.

    Like ``_collect_activations`` but computes ``find_instruction_boundary``
    per-prompt and passes it to ``_forward_collect``. Needed because the
    instruction boundary varies per prompt.

    Args:
        model: The causal language model.
        tokenizer: Tokenizer with chat template support.
        prompts: Prompts to collect activations for.
        clip_quantile: If > 0, clip per-prompt activations by this quantile.

    Returns a list of length num_layers, each element shape (d_model,).
    """
    means: list[mx.array] | None = None

    for count, prompt in enumerate(prompts, start=1):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False,
        )
        if not isinstance(text, str):
            msg = "apply_chat_template must return str when tokenize=False"
            raise TypeError(msg)
        token_ids = mx.array(tokenizer.encode(text))[None, :]
        boundary = find_instruction_boundary(tokenizer, prompt)
        residuals = _forward_collect(model, token_ids, boundary)

        if clip_quantile > 0.0:
            residuals = [_clip_activation(r, clip_quantile) for r in residuals]

        if means is None:
            means = [r.astype(mx.float32) for r in residuals]
        else:
            for i, r in enumerate(residuals):
                delta = r.astype(mx.float32) - means[i]
                means[i] = means[i] + delta / count

        if count % 16 == 0 and means is not None:
            mx.eval(*means)

    if means is None:
        msg = "No prompts provided for activation collection"
        raise ValueError(msg)

    mx.eval(*means)
    return means


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
    per_layer_bases: list[mx.array] = []
    per_layer_sv: list[list[float]] = []
    per_layer_ev: list[list[float]] = []

    for i in range(num_layers):
        # Build difference matrix: each row is (harmful[j] - harmless[j])
        n = min(harmful_per_prompt[i].shape[0], harmless_per_prompt[i].shape[0])
        diff_matrix = harmful_per_prompt[i][:n] - harmless_per_prompt[i][:n]

        # SVD on CPU for numerical stability
        u, s, vt = mx.linalg.svd(diff_matrix, stream=mx.cpu)  # type: ignore[arg-type]
        mx.eval(u, s, vt)

        # Top-k basis vectors (rows of Vt)
        actual_k = min(k, vt.shape[0])
        basis = vt[:actual_k]

        sv = [float(s[j].item()) for j in range(actual_k)]
        total_var = float(mx.sum(s * s).item())
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


def _collect_activations(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompts: list[str],
    clip_quantile: float = 0.0,
    token_position: int = -1,
) -> list[mx.array]:
    """Collect per-layer mean activations across prompts.

    Uses Welford's online algorithm for numerically stable streaming
    mean computation — O(d_model) memory per layer instead of
    O(num_prompts * d_model).

    Args:
        model: The causal language model.
        tokenizer: Tokenizer with chat template support.
        prompts: Prompts to collect activations for.
        clip_quantile: If > 0, clip per-prompt activations by this
            quantile before accumulating into the running mean.
        token_position: Token index to extract activations from.
            Defaults to -1 (last token).

    Returns a list of length num_layers, each element shape (d_model,).
    """
    means: list[mx.array] | None = None

    for count, prompt in enumerate(prompts, start=1):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False,
        )
        if not isinstance(text, str):
            msg = "apply_chat_template must return str when tokenize=False"
            raise TypeError(msg)
        token_ids = mx.array(tokenizer.encode(text))[None, :]
        residuals = _forward_collect(model, token_ids, token_position)

        if clip_quantile > 0.0:
            residuals = [_clip_activation(r, clip_quantile) for r in residuals]

        if means is None:
            means = [r.astype(mx.float32) for r in residuals]
        else:
            # Welford online mean: mean += (x - mean) / n
            for i, r in enumerate(residuals):
                delta = r.astype(mx.float32) - means[i]
                means[i] = means[i] + delta / count

        # Evaluate periodically to avoid graph buildup
        if count % 16 == 0 and means is not None:
            mx.eval(*means)

    if means is None:
        msg = "No prompts provided for activation collection"
        raise ValueError(msg)

    mx.eval(*means)
    return means


def _collect_per_prompt_activations(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompts: list[str],
    clip_quantile: float = 0.0,
    token_position: int = -1,
) -> list[mx.array]:
    """Collect per-prompt activations at each layer (no averaging).

    Args:
        model: The causal language model.
        tokenizer: Tokenizer with chat template support.
        prompts: Prompts to collect activations for.
        clip_quantile: If > 0, clip per-prompt activations by this quantile.
        token_position: Token index to extract activations from.
            Defaults to -1 (last token).

    Returns a list of length num_layers, each element shape (num_prompts, d_model).
    """
    all_residuals: list[list[mx.array]] = []

    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False,
        )
        if not isinstance(text, str):
            msg = "apply_chat_template must return str when tokenize=False"
            raise TypeError(msg)
        token_ids = mx.array(tokenizer.encode(text))[None, :]
        residuals = _forward_collect(model, token_ids, token_position)

        if clip_quantile > 0.0:
            residuals = [_clip_activation(r, clip_quantile) for r in residuals]

        all_residuals.append(residuals)

    # Stack per-prompt activations for each layer
    num_layers = len(all_residuals[0])
    per_layer: list[mx.array] = []
    for layer_idx in range(num_layers):
        stacked = mx.stack([r[layer_idx] for r in all_residuals])
        mx.eval(stacked)
        per_layer.append(stacked)

    return per_layer


def _forward_collect(
    model: CausalLM,
    token_ids: mx.array,
    token_position: int = -1,
) -> list[mx.array]:
    """Manual layer-by-layer forward pass, capturing residual stream.

    Returns per-layer activations at the given token position.
    Each element has shape (d_model,).

    Args:
        model: The causal language model.
        token_ids: Input token IDs of shape (1, seq_len).
        token_position: Token index to extract activations from.
            Defaults to -1 (last token).
    """
    transformer = model.model
    h = transformer.embed_tokens(token_ids)

    mask = nn.MultiHeadAttention.create_additive_causal_mask(
        h.shape[1],
    )
    mask = mask.astype(h.dtype)

    residuals: list[mx.array] = []
    for layer in transformer.layers:
        h = layer(h, mask)
        # Upcast to float32 for numerical stability (like Heretic)
        activation = h[0, token_position, :].astype(mx.float32)
        residuals.append(activation)

    return residuals


def detect_layer_types(model: CausalLM) -> list[str] | None:
    """Detect per-layer attention types from model config.

    Returns ``None`` for uniform models. Returns a list of type strings
    (``"global"`` / ``"sliding"``) for interleaved architectures (Cohere2).

    Detection works via ``model.model.args.sliding_window_pattern`` — present
    on Cohere2 models, absent on everything else.

    Args:
        model: The causal language model.

    Returns:
        Per-layer type list, or ``None`` if the model has uniform layers.
    """
    transformer = model.model
    args = getattr(transformer, "args", None)
    if args is None:
        return None
    pattern = getattr(args, "sliding_window_pattern", None)
    if pattern is None or pattern < 2:
        return None

    num_layers = len(transformer.layers)
    return [
        "global" if i % pattern == pattern - 1 else "sliding"
        for i in range(num_layers)
    ]


def select_target_layers(
    cosine_scores: list[float],
    strategy: str = "above_median",
    top_k: int = 10,
    layer_types: list[str] | None = None,
    type_filter: str | None = None,
) -> list[int]:
    """Select target layers for cutting based on per-layer scores.

    Strategies:
        - ``"above_median"``: layers where score > median (default).
        - ``"top_k"``: top *k* layers by score.
        - ``"silhouette"``: same as ``"above_median"`` — works with
          silhouette scores passed in place of cosine scores.

    When ``type_filter`` is set and ``layer_types`` is provided, only
    layers matching the given type are considered as candidates. The
    strategy (median, top-k) is then applied to the filtered subset.

    The ``cosine_scores`` parameter accepts any per-layer score list
    (cosine separation, silhouette scores, etc.).

    Args:
        cosine_scores: Per-layer scores from ``measure()`` or
            ``silhouette_scores()``.
        strategy: Selection strategy name.
        top_k: Number of layers to select when using ``"top_k"`` strategy.
        layer_types: Per-layer type strings from ``detect_layer_types()``.
        type_filter: If set, restrict candidates to this layer type
            (e.g. ``"global"`` or ``"sliding"``).

    Returns:
        Sorted list of layer indices to target.
    """
    if not cosine_scores:
        return []

    # Build candidate set: all layers, or filtered by type
    if type_filter is not None and layer_types is not None:
        candidates = [
            i for i, t in enumerate(layer_types) if t == type_filter
        ]
    else:
        candidates = list(range(len(cosine_scores)))

    if not candidates:
        return []

    if strategy in ("above_median", "silhouette"):
        filtered_scores = sorted(cosine_scores[i] for i in candidates)
        median = filtered_scores[len(filtered_scores) // 2]
        return [i for i in candidates if cosine_scores[i] > median]

    if strategy == "top_k":
        indexed = sorted(
            ((i, cosine_scores[i]) for i in candidates),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted(i for i, _ in indexed[:top_k])

    msg = (
        f"Unknown strategy: {strategy!r}."
        " Use 'above_median', 'top_k', or 'silhouette'."
    )
    raise ValueError(msg)


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
    group_a: mx.array,
    group_b: mx.array,
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
    def _pairwise_dist(x: mx.array, y: mx.array) -> mx.array:
        x_sq = mx.sum(x * x, axis=1, keepdims=True)  # (n, 1)
        y_sq = mx.sum(y * y, axis=1, keepdims=True)  # (m, 1)
        cross = x @ y.T  # (n, m)
        dist = x_sq + y_sq.T - 2.0 * cross
        return mx.maximum(dist, mx.array(0.0))  # clamp negatives from float error

    dist_aa = _pairwise_dist(group_a, group_a)  # (n_a, n_a)
    dist_ab = _pairwise_dist(group_a, group_b)  # (n_a, n_b)
    dist_bb = _pairwise_dist(group_b, group_b)  # (n_b, n_b)
    dist_ba = _pairwise_dist(group_b, group_a)  # (n_b, n_a)

    mx.eval(dist_aa, dist_ab, dist_bb, dist_ba)

    total_s = 0.0
    total_n = n_a + n_b

    # Silhouette for group_a samples
    for j in range(n_a):
        # Mean distance to other points in same cluster
        a_val = float(mx.sum(dist_aa[j]).item()) / (n_a - 1)
        # Mean distance to points in other cluster
        b_val = float(mx.sum(dist_ab[j]).item()) / n_b
        denom = max(a_val, b_val)
        total_s += (b_val - a_val) / denom if denom > 0 else 0.0

    # Silhouette for group_b samples
    for j in range(n_b):
        a_val = float(mx.sum(dist_bb[j]).item()) / (n_b - 1)
        b_val = float(mx.sum(dist_ba[j]).item()) / n_a
        denom = max(a_val, b_val)
        total_s += (b_val - a_val) / denom if denom > 0 else 0.0

    return total_s / total_n


def _clip_activation(activation: mx.array, quantile: float) -> mx.array:
    """Winsorize an activation vector by clamping extreme values.

    Clips each dimension to the ``[quantile, 1-quantile]`` range of
    its absolute value distribution. This tames "massive activations"
    that can distort the difference-in-means computation.

    Args:
        activation: Activation vector of shape (d_model,).
        quantile: Fraction of extremes to clip (e.g. 0.01 = clip top/bottom 1%).
    """
    abs_vals = mx.abs(activation)
    sorted_vals = mx.sort(abs_vals)
    n = sorted_vals.shape[0]
    high_idx = min(n - 1, int(n * (1.0 - quantile)))
    threshold = sorted_vals[high_idx]
    mx.eval(threshold)
    return mx.clip(activation, -threshold, threshold)


def _cosine_separation(
    harmful_mean: mx.array,
    harmless_mean: mx.array,
    direction: mx.array,
) -> mx.array:
    """Cosine separation: projection difference onto direction."""
    proj_harmful = mx.sum(harmful_mean * direction)
    proj_harmless = mx.sum(harmless_mean * direction)
    return proj_harmful - proj_harmless
