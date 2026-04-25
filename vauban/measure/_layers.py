# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Layer type detection and target layer selection."""

from vauban._forward import get_transformer
from vauban.types import CausalLM


def detect_layer_types(model: CausalLM) -> list[str] | None:
    """Detect per-layer attention types from model config.

    Returns ``None`` for uniform models. Returns a list of type strings
    (``"global"`` / ``"sliding"``) for interleaved architectures (Cohere2).

    Detection works via ``args.sliding_window_pattern`` — present
    on Cohere2 models, absent on everything else.

    Detection first checks ``transformer.args`` and then
    ``transformer.config`` so newer architectures such as Gemma 4,
    which keep their interleaving metadata on ``config``, are covered.

    Args:
        model: The causal language model.

    Returns:
        Per-layer type list, or ``None`` if the model has uniform layers.
    """
    transformer = get_transformer(model)
    metadata = getattr(transformer, "args", None)
    if metadata is None:
        metadata = getattr(transformer, "config", None)
    if metadata is None:
        return None
    num_layers = len(transformer.layers)
    layer_types = getattr(metadata, "layer_types", None)
    if isinstance(layer_types, list) and len(layer_types) == num_layers:
        normalized = [
            "global" if layer_type == "full_attention" else "sliding"
            for layer_type in layer_types
        ]
        if len(set(normalized)) > 1:
            return normalized

    pattern = getattr(metadata, "sliding_window_pattern", None)
    if pattern is None or pattern < 2:
        return None

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
