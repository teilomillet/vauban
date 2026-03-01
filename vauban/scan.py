"""Injection content scanner — per-token projection-based detection.

Extends the SIC direction projection to all tokens (not just last),
enabling span-level detection of injected content within documents.

Usage:
    direction = measure(model, tok, injected_docs, clean_docs)
    result = scan(model, tokenizer, content, config, direction.direction)
"""

import math

from vauban import _ops as ops
from vauban._array import Array
from vauban._forward import embed_and_mask, force_eval
from vauban.types import (
    CausalLM,
    ScanConfig,
    ScanResult,
    ScanSpan,
    Tokenizer,
)


def scan(
    model: CausalLM,
    tokenizer: Tokenizer,
    content: str,
    config: ScanConfig,
    direction: Array,
    layer_index: int = 0,
) -> ScanResult:
    """Scan content for injection by projecting per-token activations.

    Runs a forward pass and computes per-token projections onto the
    injection direction at the target layer. Contiguous tokens above
    ``span_threshold`` are grouped into ``ScanSpan`` objects.

    Args:
        model: The causal language model.
        tokenizer: Tokenizer with encode/decode support.
        content: Text content to scan (e.g. a document or tool output).
        config: Scan configuration.
        direction: Injection direction vector (d_model,).
        layer_index: Fallback layer index if config.target_layer is None.

    Returns:
        ScanResult with per-token projections and detected spans.
    """
    target_layer = (
        config.target_layer if config.target_layer is not None
        else layer_index
    )

    # Tokenize
    token_ids_list = tokenizer.encode(content)
    token_ids = ops.array(token_ids_list)[None, :]

    # Forward pass to target layer, collecting all token activations
    transformer = model.model
    h, mask = embed_and_mask(transformer, token_ids)

    for i, layer_module in enumerate(transformer.layers):
        h = layer_module(h, mask)
        if i == target_layer:
            break

    force_eval(h)

    # Per-token projections: dot product with direction
    # h shape: (1, seq_len, d_model), direction shape: (d_model,)
    projections = ops.sum(h[0] * direction, axis=-1)  # (seq_len,)
    force_eval(projections)
    _raw = projections.tolist()
    per_token: list[float] = (
        [float(v) for v in _raw] if isinstance(_raw, list) else [float(_raw)]
    )

    # Overall projection: mean of per-token projections
    overall_projection = sum(per_token) / len(per_token) if per_token else 0.0

    # Injection probability: sigmoid(projection - threshold)
    injection_probability = _sigmoid(overall_projection - config.threshold)

    # Span detection: contiguous tokens below span_threshold
    spans = _detect_spans(
        per_token, token_ids_list, tokenizer, config.span_threshold,
    )

    flagged = injection_probability > 0.5

    return ScanResult(
        injection_probability=injection_probability,
        overall_projection=overall_projection,
        spans=spans,
        per_token_projections=per_token,
        flagged=flagged,
    )


def calibrate_scan_threshold(
    model: CausalLM,
    tokenizer: Tokenizer,
    clean_documents: list[str],
    config: ScanConfig,
    direction: Array,
    layer_index: int = 0,
) -> float:
    """Auto-calibrate the scan threshold from known-clean documents.

    Returns ``mean - 2*std`` of per-document overall projections.

    Args:
        model: The causal language model.
        tokenizer: Tokenizer with encode/decode support.
        clean_documents: Documents known to be injection-free.
        config: Scan configuration.
        direction: Injection direction vector.
        layer_index: Fallback layer index.

    Returns:
        Calibrated threshold value.
    """
    projections: list[float] = []
    for doc in clean_documents:
        result = scan(model, tokenizer, doc, config, direction, layer_index)
        projections.append(result.overall_projection)

    if not projections:
        return config.threshold

    mean = sum(projections) / len(projections)
    variance = sum((p - mean) ** 2 for p in projections) / len(projections)
    std = math.sqrt(variance)

    return mean - 2.0 * std


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def _detect_spans(
    projections: list[float],
    token_ids: list[int],
    tokenizer: Tokenizer,
    threshold: float,
) -> list[ScanSpan]:
    """Detect contiguous spans of tokens below the threshold.

    Tokens with projections below ``threshold`` are considered
    injection-like. Consecutive such tokens form spans.

    Args:
        projections: Per-token projection values.
        token_ids: Original token IDs.
        tokenizer: Tokenizer for decoding spans.
        threshold: Projection threshold below which tokens are flagged.

    Returns:
        List of detected ScanSpan objects.
    """
    spans: list[ScanSpan] = []
    in_span = False
    start = 0
    span_projs: list[float] = []

    for i, proj in enumerate(projections):
        if proj < threshold:
            if not in_span:
                in_span = True
                start = i
                span_projs = []
            span_projs.append(proj)
        else:
            if in_span:
                text = tokenizer.decode(token_ids[start:i])
                mean_proj = (
                    sum(span_projs) / len(span_projs) if span_projs else 0.0
                )
                spans.append(ScanSpan(
                    start=start,
                    end=i,
                    text=text,
                    mean_projection=mean_proj,
                ))
                in_span = False

    # Close trailing span
    if in_span:
        text = tokenizer.decode(token_ids[start:len(projections)])
        mean_proj = (
            sum(span_projs) / len(span_projs) if span_projs else 0.0
        )
        spans.append(ScanSpan(
            start=start,
            end=len(projections),
            text=text,
            mean_projection=mean_proj,
        ))

    return spans
