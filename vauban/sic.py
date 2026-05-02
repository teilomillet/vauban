# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""SIC (Soft Instruction Control) — iterative input sanitization defense.

Detect adversarial content in prompts, rewrite to remove it, repeat until
clean or block. Direction-aware variant uses refusal projection as a fast
detection signal.

Reference: arxiv.org/abs/2510.21057
"""

import math

from vauban import _ops as ops
from vauban._array import Array
from vauban._forward import (
    embed_and_mask,
    encode_chat_prompt,
    encode_user_prompt,
    extract_logits,
    force_eval,
    get_transformer,
    make_cache,
    make_ssm_mask,
    select_mask,
    to_model_device,
)
from vauban.evaluate import DEFAULT_REFUSAL_PHRASES
from vauban.types import (
    CausalLM,
    SICConfig,
    SICPromptResult,
    SICResult,
    Tokenizer,
)


def calibrate_threshold(
    model: CausalLM,
    tokenizer: Tokenizer,
    clean_prompts: list[str],
    config: SICConfig,
    direction: Array | None,
    target_layer: int,
) -> float:
    """Auto-calibrate the detection threshold from known-clean prompts.

    Scores each clean prompt via ``_detect()``, then returns
    ``mean - 2*std`` — a conservative threshold where ~97.7% of clean
    prompts should score above.

    Args:
        model: The causal language model.
        tokenizer: Tokenizer with chat template support.
        clean_prompts: Prompts known to be benign.
        config: SIC configuration (used for detection mode/params).
        direction: Refusal direction vector (required for mode="direction").
        target_layer: Layer index for direction projection.

    Returns:
        Calibrated threshold value.
    """
    scores: list[float] = []
    for prompt in clean_prompts:
        score = _detect(model, tokenizer, prompt, config, direction, target_layer)
        scores.append(score)

    if not scores:
        return config.threshold

    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    std = math.sqrt(variance)

    return mean - 2.0 * std


def sic(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompts: list[str],
    config: SICConfig,
    direction: Array | None = None,
    layer_index: int = 0,
    calibration_prompts: list[str] | None = None,
) -> SICResult:
    """Run SIC sanitization on a batch of prompts.

    Args:
        model: The causal language model.
        tokenizer: Tokenizer with chat template support.
        prompts: Prompts to sanitize.
        config: SIC configuration.
        direction: Refusal direction vector (required for mode="direction").
        layer_index: Layer index for direction projection (used when
            config.target_layer is None).
        calibration_prompts: Known-clean prompts for threshold
            auto-calibration. Used when ``config.calibrate`` is True.

    Returns:
        SICResult with per-prompt sanitization outcomes.

    Raises:
        ValueError: If mode="direction" and no direction is provided.
    """
    if config.mode == "direction" and direction is None:
        msg = "SIC mode='direction' requires a direction vector"
        raise ValueError(msg)

    target_layer = (
        config.target_layer if config.target_layer is not None
        else layer_index
    )

    # Auto-calibrate threshold if requested
    calibrated: float | None = None
    effective_config = config
    if config.calibrate and calibration_prompts is not None:
        calibrated = calibrate_threshold(
            model, tokenizer, calibration_prompts,
            config, direction, target_layer,
        )
        # Replace threshold in config for this run
        effective_config = SICConfig(
            mode=config.mode,
            threshold=calibrated,
            max_iterations=config.max_iterations,
            max_tokens=config.max_tokens,
            target_layer=config.target_layer,
            sanitize_system_prompt=config.sanitize_system_prompt,
            max_sanitize_tokens=config.max_sanitize_tokens,
            block_on_failure=config.block_on_failure,
            calibrate=config.calibrate,
            calibrate_prompts=config.calibrate_prompts,
        )

    results: list[SICPromptResult] = []
    for prompt in prompts:
        result = sic_single(
            model, tokenizer, prompt, effective_config, direction, layer_index,
        )
        results.append(result)

    prompts_clean = [r.clean_prompt for r in results]
    prompts_blocked = [r.blocked for r in results]
    iterations_used = [r.iterations for r in results]
    initial_scores = [r.initial_score for r in results]
    final_scores = [r.final_score for r in results]

    total_blocked = sum(1 for b in prompts_blocked if b)
    total_sanitized = sum(
        1 for r in results if r.iterations > 0 and not r.blocked
    )
    total_clean = sum(
        1 for r in results if r.iterations == 0 and not r.blocked
    )

    return SICResult(
        prompts_clean=prompts_clean,
        prompts_blocked=prompts_blocked,
        iterations_used=iterations_used,
        initial_scores=initial_scores,
        final_scores=final_scores,
        total_blocked=total_blocked,
        total_sanitized=total_sanitized,
        total_clean=total_clean,
        calibrated_threshold=calibrated,
    )


def sic_single(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompt: str,
    config: SICConfig,
    direction: Array | None = None,
    layer_index: int = 0,
) -> SICPromptResult:
    """Run SIC sanitization on a single prompt.

    Args:
        model: The causal language model.
        tokenizer: Tokenizer with chat template support.
        prompt: The prompt to sanitize.
        config: SIC configuration.
        direction: Refusal direction vector (required for mode="direction").
        layer_index: Layer index for direction projection (used when
            config.target_layer is None).

    Returns:
        SICPromptResult with sanitization outcome.
    """
    target_layer = (
        config.target_layer if config.target_layer is not None
        else layer_index
    )

    return _sanitize_prompt(
        model, tokenizer, prompt, config, direction, target_layer,
    )


def _sanitize_prompt(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompt: str,
    config: SICConfig,
    direction: Array | None,
    target_layer: int,
) -> SICPromptResult:
    """Iterative sanitization loop for a single prompt.

    1. Score prompt via detection
    2. If score >= threshold: return clean (iterations=0)
    3. Else: rewrite prompt, re-score, repeat
    4. If still adversarial after max_iterations: block or return last rewrite
    """
    initial_score = _detect(
        model, tokenizer, prompt, config, direction, target_layer,
    )

    if initial_score >= config.threshold:
        return SICPromptResult(
            clean_prompt=prompt,
            blocked=False,
            iterations=0,
            initial_score=initial_score,
            final_score=initial_score,
        )

    current_prompt = prompt
    current_score = initial_score

    for i in range(1, config.max_iterations + 1):
        current_prompt = _rewrite_prompt(
            model, tokenizer, current_prompt,
            config.sanitize_system_prompt, config.max_sanitize_tokens,
        )
        current_score = _detect(
            model, tokenizer, current_prompt, config, direction, target_layer,
        )
        if current_score >= config.threshold:
            return SICPromptResult(
                clean_prompt=current_prompt,
                blocked=False,
                iterations=i,
                initial_score=initial_score,
                final_score=current_score,
            )

    # Exhausted iterations — block or return last rewrite
    return SICPromptResult(
        clean_prompt=current_prompt,
        blocked=config.block_on_failure,
        iterations=config.max_iterations,
        initial_score=initial_score,
        final_score=current_score,
    )


def _detect(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompt: str,
    config: SICConfig,
    direction: Array | None,
    target_layer: int,
) -> float:
    """Dispatch to direction, SVF, or generation detection."""
    if config.mode == "direction":
        if direction is None:
            msg = "direction mode requires a measured direction"
            raise ValueError(msg)
        return _detect_adversarial_direction(
            model, tokenizer, prompt, direction, target_layer,
        )
    if config.mode == "svf":
        return _detect_adversarial_svf(
            model, tokenizer, prompt, config, target_layer,
        )
    return _detect_adversarial_generation(
        model, tokenizer, prompt, config.max_tokens,
    )


def _detect_adversarial_direction(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompt: str,
    direction: Array,
    target_layer: int,
) -> float:
    """Detect adversarial content via refusal direction projection.

    Single forward pass, compute projection at the target layer only.
    Higher projection = more refusal-like = benign.
    Lower projection = less refusal = likely adversarial.

    Returns:
        Projection magnitude (float). Higher = more benign.
    """
    token_ids = encode_user_prompt(tokenizer, prompt)

    transformer = get_transformer(model)
    h, mask = embed_and_mask(transformer, token_ids)
    direction = ops.to_device_like(direction, h)
    ssm_mask = make_ssm_mask(transformer, h)

    for i, layer in enumerate(transformer.layers):
        h = layer(h, select_mask(layer, mask, ssm_mask))
        if i == target_layer:
            last_token = h[0, -1, :]
            proj = ops.sum(last_token * direction)
            force_eval(proj)
            return float(proj.item())

    # Fallback: if target_layer exceeds layer count, use last layer
    last_token = h[0, -1, :]
    proj = ops.sum(last_token * direction)
    force_eval(proj)
    return float(proj.item())


def _detect_adversarial_svf(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompt: str,
    config: SICConfig,
    target_layer: int,
) -> float:
    """Detect adversarial content via SVF boundary evaluation.

    Loads the SVF boundary and evaluates f(h) at the target layer.
    Higher score = more benign (positive side of boundary).
    Lower score = more adversarial (negative side).

    Returns:
        Boundary score (float). Higher = more benign.
    """
    from vauban.svf import load_svf_boundary

    if config.svf_boundary_path is None:
        msg = "SIC mode='svf' requires svf_boundary_path"
        raise ValueError(msg)

    boundary = load_svf_boundary(config.svf_boundary_path)

    transformer = get_transformer(model)
    n_layers = len(transformer.layers)

    token_ids = encode_user_prompt(tokenizer, prompt)

    h, mask = embed_and_mask(transformer, token_ids)
    ssm_mask = make_ssm_mask(transformer, h)

    for i, layer in enumerate(transformer.layers):
        h = layer(h, select_mask(layer, mask, ssm_mask))
        if i == target_layer:
            last_token = h[0, -1, :]
            score = boundary.forward(last_token, i)
            force_eval(score)
            return float(score.item())

    # Fallback: use last layer
    last_token = h[0, -1, :]
    score = boundary.forward(last_token, n_layers - 1)
    force_eval(score)
    return float(score.item())


def _detect_adversarial_generation(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_tokens: int,
) -> float:
    """Detect adversarial content via generation-based refusal check.

    Generate a response and check for refusal phrases.
    Returns 1.0 if the model refused (prompt is benign).
    Returns 0.0 if the model did not refuse (prompt is suspicious).
    """
    response = _generate_with_messages(
        model, tokenizer,
        [{"role": "user", "content": prompt}],
        max_tokens,
    )
    lower = response.lower()
    if any(phrase.lower() in lower for phrase in DEFAULT_REFUSAL_PHRASES):
        return 1.0
    return 0.0


def _rewrite_prompt(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompt: str,
    system_prompt: str,
    max_tokens: int,
) -> str:
    """Rewrite a prompt to remove adversarial content.

    Uses the model itself with a system instruction to sanitize the prompt.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    result = _generate_with_messages(model, tokenizer, messages, max_tokens)
    return result.strip() if result.strip() else prompt


def _generate_with_messages(
    model: CausalLM,
    tokenizer: Tokenizer,
    messages: list[dict[str, str]],
    max_tokens: int,
) -> str:
    """Generate text from a list of messages using KV-cached greedy decoding.

    Like evaluate._generate() but takes a full messages list (enables
    system messages for the sanitization step).
    """
    token_ids = to_model_device(model, encode_chat_prompt(tokenizer, messages))
    generated: list[int] = []

    eos_token_id: int | None = getattr(tokenizer, "eos_token_id", None)

    cache = make_cache(model)

    for _ in range(max_tokens):
        result = model(token_ids, cache=cache)  # type: ignore[call-non-callable]
        logits = extract_logits(result)
        next_token = int(ops.argmax(logits[:, -1, :], axis=-1).item())
        generated.append(next_token)
        if eos_token_id is not None and next_token == eos_token_id:
            break
        token_ids = to_model_device(model, ops.array([[next_token]]))

    return tokenizer.decode(generated)
