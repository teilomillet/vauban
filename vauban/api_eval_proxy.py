"""Defense proxy layer for API eval.

Runs local SIC/CAST filtering before sending prompts to remote API
endpoints. Prompts blocked by local defenses never reach the API;
success rates are computed over the total prompt count (including
blocked prompts) so that defense effectiveness is reflected in the
final metric.
"""

import logging

from vauban._array import Array
from vauban.api_eval import _build_api_prompt, evaluate_suffix_via_api
from vauban.evaluate import DEFAULT_REFUSAL_PHRASES
from vauban.types import (
    ApiEvalConfig,
    CausalLM,
    DefenseProxyResult,
    SICConfig,
    Tokenizer,
    TransferEvalResult,
)

logger = logging.getLogger(__name__)


def evaluate_with_defense_proxy(
    suffix_text: str,
    prompts: list[str],
    config: ApiEvalConfig,
    model: CausalLM,
    tokenizer: Tokenizer,
    direction: Array,
    layer_index: int,
    fallback_system_prompt: str | None = None,
    history: list[dict[str, str]] | None = None,
    token_position: str = "suffix",
) -> tuple[list[TransferEvalResult], DefenseProxyResult]:
    """Evaluate suffixes via API with local defense proxy filtering.

    Prompts pass through SIC and/or CAST locally first. Only survivors
    are sent to the remote API. The success rate denominator is always
    the total prompt count so that proxy-blocked prompts count as
    failures.

    Args:
        suffix_text: The optimized adversarial suffix text.
        prompts: Attack prompts to test.
        config: API evaluation configuration (includes defense_proxy settings).
        model: Local causal language model for defense evaluation.
        tokenizer: Tokenizer with chat template support.
        direction: Refusal direction vector.
        layer_index: Layer index for direction projection.
        fallback_system_prompt: System prompt fallback for API calls.
        history: Optional conversation history for multi-turn.
        token_position: Suffix placement — "prefix", "suffix", or "infix".

    Returns:
        Tuple of (api_results, proxy_result). api_results contains one
        TransferEvalResult per endpoint with adjusted success rates.
    """
    proxy_mode = config.defense_proxy or "sic"
    use_sic = proxy_mode in ("sic", "both")
    use_cast = proxy_mode in ("cast", "both")

    total = len(prompts)
    sic_blocked = 0
    sic_sanitized = 0
    cast_gated = 0
    cast_responses: list[str] = []

    # Build adversarial prompts
    adv_prompts = [
        _build_api_prompt(p, suffix_text, token_position) for p in prompts
    ]

    # --- SIC proxy ---
    surviving_prompts = list(adv_prompts)
    if use_sic:
        surviving_prompts, sic_blocked, sic_sanitized = _run_sic_proxy(
            surviving_prompts, model, tokenizer, direction, layer_index,
            config,
        )

    # --- CAST proxy (on SIC survivors) ---
    if use_cast and surviving_prompts:
        if config.defense_proxy_cast_mode == "full":
            surviving_prompts, cast_gated, cast_responses = _run_cast_full(
                surviving_prompts, model, tokenizer, direction,
                layer_index, config,
            )
        else:
            surviving_prompts, cast_gated = _run_cast_gate(
                surviving_prompts, model, tokenizer, direction,
                layer_index, config,
            )

    # Resolve proxy_mode label
    cast_mode_label = config.defense_proxy_cast_mode
    if use_sic and use_cast:
        mode_label = f"both_{cast_mode_label}"
    elif use_cast:
        mode_label = f"cast_{cast_mode_label}"
    else:
        mode_label = "sic"

    proxy_result = DefenseProxyResult(
        total_prompts=total,
        sic_blocked=sic_blocked,
        sic_sanitized=sic_sanitized,
        cast_gated=cast_gated,
        prompts_sent=len(surviving_prompts),
        proxy_mode=mode_label,
        cast_responses=cast_responses,
    )

    # --- API eval on survivors ---
    if surviving_prompts:
        # We already assembled adv_prompts — pass surviving raw prompts
        # directly as pre-assembled prompts with suffix position "suffix"
        # (they already contain the suffix).
        api_results = _evaluate_preassembled(
            surviving_prompts, config, fallback_system_prompt, history,
        )
        # Adjust success rates: denominator = total prompts
        adjusted: list[TransferEvalResult] = []
        for r in api_results:
            api_successes = int(r.success_rate * len(surviving_prompts))
            adj_rate = api_successes / total if total > 0 else 0.0
            adjusted.append(TransferEvalResult(
                model_id=r.model_id,
                success_rate=adj_rate,
                eval_responses=r.eval_responses,
            ))
        api_results = adjusted
    else:
        # All blocked — report zero success for each endpoint
        api_results = [
            TransferEvalResult(
                model_id=ep.name,
                success_rate=0.0,
                eval_responses=[],
            )
            for ep in config.endpoints
        ]

    logger.info(
        "Defense proxy: %d/%d sent (SIC blocked=%d sanitized=%d,"
        " CAST gated=%d)",
        proxy_result.prompts_sent, total, sic_blocked,
        sic_sanitized, cast_gated,
    )

    return api_results, proxy_result


def _evaluate_preassembled(
    assembled_prompts: list[str],
    config: ApiEvalConfig,
    fallback_system_prompt: str | None,
    history: list[dict[str, str]] | None,
) -> list[TransferEvalResult]:
    """Evaluate pre-assembled prompts against API endpoints.

    Since the prompts already contain the suffix, we pass an empty
    suffix_text and use suffix position so _build_api_prompt just
    appends nothing meaningful — but we override prompts directly.

    Args:
        assembled_prompts: Prompts with suffix already embedded.
        config: API eval configuration.
        fallback_system_prompt: Fallback system prompt.
        history: Optional conversation history.

    Returns:
        List of TransferEvalResult, one per endpoint.
    """
    # Use empty suffix with suffix position — _build_api_prompt will
    # just append " " which is negligible. The prompts already contain
    # the adversarial content.
    return evaluate_suffix_via_api(
        suffix_text="",
        prompts=assembled_prompts,
        config=config,
        fallback_system_prompt=fallback_system_prompt,
        history=history,
        token_position="suffix",
    )


def _run_sic_proxy(
    prompts: list[str],
    model: CausalLM,
    tokenizer: Tokenizer,
    direction: Array,
    layer_index: int,
    config: ApiEvalConfig,
) -> tuple[list[str], int, int]:
    """Run SIC sanitization on each prompt.

    Args:
        prompts: Pre-assembled adversarial prompts.
        model: Local model.
        tokenizer: Tokenizer.
        direction: Refusal direction.
        layer_index: Layer for direction projection.
        config: API eval config with defense_proxy_sic_* fields.

    Returns:
        Tuple of (surviving_prompts, blocked_count, sanitized_count).
    """
    from vauban.sic import sic_single

    sic_cfg = SICConfig(
        mode=config.defense_proxy_sic_mode,
        threshold=config.defense_proxy_sic_threshold,
        max_iterations=config.defense_proxy_sic_max_iterations,
    )

    survivors: list[str] = []
    blocked = 0
    sanitized = 0

    for prompt in prompts:
        result = sic_single(
            model, tokenizer, prompt, sic_cfg,
            direction=direction, layer_index=layer_index,
        )
        if result.blocked:
            blocked += 1
        elif result.iterations > 0:
            sanitized += 1
            survivors.append(result.clean_prompt)
        else:
            survivors.append(prompt)

    return survivors, blocked, sanitized


def _run_cast_gate(
    prompts: list[str],
    model: CausalLM,
    tokenizer: Tokenizer,
    direction: Array,
    layer_index: int,
    config: ApiEvalConfig,
) -> tuple[list[str], int]:
    """Run CAST gate detection on each prompt.

    Single forward pass per prompt — checks projection onto the refusal
    direction. Prompts with low projection (adversarial) are gated.

    Args:
        prompts: Pre-assembled adversarial prompts.
        model: Local model.
        tokenizer: Tokenizer.
        direction: Refusal direction.
        layer_index: Layer for direction projection.
        config: API eval config with defense_proxy_cast_* fields.

    Returns:
        Tuple of (surviving_prompts, gated_count).
    """
    from vauban.sic import _detect_adversarial_direction

    target_layer = (
        config.defense_proxy_cast_layers[0]
        if config.defense_proxy_cast_layers
        else layer_index
    )

    survivors: list[str] = []
    gated = 0

    for prompt in prompts:
        proj = _detect_adversarial_direction(
            model, tokenizer, prompt, direction, target_layer,
        )
        # Low projection = adversarial. CAST gates when projection
        # is below threshold (prompt looks adversarial enough to steer).
        if proj <= config.defense_proxy_cast_threshold:
            gated += 1
        else:
            survivors.append(prompt)

    return survivors, gated


def _run_cast_full(
    prompts: list[str],
    model: CausalLM,
    tokenizer: Tokenizer,
    direction: Array,
    layer_index: int,
    config: ApiEvalConfig,
) -> tuple[list[str], int, list[str]]:
    """Run full CAST generation on each prompt.

    Generates a steered response locally. If the response is a refusal,
    the prompt is considered gated (the defense worked).

    Args:
        prompts: Pre-assembled adversarial prompts.
        model: Local model.
        tokenizer: Tokenizer.
        direction: Refusal direction.
        layer_index: Layer for direction projection.
        config: API eval config with defense_proxy_cast_* fields.

    Returns:
        Tuple of (surviving_prompts, gated_count, cast_responses).
    """
    from vauban.cast import cast_generate

    layers = (
        config.defense_proxy_cast_layers
        if config.defense_proxy_cast_layers
        else [layer_index]
    )

    survivors: list[str] = []
    gated = 0
    responses: list[str] = []

    for prompt in prompts:
        result = cast_generate(
            model, tokenizer, prompt, direction, layers,
            alpha=config.defense_proxy_cast_alpha,
            threshold=config.defense_proxy_cast_threshold,
            max_tokens=config.defense_proxy_cast_max_tokens,
        )
        responses.append(result.text)
        if _is_refusal(result.text):
            gated += 1
        else:
            survivors.append(prompt)

    return survivors, gated, responses


def _is_refusal(text: str) -> bool:
    """Check if text contains refusal phrases."""
    lower = text.lower()
    return any(phrase.lower() in lower for phrase in DEFAULT_REFUSAL_PHRASES)
