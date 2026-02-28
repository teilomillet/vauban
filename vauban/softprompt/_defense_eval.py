"""Post-attack defense evaluation: test suffixes against SIC and CAST."""

from vauban._array import Array
from vauban.evaluate import DEFAULT_REFUSAL_PHRASES
from vauban.types import (
    CausalLM,
    DefenseEvalResult,
    SICConfig,
    SoftPromptConfig,
    Tokenizer,
)


def evaluate_against_defenses(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompts: list[str],
    config: SoftPromptConfig,
    direction: Array | None,
    layer_index: int,
    token_text: str | None,
) -> DefenseEvalResult:
    """Evaluate an optimized suffix against SIC and CAST defenses.

    For GCG/EGD attacks, the suffix (token_text) is appended to each
    prompt before passing to the defense modules. Continuous mode
    suffixes cannot be represented as text, so SIC is skipped and CAST
    uses the original prompts.

    Args:
        model: The causal language model.
        tokenizer: Tokenizer with chat template support.
        prompts: Original attack prompts.
        config: Soft prompt configuration with defense_eval settings.
        direction: Refusal direction vector (required for SIC/CAST).
        layer_index: Layer index for direction projection.
        token_text: Decoded suffix string (GCG/EGD), or None (continuous).

    Returns:
        DefenseEvalResult with SIC and CAST outcomes.
    """
    eval_mode = config.defense_eval or "both"
    layer = (
        config.defense_eval_layer
        if config.defense_eval_layer is not None
        else layer_index
    )

    # Build adversarial prompts with suffix appended
    if token_text is not None:
        adv_prompts = [p + " " + token_text for p in prompts]
    else:
        adv_prompts = list(prompts)

    # --- SIC evaluation ---
    sic_blocked = 0
    sic_sanitized = 0
    sic_clean = 0
    sic_bypass_rate = 0.0

    if eval_mode in ("sic", "both") and direction is not None:
        sic_blocked, sic_sanitized, sic_clean, sic_bypass_rate = (
            _eval_sic(
                model, tokenizer, adv_prompts, direction, layer,
                config.defense_eval_threshold,
                config.defense_eval_sic_mode,
                config.defense_eval_sic_max_iterations,
            )
        )

    # --- CAST evaluation ---
    cast_interventions = 0
    cast_refusal_rate = 0.0
    cast_responses: list[str] = []

    if eval_mode in ("cast", "both") and direction is not None:
        cast_layers = (
            config.defense_eval_cast_layers
            if config.defense_eval_cast_layers is not None
            else [layer]
        )
        cast_interventions, cast_refusal_rate, cast_responses = (
            _eval_cast(
                model, tokenizer, adv_prompts, direction,
                cast_layers,
                config.defense_eval_alpha,
                config.defense_eval_threshold,
                config.max_gen_tokens,
            )
        )

    return DefenseEvalResult(
        sic_blocked=sic_blocked,
        sic_sanitized=sic_sanitized,
        sic_clean=sic_clean,
        sic_bypass_rate=sic_bypass_rate,
        cast_interventions=cast_interventions,
        cast_refusal_rate=cast_refusal_rate,
        cast_responses=cast_responses,
    )


def _eval_sic(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompts: list[str],
    direction: Array,
    layer: int,
    threshold: float,
    sic_mode: str = "direction",
    sic_max_iterations: int = 3,
) -> tuple[int, int, int, float]:
    """Run SIC defense on adversarial prompts.

    Returns:
        Tuple of (blocked, sanitized, clean, bypass_rate).
    """
    from vauban.sic import sic

    sic_config = SICConfig(
        mode=sic_mode,
        threshold=threshold,
        max_iterations=sic_max_iterations,
        target_layer=layer,
    )
    result = sic(
        model, tokenizer, prompts, sic_config,
        direction=direction, layer_index=layer,
    )
    bypass_rate = (
        (result.total_sanitized + result.total_clean) / len(prompts)
        if prompts
        else 0.0
    )
    return (
        result.total_blocked,
        result.total_sanitized,
        result.total_clean,
        bypass_rate,
    )


def _eval_cast(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompts: list[str],
    direction: Array,
    layers: list[int],
    alpha: float,
    threshold: float,
    max_tokens: int,
) -> tuple[int, float, list[str]]:
    """Run CAST defense on adversarial prompts.

    Returns:
        Tuple of (total_interventions, refusal_rate, responses).
    """
    from vauban.cast import cast_generate

    total_interventions = 0
    refusals = 0
    responses: list[str] = []

    for prompt in prompts:
        result = cast_generate(
            model, tokenizer, prompt, direction,
            layers=layers, alpha=alpha,
            threshold=threshold, max_tokens=max_tokens,
        )
        total_interventions += result.interventions
        responses.append(result.text)

        lower = result.text.lower()
        if any(p.lower() in lower for p in DEFAULT_REFUSAL_PHRASES):
            refusals += 1

    refusal_rate = refusals / len(prompts) if prompts else 0.0
    return total_interventions, refusal_rate, responses
