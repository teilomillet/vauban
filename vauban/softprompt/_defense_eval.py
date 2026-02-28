"""Post-attack defense evaluation: test suffixes against SIC and CAST."""

from vauban._array import Array
from vauban.evaluate import DEFAULT_REFUSAL_PHRASES
from vauban.types import (
    AlphaTier,
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
        alpha_tiers: list[AlphaTier] | None = None
        if config.defense_eval_alpha_tiers is not None:
            alpha_tiers = [
                AlphaTier(threshold=t, alpha=a)
                for t, a in config.defense_eval_alpha_tiers
            ]
        cast_interventions, cast_refusal_rate, cast_responses = (
            _eval_cast(
                model, tokenizer, adv_prompts, direction,
                cast_layers,
                config.defense_eval_alpha,
                config.defense_eval_threshold,
                config.max_gen_tokens,
                alpha_tiers=alpha_tiers,
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


def evaluate_against_defenses_multiturn(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompts: list[str],
    config: SoftPromptConfig,
    direction: Array | None,
    layer_index: int,
    token_text: str | None,
    history: list[dict[str, str]],
) -> DefenseEvalResult:
    """Evaluate a suffix against defenses with conversation history context.

    Like ``evaluate_against_defenses`` but passes the full multi-turn
    history to SIC (as a serialized string) and CAST (as a message list).

    Args:
        model: The causal language model.
        tokenizer: Tokenizer with chat template support.
        prompts: Attack prompts for the current turn.
        config: Soft prompt configuration with defense_eval settings.
        direction: Refusal direction vector (required for SIC/CAST).
        layer_index: Layer index for direction projection.
        token_text: Decoded suffix string (GCG/EGD), or None (continuous).
        history: Previous conversation turns as role/content dicts.

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

    # For SIC: serialize history + current prompt into a single string
    # SIC works on text, so we concatenate history turns as context
    sic_blocked = 0
    sic_sanitized = 0
    sic_clean = 0
    sic_bypass_rate = 0.0

    if eval_mode in ("sic", "both") and direction is not None:
        sic_prompts = _build_sic_prompts_with_history(adv_prompts, history)
        sic_blocked, sic_sanitized, sic_clean, sic_bypass_rate = (
            _eval_sic(
                model, tokenizer, sic_prompts, direction, layer,
                config.defense_eval_threshold,
                config.defense_eval_sic_mode,
                config.defense_eval_sic_max_iterations,
            )
        )

    # For CAST: pass full message list
    cast_interventions = 0
    cast_refusal_rate = 0.0
    cast_responses: list[str] = []

    if eval_mode in ("cast", "both") and direction is not None:
        cast_layers = (
            config.defense_eval_cast_layers
            if config.defense_eval_cast_layers is not None
            else [layer]
        )
        alpha_tiers: list[AlphaTier] | None = None
        if config.defense_eval_alpha_tiers is not None:
            alpha_tiers = [
                AlphaTier(threshold=t, alpha=a)
                for t, a in config.defense_eval_alpha_tiers
            ]
        cast_interventions, cast_refusal_rate, cast_responses = (
            _eval_cast_multiturn(
                model, tokenizer, adv_prompts, direction,
                cast_layers,
                config.defense_eval_alpha,
                config.defense_eval_threshold,
                config.max_gen_tokens,
                history,
                alpha_tiers=alpha_tiers,
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


def _build_sic_prompts_with_history(
    prompts: list[str],
    history: list[dict[str, str]],
) -> list[str]:
    """Serialize conversation history into prompt strings for SIC.

    SIC operates on plain text prompts. We prepend the history as
    a formatted conversation to give SIC the full context.

    Args:
        prompts: Current-turn adversarial prompts.
        history: Previous conversation turns.

    Returns:
        List of prompts with history context prepended.
    """
    if not history:
        return list(prompts)

    history_lines: list[str] = []
    for turn in history:
        role = turn["role"]
        content = turn["content"]
        history_lines.append(f"[{role}]: {content}")
    history_text = "\n".join(history_lines)

    return [f"{history_text}\n[user]: {p}" for p in prompts]


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
    alpha_tiers: list[AlphaTier] | None = None,
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
            alpha_tiers=alpha_tiers,
        )
        total_interventions += result.interventions
        responses.append(result.text)

        lower = result.text.lower()
        if any(p.lower() in lower for p in DEFAULT_REFUSAL_PHRASES):
            refusals += 1

    refusal_rate = refusals / len(prompts) if prompts else 0.0
    return total_interventions, refusal_rate, responses


def _eval_cast_multiturn(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompts: list[str],
    direction: Array,
    layers: list[int],
    alpha: float,
    threshold: float,
    max_tokens: int,
    history: list[dict[str, str]],
    alpha_tiers: list[AlphaTier] | None = None,
) -> tuple[int, float, list[str]]:
    """Run CAST defense on adversarial prompts with conversation history.

    Passes the full message list (history + current prompt) to CAST
    so steering decisions consider the multi-turn context.

    Returns:
        Tuple of (total_interventions, refusal_rate, responses).
    """
    from vauban.cast import cast_generate_with_messages

    total_interventions = 0
    refusals = 0
    responses: list[str] = []

    for prompt in prompts:
        messages = [*history, {"role": "user", "content": prompt}]
        result = cast_generate_with_messages(
            model, tokenizer, messages, direction,
            layers=layers, alpha=alpha,
            threshold=threshold, max_tokens=max_tokens,
            alpha_tiers=alpha_tiers,
        )
        total_interventions += result.interventions
        responses.append(result.text)

        lower = result.text.lower()
        if any(p.lower() in lower for p in DEFAULT_REFUSAL_PHRASES):
            refusals += 1

    refusal_rate = refusals / len(prompts) if prompts else 0.0
    return total_interventions, refusal_rate, responses
