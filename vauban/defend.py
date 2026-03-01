"""Defense stack composition — fail-fast pipeline of all defense layers.

Composes scan (Layer 0), SIC (Layer 1), CAST (Layer 2), policy (Layer 3),
and intent (Layer 4) into a unified defense interface.

Usage:
    result = defend_content(model, tok, content, direction, config)
    result = defend_tool_call(model, tok, tool_name, args, intent, config)
"""

from vauban._array import Array
from vauban.types import (
    CausalLM,
    DefenseStackConfig,
    DefenseStackResult,
    IntentCheckResult,
    IntentState,
    PolicyDecision,
    ScanResult,
    SICPromptResult,
    Tokenizer,
)


def defend_content(
    model: CausalLM,
    tokenizer: Tokenizer,
    content: str,
    direction: Array | None,
    config: DefenseStackConfig,
    layer_index: int = 0,
) -> DefenseStackResult:
    """Run defense layers 0-2 on retrieved content.

    Evaluates content through scan, SIC, and CAST in order.
    If ``fail_fast`` is True (default), stops at the first blocking layer.

    Args:
        model: The causal language model.
        tokenizer: Tokenizer with encode/decode support.
        content: Content to defend against (e.g. tool output).
        direction: Injection/refusal direction vector.
        config: Defense stack configuration.
        layer_index: Fallback layer index for direction-based layers.

    Returns:
        DefenseStackResult with per-layer outcomes.
    """
    reasons: list[str] = []
    scan_result: ScanResult | None = None
    sic_result: SICPromptResult | None = None

    # Layer 0: Injection scanner
    if config.scan is not None and direction is not None:
        from vauban.scan import scan

        scan_result = scan(
            model, tokenizer, content, config.scan, direction, layer_index,
        )
        if scan_result.flagged:
            reasons.append(
                f"Scan: injection detected"
                f" (p={scan_result.injection_probability:.3f},"
                f" {len(scan_result.spans)} spans)",
            )
            if config.fail_fast:
                return DefenseStackResult(
                    blocked=True,
                    layer_that_blocked="scan",
                    scan_result=scan_result,
                    reasons=reasons,
                )

    # Layer 1: SIC (input sanitization)
    if config.sic is not None:
        from vauban.sic import sic_single

        sic_result = sic_single(
            model, tokenizer, content, config.sic, direction, layer_index,
        )
        if sic_result.blocked:
            reasons.append("SIC: content blocked after sanitization failed")
            if config.fail_fast:
                return DefenseStackResult(
                    blocked=True,
                    layer_that_blocked="sic",
                    scan_result=scan_result,
                    sic_result=sic_result,
                    reasons=reasons,
                )

    # Layers 1-2 passed (CAST is applied at generation time, not here)
    blocked = bool(reasons)
    blocker = None
    if blocked and scan_result is not None and scan_result.flagged:
        blocker = "scan"
    elif blocked:
        blocker = "sic"

    return DefenseStackResult(
        blocked=blocked,
        layer_that_blocked=blocker,
        scan_result=scan_result,
        sic_result=sic_result,
        reasons=reasons,
    )


def defend_tool_call(
    model: CausalLM,
    tokenizer: Tokenizer,
    tool_name: str,
    arguments: dict[str, str],
    intent_state: IntentState | None,
    config: DefenseStackConfig,
    layer_index: int = 0,
) -> DefenseStackResult:
    """Run defense layers 3-4 on a proposed tool call.

    Evaluates the tool call against the policy engine (Layer 3) and
    intent alignment (Layer 4).

    Args:
        model: The causal language model.
        tokenizer: Tokenizer with encode/decode support.
        tool_name: Name of the tool being called.
        arguments: Tool call arguments.
        intent_state: Captured user intent state, or None.
        config: Defense stack configuration.
        layer_index: Fallback layer index.

    Returns:
        DefenseStackResult with per-layer outcomes.
    """
    reasons: list[str] = []
    policy_decision: PolicyDecision | None = None
    intent_check: IntentCheckResult | None = None

    # Layer 3: Policy engine
    if config.policy is not None:
        from vauban.policy import evaluate_tool_call

        policy_decision = evaluate_tool_call(
            tool_name, arguments, config.policy,
        )
        if policy_decision.action == "block":
            reasons.extend(policy_decision.reasons)
            if config.fail_fast:
                return DefenseStackResult(
                    blocked=True,
                    layer_that_blocked="policy",
                    policy_decision=policy_decision,
                    reasons=reasons,
                )

    # Layer 4: Intent alignment
    if config.intent is not None and intent_state is not None:
        from vauban.intent import check_alignment

        action_desc = (
            f"Call {tool_name} with arguments"
            f" {arguments}"
        )
        intent_check = check_alignment(
            model, tokenizer, action_desc, intent_state,
            config.intent, layer_index,
        )
        if not intent_check.aligned:
            reasons.append(
                f"Intent: action misaligned"
                f" (score={intent_check.score:.3f})",
            )
            if config.fail_fast:
                return DefenseStackResult(
                    blocked=True,
                    layer_that_blocked="intent",
                    policy_decision=policy_decision,
                    intent_check=intent_check,
                    reasons=reasons,
                )

    blocked = bool(reasons)
    blocker = None
    if blocked and policy_decision and policy_decision.action == "block":
        blocker = "policy"
    elif blocked and intent_check and not intent_check.aligned:
        blocker = "intent"

    return DefenseStackResult(
        blocked=blocked,
        layer_that_blocked=blocker,
        policy_decision=policy_decision,
        intent_check=intent_check,
        reasons=reasons,
    )
