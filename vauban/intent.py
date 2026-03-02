"""Intent alignment module — check if actions align with user intent.

Two modes:
- **embedding**: cosine similarity between user-intent activation and
  current-action activation at target layer.
- **judge**: LLM-as-judge with formatted prompt checking ALIGNED/MISALIGNED.
"""

from typing import TYPE_CHECKING

from vauban import _ops as ops
from vauban._forward import (
    embed_and_mask,
    extract_logits,
    force_eval,
    get_transformer,
    make_cache,
    make_ssm_mask,
    select_mask,
)

if TYPE_CHECKING:
    from vauban._array import Array

from vauban.types import (
    CausalLM,
    IntentCheckResult,
    IntentConfig,
    IntentState,
    Tokenizer,
)


def _extract_activation_at_layer(
    model: CausalLM,
    tokenizer: Tokenizer,
    text_content: str,
    target_layer: int,
) -> "Array":
    """Run a forward pass and extract the last-token activation at target_layer.

    Shared helper for ``capture_intent`` and ``_check_alignment_embedding``.

    Args:
        model: The causal language model.
        tokenizer: Tokenizer with chat template support.
        text_content: Text to process (already formatted as user message).
        target_layer: Layer index to extract activation from.

    Returns:
        Activation array of shape (d_model,).
    """
    messages = [{"role": "user", "content": text_content}]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    if not isinstance(text, str):
        msg = "apply_chat_template must return str when tokenize=False"
        raise TypeError(msg)
    token_ids = ops.array(tokenizer.encode(text))[None, :]

    transformer = get_transformer(model)
    h, mask = embed_and_mask(transformer, token_ids)

    activation: Array | None = None
    ssm_mask = make_ssm_mask(transformer, h)
    for i, layer_module in enumerate(transformer.layers):
        h = layer_module(h, select_mask(layer_module, mask, ssm_mask))
        if i == target_layer:
            activation = h[0, -1, :]
            force_eval(activation)
            break

    if activation is None:
        activation = h[0, -1, :]
        force_eval(activation)

    return activation


def capture_intent(
    model: CausalLM,
    tokenizer: Tokenizer,
    user_request: str,
    config: IntentConfig,
    layer_index: int = 0,
) -> IntentState:
    """Capture the user's intent as a hidden-state activation.

    Runs a forward pass on the user request and extracts the
    activation at the target layer's last token position.

    Args:
        model: The causal language model.
        tokenizer: Tokenizer with encode/decode support.
        user_request: The original user request text.
        config: Intent configuration.
        layer_index: Fallback layer index.

    Returns:
        IntentState with the user request and captured activation.
    """
    target_layer = (
        config.target_layer if config.target_layer is not None
        else layer_index
    )

    if config.mode == "judge":
        # Judge mode doesn't need an activation
        return IntentState(
            user_request=user_request,
            activation=None,
        )

    # Embedding mode: run forward pass and extract activation
    activation = _extract_activation_at_layer(
        model, tokenizer, user_request, target_layer,
    )

    return IntentState(
        user_request=user_request,
        activation=activation,
    )


def check_alignment(
    model: CausalLM,
    tokenizer: Tokenizer,
    action_description: str,
    intent_state: IntentState,
    config: IntentConfig,
    layer_index: int = 0,
) -> IntentCheckResult:
    """Check if an action aligns with the captured user intent.

    Args:
        model: The causal language model.
        tokenizer: Tokenizer with encode/decode support.
        action_description: Description of the action to check.
        intent_state: Previously captured user intent.
        config: Intent configuration.
        layer_index: Fallback layer index.

    Returns:
        IntentCheckResult with alignment verdict and score.
    """
    if config.mode == "judge":
        return _check_alignment_judge(
            model, tokenizer, action_description, intent_state, config,
        )

    return _check_alignment_embedding(
        model, tokenizer, action_description, intent_state, config,
        layer_index,
    )


def _check_alignment_embedding(
    model: CausalLM,
    tokenizer: Tokenizer,
    action_description: str,
    intent_state: IntentState,
    config: IntentConfig,
    layer_index: int,
) -> IntentCheckResult:
    """Check alignment via cosine similarity of activations."""
    target_layer = (
        config.target_layer if config.target_layer is not None
        else layer_index
    )

    if intent_state.activation is None:
        return IntentCheckResult(
            aligned=False,
            score=0.0,
            mode="embedding",
        )

    # Get action activation at the same layer
    action_activation = _extract_activation_at_layer(
        model, tokenizer, action_description, target_layer,
    )

    # Cosine similarity
    intent_act = intent_state.activation
    dot = float(ops.sum(intent_act * action_activation).item())
    norm_intent = float(
        ops.sqrt(ops.sum(intent_act * intent_act)).item(),
    )
    norm_action = float(
        ops.sqrt(ops.sum(action_activation * action_activation)).item(),
    )

    if norm_intent < 1e-8 or norm_action < 1e-8:
        score = 0.0
    else:
        score = dot / (norm_intent * norm_action)

    aligned = score >= config.similarity_threshold

    return IntentCheckResult(
        aligned=aligned,
        score=score,
        mode="embedding",
    )


def _check_alignment_judge(
    model: CausalLM,
    tokenizer: Tokenizer,
    action_description: str,
    intent_state: IntentState,
    config: IntentConfig,
) -> IntentCheckResult:
    """Check alignment via LLM-as-judge generation."""
    prompt = (
        f"{config.judge_prompt}\n\n"
        f"User request: {intent_state.user_request}\n"
        f"Proposed action: {action_description}\n\n"
        f"Verdict:"
    )

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    if not isinstance(text, str):
        msg = "apply_chat_template must return str when tokenize=False"
        raise TypeError(msg)

    tokens = tokenizer.encode(text)
    generated: list[int] = []
    eos_token_id: int | None = getattr(tokenizer, "eos_token_id", None)

    cache = make_cache(model)
    token_ids = ops.array([tokens])

    for _ in range(config.max_tokens):
        result = model(token_ids, cache=cache)  # type: ignore[call-non-callable]
        logits = extract_logits(result)
        next_token = int(ops.argmax(logits[:, -1, :], axis=-1).item())
        generated.append(next_token)
        if eos_token_id is not None and next_token == eos_token_id:
            break
        token_ids = ops.array([[next_token]])

    response = tokenizer.decode(generated).strip().upper()
    aligned = "ALIGNED" in response and "MISALIGNED" not in response
    score = 1.0 if aligned else 0.0

    return IntentCheckResult(
        aligned=aligned,
        score=score,
        mode="judge",
    )
