"""Intent alignment module — check if actions align with user intent.

Two modes:
- **embedding**: cosine similarity between user-intent activation and
  current-action activation at target layer.
- **judge**: LLM-as-judge with formatted prompt checking ALIGNED/MISALIGNED.
"""

from typing import TYPE_CHECKING

from vauban import _ops as ops
from vauban._forward import embed_and_mask, force_eval, make_cache

if TYPE_CHECKING:
    from vauban._array import Array

from vauban.types import (
    CausalLM,
    IntentCheckResult,
    IntentConfig,
    IntentState,
    Tokenizer,
)


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

    # Embedding mode: run forward pass
    messages = [{"role": "user", "content": user_request}]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    if not isinstance(text, str):
        msg = "apply_chat_template must return str when tokenize=False"
        raise TypeError(msg)
    token_ids = ops.array(tokenizer.encode(text))[None, :]

    transformer = model.model
    h, mask = embed_and_mask(transformer, token_ids)

    activation: Array | None = None
    for i, layer_module in enumerate(transformer.layers):
        h = layer_module(h, mask)
        if i == target_layer:
            activation = h[0, -1, :]
            force_eval(activation)
            break

    if activation is None:
        # Fallback to last layer
        activation = h[0, -1, :]
        force_eval(activation)

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

    # Get action activation
    messages = [{"role": "user", "content": action_description}]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    if not isinstance(text, str):
        msg = "apply_chat_template must return str when tokenize=False"
        raise TypeError(msg)
    token_ids = ops.array(tokenizer.encode(text))[None, :]

    transformer = model.model
    h, mask = embed_and_mask(transformer, token_ids)

    action_activation: Array | None = None
    for i, layer_module in enumerate(transformer.layers):
        h = layer_module(h, mask)
        if i == target_layer:
            action_activation = h[0, -1, :]
            force_eval(action_activation)
            break

    if action_activation is None:
        action_activation = h[0, -1, :]
        force_eval(action_activation)

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

    input_ids = tokenizer.encode(text)
    input_array = ops.array(input_ids)[None, :]

    transformer = model.model
    cache = make_cache(model)

    # Prefill
    h = transformer.embed_tokens(input_array)
    mask = ops.create_additive_causal_mask(h.shape[1])

    for i, layer in enumerate(transformer.layers):
        h = layer(h, mask, cache=cache[i])

    h = transformer.norm(h)

    if hasattr(model, "lm_head"):
        logits = model.lm_head(h[:, -1:, :])  # type: ignore[attr-defined]
    else:
        logits = transformer.embed_tokens.as_linear(h[:, -1:, :])  # type: ignore[union-attr]
    force_eval(logits)

    eos_token_id: int | None = getattr(tokenizer, "eos_token_id", None)

    generated_ids: list[int] = []
    for _ in range(config.max_tokens):
        next_token = int(ops.argmax(logits[:, -1, :], axis=-1).item())
        if eos_token_id is not None and next_token == eos_token_id:
            break
        generated_ids.append(next_token)

        h = transformer.embed_tokens(ops.array([[next_token]]))
        for i, layer in enumerate(transformer.layers):
            h = layer(h, None, cache=cache[i])
        h = transformer.norm(h)

        if hasattr(model, "lm_head"):
            logits = model.lm_head(h)  # type: ignore[attr-defined]
        else:
            logits = transformer.embed_tokens.as_linear(h)  # type: ignore[union-attr]
        force_eval(logits)

    response = tokenizer.decode(generated_ids).strip().upper()
    aligned = "ALIGNED" in response and "MISALIGNED" not in response
    score = 1.0 if aligned else 0.0

    return IntentCheckResult(
        aligned=aligned,
        score=score,
        mode="judge",
    )
