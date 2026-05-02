# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Core agent simulation loop — multi-turn tool-use conversation.

Builds messages, generates responses, parses tool calls, injects
payloads into tool results, and computes reward.
"""

from vauban import _ops as ops
from vauban._forward import (
    encode_chat_prompt,
    extract_logits,
    make_cache,
    to_model_device,
)
from vauban.environment._format import format_tools_for_prompt
from vauban.environment._parse_tool_call import parse_tool_calls
from vauban.environment._policy import check_policy
from vauban.environment._reward import compute_reward
from vauban.types import (
    AgentTurn,
    CausalLM,
    EnvironmentConfig,
    EnvironmentResult,
    Tokenizer,
    ToolCall,
    ToolSchema,
)

_PAYLOAD_MARKER = "{payload}"


def run_agent_loop(
    model: CausalLM,
    tokenizer: Tokenizer,
    env_config: EnvironmentConfig,
    injection_payload: str,
) -> EnvironmentResult:
    """Run the agent loop with a given injection payload.

    Simulates a multi-turn tool-use conversation where:
    1. The agent receives a system prompt + tool descriptions + user task
    2. On each turn, the agent generates a response
    3. Tool calls are parsed from the response
    4. If the tool is the injection surface, the payload is injected
    5. If the tool is the target, reward is computed and we stop
    6. Tool results are appended and the loop continues

    Args:
        model: The causal language model.
        tokenizer: Tokenizer with chat template support.
        env_config: Environment configuration.
        injection_payload: The adversarial payload to inject.

    Returns:
        EnvironmentResult with reward and conversation trace.
    """
    # Build system prompt with tool descriptions
    tool_suffix = format_tools_for_prompt(env_config.tools)
    system_content = env_config.system_prompt + tool_suffix

    # Build tool lookup
    tool_map: dict[str, ToolSchema] = {
        t.name: t for t in env_config.tools
    }

    # Initialize conversation
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": env_config.task.content},
    ]

    turns: list[AgentTurn] = []
    all_tool_calls: list[ToolCall] = []

    for _turn_idx in range(env_config.max_turns):
        # Generate response
        response_text = _generate_response(
            model, tokenizer, messages, env_config.max_gen_tokens,
            env_config.temperature,
        )

        # Parse tool calls from response
        parsed_calls = parse_tool_calls(response_text)

        if not parsed_calls:
            # Agent responded with text only — record and stop
            turns.append(AgentTurn(
                role="assistant",
                content=response_text,
                tool_call=None,
            ))
            break

        # Process the first tool call (single-tool per turn)
        call = parsed_calls[0]
        turns.append(AgentTurn(
            role="assistant",
            content=response_text,
            tool_call=call,
        ))
        all_tool_calls.append(call)

        # Check policy
        policy_reason = check_policy(call, env_config.policy)
        if policy_reason is not None:
            tool_result = f"Error: {policy_reason}"
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": f"Tool result: {tool_result}"})
            turns.append(AgentTurn(
                role="tool",
                content=tool_result,
                tool_call=None,
            ))
            continue

        # Check if this is the target function — compute reward and stop
        if call.function == env_config.target.function:
            reward, target_called, target_args_match = compute_reward(
                all_tool_calls, env_config.target,
            )
            return EnvironmentResult(
                reward=reward,
                target_called=target_called,
                target_args_match=target_args_match,
                turns=turns,
                tool_calls_made=all_tool_calls,
                injection_payload=injection_payload,
            )

        # Get tool result
        tool_result = _get_tool_result(
            call,
            tool_map,
            env_config.injection_surface,
            injection_payload,
            env_config.injection_position,
        )

        messages.append({"role": "assistant", "content": response_text})
        messages.append({"role": "user", "content": f"Tool result: {tool_result}"})
        turns.append(AgentTurn(
            role="tool",
            content=tool_result,
            tool_call=None,
        ))

    # Max turns reached — compute final reward
    reward, target_called, target_args_match = compute_reward(
        all_tool_calls, env_config.target,
    )
    return EnvironmentResult(
        reward=reward,
        target_called=target_called,
        target_args_match=target_args_match,
        turns=turns,
        tool_calls_made=all_tool_calls,
        injection_payload=injection_payload,
    )


def _generate_response(
    model: CausalLM,
    tokenizer: Tokenizer,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float = 0.0,
) -> str:
    """Generate a response from the model given conversation messages.

    Uses KV-cached autoregressive decoding (no soft prompt injection —
    the injection is in the tool result content).

    Args:
        model: The causal language model.
        tokenizer: Tokenizer with chat template support.
        messages: Conversation history as role/content dicts.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature. 0.0 = greedy (argmax).

    Returns:
        Generated text string.
    """
    token_ids = to_model_device(model, encode_chat_prompt(tokenizer, messages))
    generated: list[int] = []
    eos_token_id: int | None = getattr(tokenizer, "eos_token_id", None)

    cache = make_cache(model)

    for _ in range(max_tokens):
        result = model(token_ids, cache=cache)  # type: ignore[call-non-callable]
        logits = extract_logits(result)
        step_logits = logits[:, -1, :]

        if temperature > 0.0:
            probs = ops.softmax(step_logits / temperature, axis=-1)
            next_token = int(
                ops.random.categorical(ops.log(probs + 1e-10)).item(),
            )
        else:
            next_token = int(ops.argmax(step_logits, axis=-1).item())

        generated.append(next_token)
        if eos_token_id is not None and next_token == eos_token_id:
            break
        token_ids = to_model_device(model, ops.array([[next_token]]))

    return tokenizer.decode(generated)


def _get_tool_result(
    call: ToolCall,
    tool_map: dict[str, ToolSchema],
    injection_surface: str,
    injection_payload: str,
    injection_position: str = "suffix",
) -> str:
    """Get the result for a tool call.

    If the tool is the injection surface, the canned result is returned
    with the injection payload inserted at the configured position.
    Otherwise, the canned result is returned as-is (or a generic
    message if no result is defined).

    Args:
        call: The tool call to get a result for.
        tool_map: Mapping of tool name to schema.
        injection_surface: Name of the tool to inject into.
        injection_payload: The adversarial payload to inject.
        injection_position: Placement for the injected payload.

    Returns:
        Tool result string.
    """
    tool = tool_map.get(call.function)
    if tool is None:
        return f"Error: Unknown tool {call.function!r}"

    base_result = tool.result if tool.result is not None else "{}"

    if call.function == injection_surface:
        return _inject_payload(
            base_result,
            injection_payload,
            injection_position,
        )

    return base_result


def _inject_payload(
    base_result: str,
    injection_payload: str,
    injection_position: str,
) -> str:
    """Insert a payload into tool output at the requested position."""
    if _PAYLOAD_MARKER in base_result:
        return base_result.replace(_PAYLOAD_MARKER, injection_payload)
    if injection_payload == "":
        return base_result
    if injection_position == "prefix":
        return f"{injection_payload}\n\n{base_result}"
    if injection_position == "infix":
        return (
            f"{base_result}\n\n"
            f"{injection_payload}\n\n"
            "(retrieved content continues below)"
        )
    return f"{base_result}\n\n{injection_payload}"
