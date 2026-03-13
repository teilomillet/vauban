"""Run the environment harness with flywheel defenses applied."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vauban.environment._format import format_tools_for_prompt
from vauban.environment._loop import _generate_response, _get_tool_result
from vauban.environment._parse_tool_call import parse_tool_calls
from vauban.environment._policy import check_policy
from vauban.environment._reward import compute_reward
from vauban.types import AgentTurn, EnvironmentResult, SICConfig

if TYPE_CHECKING:
    from vauban._array import Array
    from vauban.types import (
        CausalLM,
        EnvironmentConfig,
        FlywheelDefenseParams,
        Tokenizer,
        ToolCall,
    )


@dataclass(frozen=True, slots=True)
class DefendedEnvironmentResult:
    """Environment result augmented with defense telemetry."""

    env_result: EnvironmentResult
    cast_interventions: int
    sic_blocked: bool


def run_defended_agent_loop(
    model: CausalLM,
    tokenizer: Tokenizer,
    env_config: EnvironmentConfig,
    injection_payload: str,
    direction: Array | None,
    layer_index: int,
    defense_params: FlywheelDefenseParams,
) -> DefendedEnvironmentResult:
    """Run the environment loop with CAST on generation and SIC on content."""

    tool_suffix = format_tools_for_prompt(env_config.tools)
    system_content = env_config.system_prompt + tool_suffix
    tool_map = {tool.name: tool for tool in env_config.tools}

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": env_config.task.content},
    ]
    turns: list[AgentTurn] = []
    all_tool_calls: list[ToolCall] = []
    cast_interventions = 0
    sic_blocked = False

    for _turn_idx in range(env_config.max_turns):
        response_text, turn_cast_interventions = _generate_defended_response(
            model,
            tokenizer,
            messages,
            env_config.max_gen_tokens,
            env_config.temperature,
            direction,
            layer_index,
            defense_params,
        )
        cast_interventions += turn_cast_interventions

        parsed_calls = parse_tool_calls(response_text)
        if not parsed_calls:
            turns.append(AgentTurn(
                role="assistant",
                content=response_text,
                tool_call=None,
            ))
            break

        call = parsed_calls[0]
        turns.append(AgentTurn(
            role="assistant",
            content=response_text,
            tool_call=call,
        ))
        all_tool_calls.append(call)

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

        if call.function == env_config.target.function:
            return _finalize_result(
                turns,
                all_tool_calls,
                injection_payload,
                cast_interventions,
                sic_blocked,
                env_config,
            )

        tool_result = _get_tool_result(
            call,
            tool_map,
            env_config.injection_surface,
            injection_payload,
            env_config.injection_position,
        )

        if _should_run_sic(direction, defense_params):
            from vauban.sic import sic_single

            sic_result = sic_single(
                model,
                tokenizer,
                tool_result,
                _build_sic_config(defense_params),
                direction,
                layer_index,
            )
            if sic_result.blocked:
                sic_blocked = True
                turns.append(AgentTurn(
                    role="tool",
                    content="Tool result blocked by SIC.",
                    tool_call=None,
                ))
                return _finalize_result(
                    turns,
                    all_tool_calls,
                    injection_payload,
                    cast_interventions,
                    sic_blocked,
                    env_config,
                )
            tool_result = sic_result.clean_prompt

        messages.append({"role": "assistant", "content": response_text})
        messages.append({"role": "user", "content": f"Tool result: {tool_result}"})
        turns.append(AgentTurn(
            role="tool",
            content=tool_result,
            tool_call=None,
        ))

    return _finalize_result(
        turns,
        all_tool_calls,
        injection_payload,
        cast_interventions,
        sic_blocked,
        env_config,
    )


def _generate_defended_response(
    model: CausalLM,
    tokenizer: Tokenizer,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    direction: Array | None,
    layer_index: int,
    defense_params: FlywheelDefenseParams,
) -> tuple[str, int]:
    """Generate the next assistant response, applying CAST when available."""

    cast_layers = (
        defense_params.cast_layers
        if defense_params.cast_layers is not None
        else [layer_index]
    )
    if direction is None or len(cast_layers) == 0:
        return _generate_response(
            model,
            tokenizer,
            messages,
            max_tokens,
            temperature,
        ), 0

    try:
        from vauban.cast import cast_generate_with_messages

        result = cast_generate_with_messages(
            model,
            tokenizer,
            messages,
            direction,
            layers=cast_layers,
            alpha=defense_params.cast_alpha,
            threshold=defense_params.cast_threshold,
            max_tokens=max_tokens,
        )
        return result.text, result.interventions
    except Exception:  # CAST failure — fall back to undefended generation
        return _generate_response(
            model,
            tokenizer,
            messages,
            max_tokens,
            temperature,
        ), 0


def _build_sic_config(defense_params: FlywheelDefenseParams) -> SICConfig:
    """Build a SICConfig from the mutable flywheel defense parameters."""

    return SICConfig(
        mode=defense_params.sic_mode,
        threshold=defense_params.sic_threshold,
        max_iterations=defense_params.sic_iterations,
    )


def _should_run_sic(
    direction: Array | None,
    defense_params: FlywheelDefenseParams,
) -> bool:
    """Return whether SIC can run with the current defense settings."""

    return defense_params.sic_mode != "direction" or direction is not None


def _finalize_result(
    turns: list[AgentTurn],
    all_tool_calls: list[ToolCall],
    injection_payload: str,
    cast_interventions: int,
    sic_blocked: bool,
    env_config: EnvironmentConfig,
) -> DefendedEnvironmentResult:
    """Package the final defended-loop result."""

    reward, target_called, target_args_match = compute_reward(
        all_tool_calls,
        env_config.target,
    )
    env_result = EnvironmentResult(
        reward=reward,
        target_called=target_called,
        target_args_match=target_args_match,
        turns=turns,
        tool_calls_made=all_tool_calls,
        injection_payload=injection_payload,
    )
    return DefendedEnvironmentResult(
        env_result=env_result,
        cast_interventions=cast_interventions,
        sic_blocked=sic_blocked,
    )
