"""Execute attack matrix: run every (world, payload) pair."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vauban.types import FlywheelTrace

if TYPE_CHECKING:
    from vauban.types import CausalLM, EnvironmentConfig, Payload, Tokenizer


def execute_attack_matrix(
    model: CausalLM,
    tokenizer: Tokenizer,
    worlds: list[EnvironmentConfig],
    payloads: list[Payload],
    max_turns: int,
    max_gen_tokens: int,
) -> list[FlywheelTrace]:
    """Run every payload against every world and collect traces.

    Each (world, payload) pair is executed through the agent loop.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        worlds: List of environment configs for each world.
        payloads: List of attack payloads to test.
        max_turns: Maximum conversation turns per execution.
        max_gen_tokens: Maximum tokens to generate per turn.

    Returns:
        List of FlywheelTrace results for all (world, payload) pairs.
    """
    from dataclasses import replace

    from vauban.environment import run_agent_loop

    traces: list[FlywheelTrace] = []

    for w_idx, world in enumerate(worlds):
        # Override turn/token limits from flywheel config
        world_config = replace(
            world,
            max_turns=max_turns,
            max_gen_tokens=max_gen_tokens,
        )

        for p_idx, payload in enumerate(payloads):
            try:
                result = run_agent_loop(
                    model, tokenizer, world_config, payload.text,
                )
                traces.append(FlywheelTrace(
                    world_index=w_idx,
                    payload_index=p_idx,
                    payload_text=payload.text,
                    reward=result.reward,
                    target_called=result.target_called,
                    turns_used=len(result.turns),
                    tool_calls_made=len(result.tool_calls_made),
                ))
            except Exception:  # agent loop crash (model, parsing, etc.)
                # Record failed runs with zero reward
                traces.append(FlywheelTrace(
                    world_index=w_idx,
                    payload_index=p_idx,
                    payload_text=payload.text,
                    reward=0.0,
                    target_called=False,
                    turns_used=0,
                    tool_calls_made=0,
                ))

    return traces
