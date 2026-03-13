"""Tests for flywheel attack matrix execution."""

from __future__ import annotations

from unittest.mock import patch

from vauban.flywheel._execute import execute_attack_matrix
from vauban.types import (
    AgentTurn,
    EnvironmentConfig,
    EnvironmentResult,
    EnvironmentTarget,
    EnvironmentTask,
    Payload,
    ToolCall,
    ToolSchema,
)


def _make_world() -> EnvironmentConfig:
    """Create a minimal world config for testing."""
    return EnvironmentConfig(
        system_prompt="You are a test assistant.",
        tools=[
            ToolSchema(
                name="test_tool",
                description="A test tool.",
                parameters={"arg": "string"},
            ),
        ],
        target=EnvironmentTarget(
            function="test_tool",
            required_args=["arg"],
        ),
        task=EnvironmentTask(content="Do something."),
        injection_surface="test",
    )


def _make_payload(text: str = "inject") -> Payload:
    """Create a test payload."""
    return Payload(text=text, source="test", cycle_discovered=0)


def _make_env_result(
    *,
    reward: float = 0.9,
    target_called: bool = True,
) -> EnvironmentResult:
    """Create a synthetic EnvironmentResult."""
    return EnvironmentResult(
        reward=reward,
        target_called=target_called,
        target_args_match=target_called,
        turns=[
            AgentTurn(role="assistant", content="ok"),
        ],
        tool_calls_made=[
            ToolCall(function="test_tool", arguments={"arg": "val"}),
        ],
        injection_payload="inject",
    )


class TestExecuteAttackMatrix:
    def test_produces_correct_count(self) -> None:
        worlds = [_make_world(), _make_world()]
        payloads = [_make_payload("a"), _make_payload("b")]

        with patch(
            "vauban.environment.run_agent_loop",
            return_value=_make_env_result(),
        ):
            traces = execute_attack_matrix(
                object(), object(),  # type: ignore[arg-type]
                worlds, payloads,
                max_turns=3, max_gen_tokens=50,
            )

        assert len(traces) == 4  # 2 worlds * 2 payloads

    def test_successful_trace_fields(self) -> None:
        result = _make_env_result(reward=0.8, target_called=True)

        with patch(
            "vauban.environment.run_agent_loop",
            return_value=result,
        ):
            traces = execute_attack_matrix(
                object(), object(),  # type: ignore[arg-type]
                [_make_world()], [_make_payload("test")],
                max_turns=3, max_gen_tokens=50,
            )

        assert len(traces) == 1
        t = traces[0]
        assert t.world_index == 0
        assert t.payload_index == 0
        assert t.payload_text == "test"
        assert t.reward == 0.8
        assert t.target_called is True
        assert t.turns_used == 1
        assert t.tool_calls_made == 1

    def test_exception_records_zero_reward(self) -> None:
        with patch(
            "vauban.environment.run_agent_loop",
            side_effect=RuntimeError("model crash"),
        ):
            traces = execute_attack_matrix(
                object(), object(),  # type: ignore[arg-type]
                [_make_world()], [_make_payload()],
                max_turns=3, max_gen_tokens=50,
            )

        assert len(traces) == 1
        t = traces[0]
        assert t.reward == 0.0
        assert t.target_called is False
        assert t.turns_used == 0
        assert t.tool_calls_made == 0

    def test_empty_worlds_returns_empty(self) -> None:
        traces = execute_attack_matrix(
            object(), object(),  # type: ignore[arg-type]
            [], [_make_payload()],
            max_turns=3, max_gen_tokens=50,
        )
        assert traces == []

    def test_empty_payloads_returns_empty(self) -> None:
        traces = execute_attack_matrix(
            object(), object(),  # type: ignore[arg-type]
            [_make_world()], [],
            max_turns=3, max_gen_tokens=50,
        )
        assert traces == []

    def test_mixed_success_and_failure(self) -> None:
        call_count = 0

        def alternating_loop(*_args: object, **_kwargs: object) -> EnvironmentResult:
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                msg = "fail"
                raise RuntimeError(msg)
            return _make_env_result(reward=0.7)

        with patch(
            "vauban.environment.run_agent_loop",
            side_effect=alternating_loop,
        ):
            traces = execute_attack_matrix(
                object(), object(),  # type: ignore[arg-type]
                [_make_world()], [_make_payload("a"), _make_payload("b")],
                max_turns=3, max_gen_tokens=50,
            )

        assert len(traces) == 2
        assert traces[0].reward == 0.7  # succeeded
        assert traces[1].reward == 0.0  # failed
