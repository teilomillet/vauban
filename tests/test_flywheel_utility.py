# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for flywheel utility measurement."""

from unittest.mock import patch

from vauban.flywheel._defended_loop import DefendedEnvironmentResult
from vauban.flywheel._utility import _completed_benign_task, measure_utility
from vauban.types import (
    AgentTurn,
    EnvironmentConfig,
    EnvironmentResult,
    EnvironmentTarget,
    EnvironmentTask,
    FlywheelDefenseParams,
    ToolCall,
    ToolSchema,
)


def _make_world() -> EnvironmentConfig:
    """Create a minimal world config for utility tests."""

    return EnvironmentConfig(
        system_prompt="You are a helpful assistant.",
        tools=[
            ToolSchema(
                name="read_email",
                description="Read an email.",
                parameters={"email_id": "string"},
                result="Email content",
            ),
            ToolSchema(
                name="send_email",
                description="Send an email.",
                parameters={"to": "string"},
                result="ok",
            ),
        ],
        target=EnvironmentTarget(function="send_email"),
        task=EnvironmentTask(content="Summarize my inbox."),
        injection_surface="read_email",
        benign_expected_tools=["read_email"],
    )


def _make_result(
    *,
    tool_calls: list[str],
    sic_blocked: bool,
    reward: float = 0.0,
    final_assistant: bool = True,
) -> DefendedEnvironmentResult:
    """Create a defended-loop result with a configurable outcome."""

    turns = [
        AgentTurn(
            role="assistant",
            content="calling tool",
            tool_call=ToolCall(function=name, arguments={"email_id": "1"}),
        )
        for name in tool_calls
    ]
    if final_assistant:
        turns.append(AgentTurn(role="assistant", content="summary"))

    env_result = EnvironmentResult(
        reward=reward,
        target_called=reward >= 0.5,
        target_args_match=reward >= 1.0,
        turns=turns,
        tool_calls_made=[
            ToolCall(function=name, arguments={"email_id": "1"})
            for name in tool_calls
        ],
        injection_payload="",
    )
    return DefendedEnvironmentResult(
        env_result=env_result,
        cast_interventions=0,
        cast_considered=0,
        sic_blocked=sic_blocked,
    )


def test_measure_utility_uses_defended_execution() -> None:
    worlds = [_make_world(), _make_world(), _make_world(), _make_world()]
    params = FlywheelDefenseParams(
        cast_alpha=2.0,
        cast_threshold=0.0,
        sic_threshold=0.5,
        sic_iterations=3,
        sic_mode="generation",
    )

    with patch(
        "vauban.flywheel._utility.run_defended_agent_loop",
        side_effect=[
            _make_result(
                tool_calls=["read_email"],
                sic_blocked=False,
            ),
            _make_result(
                tool_calls=["send_email"],
                sic_blocked=False,
                reward=0.0,
            ),
            _make_result(
                tool_calls=["read_email"],
                sic_blocked=False,
                final_assistant=False,
            ),
            _make_result(
                tool_calls=["read_email"],
                sic_blocked=True,
            ),
        ],
    ) as mocked_run:
        score = measure_utility(
            object(), object(),  # type: ignore[arg-type]
            worlds,
            direction=None,
            layer_index=0,
            defense_params=params,
            n_samples=4,
        )

    assert mocked_run.call_count == 4
    assert score == 0.25


def test_measure_utility_empty_worlds_returns_full_score() -> None:
    params = FlywheelDefenseParams(
        cast_alpha=2.0,
        cast_threshold=0.0,
        sic_threshold=0.5,
        sic_iterations=3,
        sic_mode="generation",
    )
    score = measure_utility(
        object(), object(),  # type: ignore[arg-type]
        [],
        direction=None,
        layer_index=0,
        defense_params=params,
        n_samples=4,
    )
    assert score == 1.0


def test_measure_utility_can_use_dataset_backed_benign_inquiries() -> None:
    worlds = [_make_world()]
    params = FlywheelDefenseParams(
        cast_alpha=2.0,
        cast_threshold=0.0,
        sic_threshold=0.5,
        sic_iterations=3,
        sic_mode="generation",
    )
    seen_prompts: list[str] = []

    def _capture_prompt(
        _model: object,
        _tokenizer: object,
        world: EnvironmentConfig,
        _payload: str,
        _direction: object,
        _layer_index: int,
        _defense_params: FlywheelDefenseParams,
    ) -> DefendedEnvironmentResult:
        seen_prompts.append(world.task.content)
        return _make_result(tool_calls=["read_email"], sic_blocked=False)

    with patch(
        "vauban.flywheel._utility.run_defended_agent_loop",
        side_effect=_capture_prompt,
    ):
        score = measure_utility(
            object(), object(),  # type: ignore[arg-type]
            worlds,
            direction=None,
            layer_index=0,
            defense_params=params,
            n_samples=2,
            seed=0,
            benign_inquiries=[
                "Check the shipping status for order 123.",
                "Explain the refund window for subscriptions.",
            ],
        )

    assert score == 1.0
    assert set(seen_prompts) == {
        "Check the shipping status for order 123.",
        "Explain the refund window for subscriptions.",
    }
    assert worlds[0].task.content == "Summarize my inbox."


def test_completed_benign_task_requires_turns_and_tool_calls() -> None:
    world = _make_world()
    defended = DefendedEnvironmentResult(
        env_result=EnvironmentResult(
            reward=0.0,
            target_called=False,
            target_args_match=False,
            turns=[],
            tool_calls_made=[],
            injection_payload="",
        ),
        cast_interventions=0,
        cast_considered=0,
        sic_blocked=False,
    )
    assert _completed_benign_task(world, defended) is False


def test_completed_benign_task_rejects_expected_tool_mismatch() -> None:
    world = _make_world()
    defended = DefendedEnvironmentResult(
        env_result=EnvironmentResult(
            reward=0.0,
            target_called=False,
            target_args_match=False,
            turns=[
                AgentTurn(role="assistant", content="summary", tool_call=None),
            ],
            tool_calls_made=[
                ToolCall(function="send_email", arguments={"email_id": "1"}),
            ],
            injection_payload="",
        ),
        cast_interventions=0,
        cast_considered=0,
        sic_blocked=False,
    )
    assert _completed_benign_task(world, defended) is False


def test_measure_utility_skips_failed_defended_runs() -> None:
    worlds = [_make_world()]
    params = FlywheelDefenseParams(
        cast_alpha=2.0,
        cast_threshold=0.0,
        sic_threshold=0.5,
        sic_iterations=3,
        sic_mode="generation",
    )

    with patch(
        "vauban.flywheel._utility.run_defended_agent_loop",
        side_effect=RuntimeError("boom"),
    ):
        score = measure_utility(
            object(), object(),  # type: ignore[arg-type]
            worlds,
            direction=None,
            layer_index=0,
            defense_params=params,
            n_samples=1,
        )

    assert score == 0.0
