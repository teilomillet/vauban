# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for flywheel orchestrator."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

from vauban.flywheel._run import run_flywheel
from vauban.types import (
    AgentTurn,
    DefendedTrace,
    EnvironmentConfig,
    EnvironmentResult,
    EnvironmentTarget,
    EnvironmentTask,
    FlywheelConfig,
    FlywheelTrace,
    ObjectiveConfig,
    ObjectiveMetricSpec,
    Payload,
    ToolCall,
    ToolSchema,
)


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
        turns=[AgentTurn(role="assistant", content="ok")],
        tool_calls_made=[
            ToolCall(function="f", arguments={"a": "b"}),
        ],
        injection_payload="test",
    )


def _stub_generate_worlds(
    **_kwargs: object,
) -> list[tuple[EnvironmentConfig, object]]:
    """Return a single world with metadata."""
    env = EnvironmentConfig(
        system_prompt="test",
        tools=[ToolSchema(
            name="t", description="d", parameters={"x": "string"},
        )],
        target=EnvironmentTarget(function="t"),
        task=EnvironmentTask(content="task"),
        injection_surface="s",
    )
    meta = MagicMock(domain="email", skeleton="email",
                     complexity=1, position="infix", seed_offset=0)
    return [(env, meta)]


def _stub_execute(
    *_args: object, **_kwargs: object,
) -> list[FlywheelTrace]:
    """Return a single successful trace."""
    return [FlywheelTrace(
        world_index=0,
        payload_index=0,
        payload_text="test",
        reward=0.9,
        target_called=True,
        turns_used=1,
        tool_calls_made=1,
    )]


def _stub_defend(
    *_args: object, **_kwargs: object,
) -> list[DefendedTrace]:
    """Return a single evaded trace."""
    return [DefendedTrace(
        world_index=0,
        payload_index=0,
        payload_text="test",
        reward=0.9,
        target_called=True,
        turns_used=1,
        tool_calls_made=1,
        defense_blocked=False,
    )]


class TestRunFlywheel:
    def test_single_cycle_produces_result(self, tmp_path: Path) -> None:
        config = FlywheelConfig(
            n_cycles=1,
            worlds_per_cycle=1,
            payloads_per_world=1,
            skeletons=["email"],
            model_expand=False,
        )
        with (
            patch(
                "vauban.flywheel._run.generate_worlds",
                side_effect=_stub_generate_worlds,
            ),
            patch(
                "vauban.flywheel._run.execute_attack_matrix",
                side_effect=_stub_execute,
            ),
            patch(
                "vauban.flywheel._run.defend_traces",
                side_effect=_stub_defend,
            ),
            patch(
                "vauban.flywheel._run.measure_utility",
                return_value=0.95,
            ),
        ):
            result = run_flywheel(
                MagicMock(), MagicMock(), config,
                direction=None, layer_index=0,
                output_dir=tmp_path, verbose=False,
            )

        assert len(result.cycles) == 1
        assert result.total_worlds == 1
        assert result.final_defense.cast_alpha >= config.cast_alpha
        assert result.total_payloads > 0

    def test_convergence_stops_early(self, tmp_path: Path) -> None:
        config = FlywheelConfig(
            n_cycles=10,
            worlds_per_cycle=1,
            payloads_per_world=1,
            skeletons=["email"],
            model_expand=False,
            convergence_window=2,
            convergence_threshold=1.0,  # always converges
            harden=False,
        )

        def stub_defend_blocked(
            *_args: object, **_kwargs: object,
        ) -> list[DefendedTrace]:
            return [DefendedTrace(
                world_index=0, payload_index=0,
                payload_text="t", reward=0.9,
                target_called=True, turns_used=1,
                tool_calls_made=1, defense_blocked=True,
            )]

        with (
            patch(
                "vauban.flywheel._run.generate_worlds",
                side_effect=_stub_generate_worlds,
            ),
            patch(
                "vauban.flywheel._run.execute_attack_matrix",
                side_effect=_stub_execute,
            ),
            patch(
                "vauban.flywheel._run.defend_traces",
                side_effect=stub_defend_blocked,
            ),
            patch(
                "vauban.flywheel._run.measure_utility",
                return_value=0.95,
            ),
        ):
            result = run_flywheel(
                MagicMock(), MagicMock(), config,
                direction=None, layer_index=0,
                output_dir=tmp_path, verbose=False,
            )

        assert result.converged is True
        assert len(result.cycles) < config.n_cycles

    def test_writes_output_files(self, tmp_path: Path) -> None:
        config = FlywheelConfig(
            n_cycles=1,
            worlds_per_cycle=1,
            payloads_per_world=1,
            skeletons=["email"],
            model_expand=False,
            harden=False,
        )
        with (
            patch(
                "vauban.flywheel._run.generate_worlds",
                side_effect=_stub_generate_worlds,
            ),
            patch(
                "vauban.flywheel._run.execute_attack_matrix",
                side_effect=_stub_execute,
            ),
            patch(
                "vauban.flywheel._run.defend_traces",
                side_effect=_stub_defend,
            ),
            patch(
                "vauban.flywheel._run.measure_utility",
                return_value=0.95,
            ),
        ):
            run_flywheel(
                MagicMock(), MagicMock(), config,
                direction=None, layer_index=0,
                output_dir=tmp_path, verbose=False,
            )

        assert (tmp_path / "flywheel_report.json").exists()
        assert (tmp_path / "flywheel_traces.jsonl").exists()
        assert (tmp_path / "flywheel_state.json").exists()

    def test_no_hardening_when_disabled(self, tmp_path: Path) -> None:
        config = FlywheelConfig(
            n_cycles=1,
            worlds_per_cycle=1,
            payloads_per_world=1,
            skeletons=["email"],
            model_expand=False,
            harden=False,
            cast_alpha=2.0,
        )
        with (
            patch(
                "vauban.flywheel._run.generate_worlds",
                side_effect=_stub_generate_worlds,
            ),
            patch(
                "vauban.flywheel._run.execute_attack_matrix",
                side_effect=_stub_execute,
            ),
            patch(
                "vauban.flywheel._run.defend_traces",
                side_effect=_stub_defend,
            ),
            patch(
                "vauban.flywheel._run.measure_utility",
                return_value=0.95,
            ),
        ):
            result = run_flywheel(
                MagicMock(), MagicMock(), config,
                direction=None, layer_index=0,
                output_dir=tmp_path, verbose=False,
            )

        # Alpha should remain unchanged when harden=False
        assert result.final_defense.cast_alpha == 2.0

    def test_objective_assessment_is_attached_and_reported(
        self,
        tmp_path: Path,
    ) -> None:
        config = FlywheelConfig(
            n_cycles=1,
            worlds_per_cycle=1,
            payloads_per_world=1,
            skeletons=["email"],
            model_expand=False,
            harden=False,
        )
        objective = ObjectiveConfig(
            name="customer_support_gate",
            deployment="customer_support",
            access="api",
            safety=[
                ObjectiveMetricSpec(
                    metric="evasion_rate",
                    threshold=1.0,
                    comparison="at_most",
                ),
            ],
            utility=[
                ObjectiveMetricSpec(
                    metric="utility_score",
                    threshold=0.90,
                    comparison="at_least",
                ),
            ],
        )

        with (
            patch(
                "vauban.flywheel._run.generate_worlds",
                side_effect=_stub_generate_worlds,
            ),
            patch(
                "vauban.flywheel._run.execute_attack_matrix",
                side_effect=_stub_execute,
            ),
            patch(
                "vauban.flywheel._run.defend_traces",
                side_effect=_stub_defend,
            ),
            patch(
                "vauban.flywheel._run.measure_utility",
                return_value=0.95,
            ),
        ):
            result = run_flywheel(
                MagicMock(), MagicMock(), config,
                direction=None, layer_index=0,
                output_dir=tmp_path, objective=objective, verbose=False,
            )

        assert result.objective == objective
        assert result.objective_assessment is not None
        assert result.objective_assessment.passed is True
        assert result.objective_assessment.safety_passed is True
        assert result.objective_assessment.utility_passed is True

        report = json.loads((tmp_path / "flywheel_report.json").read_text())
        assert report["objective"]["name"] == "customer_support_gate"
        assert report["objective_assessment"]["summary"] == "Objective met"
        checks = report["objective_assessment"]["checks"]
        assert [check["metric"] for check in checks] == [
            "evasion_rate",
            "utility_score",
        ]

    def test_payload_selection_includes_recent(
        self, tmp_path: Path,
    ) -> None:
        """GCG-discovered payloads (appended at tail) must be selected."""
        config = FlywheelConfig(
            n_cycles=1,
            worlds_per_cycle=1,
            payloads_per_world=2,
            skeletons=["email"],
            model_expand=False,
            warmstart_gcg=True,
            harden=False,
        )

        captured_payloads: list[list[str]] = []

        def capture_execute(
            _model: object,
            _tokenizer: object,
            _worlds: object,
            payloads: object,
            *args: object,
            **kwargs: object,
        ) -> list[FlywheelTrace]:
            from vauban.types import Payload

            assert isinstance(payloads, list)
            captured_payloads.append(
                [p.text for p in payloads if isinstance(p, Payload)],
            )
            return _stub_execute()

        with (
            patch(
                "vauban.flywheel._run.generate_worlds",
                side_effect=_stub_generate_worlds,
            ),
            patch(
                "vauban.flywheel._run.execute_attack_matrix",
                side_effect=capture_execute,
            ),
            patch(
                "vauban.flywheel._run.defend_traces",
                side_effect=_stub_defend,
            ),
            patch(
                "vauban.flywheel._run.measure_utility",
                return_value=0.95,
            ),
            patch(
                "vauban.flywheel._run.warmstart_gcg_payloads",
                return_value=["gcg_discovered_payload"],
            ),
        ):
            run_flywheel(
                MagicMock(), MagicMock(), config,
                direction=None, layer_index=0,
                output_dir=tmp_path, verbose=False,
            )

        assert len(captured_payloads) == 1
        selected = captured_payloads[0]
        assert "gcg_discovered_payload" in selected

    def test_cycle_metrics_use_pre_hardening_defense(
        self, tmp_path: Path,
    ) -> None:
        config = FlywheelConfig(
            n_cycles=1,
            worlds_per_cycle=1,
            payloads_per_world=1,
            skeletons=["email"],
            model_expand=False,
            cast_alpha=2.0,
            sic_threshold=0.5,
        )

        with (
            patch(
                "vauban.flywheel._run.generate_worlds",
                side_effect=_stub_generate_worlds,
            ),
            patch(
                "vauban.flywheel._run.execute_attack_matrix",
                side_effect=_stub_execute,
            ),
            patch(
                "vauban.flywheel._run.defend_traces",
                side_effect=_stub_defend,
            ),
            patch(
                "vauban.flywheel._run.measure_utility",
                return_value=0.95,
            ),
        ):
            result = run_flywheel(
                MagicMock(), MagicMock(), config,
                direction=None, layer_index=0,
                output_dir=tmp_path, verbose=False,
            )

        assert result.cycles[0].cast_alpha == 2.0
        assert result.cycles[0].sic_threshold == 0.5
        assert result.final_defense.cast_alpha > result.cycles[0].cast_alpha

    def test_verbose_convergence_prints(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        config = FlywheelConfig(
            n_cycles=2,
            worlds_per_cycle=1,
            payloads_per_world=1,
            skeletons=["email"],
            model_expand=False,
            harden=False,
            convergence_window=1,
            convergence_threshold=1.0,
        )
        with (
            patch(
                "vauban.flywheel._run.generate_worlds",
                side_effect=_stub_generate_worlds,
            ),
            patch(
                "vauban.flywheel._run.execute_attack_matrix",
                side_effect=_stub_execute,
            ),
            patch(
                "vauban.flywheel._run.defend_traces",
                side_effect=_stub_defend,
            ),
            patch(
                "vauban.flywheel._run.measure_utility",
                return_value=0.95,
            ),
        ):
            result = run_flywheel(
                MagicMock(), MagicMock(), config,
                direction=None, layer_index=0,
                output_dir=tmp_path, verbose=True,
            )

        assert len(result.cycles) == 1
        out = capsys.readouterr().out
        assert "Flywheel cycle 1/2" in out
        assert "Converged at cycle 1" in out

    def test_count_now_blocked_counts_and_skips_errors(
        self,
    ) -> None:
        from unittest.mock import patch

        from vauban.flywheel._defended_loop import DefendedEnvironmentResult
        from vauban.flywheel._run import _count_now_blocked

        world = EnvironmentConfig(
            system_prompt="test",
            tools=[ToolSchema(
                name="t", description="d", parameters={"x": "string"},
            )],
            target=EnvironmentTarget(function="t"),
            task=EnvironmentTask(content="task"),
            injection_surface="s",
        )
        payload = Payload(text="payload", source="test", cycle_discovered=0)
        defense = MagicMock()
        defended = DefendedEnvironmentResult(
            env_result=_make_env_result(reward=0.4),
            cast_interventions=0,
            cast_considered=0,
            sic_blocked=False,
        )

        with patch(
            "vauban.flywheel._run.run_defended_agent_loop",
            side_effect=[defended, RuntimeError("boom")],
        ):
            count = _count_now_blocked(
                MagicMock(), MagicMock(),
                [(world, payload, MagicMock()), (world, payload, MagicMock())],
                direction=None,
                layer_index=0,
                defense=defense,
            )

        assert count == 1

    def test_validate_previous_calls_count_now_blocked(
        self,
        tmp_path: Path,
    ) -> None:
        config = FlywheelConfig(
            n_cycles=2,
            worlds_per_cycle=1,
            payloads_per_world=1,
            skeletons=["email"],
            model_expand=False,
            harden=False,
            validate_previous=True,
        )
        with (
            patch(
                "vauban.flywheel._run.generate_worlds",
                side_effect=_stub_generate_worlds,
            ),
            patch(
                "vauban.flywheel._run.execute_attack_matrix",
                side_effect=_stub_execute,
            ),
            patch(
                "vauban.flywheel._run.defend_traces",
                side_effect=_stub_defend,
            ),
            patch(
                "vauban.flywheel._run.measure_utility",
                return_value=0.95,
            ),
            patch(
                "vauban.flywheel._run._count_now_blocked",
                return_value=1,
            ) as mock_count_now_blocked,
        ):
            run_flywheel(
                MagicMock(), MagicMock(), config,
                direction=None, layer_index=0,
                output_dir=tmp_path, verbose=False,
            )

        assert mock_count_now_blocked.call_count == 1
