# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for pipeline dispatch: dispatch_early_mode."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from tests.conftest import make_direction_result, make_early_mode_context
from vauban._pipeline._modes import EARLY_MODE_RUNNERS, dispatch_early_mode
from vauban.config._mode_registry import (
    EARLY_MODE_SPECS,
    EarlyModePhase,
)
from vauban.types import (
    BehaviorTraceConfig,
    BehaviorTracePromptConfig,
    DepthConfig,
    InterventionEvalConfig,
    InterventionEvalPrompt,
    ProbeConfig,
    SteerConfig,
)

if TYPE_CHECKING:
    from pathlib import Path

# ===================================================================
# dispatch_early_mode — basic dispatch
# ===================================================================


class TestDispatchNoMode:
    """Returns False when no mode is active for a phase."""

    def test_standalone_no_mode(self, tmp_path: Path) -> None:
        ctx = make_early_mode_context(tmp_path)
        assert dispatch_early_mode("standalone", ctx) is False

    def test_before_prompts_no_mode(self, tmp_path: Path) -> None:
        ctx = make_early_mode_context(tmp_path)
        assert dispatch_early_mode("before_prompts", ctx) is False

    def test_after_measure_no_mode(self, tmp_path: Path) -> None:
        ctx = make_early_mode_context(tmp_path)
        assert dispatch_early_mode("after_measure", ctx) is False


class TestDispatchCallsRunner:
    """Returns True and calls the runner when a mode is active."""

    def test_probe_mode_dispatched(self, tmp_path: Path) -> None:
        probe_cfg = ProbeConfig(prompts=["test"])
        dr = make_direction_result()
        ctx = make_early_mode_context(tmp_path, probe=probe_cfg, direction_result=dr)
        with patch.dict(
            EARLY_MODE_RUNNERS,
            {"probe": MagicMock()},
        ):
            result = dispatch_early_mode("after_measure", ctx)
            assert result is True
            EARLY_MODE_RUNNERS["probe"].assert_called_once_with(ctx)

    def test_steer_mode_dispatched(self, tmp_path: Path) -> None:
        steer_cfg = SteerConfig(prompts=["test"])
        dr = make_direction_result()
        ctx = make_early_mode_context(tmp_path, steer=steer_cfg, direction_result=dr)
        with patch.dict(
            EARLY_MODE_RUNNERS,
            {"steer": MagicMock()},
        ):
            result = dispatch_early_mode("after_measure", ctx)
            assert result is True
            EARLY_MODE_RUNNERS["steer"].assert_called_once_with(ctx)

    def test_intervention_eval_mode_dispatched(self, tmp_path: Path) -> None:
        eval_cfg = InterventionEvalConfig(
            prompts=[
                InterventionEvalPrompt(
                    prompt_id="p1",
                    prompt="Explain rainbows.",
                ),
            ],
        )
        dr = make_direction_result()
        ctx = make_early_mode_context(
            tmp_path,
            intervention_eval=eval_cfg,
            direction_result=dr,
        )
        with patch.dict(
            EARLY_MODE_RUNNERS,
            {"intervention_eval": MagicMock()},
        ):
            result = dispatch_early_mode("after_measure", ctx)
            assert result is True
            EARLY_MODE_RUNNERS["intervention_eval"].assert_called_once_with(ctx)

    def test_behavior_trace_mode_dispatched(self, tmp_path: Path) -> None:
        trace_cfg = BehaviorTraceConfig(
            prompts=[
                BehaviorTracePromptConfig(
                    prompt_id="p1",
                    text="Explain rainbows.",
                ),
            ],
        )
        ctx = make_early_mode_context(tmp_path, behavior_trace=trace_cfg)
        with patch.dict(
            EARLY_MODE_RUNNERS,
            {"behavior_trace": MagicMock()},
        ):
            result = dispatch_early_mode("before_prompts", ctx)
            assert result is True
            EARLY_MODE_RUNNERS["behavior_trace"].assert_called_once_with(ctx)


class TestDispatchRequiresDirection:
    """Returns False when requires_direction=True but direction_result=None."""

    def test_probe_without_direction(self, tmp_path: Path) -> None:
        probe_cfg = ProbeConfig(prompts=["test"])
        ctx = make_early_mode_context(tmp_path, probe=probe_cfg)
        # probe requires_direction=True, no direction_result
        result = dispatch_early_mode("after_measure", ctx)
        assert result is False

    def test_depth_dispatches_without_direction(self, tmp_path: Path) -> None:
        """Modes with requires_direction=False dispatch even without direction."""
        depth_cfg = DepthConfig(prompts=["test"])
        ctx = make_early_mode_context(tmp_path, depth=depth_cfg)
        with patch.dict(
            EARLY_MODE_RUNNERS,
            {"depth": MagicMock()},
        ):
            result = dispatch_early_mode("before_prompts", ctx)
            assert result is True
            EARLY_MODE_RUNNERS["depth"].assert_called_once_with(ctx)


# ===================================================================
# Registry consistency
# ===================================================================


class TestRegistryConsistency:
    """EARLY_MODE_RUNNERS keys match EARLY_MODE_SPECS modes."""

    def test_all_specs_have_runners(self) -> None:
        spec_modes = {spec.mode for spec in EARLY_MODE_SPECS}
        runner_modes = set(EARLY_MODE_RUNNERS.keys())
        missing = spec_modes - runner_modes
        assert not missing, f"Specs without runners: {missing}"

    def test_all_runners_have_specs(self) -> None:
        spec_modes = {spec.mode for spec in EARLY_MODE_SPECS}
        runner_modes = set(EARLY_MODE_RUNNERS.keys())
        extra = runner_modes - spec_modes
        assert not extra, f"Runners without specs: {extra}"

    def test_valid_phases(self) -> None:
        valid_phases: set[EarlyModePhase] = {
            "standalone",
            "before_prompts",
            "after_measure",
        }
        for spec in EARLY_MODE_SPECS:
            assert spec.phase in valid_phases, (
                f"spec {spec.mode} has invalid phase {spec.phase}"
            )

    def test_runner_callables(self) -> None:
        for mode, runner in EARLY_MODE_RUNNERS.items():
            assert callable(runner), f"Runner for {mode} is not callable"
