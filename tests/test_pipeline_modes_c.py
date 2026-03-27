# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for mode runners C: optimize, compose_optimize, softprompt."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import make_direction_result, make_early_mode_context
from vauban._pipeline._mode_compose_optimize import _run_compose_optimize_mode
from vauban._pipeline._mode_optimize import _run_optimize_mode
from vauban._pipeline._mode_softprompt import _run_softprompt_mode
from vauban.types import (
    ComposeOptimizeConfig,
    CompositionTrialResult,
    OptimizeConfig,
    SoftPromptConfig,
    SoftPromptResult,
)

if TYPE_CHECKING:
    from pathlib import Path

# ===================================================================
# Optimize mode
# ===================================================================


class TestOptimizeMode:
    """Tests for _run_optimize_mode."""

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        ctx = make_early_mode_context(tmp_path)
        with pytest.raises(ValueError, match="optimize config is required"):
            _run_optimize_mode(ctx)

    def test_missing_direction_raises(self, tmp_path: Path) -> None:
        opt_cfg = OptimizeConfig(n_trials=5)
        ctx = make_early_mode_context(tmp_path, optimize=opt_cfg)
        with pytest.raises(
            ValueError, match="direction_result is required",
        ):
            _run_optimize_mode(ctx)

    def test_missing_harmful_raises(self, tmp_path: Path) -> None:
        opt_cfg = OptimizeConfig(n_trials=5)
        dr = make_direction_result()
        ctx = make_early_mode_context(
            tmp_path,
            direction_result=dr,
            harmful=None,
            optimize=opt_cfg,
        )
        with pytest.raises(
            ValueError, match="harmful prompts are required",
        ):
            _run_optimize_mode(ctx)

    def test_happy_path(self, tmp_path: Path) -> None:
        opt_cfg = OptimizeConfig(n_trials=5)
        dr = make_direction_result()
        ctx = make_early_mode_context(
            tmp_path, direction_result=dr, optimize=opt_cfg,
        )

        mock_result = MagicMock()
        with (
            patch(
                "vauban.optimize.optimize",
                return_value=mock_result,
            ),
            patch(
                "vauban._pipeline._mode_optimize.finish_mode_run",
            ) as mock_finish,
            patch(
                "vauban._pipeline._mode_optimize._optimize_to_dict",
                return_value={},
            ),
        ):
            _run_optimize_mode(ctx)
            assert (tmp_path / "optimize_report.json").exists()
            metadata = mock_finish.call_args[0][3]
            assert metadata["n_trials"] == 5.0


# ===================================================================
# Compose optimize mode
# ===================================================================


class TestComposeOptimizeMode:
    """Tests for _run_compose_optimize_mode."""

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        ctx = make_early_mode_context(tmp_path)
        with pytest.raises(
            ValueError, match="compose_optimize config is required",
        ):
            _run_compose_optimize_mode(ctx)

    def test_missing_harmful_raises(self, tmp_path: Path) -> None:
        co_cfg = ComposeOptimizeConfig(
            bank_path="bank.safetensors", n_trials=5,
        )
        ctx = make_early_mode_context(
            tmp_path, harmful=None, compose_optimize=co_cfg,
        )
        with pytest.raises(
            ValueError, match="harmful prompts are required",
        ):
            _run_compose_optimize_mode(ctx)

    def test_happy_path(self, tmp_path: Path) -> None:
        co_cfg = ComposeOptimizeConfig(
            bank_path="bank.safetensors", n_trials=3,
        )
        ctx = make_early_mode_context(tmp_path, compose_optimize=co_cfg)

        mock_result = MagicMock()
        mock_result.n_trials = 3
        mock_result.bank_entries = ["entry1"]
        mock_result.best_refusal = None
        mock_result.best_balanced = None

        with (
            patch(
                "vauban.optimize.optimize_composition",
                return_value=mock_result,
            ),
            patch(
                "vauban._pipeline._mode_compose_optimize.finish_mode_run",
            ) as mock_finish,
        ):
            _run_compose_optimize_mode(ctx)
            assert (
                tmp_path / "compose_optimize_report.json"
            ).exists()
            metadata = mock_finish.call_args[0][3]
            assert metadata["n_trials"] == 3.0


# ===================================================================
# Softprompt mode
# ===================================================================


class TestSoftpromptMode:
    """Tests for _run_softprompt_mode."""

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        ctx = make_early_mode_context(tmp_path)
        with pytest.raises(
            ValueError, match="softprompt config is required",
        ):
            _run_softprompt_mode(ctx)

    def test_missing_harmful_raises(self, tmp_path: Path) -> None:
        sp_cfg = SoftPromptConfig(n_steps=10, n_tokens=4)
        ctx = make_early_mode_context(
            tmp_path, harmful=None, softprompt=sp_cfg,
        )
        with pytest.raises(
            ValueError, match="harmful prompts are required",
        ):
            _run_softprompt_mode(ctx)

    def test_simplest_path_no_transfer_no_gan(
        self, tmp_path: Path,
    ) -> None:
        """Minimal softprompt: no transfer, no api_eval, no GAN."""
        sp_cfg = SoftPromptConfig(n_steps=10, n_tokens=4)
        ctx = make_early_mode_context(tmp_path, softprompt=sp_cfg)

        mock_result = SoftPromptResult(
            mode="continuous",
            success_rate=0.5,
            final_loss=1.23,
            loss_history=[2.0, 1.5, 1.23],
            n_steps=10,
            n_tokens=4,
            embeddings=None,
            token_ids=None,
            token_text=None,
            eval_responses=["response1"],
            accessibility_score=0.0,
            per_prompt_losses=[1.23],
            early_stopped=False,
            transfer_results=[],
            defense_eval=None,
            gan_history=[],
        )
        with (
            patch(
                "vauban.softprompt.softprompt_attack",
                return_value=mock_result,
            ),
            patch(
                "vauban._pipeline._mode_softprompt.finish_mode_run",
            ) as mock_finish,
            patch(
                "vauban._pipeline._mode_softprompt._softprompt_to_dict",
                return_value={},
            ),
        ):
            _run_softprompt_mode(ctx)
            assert (tmp_path / "softprompt_report.json").exists()
            metadata = mock_finish.call_args[0][3]
            assert metadata["success_rate"] == 0.5


# ===================================================================
# Compose optimize — non-None results
# ===================================================================


class TestComposeOptimizeResults:
    """Tests for non-None best_refusal/best_balanced paths."""

    def test_best_refusal_in_report(self, tmp_path: Path) -> None:
        """Non-None best_refusal is serialized into the report."""
        import json

        co_cfg = ComposeOptimizeConfig(
            bank_path="bank.safetensors", n_trials=2,
        )
        ctx = make_early_mode_context(tmp_path, compose_optimize=co_cfg)

        trial = CompositionTrialResult(
            trial_number=1,
            weights={"dir_a": 0.7},
            refusal_rate=0.1,
            perplexity=15.0,
        )
        mock_result = MagicMock()
        mock_result.n_trials = 2
        mock_result.bank_entries = ["dir_a"]
        mock_result.best_refusal = trial
        mock_result.best_balanced = None

        with (
            patch(
                "vauban.optimize.optimize_composition",
                return_value=mock_result,
            ),
            patch(
                "vauban._pipeline._mode_compose_optimize.finish_mode_run",
            ),
        ):
            _run_compose_optimize_mode(ctx)
            report = json.loads(
                (tmp_path / "compose_optimize_report.json").read_text(),
            )
            assert report["best_refusal"] is not None
            assert report["best_refusal"]["refusal_rate"] == 0.1
            assert report["best_balanced"] is None

    def test_best_balanced_in_report(self, tmp_path: Path) -> None:
        """Non-None best_balanced is serialized into the report."""
        import json

        co_cfg = ComposeOptimizeConfig(
            bank_path="bank.safetensors", n_trials=2,
        )
        ctx = make_early_mode_context(tmp_path, compose_optimize=co_cfg)

        trial = CompositionTrialResult(
            trial_number=1,
            weights={"dir_a": 0.5, "dir_b": 0.5},
            refusal_rate=0.3,
            perplexity=12.0,
        )
        mock_result = MagicMock()
        mock_result.n_trials = 2
        mock_result.bank_entries = ["dir_a", "dir_b"]
        mock_result.best_refusal = None
        mock_result.best_balanced = trial

        with (
            patch(
                "vauban.optimize.optimize_composition",
                return_value=mock_result,
            ),
            patch(
                "vauban._pipeline._mode_compose_optimize.finish_mode_run",
            ),
        ):
            _run_compose_optimize_mode(ctx)
            report = json.loads(
                (tmp_path / "compose_optimize_report.json").read_text(),
            )
            assert report["best_refusal"] is None
            assert report["best_balanced"] is not None
            assert report["best_balanced"]["perplexity"] == 12.0
