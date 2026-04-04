# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for mode runners C: optimize, compose_optimize, softprompt."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import make_direction_result, make_early_mode_context
from vauban import _ops as ops
from vauban._pipeline._mode_compose_optimize import _run_compose_optimize_mode
from vauban._pipeline._mode_optimize import _run_optimize_mode
from vauban._pipeline._mode_softprompt import _run_softprompt_mode
from vauban.types import (
    ApiEvalConfig,
    ComposeOptimizeConfig,
    CompositionTrialResult,
    EvalConfig,
    OptimizeConfig,
    SoftPromptConfig,
    SoftPromptResult,
    TransferEvalResult,
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

    def test_uses_eval_prompts_path_when_present(self, tmp_path: Path) -> None:
        opt_cfg = OptimizeConfig(n_trials=2)
        dr = make_direction_result()
        ctx = make_early_mode_context(
            tmp_path,
            direction_result=dr,
            optimize=opt_cfg,
            eval=EvalConfig(prompts_path=tmp_path / "eval.jsonl", num_prompts=1),
        )

        mock_result = MagicMock()
        with (
            patch(
                "vauban.measure.load_prompts",
                return_value=["eval-1", "eval-2"],
            ) as mock_load,
            patch(
                "vauban.optimize.optimize",
                return_value=mock_result,
            ) as mock_optimize,
            patch("vauban._pipeline._mode_optimize.finish_mode_run"),
            patch(
                "vauban._pipeline._mode_optimize._optimize_to_dict",
                return_value={},
            ),
        ):
            _run_optimize_mode(ctx)

        assert mock_load.call_args == ((tmp_path / "eval.jsonl",),)
        assert mock_optimize.call_args[0][3] == ["eval-1", "eval-2"]


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

    def test_uses_eval_prompts_path_when_present(self, tmp_path: Path) -> None:
        co_cfg = ComposeOptimizeConfig(
            bank_path="bank.safetensors", n_trials=3,
        )
        ctx = make_early_mode_context(
            tmp_path,
            compose_optimize=co_cfg,
            eval=EvalConfig(prompts_path=tmp_path / "eval.jsonl", num_prompts=1),
        )

        mock_result = MagicMock()
        mock_result.n_trials = 3
        mock_result.bank_entries = []
        mock_result.best_refusal = None
        mock_result.best_balanced = None

        with (
            patch(
                "vauban.measure.load_prompts",
                return_value=["eval-1", "eval-2"],
            ) as mock_load,
            patch(
                "vauban.optimize.optimize_composition",
                return_value=mock_result,
            ) as mock_optimize,
            patch("vauban._pipeline._mode_compose_optimize.finish_mode_run"),
        ):
            _run_compose_optimize_mode(ctx)

        assert mock_load.call_args == ((tmp_path / "eval.jsonl",),)
        assert mock_optimize.call_args[0][2] == ["eval-1", "eval-2"]


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

    def test_externality_transfer_api_and_arena_paths(
        self,
        tmp_path: Path,
    ) -> None:
        """Externality loading, transfer eval, API eval, and arena output compose."""
        sp_cfg = SoftPromptConfig(
            n_steps=10,
            n_tokens=4,
            loss_mode="externality",
            externality_target=str(tmp_path / "direction.safetensors"),
            ref_model="ref-model",
            transfer_models=["transfer-model"],
            svf_boundary_path=str(tmp_path / "boundary.safetensors"),
            system_prompt="SYS",
            token_position="prefix",
        )
        api_cfg = ApiEvalConfig(endpoints=[])
        eval_cfg = EvalConfig(prompts_path=tmp_path / "prompts.txt", num_prompts=1)
        ctx = make_early_mode_context(
            tmp_path,
            direction_result=make_direction_result(),
            softprompt=sp_cfg,
            api_eval=api_cfg,
            eval=eval_cfg,
        )

        main_model = object()
        main_tokenizer = object()
        ref_model = object()
        transfer_model = object()
        transfer_tokenizer = object()
        ctx.model = main_model
        ctx.tokenizer = main_tokenizer

        externality_direction = ops.array([1.0, 0.0, 0.0, 0.0])
        embeddings = ops.ones((1, 3, 4))
        projected_ids = [7, 8, 9]
        transfer_embeds = ops.ones((1, 3, 4))
        api_results = [
            TransferEvalResult(
                model_id="api-model",
                success_rate=0.75,
                eval_responses=["api response"],
            ),
        ]
        attack_result = SoftPromptResult(
            mode="continuous",
            success_rate=0.5,
            final_loss=1.23,
            loss_history=[2.0, 1.23],
            n_steps=10,
            n_tokens=4,
            embeddings=embeddings,
            token_ids=None,
            token_text="TOKENS",
            eval_responses=["response1"],
            accessibility_score=0.2,
            per_prompt_losses=[1.23],
            early_stopped=False,
            transfer_results=[],
            defense_eval=None,
            gan_history=[],
        )

        main_transformer = MagicMock()
        main_transformer.embed_tokens.weight = ops.ones((16, 4))
        transfer_transformer = MagicMock()
        transfer_transformer.embed_tokens = MagicMock(return_value=transfer_embeds)

        def _fake_load_model(model_id: str) -> tuple[object, object]:
            if model_id == "ref-model":
                return ref_model, object()
            assert model_id == "transfer-model"
            return transfer_model, transfer_tokenizer

        def _fake_is_quantized(model: object) -> bool:
            return model in {ref_model, transfer_model}

        def _fake_get_transformer(model: object) -> MagicMock:
            if model is main_model:
                return main_transformer
            assert model is transfer_model
            return transfer_transformer

        with (
            patch(
                "vauban._ops.load",
                return_value={"direction": externality_direction},
            ),
            patch(
                "vauban.measure.load_prompts",
                return_value=["prompt-a", "prompt-b"],
            ) as mock_load_prompts,
            patch(
                "vauban._model_io.load_model",
                side_effect=_fake_load_model,
            ),
            patch(
                "vauban.dequantize.is_quantized",
                side_effect=_fake_is_quantized,
            ),
            patch("vauban.dequantize.dequantize_model") as mock_dequantize,
            patch(
                "vauban.svf.load_svf_boundary",
                return_value="boundary",
            ),
            patch(
                "vauban.softprompt.softprompt_attack",
                return_value=attack_result,
            ) as mock_attack,
            patch(
                "vauban.softprompt._runtime._project_to_tokens",
                return_value=projected_ids,
            ) as mock_project,
            patch(
                "vauban._forward.get_transformer",
                side_effect=_fake_get_transformer,
            ),
            patch("vauban._forward.force_eval"),
            patch(
                "vauban.softprompt._generation._evaluate_attack",
                return_value=(0.4, ["transfer response"]),
            ) as mock_eval_attack,
            patch(
                "vauban.api_eval.evaluate_suffix_via_api",
                return_value=api_results,
            ) as mock_api_eval,
            patch(
                "vauban._pipeline._mode_softprompt.write_mode_report",
                return_value=tmp_path / "softprompt_report.json",
            ),
            patch(
                "vauban._pipeline._mode_softprompt.write_arena_card",
            ) as mock_arena,
            patch(
                "vauban._pipeline._mode_softprompt.finish_mode_run",
            ) as mock_finish,
            patch(
                "vauban._pipeline._mode_softprompt._softprompt_to_dict",
                return_value={},
            ),
        ):
            _run_softprompt_mode(ctx)

        mock_load_prompts.assert_called_once_with(eval_cfg.prompts_path)
        assert mock_dequantize.call_args_list == [
            ((ref_model,), {}),
            ((transfer_model,), {}),
        ]
        assert mock_attack.call_args.args[2] == ["prompt-a", "prompt-b"]
        assert ops.allclose(mock_attack.call_args.args[4], externality_direction)
        mock_project.assert_called_once_with(
            embeddings,
            main_transformer.embed_tokens.weight,
        )
        mock_eval_attack.assert_called_once_with(
            transfer_model,
            transfer_tokenizer,
            ["prompt-a", "prompt-b"],
            transfer_embeds,
            sp_cfg,
        )
        mock_api_eval.assert_called_once_with(
            "TOKENS",
            ["prompt-a", "prompt-b"],
            api_cfg,
            "SYS",
            token_position="prefix",
        )
        mock_arena.assert_called_once()
        report_files = mock_finish.call_args.args[2]
        assert report_files == ["softprompt_report.json", "arena_card.txt"]
        metadata = mock_finish.call_args.args[3]
        assert metadata["success_rate"] == 0.5

    def test_externality_tuple_raises(self, tmp_path: Path) -> None:
        """Tuple payloads from externality targets should fail fast."""
        sp_cfg = SoftPromptConfig(
            n_steps=10,
            n_tokens=4,
            loss_mode="externality",
            externality_target=str(tmp_path / "direction.safetensors"),
        )
        ctx = make_early_mode_context(tmp_path, softprompt=sp_cfg)

        with (
            patch(
                "vauban._ops.load",
                return_value=(ops.array([1.0]), ops.array([0.0])),
            ),
            pytest.raises(TypeError, match="Expected array or dict"),
        ):
            _run_softprompt_mode(ctx)

    def test_externality_array_and_empty_transfer_ids_path(
        self,
        tmp_path: Path,
    ) -> None:
        """Raw-array externality targets and empty transfer token ids are handled."""
        sp_cfg = SoftPromptConfig(
            n_steps=10,
            n_tokens=4,
            loss_mode="externality",
            externality_target=str(tmp_path / "direction.safetensors"),
            transfer_models=["transfer-model"],
        )
        ctx = make_early_mode_context(tmp_path, softprompt=sp_cfg)

        transfer_model = object()
        transfer_tokenizer = object()
        ctx.model = object()
        ctx.tokenizer = object()

        externality_direction = ops.array([0.0, 1.0, 0.0, 0.0])
        transfer_embeds = ops.ones((1, 0, 4))
        attack_result = SoftPromptResult(
            mode="continuous",
            success_rate=0.4,
            final_loss=1.0,
            loss_history=[1.0],
            n_steps=10,
            n_tokens=4,
            embeddings=None,
            token_ids=None,
            token_text=None,
            eval_responses=["response"],
            accessibility_score=0.0,
            per_prompt_losses=[1.0],
            early_stopped=False,
            transfer_results=[],
            defense_eval=None,
            gan_history=[],
        )

        transfer_transformer = MagicMock()
        transfer_transformer.embed_tokens = MagicMock(return_value=transfer_embeds)

        with (
            patch("vauban._ops.load", return_value=externality_direction),
            patch(
                "vauban._model_io.load_model",
                return_value=(transfer_model, transfer_tokenizer),
            ),
            patch("vauban.dequantize.is_quantized", return_value=False),
            patch(
                "vauban.softprompt.softprompt_attack",
                return_value=attack_result,
            ) as mock_attack,
            patch(
                "vauban._forward.get_transformer",
                return_value=transfer_transformer,
            ),
            patch("vauban._forward.force_eval"),
            patch(
                "vauban.softprompt._generation._evaluate_attack",
                return_value=(0.25, ["transfer response"]),
            ) as mock_eval_attack,
            patch(
                "vauban._pipeline._mode_softprompt.finish_mode_run",
            ),
            patch(
                "vauban._pipeline._mode_softprompt._softprompt_to_dict",
                return_value={},
            ),
        ):
            _run_softprompt_mode(ctx)

        assert ops.allclose(mock_attack.call_args.args[4], externality_direction)
        mock_eval_attack.assert_called_once_with(
            transfer_model,
            transfer_tokenizer,
            ["test prompt"],
            transfer_embeds,
            sp_cfg,
        )
        token_array = transfer_transformer.embed_tokens.call_args.args[0]
        assert tuple(token_array.shape) == (1, 0)

    def test_transfer_token_ids_skip_projection(
        self,
        tmp_path: Path,
    ) -> None:
        """Existing token ids should be reused directly for transfer evaluation."""
        sp_cfg = SoftPromptConfig(
            n_steps=10,
            n_tokens=4,
            transfer_models=["transfer-model"],
        )
        ctx = make_early_mode_context(tmp_path, softprompt=sp_cfg)

        transfer_model = object()
        transfer_tokenizer = object()
        ctx.model = object()
        ctx.tokenizer = object()

        attack_result = SoftPromptResult(
            mode="continuous",
            success_rate=0.4,
            final_loss=1.0,
            loss_history=[1.0],
            n_steps=10,
            n_tokens=4,
            embeddings=ops.ones((1, 2, 4)),
            token_ids=[4, 5],
            token_text=None,
            eval_responses=["response"],
            accessibility_score=0.0,
            per_prompt_losses=[1.0],
            early_stopped=False,
            transfer_results=[],
            defense_eval=None,
            gan_history=[],
        )
        transfer_embeds = ops.ones((1, 2, 4))
        transfer_transformer = MagicMock()
        transfer_transformer.embed_tokens = MagicMock(return_value=transfer_embeds)

        with (
            patch(
                "vauban._model_io.load_model",
                return_value=(transfer_model, transfer_tokenizer),
            ),
            patch("vauban.dequantize.is_quantized", return_value=False),
            patch(
                "vauban.softprompt.softprompt_attack",
                return_value=attack_result,
            ),
            patch(
                "vauban._forward.get_transformer",
                return_value=transfer_transformer,
            ),
            patch("vauban._forward.force_eval"),
            patch(
                "vauban.softprompt._generation._evaluate_attack",
                return_value=(0.25, ["transfer response"]),
            ),
            patch(
                "vauban.softprompt._runtime._project_to_tokens",
            ) as mock_project,
            patch(
                "vauban._pipeline._mode_softprompt.finish_mode_run",
            ),
            patch(
                "vauban._pipeline._mode_softprompt._softprompt_to_dict",
                return_value={},
            ),
        ):
            _run_softprompt_mode(ctx)

        mock_project.assert_not_called()
        token_array = transfer_transformer.embed_tokens.call_args.args[0]
        assert token_array.tolist() == [[4, 5]]


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
