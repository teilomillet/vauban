# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for the softprompt dispatcher orchestration layer."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest
from conftest import MockTokenizer

from vauban import _ops as ops
from vauban.softprompt._dispatcher import _run_single_attack, softprompt_attack
from vauban.types import (
    ApiEvalConfig,
    CausalLM,
    DefenseEvalResult,
    EnvironmentConfig,
    EnvironmentResult,
    EnvironmentTarget,
    EnvironmentTask,
    SoftPromptConfig,
    SoftPromptResult,
    Tokenizer,
    ToolSchema,
)

if TYPE_CHECKING:
    from vauban._array import Array


def _model() -> CausalLM:
    """Build a typed stand-in model object for dispatcher tests."""
    return cast("CausalLM", object())


def _tokenizer() -> Tokenizer:
    """Build a typed stand-in tokenizer object for dispatcher tests."""
    return cast("Tokenizer", MockTokenizer(32))


def _direction() -> Array:
    """Build a small direction vector for dispatcher tests."""
    direction = ops.array([1.0, 0.0, 0.0, 0.0])
    ops.eval(direction)
    return direction


def _environment_config() -> EnvironmentConfig:
    """Build a minimal environment config for rollout wiring tests."""
    return EnvironmentConfig(
        system_prompt="system",
        tools=[ToolSchema(name="noop", description="no-op", parameters={})],
        target=EnvironmentTarget(function="noop"),
        task=EnvironmentTask(content="do something"),
        injection_surface="tool_output",
    )


def _softprompt_result(
    mode: str,
    *,
    success_rate: float = 0.25,
    token_text: str | None = "TOKENS",
) -> SoftPromptResult:
    """Build a minimal soft prompt result for dispatcher stubs."""
    return SoftPromptResult(
        mode=mode,
        success_rate=success_rate,
        final_loss=1.0,
        loss_history=[1.0],
        n_steps=1,
        n_tokens=2,
        embeddings=None,
        token_ids=None,
        token_text=token_text,
        eval_responses=["response"],
    )


class TestRunSingleAttackDispatch:
    """Tests for the raw mode dispatcher."""

    @pytest.mark.parametrize(
        ("mode", "attack_name"),
        [
            pytest.param("continuous", "_continuous_attack", id="continuous"),
            pytest.param("gcg", "_gcg_attack", id="gcg"),
            pytest.param("egd", "_egd_attack", id="egd"),
            pytest.param("cold", "_cold_attack", id="cold"),
            pytest.param("amplecgc", "_amplecgc_attack", id="amplecgc"),
        ],
    )
    def test_dispatches_to_each_mode(
        self,
        mode: str,
        attack_name: str,
    ) -> None:
        """Each configured mode should dispatch to the matching backend."""
        model = _model()
        tokenizer = _tokenizer()
        prompts = ["prompt"]
        config = SoftPromptConfig(mode=mode)
        direction = _direction()
        ref_model = _model()
        transfer_models = [("transfer", model, tokenizer)]
        environment_config = _environment_config()
        expected = _softprompt_result(mode)

        with (
            patch(
                "vauban.softprompt._dispatcher._resolve_injection_ids",
                return_value=None,
            ) as mock_resolve_ids,
            patch(
                f"vauban.softprompt._dispatcher.{attack_name}",
                return_value=expected,
            ) as mock_attack,
        ):
            result = _run_single_attack(
                model,
                tokenizer,
                prompts,
                config,
                direction,
                ref_model,
                transfer_models=transfer_models,
                environment_config=environment_config,
            )

        assert result is expected
        mock_resolve_ids.assert_called_once_with(config, tokenizer, prompts)
        if mode == "continuous":
            mock_attack.assert_called_once_with(
                model,
                tokenizer,
                prompts,
                config,
                direction,
                ref_model,
            )
        else:
            mock_attack.assert_called_once_with(
                model,
                tokenizer,
                prompts,
                config,
                direction,
                ref_model,
                all_prompt_ids_override=None,
                transfer_models=transfer_models,
                infix_map=None,
                environment_config=environment_config,
            )

    def test_paraphrase_and_infix_overrides_are_forwarded(self) -> None:
        """Paraphrase expansion and infix override resolution should compose."""
        model = _model()
        tokenizer = _tokenizer()
        prompts = ["prompt {suffix}"]
        config = SoftPromptConfig(
            mode="gcg",
            paraphrase_strategies=["para6"],
            token_position="infix",
            system_prompt="system",
        )
        direction = _direction()
        transfer_models = [("transfer", model, tokenizer)]
        environment_config = _environment_config()
        expected = _softprompt_result("gcg")
        infix_ids = [ops.array([7, 8, 9])[None, :]]
        infix_map = {id(infix_ids[0]): 3}

        with (
            patch(
                "vauban.softprompt._dispatcher.paraphrase_prompts",
                return_value=["prompt para6"],
            ) as mock_paraphrase,
            patch(
                "vauban.softprompt._dispatcher._resolve_injection_ids",
                return_value=None,
            ) as mock_resolve_ids,
            patch(
                "vauban.softprompt._dispatcher._resolve_infix_overrides",
                return_value=(infix_ids, infix_map),
            ) as mock_resolve_infix,
            patch(
                "vauban.softprompt._dispatcher._gcg_attack",
                return_value=expected,
            ) as mock_attack,
        ):
            result = _run_single_attack(
                model,
                tokenizer,
                prompts,
                config,
                direction,
                None,
                transfer_models=transfer_models,
                environment_config=environment_config,
            )

        assert result is expected
        mock_paraphrase.assert_called_once_with(prompts, ["para6"])
        mock_resolve_ids.assert_called_once_with(config, tokenizer, ["prompt para6"])
        mock_resolve_infix.assert_called_once_with(
            tokenizer,
            ["prompt para6"],
            "system",
        )
        mock_attack.assert_called_once_with(
            model,
            tokenizer,
            ["prompt para6"],
            config,
            direction,
            None,
            all_prompt_ids_override=infix_ids,
            transfer_models=transfer_models,
            infix_map=infix_map,
            environment_config=environment_config,
        )

    def test_unknown_mode_raises_value_error(self) -> None:
        """An unknown mode should fail loudly instead of guessing."""
        model = _model()
        tokenizer = _tokenizer()
        config = SoftPromptConfig(mode="unknown")

        with (
            patch(
                "vauban.softprompt._dispatcher._resolve_injection_ids",
                return_value=None,
            ),
            pytest.raises(ValueError, match="Unknown soft prompt mode"),
        ):
            _run_single_attack(
                model,
                tokenizer,
                ["prompt"],
                config,
                None,
            )


class TestSoftPromptAttackDispatch:
    """Tests for the top-level softprompt orchestration wrapper."""

    def test_largo_branch_seeds_and_returns_early(self) -> None:
        """LARGO should short-circuit before single-attack dispatch."""
        model = _model()
        tokenizer = _tokenizer()
        config = SoftPromptConfig(
            mode="continuous",
            seed=11,
            largo_reflection_rounds=2,
        )
        expected = _softprompt_result("continuous")

        with (
            patch("vauban.softprompt._dispatcher.ops.random.seed") as mock_seed,
            patch("vauban.softprompt._dispatcher.random.seed") as mock_py_seed,
            patch(
                "vauban.softprompt._largo.largo_loop",
                return_value=expected,
            ) as mock_loop,
            patch("vauban.softprompt._dispatcher._run_single_attack") as mock_single,
        ):
            result = softprompt_attack(model, tokenizer, ["prompt"], config, None)

        assert result is expected
        mock_seed.assert_called_once_with(11)
        mock_py_seed.assert_called_once_with(11)
        mock_loop.assert_called_once()
        mock_single.assert_not_called()

    def test_gan_branch_returns_early(self) -> None:
        """GAN mode should short-circuit before the single-attack path."""
        model = _model()
        tokenizer = _tokenizer()
        config = SoftPromptConfig(mode="gcg", gan_rounds=3)
        api_eval_config = ApiEvalConfig(endpoints=[])
        environment_config = _environment_config()
        expected = _softprompt_result("gcg")

        with (
            patch(
                "vauban.softprompt._dispatcher.gan_loop",
                return_value=expected,
            ) as mock_loop,
            patch("vauban.softprompt._dispatcher._run_single_attack") as mock_single,
        ):
            result = softprompt_attack(
                model,
                tokenizer,
                ["prompt"],
                config,
                None,
                api_eval_config=api_eval_config,
                environment_config=environment_config,
            )

        assert result is expected
        assert mock_loop.call_args is not None
        assert mock_loop.call_args.args == (
            model,
            tokenizer,
            ["prompt"],
            config,
            None,
            None,
        )
        assert mock_loop.call_args.kwargs["transfer_models"] is None
        assert mock_loop.call_args.kwargs["api_eval_config"] is api_eval_config
        assert mock_loop.call_args.kwargs["environment_config"] is (
            environment_config
        )
        mock_single.assert_not_called()

    def test_environment_rollout_and_defense_eval_update_result(self) -> None:
        """Continuous mode should roll out the environment and attach defenses."""
        model = _model()
        tokenizer = _tokenizer()
        direction = _direction()
        config = SoftPromptConfig(
            mode="continuous",
            defense_eval="both",
            defense_eval_layer=7,
        )
        environment_config = _environment_config()
        original = _softprompt_result(
            "continuous",
            success_rate=0.1,
            token_text="TOKENS",
        )
        env_result = EnvironmentResult(
            reward=0.8,
            target_called=True,
            target_args_match=True,
            turns=[],
            tool_calls_made=[],
            injection_payload="TOKENS",
        )
        defense_result = DefenseEvalResult(
            sic_blocked=1,
            sic_sanitized=0,
            sic_clean=2,
            sic_bypass_rate=0.5,
            cast_interventions=3,
            cast_refusal_rate=0.25,
            cast_responses=["refusal"],
        )

        with (
            patch(
                "vauban.softprompt._dispatcher._run_single_attack",
                return_value=original,
            ) as mock_single,
            patch(
                "vauban.environment.run_agent_loop",
                return_value=env_result,
            ) as mock_rollout,
            patch(
                "vauban.softprompt._dispatcher.evaluate_against_defenses",
                return_value=defense_result,
            ) as mock_defense,
        ):
            result = softprompt_attack(
                model,
                tokenizer,
                ["prompt"],
                config,
                direction,
                environment_config=environment_config,
            )

        assert result is not original
        assert result.success_rate == 0.8
        assert result.defense_eval is defense_result
        mock_single.assert_called_once_with(
            model,
            tokenizer,
            ["prompt"],
            config,
            direction,
            None,
            transfer_models=None,
            environment_config=environment_config,
            svf_boundary=None,
        )
        mock_rollout.assert_called_once_with(
            model,
            tokenizer,
            environment_config,
            "TOKENS",
        )
        mock_defense.assert_called_once_with(
            model,
            tokenizer,
            ["prompt"],
            config,
            direction,
            7,
            "TOKENS",
        )

    def test_environment_rollout_keeps_success_rate_for_nonpositive_reward(
        self,
    ) -> None:
        """Nonpositive rollout rewards should not raise the success rate."""
        model = _model()
        tokenizer = _tokenizer()
        config = SoftPromptConfig(mode="continuous")
        environment_config = _environment_config()
        original = _softprompt_result(
            "continuous",
            success_rate=0.4,
            token_text="TOKENS",
        )
        env_result = EnvironmentResult(
            reward=0.0,
            target_called=False,
            target_args_match=False,
            turns=[],
            tool_calls_made=[],
            injection_payload="TOKENS",
        )

        with (
            patch(
                "vauban.softprompt._dispatcher._run_single_attack",
                return_value=original,
            ) as mock_single,
            patch(
                "vauban.environment.run_agent_loop",
                return_value=env_result,
            ) as mock_rollout,
            patch(
                "vauban.softprompt._dispatcher.evaluate_against_defenses",
            ) as mock_defense,
        ):
            result = softprompt_attack(
                model,
                tokenizer,
                ["prompt"],
                config,
                None,
                environment_config=environment_config,
            )

        assert result is original
        assert result.success_rate == 0.4
        mock_single.assert_called_once()
        mock_rollout.assert_called_once_with(
            model,
            tokenizer,
            environment_config,
            "TOKENS",
        )
        mock_defense.assert_not_called()

    def test_defense_eval_defaults_to_layer_zero_when_unspecified(self) -> None:
        """Defense evaluation should default to layer zero when unset."""
        model = _model()
        tokenizer = _tokenizer()
        direction = _direction()
        config = SoftPromptConfig(
            mode="continuous",
            defense_eval="sic",
        )
        original = _softprompt_result("continuous", token_text="TOKENS")
        defense_result = DefenseEvalResult(
            sic_blocked=0,
            sic_sanitized=1,
            sic_clean=2,
            sic_bypass_rate=1.0,
            cast_interventions=0,
            cast_refusal_rate=0.0,
            cast_responses=[],
        )

        with (
            patch(
                "vauban.softprompt._dispatcher._run_single_attack",
                return_value=original,
            ),
            patch(
                "vauban.softprompt._dispatcher.evaluate_against_defenses",
                return_value=defense_result,
            ) as mock_defense,
        ):
            result = softprompt_attack(
                model,
                tokenizer,
                ["prompt"],
                config,
                direction,
            )

        assert result.defense_eval is defense_result
        mock_defense.assert_called_once_with(
            model,
            tokenizer,
            ["prompt"],
            config,
            direction,
            0,
            "TOKENS",
        )

    def test_defense_eval_is_skipped_without_configuration(self) -> None:
        """A missing defense-eval config should skip the post-attack hook."""
        model = _model()
        tokenizer = _tokenizer()
        direction = _direction()
        config = SoftPromptConfig(mode="continuous")
        original = _softprompt_result("continuous", token_text="TOKENS")

        with (
            patch(
                "vauban.softprompt._dispatcher._run_single_attack",
                return_value=original,
            ),
            patch(
                "vauban.softprompt._dispatcher.evaluate_against_defenses",
            ) as mock_defense,
        ):
            result = softprompt_attack(
                model,
                tokenizer,
                ["prompt"],
                config,
                direction,
            )

        assert result is original
        mock_defense.assert_not_called()
