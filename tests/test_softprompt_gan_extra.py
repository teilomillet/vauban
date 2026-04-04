# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Extra tests for vauban.softprompt GAN orchestration branches."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest

from tests.conftest import (
    D_MODEL,
    NUM_HEADS,
    NUM_LAYERS,
    VOCAB_SIZE,
    MockCausalLM,
    MockTokenizer,
)
from vauban import _ops as ops
from vauban.softprompt._gan import (
    _dispatch_attack,
    _dispatch_attack_multiturn,
    gan_loop,
)
from vauban.types import (
    ApiEvalConfig,
    CausalLM,
    DefenseEvalResult,
    DefenseProxyResult,
    EnvironmentConfig,
    EnvironmentResult,
    EnvironmentTarget,
    EnvironmentTask,
    SoftPromptConfig,
    SoftPromptResult,
    Tokenizer,
    ToolSchema,
    TransferEvalResult,
)

if TYPE_CHECKING:
    from vauban._array import Array


def _make_model() -> MockCausalLM:
    """Create and eval a small mock model."""
    model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
    ops.eval(model.parameters())
    return model


def _make_tokenizer() -> MockTokenizer:
    """Create the matching mock tokenizer."""
    return MockTokenizer(VOCAB_SIZE)


def _make_direction() -> Array:
    """Create a deterministic direction vector."""
    direction = ops.array([1.0, 0.0, 0.0, 0.0])
    ops.eval(direction)
    return direction


def _make_attack_result(
    *,
    success_rate: float = 0.5,
    token_text: str = "TOKENS",
    token_ids: list[int] | None = None,
    embeddings: Array | None = None,
    eval_responses: list[str] | None = None,
) -> SoftPromptResult:
    """Build a minimal softprompt attack result."""
    return SoftPromptResult(
        mode="gcg",
        success_rate=success_rate,
        final_loss=1.0,
        loss_history=[1.0],
        n_steps=1,
        n_tokens=4,
        embeddings=embeddings,
        token_ids=token_ids,
        token_text=token_text,
        eval_responses=eval_responses if eval_responses is not None else ["resp"],
        accessibility_score=0.0,
        per_prompt_losses=[1.0],
        early_stopped=False,
        transfer_results=[],
        defense_eval=None,
        gan_history=[],
    )


def _make_environment_config() -> EnvironmentConfig:
    """Build a small environment config for GAN-loop evaluation paths."""
    return EnvironmentConfig(
        system_prompt="SYS",
        tools=[
            ToolSchema(
                name="read_inbox",
                description="Read inbox",
                parameters={},
                result="mail",
            ),
            ToolSchema(
                name="send_email",
                description="Send email",
                parameters={"to": "string"},
                result=None,
            ),
        ],
        target=EnvironmentTarget(
            function="send_email",
            required_args=["to"],
            arg_contains={"to": "attacker@"},
        ),
        task=EnvironmentTask(content="Check inbox"),
        injection_surface="read_inbox",
        max_turns=1,
        max_gen_tokens=4,
    )


class TestDispatchAttack:
    """Tests for _dispatch_attack routing and prompt preprocessing."""

    def test_continuous_dispatch_routes_paraphrases(self) -> None:
        model = cast("CausalLM", _make_model())
        tokenizer = cast("Tokenizer", _make_tokenizer())
        config = SoftPromptConfig(
            mode="continuous",
            paraphrase_strategies=["para1"],
        )
        attack_result = _make_attack_result()

        with (
            patch(
                "vauban.softprompt._gan.paraphrase_prompts",
                return_value=["paraphrased"],
            ) as mock_para,
            patch(
                "vauban.softprompt._gan._resolve_injection_ids",
                return_value=None,
            ) as mock_injection_ids,
            patch(
                "vauban.softprompt._gan._continuous_attack",
                return_value=attack_result,
            ) as mock_attack,
        ):
            result = _dispatch_attack(
                model,
                tokenizer,
                ["prompt"],
                config,
                direction=None,
                ref_model=None,
            )

        assert result == attack_result
        mock_para.assert_called_once_with(["prompt"], ["para1"])
        mock_injection_ids.assert_called_once()
        mock_attack.assert_called_once()
        assert mock_attack.call_args.args[2] == ["paraphrased"]

    def test_cold_dispatch_uses_infix_resolution(self) -> None:
        model = cast("CausalLM", _make_model())
        tokenizer = cast("Tokenizer", _make_tokenizer())
        config = SoftPromptConfig(
            mode="cold",
            token_position="infix",
            system_prompt="SYS",
        )
        attack_result = _make_attack_result()

        with (
            patch(
                "vauban.softprompt._gan._resolve_injection_ids",
                return_value=None,
            ) as mock_injection_ids,
            patch(
                "vauban.softprompt._gan._resolve_infix_overrides",
                return_value=(ops.array([[9, 8]]), {0: 1}),
            ) as mock_infix,
            patch(
                "vauban.softprompt._gan._cold_attack",
                return_value=attack_result,
            ) as mock_attack,
        ):
            result = _dispatch_attack(
                model,
                tokenizer,
                ["prompt"],
                config,
                direction=None,
                ref_model=None,
            )

        assert result == attack_result
        mock_injection_ids.assert_called_once()
        mock_infix.assert_called_once_with(tokenizer, ["prompt"], "SYS")
        assert mock_attack.call_args.kwargs["all_prompt_ids_override"].shape == (1, 2)
        assert mock_attack.call_args.kwargs["infix_map"] == {0: 1}

    def test_gcg_infix_dispatch_forwards_paraphrases_and_infix_map(self) -> None:
        model = cast("CausalLM", _make_model())
        tokenizer = cast("Tokenizer", _make_tokenizer())
        config = SoftPromptConfig(
            mode="gcg",
            n_tokens=4,
            n_steps=1,
            batch_size=4,
            top_k=8,
            token_position="infix",
            paraphrase_strategies=["para1"],
            system_prompt="SYS",
        )
        attack_result = _make_attack_result()

        with (
            patch(
                "vauban.softprompt._gan.paraphrase_prompts",
                return_value=["paraphrased"],
            ) as mock_para,
            patch(
                "vauban.softprompt._gan._resolve_injection_ids",
                return_value=None,
            ) as mock_injection_ids,
            patch(
                "vauban.softprompt._gan._resolve_infix_overrides",
                return_value=(ops.array([[1, 2, 3]]), {0: 2}),
            ) as mock_infix,
            patch(
                "vauban.softprompt._gan._gcg_attack",
                return_value=attack_result,
            ) as mock_attack,
        ):
            result = _dispatch_attack(
                model,
                tokenizer,
                ["prompt"],
                config,
                direction=None,
                ref_model=None,
                transfer_models=[("transfer", model, tokenizer)],
                environment_config=None,
            )

        assert result == attack_result
        mock_para.assert_called_once_with(["prompt"], ["para1"])
        mock_injection_ids.assert_called_once()
        mock_infix.assert_called_once_with(tokenizer, ["paraphrased"], "SYS")
        assert mock_attack.call_args.args[2] == ["paraphrased"]
        assert mock_attack.call_args.kwargs["all_prompt_ids_override"].shape == (1, 3)
        assert mock_attack.call_args.kwargs["infix_map"] == {0: 2}

    def test_unknown_mode_raises(self) -> None:
        model = cast("CausalLM", _make_model())
        tokenizer = cast("Tokenizer", _make_tokenizer())
        config = SoftPromptConfig(mode="unknown")

        with pytest.raises(ValueError, match="Unknown soft prompt mode"):
            _dispatch_attack(
                model,
                tokenizer,
                ["prompt"],
                config,
                direction=None,
                ref_model=None,
            )


class TestDispatchAttackMultiturn:
    """Tests for history-aware dispatch."""

    def test_egd_history_path_forwards_preencoded_ids(self) -> None:
        model = cast("CausalLM", _make_model())
        tokenizer = cast("Tokenizer", _make_tokenizer())
        config = SoftPromptConfig(mode="egd", token_position="suffix")
        history = [{"role": "assistant", "content": "prev"}]
        attack_result = _make_attack_result(token_ids=[7, 8])

        with (
            patch(
                "vauban.softprompt._gan._pre_encode_prompts_with_history",
                return_value=ops.array([[1, 2, 3]]),
            ) as mock_encode,
            patch(
                "vauban.softprompt._gan._egd_attack",
                return_value=attack_result,
            ) as mock_attack,
        ):
            result = _dispatch_attack_multiturn(
                model,
                tokenizer,
                ["prompt"],
                config,
                direction=None,
                ref_model=None,
                history=history,
            )

        assert result == attack_result
        mock_encode.assert_called_once_with(
            tokenizer,
            ["prompt"],
            history,
            None,
        )
        assert mock_attack.call_args.kwargs["all_prompt_ids_override"].shape == (1, 3)

    def test_invalid_mode_raises(self) -> None:
        model = cast("CausalLM", _make_model())
        tokenizer = cast("Tokenizer", _make_tokenizer())
        config = SoftPromptConfig(mode="continuous")

        with pytest.raises(ValueError, match="Multi-turn attack requires"):
            _dispatch_attack_multiturn(
                model,
                tokenizer,
                ["prompt"],
                config,
                direction=None,
                ref_model=None,
                history=[],
            )


class TestGanLoop:
    """Tests for the GAN round orchestration."""

    def test_attack_escalates_after_failed_round(self) -> None:
        model = cast("CausalLM", _make_model())
        tokenizer = cast("Tokenizer", _make_tokenizer())
        config = SoftPromptConfig(
            mode="gcg",
            gan_rounds=2,
            n_tokens=4,
            n_steps=10,
            direction_weight=0.25,
            gan_step_multiplier=2.0,
            gan_direction_escalation=0.5,
            gan_token_escalation=3,
        )
        first = _make_attack_result(token_ids=[11, 12], success_rate=0.1)
        second = _make_attack_result(token_ids=[21, 22], success_rate=0.2)

        with patch(
            "vauban.softprompt._gan._dispatch_attack",
            side_effect=[first, second],
        ) as mock_attack:
            result = gan_loop(
                model,
                tokenizer,
                ["a", "b"],
                config,
                direction=None,
            )

        assert len(result.gan_history) == 2
        second_config = mock_attack.call_args_list[1].args[3]
        assert second_config.n_steps == 20
        assert second_config.direction_weight == pytest.approx(0.75)
        assert second_config.n_tokens == 7
        assert second_config.init_tokens == [11, 12]

    def test_defense_win_breaks_when_escalation_off(self) -> None:
        model = cast("CausalLM", _make_model())
        tokenizer = cast("Tokenizer", _make_tokenizer())
        direction = _make_direction()
        config = SoftPromptConfig(
            mode="gcg",
            gan_rounds=2,
            defense_eval="both",
        )
        first = _make_attack_result(token_ids=[1, 2], success_rate=0.3)
        defense = DefenseEvalResult(
            sic_blocked=0,
            sic_sanitized=0,
            sic_clean=1,
            sic_bypass_rate=1.0,
            cast_interventions=0,
            cast_refusal_rate=0.0,
            cast_responses=["ok"],
        )

        with (
            patch(
                "vauban.softprompt._gan._dispatch_attack",
                return_value=first,
            ) as mock_attack,
            patch(
                "vauban.softprompt._gan.evaluate_against_defenses",
                return_value=defense,
            ) as mock_defense,
        ):
            result = gan_loop(
                model,
                tokenizer,
                ["prompt"],
                config,
                direction=direction,
            )

        assert len(result.gan_history) == 1
        assert mock_attack.call_count == 1
        mock_defense.assert_called_once()

    def test_defense_win_escalates_when_enabled(self) -> None:
        model = cast("CausalLM", _make_model())
        tokenizer = cast("Tokenizer", _make_tokenizer())
        direction = _make_direction()
        config = SoftPromptConfig(
            mode="gcg",
            gan_rounds=2,
            defense_eval="both",
            gan_defense_escalation=True,
            defense_eval_alpha=1.0,
            defense_eval_threshold=0.0,
            defense_eval_sic_max_iterations=3,
        )
        first = _make_attack_result(token_ids=[1, 2], success_rate=0.3)
        second = _make_attack_result(token_ids=[3, 4], success_rate=0.4)
        defense = DefenseEvalResult(
            sic_blocked=0,
            sic_sanitized=0,
            sic_clean=1,
            sic_bypass_rate=1.0,
            cast_interventions=0,
            cast_refusal_rate=0.0,
            cast_responses=["ok"],
        )

        with (
            patch(
                "vauban.softprompt._gan._dispatch_attack",
                side_effect=[first, second],
            ) as mock_attack,
            patch(
                "vauban.softprompt._gan.evaluate_against_defenses",
                return_value=defense,
            ),
        ):
            result = gan_loop(
                model,
                tokenizer,
                ["prompt"],
                config,
                direction=direction,
            )

        assert len(result.gan_history) == 2
        second_config = mock_attack.call_args_list[1].args[3]
        assert second_config.defense_eval_alpha == pytest.approx(1.5)
        assert second_config.defense_eval_threshold == pytest.approx(-0.5)
        assert second_config.defense_eval_sic_max_iterations == 4

    def test_transfer_api_and_environment_paths(self) -> None:
        model = _make_model()
        tokenizer = _make_tokenizer()
        direction = _make_direction()
        transfer_model = _make_model()
        transfer_tokenizer = _make_tokenizer()
        config = SoftPromptConfig(
            mode="gcg",
            gan_rounds=1,
            defense_eval="both",
        )
        api_cfg = ApiEvalConfig(endpoints=[])
        env_cfg = _make_environment_config()
        embeddings = ops.ones((1, 2, D_MODEL))
        ops.eval(embeddings)
        attack = _make_attack_result(
            token_ids=None,
            embeddings=embeddings,
            eval_responses=["assistant reply"],
        )
        defense = DefenseEvalResult(
            sic_blocked=1,
            sic_sanitized=0,
            sic_clean=0,
            sic_bypass_rate=0.0,
            cast_interventions=0,
            cast_refusal_rate=0.0,
            cast_responses=[],
        )
        env_result = EnvironmentResult(
            reward=1.0,
            target_called=True,
            target_args_match=True,
            turns=[],
            tool_calls_made=[],
            injection_payload="TOKENS",
        )
        proxy_result = DefenseProxyResult(
            total_prompts=1,
            sic_blocked=0,
            sic_sanitized=0,
            cast_gated=0,
            prompts_sent=1,
            proxy_mode="sic",
            cast_responses=[],
        )
        api_results = [
            TransferEvalResult(
                model_id="api",
                success_rate=0.8,
                eval_responses=["api response"],
            ),
        ]
        main_transformer = model.model
        transfer_transformer = transfer_model.model

        with (
            patch(
                "vauban.softprompt._gan._dispatch_attack",
                return_value=attack,
            ),
            patch(
                "vauban.softprompt._gan.evaluate_against_defenses",
                return_value=defense,
            ),
            patch(
                "vauban.softprompt._gan.get_transformer",
                side_effect=lambda current: (
                    main_transformer if current is model else transfer_transformer
                ),
            ),
            patch(
                "vauban.softprompt._gan._project_to_tokens",
                return_value=[7, 8],
            ) as mock_project,
            patch(
                "vauban.softprompt._gan._evaluate_attack",
                return_value=(0.6, ["transfer eval"]),
            ) as mock_eval_attack,
            patch(
                "vauban.environment.run_agent_loop",
                return_value=env_result,
            ) as mock_env,
            patch(
                "vauban.api_eval.evaluate_suffix_via_api",
                return_value=api_results,
            ) as mock_api,
            patch(
                "vauban.api_eval_proxy.evaluate_with_defense_proxy",
                return_value=(api_results, proxy_result),
            ) as mock_proxy,
        ):
            result = gan_loop(
                model,
                tokenizer,
                ["prompt"],
                config,
                direction=direction,
                transfer_models=[("transfer", transfer_model, transfer_tokenizer)],
                api_eval_config=api_cfg,
                environment_config=env_cfg,
            )

        assert mock_project.called
        assert mock_eval_attack.called
        assert mock_env.called
        assert mock_api.called
        assert not mock_proxy.called
        assert len(result.gan_history) == 1
        round_result = result.gan_history[0]
        assert round_result.environment_result is not None
        assert round_result.environment_result.reward == 1.0
        assert round_result.transfer_results == [
            TransferEvalResult(
                model_id="transfer",
                success_rate=0.6,
                eval_responses=["transfer eval"],
            ),
            *api_results,
        ]

    def test_api_only_path_recomputes_mean_transfer(self) -> None:
        model = cast("CausalLM", _make_model())
        tokenizer = cast("Tokenizer", _make_tokenizer())
        config = SoftPromptConfig(
            mode="gcg",
            gan_rounds=1,
            defense_eval="both",
        )
        api_cfg = ApiEvalConfig(endpoints=[])
        attack = _make_attack_result(token_ids=[1, 2], eval_responses=["ok"])
        api_results = [
            TransferEvalResult(
                model_id="api",
                success_rate=0.75,
                eval_responses=["api response"],
            ),
        ]

        with (
            patch(
                "vauban.softprompt._gan._dispatch_attack",
                return_value=attack,
            ),
            patch(
                "vauban.api_eval.evaluate_suffix_via_api",
                return_value=api_results,
            ) as mock_api,
        ):
            result = gan_loop(
                model,
                tokenizer,
                ["prompt"],
                config,
                direction=None,
                api_eval_config=api_cfg,
            )

        assert mock_api.called
        assert len(result.gan_history) == 1
        round_result = result.gan_history[0]
        assert round_result.transfer_results == api_results

    def test_environment_reward_without_direction_marks_win(self) -> None:
        model = cast("CausalLM", _make_model())
        tokenizer = cast("Tokenizer", _make_tokenizer())
        config = SoftPromptConfig(mode="gcg", gan_rounds=1)
        attack = _make_attack_result(token_ids=[3, 4], eval_responses=["ok"])
        env_result = EnvironmentResult(
            reward=1.0,
            target_called=True,
            target_args_match=True,
            turns=[],
            tool_calls_made=[],
            injection_payload="TOKENS",
        )

        with (
            patch(
                "vauban.softprompt._gan._dispatch_attack",
                return_value=attack,
            ),
            patch(
                "vauban.environment.run_agent_loop",
                return_value=env_result,
            ) as mock_env,
        ):
            result = gan_loop(
                model,
                tokenizer,
                ["prompt"],
                config,
                direction=None,
                environment_config=_make_environment_config(),
            )

        assert mock_env.called
        assert len(result.gan_history) == 1
        assert result.gan_history[0].attacker_won is True

    def test_proxy_path_uses_defense_proxy(self) -> None:
        model = _make_model()
        tokenizer = _make_tokenizer()
        direction = _make_direction()
        config = SoftPromptConfig(
            mode="gcg",
            gan_rounds=1,
            defense_eval="both",
            gan_multiturn=True,
        )
        api_cfg = ApiEvalConfig(
            endpoints=[],
            defense_proxy="sic",
            token_position="suffix",
        )
        attack = _make_attack_result(token_ids=[1, 2], eval_responses=["ok"])
        defense = DefenseEvalResult(
            sic_blocked=0,
            sic_sanitized=0,
            sic_clean=1,
            sic_bypass_rate=1.0,
            cast_interventions=0,
            cast_refusal_rate=0.0,
            cast_responses=[],
        )
        proxy_result = DefenseProxyResult(
            total_prompts=1,
            sic_blocked=1,
            sic_sanitized=0,
            cast_gated=0,
            prompts_sent=0,
            proxy_mode="sic",
            cast_responses=[],
        )
        api_results = [
            TransferEvalResult(
                model_id="proxy-model",
                success_rate=0.5,
                eval_responses=["proxy response"],
            ),
        ]

        with (
            patch(
                "vauban.softprompt._gan._dispatch_attack",
                return_value=attack,
            ),
            patch(
                "vauban.softprompt._gan.evaluate_against_defenses",
                return_value=defense,
            ),
            patch(
                "vauban.api_eval_proxy.evaluate_with_defense_proxy",
                return_value=(api_results, proxy_result),
            ) as mock_proxy,
        ):
            result = gan_loop(
                model,
                tokenizer,
                ["prompt"],
                config,
                direction=direction,
                api_eval_config=api_cfg,
            )

        assert mock_proxy.called
        assert len(result.gan_history) == 1
        assert result.gan_history[0].defense_proxy_result == proxy_result

    def test_zero_rounds_raises(self) -> None:
        model = cast("CausalLM", _make_model())
        tokenizer = cast("Tokenizer", _make_tokenizer())
        config = SoftPromptConfig(mode="gcg", gan_rounds=0)

        with pytest.raises(ValueError, match="gan_loop called with gan_rounds=0"):
            gan_loop(model, tokenizer, ["prompt"], config, direction=None)
