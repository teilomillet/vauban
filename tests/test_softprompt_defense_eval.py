# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for softprompt post-attack defense evaluation."""

from typing import cast
from unittest.mock import patch

from vauban import _ops as ops
from vauban._array import Array
from vauban.evaluate import DEFAULT_REFUSAL_PHRASES
from vauban.softprompt._defense_eval import (
    _eval_cast,
    _eval_cast_multiturn,
    _eval_sic,
    evaluate_against_defenses,
    evaluate_against_defenses_multiturn,
)
from vauban.types import CastResult, CausalLM, SICResult, SoftPromptConfig, Tokenizer


def _direction() -> Array:
    """Build a small direction vector for defense-eval tests."""
    direction = ops.array([1.0, 0.0, 0.0, 0.0])
    ops.eval(direction)
    return direction


def _model() -> CausalLM:
    """Build a typed stand-in model object for patched tests."""
    return cast("CausalLM", object())


def _tokenizer() -> Tokenizer:
    """Build a typed stand-in tokenizer object for patched tests."""
    return cast("Tokenizer", object())


class TestEvaluateAgainstDefenses:
    """Tests for single-turn defense evaluation orchestration."""

    def test_runs_sic_and_cast_with_configured_thresholds(self) -> None:
        model = _model()
        tokenizer = _tokenizer()
        direction = _direction()
        config = SoftPromptConfig(
            defense_eval="both",
            defense_eval_layer=5,
            defense_eval_threshold=0.1,
            defense_eval_sic_threshold=0.3,
            defense_eval_sic_mode="generation",
            defense_eval_sic_max_iterations=4,
            defense_eval_cast_layers=[1, 2],
            defense_eval_alpha=1.5,
            defense_eval_alpha_tiers=[(0.2, 2.0)],
            token_position="infix",
            max_gen_tokens=6,
        )

        with (
            patch(
                "vauban.softprompt._defense_eval._eval_sic",
                return_value=(1, 2, 3, 0.75),
            ) as mock_sic,
            patch(
                "vauban.softprompt._defense_eval._eval_cast",
                return_value=(4, 0.5, ["refusal"]),
            ) as mock_cast,
        ):
            result = evaluate_against_defenses(
                model,
                tokenizer,
                ["prompt {suffix}"],
                config,
                direction,
                layer_index=2,
                token_text="TOKENS",
            )

        assert result.sic_blocked == 1
        assert result.sic_sanitized == 2
        assert result.sic_clean == 3
        assert result.sic_bypass_rate == 0.75
        assert result.cast_interventions == 4
        assert result.cast_refusal_rate == 0.5
        assert result.cast_responses == ["refusal"]
        mock_sic.assert_called_once_with(
            model,
            tokenizer,
            ["prompt TOKENS"],
            direction,
            5,
            0.3,
            "generation",
            4,
        )
        cast_tiers = mock_cast.call_args.kwargs["alpha_tiers"]
        assert cast_tiers is not None
        assert len(cast_tiers) == 1
        assert cast_tiers[0].threshold == 0.2
        assert cast_tiers[0].alpha == 2.0

    def test_multiturn_uses_history_for_sic_and_messages_for_cast(self) -> None:
        model = _model()
        tokenizer = _tokenizer()
        direction = _direction()
        history = [{"role": "assistant", "content": "previous answer"}]
        config = SoftPromptConfig(
            defense_eval="both",
            defense_eval_threshold=0.2,
            defense_eval_alpha=0.7,
            token_position="suffix",
        )

        with (
            patch(
                "vauban.softprompt._defense_eval._build_sic_prompts_with_history",
                return_value=["history prompt"],
            ) as mock_build,
            patch(
                "vauban.softprompt._defense_eval._eval_sic",
                return_value=(0, 1, 0, 0.5),
            ) as mock_sic,
            patch(
                "vauban.softprompt._defense_eval._eval_cast_multiturn",
                return_value=(2, 0.5, ["refusal"]),
            ) as mock_cast,
        ):
            result = evaluate_against_defenses_multiturn(
                model,
                tokenizer,
                ["prompt"],
                config,
                direction,
                layer_index=7,
                token_text="TOKENS",
                history=history,
            )

        assert result.sic_sanitized == 1
        assert result.cast_interventions == 2
        mock_build.assert_called_once_with(["prompt TOKENS"], history)
        mock_sic.assert_called_once_with(
            model,
            tokenizer,
            ["history prompt"],
            direction,
            7,
            0.2,
            "direction",
            3,
        )
        mock_cast.assert_called_once_with(
            model,
            tokenizer,
            ["prompt TOKENS"],
            direction,
            [7],
            0.7,
            0.2,
            config.max_gen_tokens,
            history,
            alpha_tiers=None,
        )


class TestEvalSic:
    """Tests for SIC metrics aggregation."""

    def test_bypass_rate_uses_sanitized_plus_clean(self) -> None:
        result = SICResult(
            prompts_clean=["clean-1", "clean-2"],
            prompts_blocked=[False, True],
            iterations_used=[1, 2],
            initial_scores=[0.4, 0.7],
            final_scores=[0.1, 0.8],
            total_blocked=1,
            total_sanitized=1,
            total_clean=1,
        )

        with patch("vauban.sic.sic", return_value=result) as mock_sic:
            metrics = _eval_sic(
                _model(),
                _tokenizer(),
                ["p1", "p2"],
                _direction(),
                layer=3,
                threshold=0.5,
                sic_mode="generation",
                sic_max_iterations=4,
            )

        assert metrics == (1, 1, 1, 1.0)
        mock_sic.assert_called_once()

    def test_empty_prompt_list_returns_zero_bypass_rate(self) -> None:
        result = SICResult(
            prompts_clean=[],
            prompts_blocked=[],
            iterations_used=[],
            initial_scores=[],
            final_scores=[],
            total_blocked=0,
            total_sanitized=0,
            total_clean=0,
        )

        with patch("vauban.sic.sic", return_value=result):
            assert _eval_sic(
                _model(),
                _tokenizer(),
                [],
                _direction(),
                layer=0,
                threshold=0.1,
            ) == (0, 0, 0, 0.0)


class TestEvalCast:
    """Tests for CAST metrics aggregation."""

    def test_counts_interventions_and_refusals(self) -> None:
        refusal_text = DEFAULT_REFUSAL_PHRASES[0]
        responses = [
            CastResult(
                prompt="p1",
                text=refusal_text,
                projections_before=[],
                projections_after=[],
                interventions=2,
                considered=1,
            ),
            CastResult(
                prompt="p2",
                text="safe answer",
                projections_before=[],
                projections_after=[],
                interventions=1,
                considered=1,
            ),
        ]

        with patch(
            "vauban.cast.cast_generate",
            side_effect=responses,
        ) as mock_cast:
            interventions, refusal_rate, texts = _eval_cast(
                _model(),
                _tokenizer(),
                ["p1", "p2"],
                _direction(),
                layers=[0],
                alpha=1.0,
                threshold=0.0,
                max_tokens=8,
            )

        assert interventions == 3
        assert refusal_rate == 0.5
        assert texts == [refusal_text, "safe answer"]
        assert mock_cast.call_count == 2

    def test_empty_prompt_list_returns_zero_refusal_rate(self) -> None:
        assert _eval_cast(
            _model(),
            _tokenizer(),
            [],
            _direction(),
            layers=[0],
            alpha=1.0,
            threshold=0.0,
            max_tokens=8,
        ) == (0, 0.0, [])

    def test_multiturn_builds_messages_and_counts_refusals(self) -> None:
        refusal_text = DEFAULT_REFUSAL_PHRASES[0]
        history = [{"role": "assistant", "content": "before"}]
        responses = [
            CastResult(
                prompt="p1",
                text=refusal_text,
                projections_before=[],
                projections_after=[],
                interventions=1,
                considered=1,
            ),
            CastResult(
                prompt="p2",
                text="safe",
                projections_before=[],
                projections_after=[],
                interventions=3,
                considered=1,
            ),
        ]

        with patch(
            "vauban.cast.cast_generate_with_messages",
            side_effect=responses,
        ) as mock_cast:
            interventions, refusal_rate, texts = _eval_cast_multiturn(
                _model(),
                _tokenizer(),
                ["prompt-1", "prompt-2"],
                _direction(),
                layers=[1],
                alpha=0.7,
                threshold=0.2,
                max_tokens=5,
                history=history,
            )

        assert interventions == 4
        assert refusal_rate == 0.5
        assert texts == [refusal_text, "safe"]
        first_messages = mock_cast.call_args_list[0].args[2]
        assert first_messages == [
            {"role": "assistant", "content": "before"},
            {"role": "user", "content": "prompt-1"},
        ]

    def test_multiturn_empty_prompt_list_returns_zero_refusal_rate(self) -> None:
        assert _eval_cast_multiturn(
            _model(),
            _tokenizer(),
            [],
            _direction(),
            layers=[1],
            alpha=0.7,
            threshold=0.2,
            max_tokens=5,
            history=[{"role": "assistant", "content": "before"}],
        ) == (0, 0.0, [])
