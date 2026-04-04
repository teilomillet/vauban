# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Extra edge-case coverage for the remaining softprompt helper branches."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest
from conftest import (
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
from vauban.softprompt._gcg_candidates import (
    _initialize_restart_tokens,
    _select_step_prompts,
)
from vauban.softprompt._gcg_objective import GCGDefenseConfig, GCGSharedState
from vauban.softprompt._largo import largo_loop
from vauban.softprompt._loss import _compute_defensive_loss, _compute_loss
from vauban.types import (
    ApiEvalConfig,
    CausalLM,
    DefenseEvalResult,
    SoftPromptConfig,
    SoftPromptResult,
    Tokenizer,
    TransferEvalResult,
)

if TYPE_CHECKING:
    from vauban._array import Array


def _make_model() -> MockCausalLM:
    """Build a small mock model for softprompt helper tests."""
    model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
    ops.eval(model.parameters())
    return model


def _make_tokenizer() -> MockTokenizer:
    """Build a matching tokenizer for the mock model."""
    return MockTokenizer(VOCAB_SIZE)


def _make_attack_result(
    *,
    success_rate: float = 0.5,
    token_ids: list[int] | None = None,
    embeddings: Array | None = None,
    eval_responses: list[str] | None = None,
) -> SoftPromptResult:
    """Construct a minimal SoftPromptResult for GAN/LARGO tests."""
    return SoftPromptResult(
        mode="gcg",
        success_rate=success_rate,
        final_loss=1.0,
        loss_history=[1.0],
        n_steps=1,
        n_tokens=2,
        embeddings=embeddings,
        token_ids=token_ids,
        token_text="TOKENS" if token_ids is not None else "",
        eval_responses=(
            eval_responses
            if eval_responses is not None
            else ["This is a detailed answer."]
        ),
        accessibility_score=0.0,
        per_prompt_losses=[1.0],
        early_stopped=False,
        transfer_results=[],
        defense_eval=None,
        gan_history=[],
    )


def _prompt_ids(tokenizer: Tokenizer, text: str) -> Array:
    """Encode a single chat prompt into token IDs."""
    messages = [{"role": "user", "content": text}]
    prompt_text = cast(
        "str",
        tokenizer.apply_chat_template(messages, tokenize=False),
    )
    return ops.array(tokenizer.encode(prompt_text))[None, :]


def _soft_embeds(n_tokens: int) -> Array:
    """Create a small soft-embedding tensor."""
    embeds = ops.ones((1, n_tokens, D_MODEL)) * 0.1
    ops.eval(embeds)
    return embeds


class _FakeBoundary:
    """Tiny SVF boundary stub that records calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, tuple[int, ...]]] = []

    def forward(self, hidden: Array, layer_idx: int) -> Array:
        self.calls.append((layer_idx, hidden.shape))
        return ops.array(0.5)


class TestGcgCandidateEdges:
    """Branches in the GCG candidate helper module."""

    def test_initialize_restart_tokens_extends_short_init(self) -> None:
        config = SoftPromptConfig(mode="gcg", n_tokens=4, init_tokens=[1, 2])

        with patch("random.choice", return_value=9):
            tokens = _initialize_restart_tokens(config, [9], restart_idx=0)

        assert tokens == [1, 2, 9, 9]

    def test_select_step_prompts_sample_branch(self) -> None:
        config = SoftPromptConfig(mode="gcg", prompt_strategy="sample")
        model = _make_model()
        state = GCGSharedState(
            model=model,
            target_ids=ops.array([1]),
            n_tokens=1,
            loss_mode="targeted",
            direction=None,
            direction_weight=0.0,
            direction_mode="last",
            direction_layers=None,
            eos_token_id=None,
            eos_loss_mode="none",
            eos_loss_weight=0.0,
            ref_model=None,
            kl_ref_weight=0.0,
            refusal_ids=None,
            defense=GCGDefenseConfig(
                weight=0.0,
                sic_layer=None,
                sic_threshold=0.0,
                cast_layers=None,
                cast_threshold=0.0,
            ),
            perplexity_weight=0.0,
            token_position="prefix",
            infix_map=None,
        )
        all_prompt_ids = [ops.array([1]), ops.array([2])]
        soft_embeds = ops.zeros((1, 1, D_MODEL))

        with patch(
            "vauban.softprompt._gcg_candidates._sample_prompt_ids",
            return_value=[all_prompt_ids[1]],
        ) as mock_sample:
            selected = _select_step_prompts(
                config,
                state,
                all_prompt_ids,
                soft_embeds,
                step=0,
            )

        assert selected == [all_prompt_ids[1]]
        mock_sample.assert_called_once_with(all_prompt_ids, config.worst_k)


class TestLossCommonEdges:
    """Branches in shared softprompt loss helpers."""

    def test_svf_boundary_branch_in_raid_mode(self) -> None:
        model = _make_model()
        tokenizer = _make_tokenizer()
        soft_embeds = _soft_embeds(2)
        prompt_ids = _prompt_ids(tokenizer, "hello")
        target_ids = ops.array([tokenizer.encode("Sure")[0]])
        boundary = _FakeBoundary()

        loss = _compute_loss(
            model,
            soft_embeds,
            prompt_ids,
            target_ids,
            n_tokens=2,
            direction=None,
            direction_weight=1.0,
            direction_mode="raid",
            direction_layers={0},
            svf_boundary=boundary,
            defense_aware_weight=1.0,
            perplexity_weight=0.0,
        )
        ops.eval(loss)

        assert boundary.calls
        assert boundary.calls[0][0] == 0

    def test_svf_boundary_branch_in_last_mode(self) -> None:
        model = _make_model()
        tokenizer = _make_tokenizer()
        soft_embeds = _soft_embeds(2)
        prompt_ids = _prompt_ids(tokenizer, "hello")
        target_ids = ops.array([tokenizer.encode("Sure")[0]])
        boundary = _FakeBoundary()

        loss = _compute_loss(
            model,
            soft_embeds,
            prompt_ids,
            target_ids,
            n_tokens=2,
            direction=None,
            direction_weight=1.0,
            direction_mode="last",
            svf_boundary=boundary,
            defense_aware_weight=1.0,
            perplexity_weight=0.0,
        )
        ops.eval(loss)

        assert boundary.calls
        assert boundary.calls[0][0] == NUM_LAYERS - 1

    def test_defensive_loss_wrapper_runs(self) -> None:
        model = _make_model()
        tokenizer = _make_tokenizer()
        soft_embeds = _soft_embeds(2)
        prompt_ids = _prompt_ids(tokenizer, "hello")
        refusal_ids = ops.array([0])

        loss = _compute_defensive_loss(
            model,
            soft_embeds,
            prompt_ids,
            n_tokens=2,
            refusal_ids=refusal_ids,
            direction=None,
            direction_weight=0.0,
        )
        ops.eval(loss)

        assert float(loss.item()) >= 0.0


class TestLargoEdges:
    """Branches in the LARGO reflection loop."""

    def test_embeddings_none_uses_eval_responses_and_stops_early(self) -> None:
        model = _make_model()
        tokenizer = _make_tokenizer()
        config = SoftPromptConfig(
            mode="continuous",
            n_tokens=2,
            n_steps=3,
            largo_reflection_rounds=2,
            largo_objective="targeted",
        )
        attack_result = _make_attack_result(
            embeddings=None,
            eval_responses=[
                "This is a detailed answer that is not a refusal.",
            ],
        )

        with (
            patch(
                "vauban.softprompt._largo._continuous_attack",
                return_value=attack_result,
            ) as mock_attack,
            patch(
                "vauban.softprompt._largo._evaluate_attack",
            ) as mock_evaluate,
        ):
            result = largo_loop(
                model,
                tokenizer,
                ["prompt"],
                config,
                direction=None,
            )

        assert result.loss_history == [1.0]
        mock_attack.assert_called_once()
        mock_evaluate.assert_not_called()

    def test_zero_rounds_raises(self) -> None:
        model = _make_model()
        tokenizer = _make_tokenizer()
        config = SoftPromptConfig(mode="continuous", largo_reflection_rounds=0)

        with pytest.raises(
            RuntimeError,
            match="LARGO loop completed without producing any result",
        ):
            largo_loop(model, tokenizer, ["prompt"], config, direction=None)


class TestGanEdges:
    """Branches in GAN dispatch and orchestration."""

    def test_dispatch_attack_uses_egd_branch(self) -> None:
        model = cast(CausalLM, _make_model())  # noqa: TC006
        tokenizer = cast(Tokenizer, _make_tokenizer())  # noqa: TC006
        config = SoftPromptConfig(mode="egd")
        attack_result = _make_attack_result(token_ids=[1, 2])

        with patch(
            "vauban.softprompt._gan._egd_attack",
            return_value=attack_result,
        ) as mock_attack:
            result = _dispatch_attack(
                model,
                tokenizer,
                ["prompt"],
                config,
                direction=None,
                ref_model=None,
            )

        assert result == attack_result
        mock_attack.assert_called_once()

    def test_dispatch_attack_uses_amplecgc_branch(self) -> None:
        model = cast(CausalLM, _make_model())  # noqa: TC006
        tokenizer = cast(Tokenizer, _make_tokenizer())  # noqa: TC006
        config = SoftPromptConfig(mode="amplecgc")
        attack_result = _make_attack_result(token_ids=[1, 2])

        with patch(
            "vauban.softprompt._gan._amplecgc_attack",
            return_value=attack_result,
        ) as mock_attack:
            result = _dispatch_attack(
                model,
                tokenizer,
                ["prompt"],
                config,
                direction=None,
                ref_model=None,
            )

        assert result == attack_result
        mock_attack.assert_called_once()

    def test_multiturn_dispatch_uses_cold_branch(self) -> None:
        model = cast(CausalLM, _make_model())  # noqa: TC006
        tokenizer = cast(Tokenizer, _make_tokenizer())  # noqa: TC006
        config = SoftPromptConfig(mode="cold", token_position="suffix")
        history = [{"role": "assistant", "content": "prev"}]
        attack_result = _make_attack_result(token_ids=[7, 8])

        with (
            patch(
                "vauban.softprompt._gan._pre_encode_prompts_with_history",
                return_value=ops.array([[1, 2]]),
            ) as mock_encode,
            patch(
                "vauban.softprompt._gan._cold_attack",
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
        mock_encode.assert_called_once_with(tokenizer, ["prompt"], history, None)
        mock_attack.assert_called_once()

    def test_multiturn_dispatch_uses_amplecgc_branch(self) -> None:
        model = cast(CausalLM, _make_model())  # noqa: TC006
        tokenizer = cast(Tokenizer, _make_tokenizer())  # noqa: TC006
        config = SoftPromptConfig(mode="amplecgc", token_position="suffix")
        history = [{"role": "assistant", "content": "prev"}]
        attack_result = _make_attack_result(token_ids=[7, 8])

        with (
            patch(
                "vauban.softprompt._gan._pre_encode_prompts_with_history",
                return_value=ops.array([[1, 2]]),
            ) as mock_encode,
            patch(
                "vauban.softprompt._gan._amplecgc_attack",
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
        mock_encode.assert_called_once_with(tokenizer, ["prompt"], history, None)
        mock_attack.assert_called_once()

    @pytest.mark.parametrize("defense_mode", ["sic", "cast"])
    def test_gan_loop_defense_modes(self, defense_mode: str) -> None:
        model = _make_model()
        tokenizer = _make_tokenizer()
        direction = ops.array([1.0] + [0.0] * (D_MODEL - 1))
        ops.eval(direction)
        config = SoftPromptConfig(
            mode="gcg",
            gan_rounds=1,
            defense_eval=defense_mode,
        )
        attack = _make_attack_result(token_ids=[1, 2], eval_responses=["ok"])
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
                return_value=attack,
            ),
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

        assert len(result.gan_history) == 1
        assert result.gan_history[0].defense_result == defense

    def test_gan_loop_multiturn_uses_history_and_trims(self) -> None:
        model = _make_model()
        tokenizer = _make_tokenizer()
        direction = ops.array([1.0] + [0.0] * (D_MODEL - 1))
        ops.eval(direction)
        config = SoftPromptConfig(
            mode="gcg",
            gan_rounds=2,
            gan_multiturn=True,
            gan_multiturn_max_turns=1,
            defense_eval="both",
        )
        first = _make_attack_result(token_ids=[1, 2], eval_responses=["first"])
        second = _make_attack_result(token_ids=[3, 4], eval_responses=["second"])
        defense = DefenseEvalResult(
            sic_blocked=1,
            sic_sanitized=0,
            sic_clean=0,
            sic_bypass_rate=0.0,
            cast_interventions=0,
            cast_refusal_rate=1.0,
            cast_responses=["blocked"],
        )

        with (
            patch(
                "vauban.softprompt._gan._dispatch_attack",
                return_value=first,
            ) as mock_dispatch,
            patch(
                "vauban.softprompt._gan._dispatch_attack_multiturn",
                return_value=second,
            ) as mock_dispatch_multiturn,
            patch(
                "vauban.softprompt._gan.evaluate_against_defenses",
                return_value=defense,
            ),
            patch(
                "vauban.softprompt._gan.evaluate_against_defenses_multiturn",
                return_value=defense,
            ) as mock_defense_multiturn,
        ):
            result = gan_loop(
                model,
                tokenizer,
                ["prompt"],
                config,
                direction=direction,
            )

        assert len(result.gan_history) == 2
        assert mock_dispatch.called
        assert mock_dispatch_multiturn.called
        assert mock_defense_multiturn.called

    def test_gan_loop_uses_transfer_token_ids_path(self) -> None:
        model = _make_model()
        tokenizer = _make_tokenizer()
        transfer_model = _make_model()
        transfer_tokenizer = _make_tokenizer()
        config = SoftPromptConfig(mode="gcg", gan_rounds=1, defense_eval="both")
        attack = _make_attack_result(token_ids=[1, 2], eval_responses=["ok"])
        api_cfg = ApiEvalConfig(endpoints=[])
        api_results = [
            TransferEvalResult(
                model_id="api",
                success_rate=0.5,
                eval_responses=["api response"],
            ),
        ]

        with (
            patch(
                "vauban.softprompt._gan._dispatch_attack",
                return_value=attack,
            ),
            patch(
                "vauban.softprompt._gan._evaluate_attack",
                return_value=(0.6, ["transfer eval"]),
            ) as mock_eval_attack,
            patch(
                "vauban.softprompt._gan.get_transformer",
                side_effect=lambda current: current.model,
            ),
            patch(
                "vauban.api_eval.evaluate_suffix_via_api",
                return_value=api_results,
            ),
        ):
            result = gan_loop(
                model,
                tokenizer,
                ["prompt"],
                config,
                direction=None,
                transfer_models=[("transfer", transfer_model, transfer_tokenizer)],
                api_eval_config=api_cfg,
            )

        assert mock_eval_attack.called
        assert len(result.gan_history) == 1
        assert result.gan_history[0].transfer_results[0].success_rate == 0.6

    def test_gan_loop_falls_back_to_last_attack_result(self) -> None:
        model = _make_model()
        tokenizer = _make_tokenizer()
        config = SoftPromptConfig(mode="gcg", gan_rounds=1)
        attack = _make_attack_result(success_rate=math.nan)

        with patch(
            "vauban.softprompt._gan._dispatch_attack",
            return_value=attack,
        ):
            result = gan_loop(
                model,
                tokenizer,
                ["prompt"],
                config,
                direction=None,
            )

        assert len(result.gan_history) == 1
        assert math.isnan(result.success_rate)
