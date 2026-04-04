# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tail coverage for remaining softprompt helper branches."""

from __future__ import annotations

from dataclasses import dataclass
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
from vauban.softprompt._attack_init import AttackInitState
from vauban.softprompt._constraints import (
    _build_vocab_mask,
    _detect_glitch_token_ids,
    _is_emoji_char,
    _is_invisible_char,
)
from vauban.softprompt._continuous import _continuous_attack
from vauban.softprompt._defense_eval import (
    _build_adv_prompts,
    _build_sic_prompts_with_history,
    _eval_cast,
    _eval_cast_multiturn,
    _eval_sic,
    evaluate_against_defenses,
    evaluate_against_defenses_multiturn,
)
from vauban.softprompt._encoding import (
    _compute_infix_split,
    _encode_messages,
    _resolve_infix_overrides,
    _resolve_infix_overrides_with_history,
)
from vauban.softprompt._gcg import _gcg_attack
from vauban.softprompt._gcg_objective import (
    GCGDefenseConfig,
    GCGSharedState,
    _compute_prompt_objective_loss,
)
from vauban.softprompt._generation import (
    _evaluate_attack_with_history,
)
from vauban.softprompt._runtime import (
    _build_one_hot,
    _compute_temperature,
    _project_to_tokens,
    _resolve_init_ids,
    _sample_random_init_ids,
)
from vauban.types import AlphaTier, DefenseEvalResult, SoftPromptConfig, Tokenizer

if TYPE_CHECKING:
    from collections.abc import Callable

    from vauban._array import Array


def _make_model() -> MockCausalLM:
    model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
    ops.eval(model.parameters())
    return model


def _make_tokenizer() -> MockTokenizer:
    return MockTokenizer(VOCAB_SIZE)


def _make_logits(next_token: int) -> Array:
    logits = ops.zeros((1, 1, VOCAB_SIZE))
    logits[0, 0, next_token] = 1.0
    ops.eval(logits)
    return logits


def _fake_value_and_grad(
    fn: Callable[[Array], Array],
) -> Callable[[Array], tuple[Array, Array]]:
    def wrapped(x: Array) -> tuple[Array, Array]:
        loss = fn(x)
        return loss, ops.zeros_like(x)

    return wrapped


@dataclass(frozen=True, slots=True)
class _FakeSICResult:
    total_blocked: int
    total_sanitized: int
    total_clean: int


@dataclass(frozen=True, slots=True)
class _FakeCastResult:
    interventions: int
    text: str


class _ListTemplateTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(c) % VOCAB_SIZE for c in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(65 + (tid % 26)) for tid in token_ids)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = True,
    ) -> str | list[int]:
        del messages
        if tokenize:
            return [1, 2, 3]
        return [1, 2, 3]


class _ConstantEncodeTokenizer:
    def encode(self, text: str) -> list[int]:
        del text
        return [7, 8, 9]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(str(t) for t in token_ids)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = True,
    ) -> str | list[int]:
        del tokenize
        return "".join(m["content"] for m in messages)


def _make_attack_init(
    model: MockCausalLM,
    tokenizer: MockTokenizer,
    *,
    target_ids: Array | None = None,
    all_prompt_ids: list[Array] | None = None,
    objective_state: GCGSharedState | None = None,
) -> AttackInitState:
    transformer = model.model
    embed_matrix = transformer.embed_tokens.weight
    resolved_target_ids = target_ids if target_ids is not None else ops.array([1])
    resolved_prompts = (
        all_prompt_ids
        if all_prompt_ids is not None
        else [ops.array([[1, 2]])]
    )
    if objective_state is None:
        objective_state = GCGSharedState(
            model=model,
            target_ids=resolved_target_ids,
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
    ops.eval(embed_matrix, resolved_target_ids)
    return AttackInitState(
        transformer=transformer,
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        embed_matrix=embed_matrix,
        target_ids=resolved_target_ids,
        all_prompt_ids=resolved_prompts,
        objective_state=objective_state,
        transfer_data=[],
    )


class TestConstraintTail:
    def test_invisible_space_branch(self) -> None:
        assert _is_invisible_char("\u00a0") is True
        assert _is_invisible_char(" ") is False

    def test_emoji_range_fallback_branches(self) -> None:
        import unicodedata

        chars = [
            "\U0001F600",
            "\U0001F300",
            "\U0001F680",
            "\U0001F900",
            "\u2600",
            "\u2702",
        ]
        original_category = unicodedata.category

        def fake_category(ch: str) -> str:
            if ch in chars:
                return "Lu"
            return original_category(ch)

        with patch(
            "vauban.softprompt._constraints.unicodedata.category",
            side_effect=fake_category,
        ):
            for ch in chars:
                assert _is_emoji_char(ch) is True

    def test_detect_glitch_ids(self) -> None:
        embed_matrix = ops.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [20.0, 0.0],
            ],
        )
        ops.eval(embed_matrix)
        glitch_ids = _detect_glitch_token_ids(embed_matrix, sigma_threshold=1.0)
        assert glitch_ids == {2}

    def test_build_vocab_mask_exclude_glitch_fallback(self) -> None:
        tokenizer = _make_tokenizer()
        mask = _build_vocab_mask(
            tokenizer,
            2,
            "exclude_glitch",
            glitch_token_ids={1},
        )
        assert mask is not None
        ops.eval(mask)
        assert bool(mask[0].item()) is True
        assert bool(mask[1].item()) is False


class TestEncodingTail:
    def test_encode_messages_type_error(self) -> None:
        tokenizer = _ListTemplateTokenizer()
        with pytest.raises(TypeError, match="apply_chat_template"):
            _encode_messages(
                cast("Tokenizer", tokenizer),
                [{"role": "user", "content": "x"}],
            )

    def test_token_list_scalar_branch(self) -> None:
        from vauban.softprompt._encoding import _token_list

        tokens = _token_list(ops.array([5]))
        assert tokens == [5]

    def test_compute_infix_split_error_and_equal_branch(self) -> None:
        tokenizer = _ConstantEncodeTokenizer()
        with pytest.raises(
            ValueError,
            match=r"requires prompts with a \{suffix\}",
        ):
            _compute_infix_split(cast("Tokenizer", tokenizer), "no marker")

        clean_ids, split_idx = _compute_infix_split(
            cast("Tokenizer", tokenizer),
            "prefix {suffix} suffix",
        )
        assert clean_ids == [7, 8, 9]
        assert split_idx == len(clean_ids)

    def test_resolve_infix_overrides_with_history(self) -> None:
        tokenizer = _make_tokenizer()
        prompts = ["hello {suffix}"]
        history = [{"role": "assistant", "content": "prev"}]
        encoded, infix_map = _resolve_infix_overrides_with_history(
            tokenizer,
            prompts,
            history,
            system_prompt="SYS",
        )
        assert len(encoded) == 1
        assert id(encoded[0]) in infix_map

    def test_resolve_infix_overrides_without_history(self) -> None:
        tokenizer = _make_tokenizer()
        prompts = ["hello {suffix}"]
        encoded, infix_map = _resolve_infix_overrides(
            tokenizer,
            prompts,
            system_prompt="SYS",
        )
        assert len(encoded) == 1
        assert id(encoded[0]) in infix_map


class TestRuntimeTail:
    def test_temperature_schedules(self) -> None:
        assert _compute_temperature(1.0, 0, 4, "constant") == 1.0
        assert _compute_temperature(1.0, 2, 4, "linear") > 1.0
        assert _compute_temperature(1.0, 2, 4, "cosine") <= 2.0

    def test_sample_random_init_ids_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="No allowed token indices"):
            _sample_random_init_ids(2, [])

    def test_resolve_init_ids_padding_and_random(self) -> None:
        padded = _resolve_init_ids([1], 3, None, VOCAB_SIZE)
        assert padded == [1, 0, 0]

        with (
            patch(
                "vauban.softprompt._gcg_candidates._allowed_indices_from_mask",
                return_value=[4],
            ),
            patch("random.choice", return_value=4),
        ):
            random_ids = _resolve_init_ids(None, 2, None, VOCAB_SIZE)

        assert random_ids == [4, 4]

    def test_build_one_hot_falls_back_when_array_alias_changes(self) -> None:
        with patch("vauban.softprompt._runtime.Array", tuple):
            one_hot = _build_one_hot([1, 0], 3)
            ops.eval(one_hot)

        assert one_hot.shape == (2, 3)
        assert float(one_hot[0, 1].item()) == 1.0
        assert float(one_hot[1, 0].item()) == 1.0

    def test_project_to_tokens_scalar_branch(self) -> None:
        soft_embeds = ops.ones((1, 1, D_MODEL))
        embed_matrix = ops.ones((VOCAB_SIZE, D_MODEL))
        with patch(
            "vauban.softprompt._runtime.ops.argmax",
            return_value=ops.array(7),
        ):
            tokens = _project_to_tokens(soft_embeds, embed_matrix)
        assert tokens == [7]


class TestDefenseEvalTail:
    def test_build_adv_prompts_and_history(self) -> None:
        prompts = ["hello {suffix}", "fallback"]
        history = [{"role": "assistant", "content": "prev"}]
        assert _build_adv_prompts(prompts, None, "infix") == prompts
        assert _build_adv_prompts(prompts, "TOK", "infix") == [
            "hello TOK",
            "fallback TOK",
        ]
        assert _build_sic_prompts_with_history(prompts, history) == [
            "[assistant]: prev\n[user]: hello {suffix}",
            "[assistant]: prev\n[user]: fallback",
        ]

    def test_eval_helpers_call_underlying_modules(self) -> None:
        model = _make_model()
        tokenizer = _make_tokenizer()
        direction = ops.ones((D_MODEL,))
        ops.eval(direction)

        fake_sic = _FakeSICResult(1, 1, 0)
        fake_cast = _FakeCastResult(4, "safe")
        fake_cast_multi = _FakeCastResult(5, "multi")

        with (
            patch("vauban.sic.sic", return_value=fake_sic) as mock_sic,
            patch(
                "vauban.cast.cast_generate",
                return_value=fake_cast,
            ) as mock_cast,
            patch(
                "vauban.cast.cast_generate_with_messages",
                return_value=fake_cast_multi,
            ) as mock_cast_multi,
        ):
            sic_result = _eval_sic(
                model,
                tokenizer,
                ["prompt"],
                direction,
                layer=1,
                threshold=0.5,
                sic_mode="direction",
                sic_max_iterations=2,
            )
            cast_result = _eval_cast(
                model,
                tokenizer,
                ["prompt"],
                direction,
                layers=[1],
                alpha=0.3,
                threshold=0.4,
                max_tokens=2,
                alpha_tiers=[AlphaTier(0.1, 0.2)],
            )
            cast_multi_result = _eval_cast_multiturn(
                model,
                tokenizer,
                ["prompt"],
                direction,
                layers=[1],
                alpha=0.3,
                threshold=0.4,
                max_tokens=2,
                history=[{"role": "assistant", "content": "prev"}],
                alpha_tiers=[AlphaTier(0.1, 0.2)],
            )

        assert sic_result == (1, 1, 0, 1.0)
        assert cast_result == (4, 0.0, ["safe"])
        assert cast_multi_result == (5, 0.0, ["multi"])
        mock_sic.assert_called_once()
        mock_cast.assert_called_once()
        mock_cast_multi.assert_called_once()

    def test_eval_cast_refusal_detection_branches(self) -> None:
        model = _make_model()
        tokenizer = _make_tokenizer()
        direction = ops.ones((D_MODEL,))
        ops.eval(direction)
        refusal = _FakeCastResult(2, "I cannot comply")
        refusal_multi = _FakeCastResult(3, "I cannot comply")

        with (
            patch("vauban.cast.cast_generate", return_value=refusal),
            patch(
                "vauban.cast.cast_generate_with_messages",
                return_value=refusal_multi,
            ),
        ):
            cast_result = _eval_cast(
                model,
                tokenizer,
                ["prompt"],
                direction,
                layers=[1],
                alpha=0.3,
                threshold=0.4,
                max_tokens=2,
            )
            cast_multi_result = _eval_cast_multiturn(
                model,
                tokenizer,
                ["prompt"],
                direction,
                layers=[1],
                alpha=0.3,
                threshold=0.4,
                max_tokens=2,
                history=[{"role": "assistant", "content": "prev"}],
            )

        assert cast_result == (2, 1.0, ["I cannot comply"])
        assert cast_multi_result == (3, 1.0, ["I cannot comply"])

    def test_public_evaluators_use_helpers(self) -> None:
        model = _make_model()
        tokenizer = _make_tokenizer()
        direction = ops.ones((D_MODEL,))
        ops.eval(direction)
        config = SoftPromptConfig(
            mode="gcg",
            defense_eval="both",
            token_position="infix",
            defense_eval_sic_threshold=0.25,
            defense_eval_cast_layers=[1],
            defense_eval_alpha_tiers=[(0.1, 0.2)],
            max_gen_tokens=1,
        )
        with (
            patch(
                "vauban.softprompt._defense_eval._eval_sic",
                return_value=(1, 2, 0, 1.0),
            ) as mock_sic,
            patch(
                "vauban.softprompt._defense_eval._eval_cast",
                return_value=(3, 0.0, ["safe"]),
            ) as mock_cast,
            patch(
                "vauban.softprompt._defense_eval._eval_cast_multiturn",
                return_value=(4, 0.0, ["safe-multi"]),
            ) as mock_cast_multi,
        ):
            result = evaluate_against_defenses(
                model,
                tokenizer,
                ["hello {suffix}"],
                config,
                direction,
                layer_index=0,
                token_text="TOK",
            )
            result_multi = evaluate_against_defenses_multiturn(
                model,
                tokenizer,
                ["hello {suffix}"],
                config,
                direction,
                layer_index=0,
                token_text="TOK",
                history=[{"role": "assistant", "content": "prev"}],
            )

        assert isinstance(result, DefenseEvalResult)
        assert result.sic_blocked == 1
        assert result.cast_interventions == 3
        assert isinstance(result_multi, DefenseEvalResult)
        assert result_multi.cast_interventions == 4
        assert mock_sic.call_count == 2
        mock_cast.assert_called_once()
        mock_cast_multi.assert_called_once()


class TestGcgObjectiveTail:
    def test_defensive_branch_dispatches(self) -> None:
        model = _make_model()
        direction = ops.ones((D_MODEL,))
        ops.eval(direction)
        state = GCGSharedState(
            model=model,
            target_ids=ops.array([1]),
            n_tokens=1,
            loss_mode="defensive",
            direction=direction,
            direction_weight=0.5,
            direction_mode="last",
            direction_layers=None,
            eos_token_id=None,
            eos_loss_mode="none",
            eos_loss_weight=0.0,
            ref_model=None,
            kl_ref_weight=0.0,
            refusal_ids=ops.array([0]),
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
        soft_embeds = ops.ones((1, 1, D_MODEL))
        prompt_ids = ops.array([[1]])
        with patch(
            "vauban.softprompt._gcg_objective._compute_defensive_loss",
            return_value=ops.array(1.0),
        ) as mock_defensive:
            loss = _compute_prompt_objective_loss(state, soft_embeds, prompt_ids)
        ops.eval(loss)
        assert float(loss.item()) == 1.0
        mock_defensive.assert_called_once()


class TestGenerationTail:
    def test_evaluate_attack_with_history_eos_breaks(self) -> None:
        model = _make_model()
        tokenizer = _make_tokenizer()
        soft_embeds = ops.ones((1, 1, D_MODEL))
        config = SoftPromptConfig(
            mode="continuous",
            loss_mode="targeted",
            max_gen_tokens=2,
            system_prompt="SYS",
            token_position="suffix",
        )
        logits_eos = _make_logits(tokenizer.eos_token_id)
        captured: list[list[dict[str, str]]] = []

        def fake_encode(_tokenizer: object, messages: list[dict[str, str]]) -> Array:
            captured.append(messages)
            return ops.array([1, 2, 3])

        with (
            patch(
                "vauban.softprompt._generation.get_transformer",
                return_value=cast("object", model.model),
            ),
            patch("vauban.softprompt._generation.make_cache", return_value=[object()]),
            patch(
                "vauban.softprompt._generation.make_ssm_mask",
                return_value=ops.zeros((1, 1, 1)),
            ),
            patch(
                "vauban.softprompt._generation.encode_chat_prompt",
                side_effect=fake_encode,
            ),
            patch(
                "vauban.softprompt._generation._prefill_with_cache",
                return_value=logits_eos,
            ),
        ):
            success_rate, responses = _evaluate_attack_with_history(
                model,
                tokenizer,
                ["hello"],
                soft_embeds,
                config,
                [{"role": "assistant", "content": "prev"}],
            )

        assert success_rate == 1.0
        assert responses == [""]
        assert captured[0][0]["role"] == "system"
        assert captured[0][1]["role"] == "assistant"

    def test_evaluate_attack_with_history_defensive_success(self) -> None:
        model = _make_model()
        tokenizer = _make_tokenizer()
        soft_embeds = ops.ones((1, 1, D_MODEL))
        config = SoftPromptConfig(
            mode="continuous",
            loss_mode="defensive",
            max_gen_tokens=1,
            system_prompt="SYS",
            token_position="suffix",
        )
        logits_token = _make_logits(1)
        tokenizer.decode = lambda _token_ids: "I cannot help"  # type: ignore[method-assign]

        with (
            patch(
                "vauban.softprompt._generation.get_transformer",
                return_value=cast("object", model.model),
            ),
            patch("vauban.softprompt._generation.make_cache", return_value=[object()]),
            patch(
                "vauban.softprompt._generation.make_ssm_mask",
                return_value=ops.zeros((1, 1, 1)),
            ),
            patch(
                "vauban.softprompt._generation.encode_chat_prompt",
                return_value=ops.array([1, 2, 3]),
            ),
            patch(
                "vauban.softprompt._generation._prefill_with_cache",
                return_value=logits_token,
            ),
            patch(
                "vauban.softprompt._generation._decode_step",
                return_value=logits_token,
            ),
        ):
            success_rate, responses = _evaluate_attack_with_history(
                model,
                tokenizer,
                ["hello"],
                soft_embeds,
                config,
                [{"role": "assistant", "content": "prev"}],
            )

        assert success_rate == 1.0
        assert responses == ["I cannot help"]


class TestContinuousTail:
    def test_sample_prompt_strategy_branch(self) -> None:
        model = _make_model()
        tokenizer = _make_tokenizer()
        config = SoftPromptConfig(
            mode="continuous",
            n_tokens=1,
            n_steps=1,
            prompt_strategy="sample",
            max_gen_tokens=1,
        )
        with (
            patch(
                "vauban.softprompt._continuous._sample_prompt_ids",
                return_value=[ops.array([[1, 2]])],
            ),
            patch(
                "vauban.softprompt._continuous._compute_average_objective_loss",
                return_value=ops.array(1.0),
            ),
            patch(
                "vauban.softprompt._continuous.ops.value_and_grad",
                side_effect=_fake_value_and_grad,
            ),
            patch(
                "vauban.softprompt._continuous._compute_per_prompt_losses",
                return_value=[0.0],
            ),
            patch(
                "vauban.softprompt._continuous._evaluate_attack",
                return_value=(0.0, ["ok"]),
            ),
        ):
            result = _continuous_attack(
                model,
                tokenizer,
                ["prompt"],
                config,
                direction=None,
            )

        assert result.mode == "continuous"
        assert len(result.loss_history) == 1


class TestGcgTail:
    def test_early_stop_branch(self) -> None:
        model = _make_model()
        tokenizer = _make_tokenizer()
        init = _make_attack_init(model, tokenizer)
        config = SoftPromptConfig(
            mode="gcg",
            n_tokens=1,
            n_steps=2,
            n_restarts=1,
            patience=1,
            beam_width=1,
            batch_size=1,
            top_k=1,
            grad_accum_steps=1,
            max_gen_tokens=1,
        )
        with (
            patch("vauban.softprompt._gcg.prepare_attack", return_value=init),
            patch(
                "vauban.softprompt._gcg._compute_average_objective_loss",
                side_effect=[ops.array(1.0), ops.array(2.0)],
            ),
            patch(
                "vauban.softprompt._gcg.ops.value_and_grad",
                side_effect=_fake_value_and_grad,
            ),
            patch(
                "vauban.softprompt._gcg._score_token_candidates",
                return_value=ops.zeros((1, VOCAB_SIZE)),
            ),
            patch(
                "vauban.softprompt._gcg._top_candidate_indices",
                return_value=(ops.array([[0]]), 1),
            ),
            patch(
                "vauban.softprompt._gcg._sample_greedy_candidates",
                return_value=[[0]],
            ),
            patch(
                "vauban.softprompt._gcg._evaluate_candidate_loss",
                return_value=0.5,
            ),
            patch(
                "vauban.softprompt._gcg._apply_transfer_reranking",
            ),
            patch(
                "vauban.softprompt._gcg._apply_rollout_reranking",
            ),
            patch(
                "vauban.softprompt._gcg._compute_per_prompt_losses",
                return_value=[0.0],
            ),
            patch(
                "vauban.softprompt._gcg._evaluate_attack",
                return_value=(0.0, ["ok"]),
            ),
        ):
            result = _gcg_attack(
                model,
                tokenizer,
                ["prompt"],
                config,
                direction=None,
            )

        assert result.early_stopped is True
        assert len(result.loss_history) == 2
