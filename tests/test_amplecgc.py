# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for the AmpleGCG attack, collection, and generator training slice."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest
from conftest import MockTokenizer

from vauban import _ops as ops
from vauban.softprompt._amplecgc import _amplecgc_attack
from vauban.softprompt._amplecgc_collect import collect_gcg_suffixes
from vauban.softprompt._amplecgc_train import (
    AmpleGCGGenerator,
    train_amplecgc_generator,
)
from vauban.types import (
    CausalLM,
    EnvironmentConfig,
    SoftPromptConfig,
    Tokenizer,
)

if TYPE_CHECKING:
    from vauban._array import Array


class _FakeEmbedTokens:
    """Fake embedding lookup with a fixed weight shape."""

    def __init__(self, d_model: int, vocab_size: int) -> None:
        self.weight = ops.zeros((vocab_size, d_model))

    def __call__(self, token_array: Array) -> Array:
        batch, seq = token_array.shape
        return ops.ones((batch, seq, self.weight.shape[1])) * 0.5


class _FakeTransformer:
    """Minimal transformer stand-in for the AmpleGCG tests."""

    def __init__(self, d_model: int, vocab_size: int) -> None:
        self.embed_tokens = _FakeEmbedTokens(d_model, vocab_size)
        self.layers: list[object] = []


class _ScalarIndex:
    """Fake tensor-like index object whose ``tolist`` returns a scalar."""

    def __init__(self, value: int) -> None:
        self._value = value

    def tolist(self) -> int:
        return self._value


class _ListIndex:
    """Fake tensor-like index object whose ``tolist`` returns a list."""

    def __init__(self, values: list[int]) -> None:
        self._values = values

    def tolist(self) -> list[int]:
        return self._values


class _FakeGenerator:
    """Generator stub for the attack path."""

    def __init__(self, candidates: list[list[int]]) -> None:
        self.candidates = candidates
        self.calls: list[tuple[Array, int, float]] = []

    def sample(
        self,
        prompt_embed: Array,
        n_candidates: int,
        temperature: float = 1.0,
    ) -> list[list[int]]:
        self.calls.append((prompt_embed, n_candidates, temperature))
        return [candidate[:] for candidate in self.candidates]


def _model() -> CausalLM:
    """Return an opaque stand-in model for the AmpleGCG slice."""
    return cast("CausalLM", object())


def _tokenizer() -> Tokenizer:
    """Return a small tokenizer stand-in for the AmpleGCG slice."""
    return cast("Tokenizer", MockTokenizer(32))


def _fake_objective_state(perplexity_weight: float) -> SimpleNamespace:
    """Build the tiny objective state needed by the collection helper."""
    return SimpleNamespace(perplexity_weight=perplexity_weight)


def _fake_attack_state() -> SimpleNamespace:
    """Build the minimal prepared-attack state used by the top-level attack."""
    return SimpleNamespace(
        vocab_size=8,
        target_ids=ops.array([[1, 2]]),
        all_prompt_ids=[ops.array([3, 4]), ops.array([5, 6])],
        objective_state=SimpleNamespace(
            direction_layers=[0],
            eos_token_id=31,
            refusal_ids=[9],
            defense=SimpleNamespace(
                weight=0.25,
                sic_layer=1,
                sic_threshold=0.15,
                cast_layers=[0],
                cast_threshold=0.35,
            ),
            perplexity_weight=0.0,
            token_position="suffix",
        ),
        transformer=_FakeTransformer(4, 8),
    )


class TestAmpleGCGAttack:
    """Tests for the top-level AmpleGCG orchestration function."""

    def test_returns_minimal_result_when_collection_is_empty(self) -> None:
        model = _model()
        tokenizer = _tokenizer()
        config = SoftPromptConfig(mode="amplecgc", n_tokens=3)
        transformer = _FakeTransformer(4, 8)

        with (
            patch(
                "vauban.softprompt._amplecgc.get_transformer",
                return_value=transformer,
            ) as mock_get_transformer,
            patch(
                "vauban.softprompt._amplecgc.collect_gcg_suffixes",
                return_value=[],
            ) as mock_collect,
            patch(
                "vauban.softprompt._amplecgc.prepare_attack",
            ) as mock_prepare,
            patch(
                "vauban.softprompt._amplecgc.train_amplecgc_generator",
            ) as mock_train,
        ):
            result = _amplecgc_attack(
                model,
                tokenizer,
                ["prompt"],
                config,
                direction=None,
            )

        assert result.mode == "amplecgc"
        assert result.success_rate == 0.0
        assert result.final_loss == float("inf")
        assert result.token_ids == []
        assert result.token_text == ""
        assert result.n_steps == 0
        mock_get_transformer.assert_called_once_with(model)
        mock_collect.assert_called_once()
        mock_prepare.assert_not_called()
        mock_train.assert_not_called()

    def test_runs_full_flow_and_selects_best_candidate(self) -> None:
        model = _model()
        tokenizer = _tokenizer()
        config = SoftPromptConfig(
            mode="amplecgc",
            n_tokens=2,
            amplecgc_n_candidates=2,
            amplecgc_hidden_dim=4,
            amplecgc_train_steps=1,
            amplecgc_collect_steps=1,
            amplecgc_collect_restarts=1,
            amplecgc_collect_threshold=1.5,
            amplecgc_sample_temperature=0.75,
        )
        transformer = _FakeTransformer(4, 8)
        collected = [([5, 6], 1.1), ([7, 8], 1.2)]
        prepared = _fake_attack_state()
        generator = _FakeGenerator([[9, 9], [8, 8]])
        candidate_losses = {
            (9, 9): 1.4,
            (8, 8): 0.4,
            (5, 6): 0.9,
            (7, 8): 1.0,
        }

        def _fake_candidate_loss(
            _objective_state: object,
            candidate: list[int],
            _all_prompt_ids: list[Array],
        ) -> float:
            return candidate_losses[tuple(candidate)]

        with (
            patch(
                "vauban.softprompt._amplecgc.get_transformer",
                return_value=transformer,
            ) as mock_get_transformer,
            patch(
                "vauban.softprompt._amplecgc.collect_gcg_suffixes",
                return_value=collected,
            ) as mock_collect,
            patch(
                "vauban.softprompt._amplecgc.prepare_attack",
                return_value=prepared,
            ) as mock_prepare,
            patch(
                "vauban.softprompt._amplecgc.train_amplecgc_generator",
                return_value=generator,
            ) as mock_train,
            patch(
                "vauban.softprompt._amplecgc._evaluate_candidate_loss",
                side_effect=_fake_candidate_loss,
            ) as mock_candidate_loss,
            patch(
                "vauban.softprompt._amplecgc._compute_per_prompt_losses",
                return_value=[0.6, 0.7],
            ) as mock_per_prompt_losses,
            patch(
                "vauban.softprompt._amplecgc._evaluate_attack",
                return_value=(0.75, ["r1", "r2"]),
            ) as mock_evaluate_attack,
            patch(
                "vauban.softprompt._amplecgc._compute_accessibility_score",
                return_value=0.88,
            ) as mock_accessibility,
        ):
            result = _amplecgc_attack(
                model,
                tokenizer,
                ["prompt"],
                config,
                direction=None,
                ref_model=model,
                transfer_models=[("t", model, tokenizer)],
                infix_map={0: 2},
                environment_config=cast(
                    "EnvironmentConfig",
                    SimpleNamespace(mode="env"),
                ),
            )

        assert result.mode == "amplecgc"
        assert result.token_ids == [8, 8]
        assert result.token_text == "II"
        assert result.success_rate == 0.75
        assert result.final_loss == 0.4
        assert result.loss_history == [1.1, 1.2, 1.4, 0.4, 0.9, 1.0]
        assert result.n_steps == 6
        assert result.accessibility_score == 0.88
        assert result.per_prompt_losses == [0.6, 0.7]
        assert result.eval_responses == ["r1", "r2"]
        mock_get_transformer.assert_called_once_with(model)
        mock_collect.assert_called_once()
        mock_prepare.assert_called_once()
        mock_train.assert_called_once()
        mock_candidate_loss.assert_called()
        mock_per_prompt_losses.assert_called_once()
        mock_evaluate_attack.assert_called_once()
        mock_accessibility.assert_called_once_with(0.4)
        assert generator.calls[0][1:] == (
            config.amplecgc_n_candidates,
            config.amplecgc_sample_temperature,
        )


class TestCollectGcgSuffixes:
    """Tests for the GCG data collection helper."""

    def test_defaults_to_hello_and_collects_best_suffix(self) -> None:
        model = _model()
        tokenizer = _tokenizer()
        config = SoftPromptConfig(
            mode="amplecgc",
            n_tokens=2,
            target_prefixes=["Sure"],
            target_repeat_count=1,
            system_prompt="system",
            grad_accum_steps=2,
            amplecgc_collect_steps=2,
            amplecgc_collect_restarts=1,
            amplecgc_collect_threshold=1.5,
            patience=0,
        )
        transformer = _FakeTransformer(4, 8)
        objective_state = _fake_objective_state(perplexity_weight=1.0)
        loss_calls: list[Array | None] = []

        def _fake_average_loss(
            _objective_state: object,
            embeds: Array,
            _selected: list[Array],
            *,
            suffix_token_ids: Array | None = None,
        ) -> Array:
            loss_calls.append(suffix_token_ids)
            return ops.sum(embeds) * 0.0 + 1.1 + (
                0.1 if suffix_token_ids is not None else 0.0
            )

        candidate_losses = {
            (9, 9): 1.4,
            (8, 8): 0.4,
        }

        def _fake_candidate_loss(
            _objective_state: object,
            candidate: list[int],
            _ids: list[Array],
        ) -> float:
            return candidate_losses[tuple(candidate)]

        with (
            patch(
                "vauban.softprompt._amplecgc_collect.get_transformer",
                return_value=transformer,
            ) as mock_get_transformer,
            patch(
                "vauban.softprompt._amplecgc_collect._encode_targets",
                return_value=ops.array([[1, 2]]),
            ) as mock_encode_targets,
            patch(
                "vauban.softprompt._amplecgc_collect._pre_encode_prompts",
                return_value=[ops.array([3, 4]), ops.array([5, 6])],
            ) as mock_pre_encode,
            patch(
                "vauban.softprompt._amplecgc_collect._build_gcg_shared_state",
                return_value=objective_state,
            ) as mock_build_state,
            patch(
                "vauban.softprompt._amplecgc_collect._build_vocab_mask",
                return_value=ops.array([True] * 8),
            ) as mock_vocab_mask,
            patch(
                "vauban.softprompt._amplecgc_collect._allowed_indices_from_mask",
                return_value=[0, 1, 2, 3],
            ) as mock_allowed_indices,
            patch(
                "vauban.softprompt._amplecgc_collect._initialize_restart_tokens",
                return_value=[5, 6],
            ) as mock_init_tokens,
            patch(
                "vauban.softprompt._amplecgc_collect._select_step_prompts",
                return_value=[ops.array([0]), ops.array([1]), ops.array([2])],
            ) as mock_select_prompts,
            patch(
                "vauban.softprompt._amplecgc_collect._compute_average_objective_loss",
                side_effect=_fake_average_loss,
            ) as mock_average_loss,
            patch(
                "vauban.softprompt._amplecgc_collect._score_token_candidates",
                return_value=ops.array([0.1, 0.9, 0.2, 0.3]),
            ) as mock_score_candidates,
            patch(
                "vauban.softprompt._amplecgc_collect._top_candidate_indices",
                return_value=([1, 2], 2),
            ) as mock_top_indices,
            patch(
                "vauban.softprompt._amplecgc_collect._sample_greedy_candidates",
                return_value=[[9, 9], [8, 8]],
            ) as mock_sample_candidates,
            patch(
                "vauban.softprompt._amplecgc_collect._evaluate_candidate_loss",
                side_effect=_fake_candidate_loss,
            ) as mock_candidate_loss,
        ):
            result = collect_gcg_suffixes(
                model,
                tokenizer,
                [],
                config,
                direction=None,
            )

        assert result[0][0] == [5, 6]
        assert result[0][1] == pytest.approx(1.2)
        assert result[1][0] == [8, 8]
        assert result[1][1] == pytest.approx(1.2)
        assert result[2][0] == [5, 6]
        assert result[2][1] == pytest.approx(1.2)
        assert len(loss_calls) == 4
        assert all(call is not None for call in loss_calls)
        mock_get_transformer.assert_called_once_with(model)
        mock_encode_targets.assert_called_once()
        mock_pre_encode.assert_called_once_with(tokenizer, ["Hello"], "system")
        mock_build_state.assert_called_once()
        mock_vocab_mask.assert_called_once()
        mock_allowed_indices.assert_called_once()
        mock_init_tokens.assert_called_once()
        mock_select_prompts.assert_called()
        mock_average_loss.assert_called()
        mock_score_candidates.assert_called()
        mock_top_indices.assert_called()
        mock_sample_candidates.assert_called()
        mock_candidate_loss.assert_called()

    def test_patience_stops_without_collecting_when_threshold_not_met(self) -> None:
        model = _model()
        tokenizer = _tokenizer()
        config = SoftPromptConfig(
            mode="amplecgc",
            n_tokens=2,
            target_prefixes=["Sure"],
            target_repeat_count=1,
            system_prompt="system",
            grad_accum_steps=1,
            amplecgc_collect_steps=2,
            amplecgc_collect_restarts=1,
            amplecgc_collect_threshold=0.5,
            patience=1,
        )
        transformer = _FakeTransformer(4, 8)
        objective_state = _fake_objective_state(perplexity_weight=0.0)
        suffix_args: list[Array | None] = []

        def _fake_average_loss(
            _objective_state: object,
            embeds: Array,
            _selected: list[Array],
            *,
            suffix_token_ids: Array | None = None,
        ) -> Array:
            suffix_args.append(suffix_token_ids)
            return ops.sum(embeds) * 0.0 + 1.0

        with (
            patch(
                "vauban.softprompt._amplecgc_collect.get_transformer",
                return_value=transformer,
            ),
            patch(
                "vauban.softprompt._amplecgc_collect._encode_targets",
                return_value=ops.array([[1, 2]]),
            ),
            patch(
                "vauban.softprompt._amplecgc_collect._pre_encode_prompts",
                return_value=[ops.array([3, 4])],
            ),
            patch(
                "vauban.softprompt._amplecgc_collect._build_gcg_shared_state",
                return_value=objective_state,
            ),
            patch(
                "vauban.softprompt._amplecgc_collect._build_vocab_mask",
                return_value=ops.array([True] * 8),
            ),
            patch(
                "vauban.softprompt._amplecgc_collect._allowed_indices_from_mask",
                return_value=[0, 1, 2, 3],
            ),
            patch(
                "vauban.softprompt._amplecgc_collect._initialize_restart_tokens",
                return_value=[5, 6],
            ),
            patch(
                "vauban.softprompt._amplecgc_collect._select_step_prompts",
                return_value=[ops.array([0])],
            ),
            patch(
                "vauban.softprompt._amplecgc_collect._compute_average_objective_loss",
                side_effect=_fake_average_loss,
            ),
            patch(
                "vauban.softprompt._amplecgc_collect._score_token_candidates",
                return_value=ops.array([0.1, 0.9, 0.2, 0.3]),
            ),
            patch(
                "vauban.softprompt._amplecgc_collect._top_candidate_indices",
                return_value=([1, 2], 2),
            ),
            patch(
                "vauban.softprompt._amplecgc_collect._sample_greedy_candidates",
                return_value=[[9, 9], [8, 8]],
            ),
            patch(
                "vauban.softprompt._amplecgc_collect._evaluate_candidate_loss",
                return_value=2.0,
            ),
        ):
            result = collect_gcg_suffixes(
                model,
                tokenizer,
                ["alpha"],
                config,
                direction=None,
            )

        assert result == []
        assert suffix_args == [None, None]


class TestTrainAmpleGCGGenerator:
    """Tests for the generator training helper and its sampling branches."""

    def test_train_amplecgc_generator_runs_with_scalar_index(self) -> None:
        collected = [([1, 2], 0.5)]

        def _fake_normal(shape: tuple[int, ...]) -> Array:
            return ops.zeros(shape)

        with (
            patch(
                "vauban.softprompt._amplecgc_train.ops.random.normal",
                side_effect=_fake_normal,
            ),
            patch(
                "vauban.softprompt._amplecgc_train.ops.random.randint",
                return_value=_ScalarIndex(0),
            ),
            patch(
                "vauban.softprompt._amplecgc_train.force_eval",
                lambda *args: None,
            ),
        ):
            generator = train_amplecgc_generator(
                collected,
                embed_dim=2,
                n_tokens=2,
                vocab_size=3,
                hidden_dim=4,
                train_steps=1,
                learning_rate=0.0,
            )

        assert isinstance(generator, AmpleGCGGenerator)
        assert generator.n_tokens == 2
        assert generator.vocab_size == 3
        assert len(generator.sample(ops.zeros((2,)), 1, temperature=0.0)) == 1

    @pytest.mark.parametrize(
        ("temperature", "argmax_return", "expected"),
        [
            pytest.param(0.0, _ScalarIndex(7), [[7]], id="scalar-argmax"),
            pytest.param(0.5, _ListIndex([1, 2]), [[1, 2]], id="list-argmax"),
        ],
    )
    def test_generator_sample_covers_temperature_branches(
        self,
        temperature: float,
        argmax_return: _ScalarIndex | _ListIndex,
        expected: list[list[int]],
    ) -> None:
        generator = AmpleGCGGenerator(
            w1=ops.zeros((2, 4)),
            b1=ops.zeros((4,)),
            w2=ops.zeros((4, 4)),
            b2=ops.array([3.0, 1.0, 0.0, 2.0]),
            n_tokens=2,
            vocab_size=2,
        )
        prompt_embed = ops.zeros((2,))

        with (
            patch(
                "vauban.softprompt._amplecgc_train.force_eval",
                lambda *args: None,
            ),
            patch(
                "vauban.softprompt._amplecgc_train.ops.argmax",
                return_value=argmax_return,
            ),
            patch(
                "vauban.softprompt._amplecgc_train.ops.random.uniform",
                return_value=ops.zeros((2, 2)) + 0.5,
            ),
        ):
            result = generator.sample(prompt_embed, 1, temperature=temperature)

        assert result == expected
