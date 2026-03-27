# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared attack initialization (AttackInitState + prepare_attack)."""

from __future__ import annotations

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
from vauban.softprompt._attack_init import AttackInitState, prepare_attack
from vauban.types import SoftPromptConfig


def _make_config(**overrides: object) -> SoftPromptConfig:
    """Build a SoftPromptConfig with minimal defaults."""
    defaults: dict[str, object] = {
        "n_tokens": 4,
        "n_steps": 1,
    }
    defaults.update(overrides)
    return SoftPromptConfig(**defaults)  # type: ignore[arg-type]


@pytest.fixture()
def model() -> MockCausalLM:
    """Shared mock model for attack init tests."""
    return MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)


@pytest.fixture()
def tok() -> MockTokenizer:
    """Shared mock tokenizer for attack init tests."""
    return MockTokenizer(VOCAB_SIZE)


class TestPrepareAttack:
    """Tests for the prepare_attack factory."""

    def test_returns_attack_init_state(
        self, model: MockCausalLM, tok: MockTokenizer,
    ) -> None:
        config = _make_config()
        result = prepare_attack(model, tok, ["Hello"], config, direction=None)
        assert isinstance(result, AttackInitState)

    def test_vocab_size_and_d_model(
        self, model: MockCausalLM, tok: MockTokenizer,
    ) -> None:
        config = _make_config()
        result = prepare_attack(model, tok, ["Hello"], config, direction=None)
        assert result.vocab_size == VOCAB_SIZE
        assert result.d_model == D_MODEL

    def test_embed_matrix_shape(
        self, model: MockCausalLM, tok: MockTokenizer,
    ) -> None:
        config = _make_config()
        result = prepare_attack(model, tok, ["Hello"], config, direction=None)
        assert result.embed_matrix.shape == (VOCAB_SIZE, D_MODEL)

    def test_target_ids_nonempty(
        self, model: MockCausalLM, tok: MockTokenizer,
    ) -> None:
        config = _make_config()
        result = prepare_attack(model, tok, ["Hello"], config, direction=None)
        assert result.target_ids.shape[0] > 0

    def test_all_prompt_ids_length(
        self, model: MockCausalLM, tok: MockTokenizer,
    ) -> None:
        config = _make_config()
        result = prepare_attack(
            model, tok, ["Hello", "World"], config, direction=None,
        )
        assert len(result.all_prompt_ids) == 2

    def test_prompt_ids_override(
        self, model: MockCausalLM, tok: MockTokenizer,
    ) -> None:
        config = _make_config()
        override = [ops.array([1, 2, 3]), ops.array([4, 5])]
        result = prepare_attack(
            model, tok, ["ignored"], config, direction=None,
            all_prompt_ids_override=override,
        )
        assert len(result.all_prompt_ids) == 2

    def test_objective_state_populated(
        self, model: MockCausalLM, tok: MockTokenizer,
    ) -> None:
        config = _make_config()
        result = prepare_attack(model, tok, ["Hello"], config, direction=None)
        assert result.objective_state.n_tokens == 4
        assert result.objective_state.model is model

    def test_transfer_data_empty_by_default(
        self, model: MockCausalLM, tok: MockTokenizer,
    ) -> None:
        config = _make_config()
        result = prepare_attack(model, tok, ["Hello"], config, direction=None)
        assert result.transfer_data == []

    def test_empty_prompts_defaults_to_hello(
        self, model: MockCausalLM, tok: MockTokenizer,
    ) -> None:
        config = _make_config()
        result = prepare_attack(model, tok, [], config, direction=None)
        assert len(result.all_prompt_ids) == 1

    def test_perplexity_weight_override(
        self, model: MockCausalLM, tok: MockTokenizer,
    ) -> None:
        config = _make_config(perplexity_weight=1.0)
        result = prepare_attack(
            model, tok, ["Hello"], config, direction=None,
            perplexity_weight_override=0.0,
        )
        assert result.objective_state.perplexity_weight == 0.0

    def test_direction_is_stored_in_objective_state(
        self, model: MockCausalLM, tok: MockTokenizer,
    ) -> None:
        """Passing a direction should store it in objective_state."""
        config = _make_config()
        direction = ops.random.normal((D_MODEL,))
        direction = direction / ops.linalg.norm(direction)
        ops.eval(direction)

        result = prepare_attack(
            model, tok, ["Hello"], config, direction=direction,
        )

        assert result.objective_state.direction is not None
        assert result.objective_state.direction.shape == (D_MODEL,)

    def test_transfer_models_populate_transfer_data(
        self, model: MockCausalLM, tok: MockTokenizer,
    ) -> None:
        """Passing transfer_models should produce non-empty transfer_data."""
        config = _make_config(transfer_loss_weight=1.0)
        transfer_model = MockCausalLM(
            D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS,
        )
        transfer_tok = MockTokenizer(VOCAB_SIZE)
        transfer_models = [("transfer", transfer_model, transfer_tok)]

        result = prepare_attack(
            model, tok, ["Hello"], config, direction=None,
            transfer_models=transfer_models,
        )

        assert len(result.transfer_data) == 1

    def test_infix_map_passed_to_objective_state(
        self, model: MockCausalLM, tok: MockTokenizer,
    ) -> None:
        """infix_map should be forwarded to the objective state."""
        config = _make_config()
        infix_map = {0: 5}

        result = prepare_attack(
            model, tok, ["Hello"], config, direction=None,
            infix_map=infix_map,
        )

        assert result.objective_state.infix_map == {0: 5}
