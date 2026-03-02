"""Tests for shared attack initialization (AttackInitState + prepare_attack)."""

from __future__ import annotations

from conftest import (
    D_MODEL,
    NUM_HEADS,
    NUM_LAYERS,
    VOCAB_SIZE,
    MockCausalLM,
    MockTokenizer,
)

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


class TestPrepareAttack:
    """Tests for the prepare_attack factory."""

    def test_returns_attack_init_state(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        tok = MockTokenizer(VOCAB_SIZE)
        config = _make_config()

        result = prepare_attack(model, tok, ["Hello"], config, direction=None)

        assert isinstance(result, AttackInitState)

    def test_vocab_size_and_d_model(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        tok = MockTokenizer(VOCAB_SIZE)
        config = _make_config()

        result = prepare_attack(model, tok, ["Hello"], config, direction=None)

        assert result.vocab_size == VOCAB_SIZE
        assert result.d_model == D_MODEL

    def test_embed_matrix_shape(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        tok = MockTokenizer(VOCAB_SIZE)
        config = _make_config()

        result = prepare_attack(model, tok, ["Hello"], config, direction=None)

        assert result.embed_matrix.shape == (VOCAB_SIZE, D_MODEL)

    def test_target_ids_nonempty(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        tok = MockTokenizer(VOCAB_SIZE)
        config = _make_config()

        result = prepare_attack(model, tok, ["Hello"], config, direction=None)

        assert result.target_ids.shape[0] > 0

    def test_all_prompt_ids_length(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        tok = MockTokenizer(VOCAB_SIZE)
        config = _make_config()
        prompts = ["Hello", "World"]

        result = prepare_attack(model, tok, prompts, config, direction=None)

        assert len(result.all_prompt_ids) == 2

    def test_prompt_ids_override(self) -> None:
        from vauban import _ops as ops

        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        tok = MockTokenizer(VOCAB_SIZE)
        config = _make_config()
        override = [ops.array([1, 2, 3]), ops.array([4, 5])]

        result = prepare_attack(
            model, tok, ["ignored"], config, direction=None,
            all_prompt_ids_override=override,
        )

        assert len(result.all_prompt_ids) == 2

    def test_objective_state_populated(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        tok = MockTokenizer(VOCAB_SIZE)
        config = _make_config()

        result = prepare_attack(model, tok, ["Hello"], config, direction=None)

        assert result.objective_state.n_tokens == 4
        assert result.objective_state.model is model

    def test_transfer_data_empty_by_default(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        tok = MockTokenizer(VOCAB_SIZE)
        config = _make_config()

        result = prepare_attack(model, tok, ["Hello"], config, direction=None)

        assert result.transfer_data == []

    def test_empty_prompts_defaults_to_hello(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        tok = MockTokenizer(VOCAB_SIZE)
        config = _make_config()

        result = prepare_attack(model, tok, [], config, direction=None)

        assert len(result.all_prompt_ids) == 1

    def test_perplexity_weight_override(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        tok = MockTokenizer(VOCAB_SIZE)
        config = _make_config(perplexity_weight=1.0)

        result = prepare_attack(
            model, tok, ["Hello"], config, direction=None,
            perplexity_weight_override=0.0,
        )

        assert result.objective_state.perplexity_weight == 0.0
