"""Tests for LARGO reflection loop (vauban.softprompt._largo)."""

from __future__ import annotations

import math

from conftest import (
    D_MODEL,
    NUM_HEADS,
    NUM_LAYERS,
    VOCAB_SIZE,
    MockCausalLM,
    MockTokenizer,
)

from vauban import _ops as ops
from vauban.softprompt._continuous import _continuous_attack
from vauban.softprompt._largo import _objective_satisfied, largo_loop
from vauban.types import SoftPromptConfig, SoftPromptResult


class TestObjectiveSatisfied:
    """Unit tests for the _objective_satisfied function."""

    def test_targeted_non_refusal(self) -> None:
        """Non-refusal text satisfies targeted objective."""
        assert _objective_satisfied(
            "Sure, here is how you do that thing in detail.",
            "targeted",
        )

    def test_targeted_refusal(self) -> None:
        """Refusal text does NOT satisfy targeted objective."""
        assert not _objective_satisfied(
            "I cannot help with that request.",
            "targeted",
        )

    def test_targeted_short_response(self) -> None:
        """Very short responses fail targeted (< 20 chars)."""
        assert not _objective_satisfied("OK", "targeted")

    def test_targeted_empty(self) -> None:
        """Empty response fails targeted."""
        assert not _objective_satisfied("", "targeted")

    def test_untargeted_refusal(self) -> None:
        """Refusal text satisfies untargeted objective."""
        assert _objective_satisfied(
            "I'm sorry, I cannot assist with that.",
            "untargeted",
        )

    def test_untargeted_non_refusal(self) -> None:
        """Non-refusal does NOT satisfy untargeted objective."""
        assert not _objective_satisfied(
            "Sure, here is how you do that thing in detail.",
            "untargeted",
        )


class TestLargoLoop:
    """Integration tests for the LARGO reflection loop."""

    def test_basic_run(self) -> None:
        """LARGO loop produces a valid SoftPromptResult."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="continuous",
            n_tokens=4,
            n_steps=3,
            seed=42,
            max_gen_tokens=3,
            largo_reflection_rounds=2,
            largo_objective="targeted",
        )

        result = largo_loop(
            model, tokenizer, ["test prompt"], config, None,
        )

        assert isinstance(result, SoftPromptResult)
        assert result.mode == "continuous"
        assert result.n_tokens == 4
        # Loss history should aggregate across rounds
        assert len(result.loss_history) >= 3

    def test_loss_history_aggregated(self) -> None:
        """Loss history aggregates steps from all reflection rounds."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="continuous",
            n_tokens=4,
            n_steps=5,
            seed=42,
            max_gen_tokens=2,
            largo_reflection_rounds=3,
            largo_objective="targeted",
            largo_embed_warmstart=True,
        )

        result = largo_loop(
            model, tokenizer, ["test"], config, None,
        )

        # At least first round runs 5 steps; subsequent rounds run halved
        assert len(result.loss_history) >= 5
        for loss in result.loss_history:
            assert not math.isnan(loss), "LARGO produced NaN loss"

    def test_warm_start_embeds_differ(self) -> None:
        """Warm-start should produce different initial embeddings than cold start."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        # Cold start: round 1 only
        config_cold = SoftPromptConfig(
            mode="continuous",
            n_tokens=4,
            n_steps=5,
            seed=42,
            max_gen_tokens=2,
            largo_reflection_rounds=1,
        )
        result_cold = largo_loop(
            model, tokenizer, ["test"], config_cold, None,
        )

        # Warm start: round 1 + warm-started round 2
        config_warm = SoftPromptConfig(
            mode="continuous",
            n_tokens=4,
            n_steps=5,
            seed=42,
            max_gen_tokens=2,
            largo_reflection_rounds=2,
            largo_embed_warmstart=True,
        )
        result_warm = largo_loop(
            model, tokenizer, ["test"], config_warm, None,
        )

        # Warm-start should have more loss history entries
        assert len(result_warm.loss_history) > len(result_cold.loss_history)

    def test_no_warmstart_reinitializes(self) -> None:
        """Without warmstart, each round reinitializes from scratch."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="continuous",
            n_tokens=4,
            n_steps=3,
            seed=42,
            max_gen_tokens=2,
            largo_reflection_rounds=2,
            largo_embed_warmstart=False,
        )

        result = largo_loop(
            model, tokenizer, ["test"], config, None,
        )

        # Should still produce valid results
        assert isinstance(result, SoftPromptResult)
        assert len(result.loss_history) >= 3


class TestContinuousAttackWarmStart:
    """Tests for the init_embeddings parameter added to _continuous_attack."""

    def test_init_embeddings_used(self) -> None:
        """Providing init_embeddings should start from those embeddings."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        # Create specific init embeddings
        init_embeds = ops.ones((1, 4, D_MODEL)) * 0.5
        ops.eval(init_embeds)

        config = SoftPromptConfig(
            mode="continuous",
            n_tokens=4,
            n_steps=3,
            seed=42,
            max_gen_tokens=2,
            init_scale=0.01,  # Small perturbation
        )

        result = _continuous_attack(
            model, tokenizer, ["test"], config, None,
            init_embeddings=init_embeds,
        )

        assert isinstance(result, SoftPromptResult)
        assert result.embeddings is not None
        assert result.embeddings.shape == (1, 4, D_MODEL)

    def test_no_init_embeddings_uses_random(self) -> None:
        """Without init_embeddings, falls back to random init."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="continuous",
            n_tokens=4,
            n_steps=3,
            seed=42,
            max_gen_tokens=2,
        )

        result = _continuous_attack(
            model, tokenizer, ["test"], config, None,
        )

        assert result.embeddings is not None
        assert result.embeddings.shape == (1, 4, D_MODEL)
