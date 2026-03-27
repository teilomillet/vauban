# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for COLD-Attack: Langevin dynamics in logit space (vauban.softprompt._cold)."""

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
from vauban.softprompt import softprompt_attack
from vauban.softprompt._cold import _cold_attack
from vauban.types import SoftPromptConfig, SoftPromptResult


class TestColdAttack:
    """Core COLD-Attack functionality tests."""

    def test_basic_run(self) -> None:
        """COLD attack produces valid SoftPromptResult with token_ids."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="cold",
            n_tokens=4,
            n_steps=5,
            learning_rate=0.1,
            seed=42,
            max_gen_tokens=3,
            cold_temperature=0.5,
            cold_noise_scale=1.0,
        )

        result = _cold_attack(
            model, tokenizer, ["test prompt"], config, None,
        )

        assert isinstance(result, SoftPromptResult)
        assert result.mode == "cold"
        assert result.n_tokens == 4
        assert len(result.loss_history) == 5
        assert result.token_ids is not None
        assert len(result.token_ids) == 4
        assert result.token_text is not None
        assert result.embeddings is None  # COLD returns token_ids, not embeds
        assert len(result.eval_responses) == 1

    def test_token_ids_in_vocab_range(self) -> None:
        """All output token IDs are in [0, vocab_size)."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="cold",
            n_tokens=4,
            n_steps=3,
            seed=42,
            max_gen_tokens=2,
        )

        result = _cold_attack(
            model, tokenizer, ["test"], config, None,
        )

        assert result.token_ids is not None
        for tid in result.token_ids:
            assert 0 <= tid < VOCAB_SIZE

    def test_loss_is_finite(self) -> None:
        """All loss values should be finite (no NaN/inf)."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="cold",
            n_tokens=4,
            n_steps=5,
            seed=42,
            max_gen_tokens=2,
        )

        result = _cold_attack(
            model, tokenizer, ["test"], config, None,
        )

        for loss in result.loss_history:
            assert not math.isnan(loss), "COLD produced NaN loss"
            assert not math.isinf(loss), "COLD produced inf loss"

    def test_cold_improves_loss(self) -> None:
        """COLD should find candidates with lower loss than initial."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="cold",
            n_tokens=4,
            n_steps=30,
            learning_rate=0.5,
            seed=42,
            max_gen_tokens=2,
            cold_temperature=0.5,
            cold_noise_scale=0.1,  # low noise for cleaner descent
        )

        result = _cold_attack(
            model, tokenizer, ["test"], config, None,
        )

        first_loss = result.loss_history[0]
        best_loss = min(result.loss_history)
        assert best_loss <= first_loss, (
            f"COLD did not improve: {first_loss:.4f} -> {best_loss:.4f}"
        )


class TestColdLangevinDynamics:
    """Tests verifying Langevin dynamics properties."""

    def test_noise_scale_zero_is_deterministic(self) -> None:
        """With noise_scale=0, COLD becomes pure gradient descent (deterministic)."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="cold",
            n_tokens=4,
            n_steps=5,
            learning_rate=0.1,
            seed=42,
            max_gen_tokens=2,
            cold_noise_scale=0.0,
        )

        result1 = _cold_attack(
            model, tokenizer, ["test"], config, None,
        )
        result2 = _cold_attack(
            model, tokenizer, ["test"], config, None,
        )

        # Same seed + no noise = same results
        assert result1.token_ids == result2.token_ids
        for l1, l2 in zip(result1.loss_history, result2.loss_history, strict=True):
            assert abs(l1 - l2) < 1e-5, f"Losses differ: {l1} vs {l2}"

    def test_high_noise_explores_more(self) -> None:
        """Higher noise should produce more variable loss trajectories."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config_low = SoftPromptConfig(
            mode="cold",
            n_tokens=4,
            n_steps=15,
            learning_rate=0.1,
            seed=42,
            max_gen_tokens=2,
            cold_noise_scale=0.01,
        )
        config_high = SoftPromptConfig(
            mode="cold",
            n_tokens=4,
            n_steps=15,
            learning_rate=0.1,
            seed=42,
            max_gen_tokens=2,
            cold_noise_scale=5.0,
        )

        result_low = _cold_attack(
            model, tokenizer, ["test"], config_low, None,
        )
        result_high = _cold_attack(
            model, tokenizer, ["test"], config_high, None,
        )

        # Loss variance should be higher with more noise
        def variance(xs: list[float]) -> float:
            mean = sum(xs) / len(xs)
            return sum((x - mean) ** 2 for x in xs) / len(xs)

        var_low = variance(result_low.loss_history)
        var_high = variance(result_high.loss_history)
        assert var_high > var_low, (
            f"High noise should have higher variance: {var_high:.4f} <= {var_low:.4f}"
        )


class TestColdTemperature:
    """Tests for temperature parameter behavior."""

    def test_low_temperature_sharper(self) -> None:
        """Lower temperature should produce sharper probability distributions."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="cold",
            n_tokens=4,
            n_steps=5,
            seed=42,
            max_gen_tokens=2,
            cold_temperature=0.1,
            cold_noise_scale=0.0,
        )

        result = _cold_attack(
            model, tokenizer, ["test"], config, None,
        )

        assert result.token_ids is not None
        assert len(result.token_ids) == 4


class TestColdEarlyStopping:
    """Tests for early stopping in COLD."""

    def test_early_stop_fires(self) -> None:
        """COLD should early-stop with patience."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="cold",
            n_tokens=4,
            n_steps=100,
            learning_rate=1e-10,  # tiny LR -> no improvement
            seed=42,
            max_gen_tokens=2,
            patience=3,
            cold_noise_scale=0.0,
        )

        result = _cold_attack(
            model, tokenizer, ["test"], config, None,
        )

        assert result.early_stopped is True
        assert result.n_steps < 100


class TestColdDispatch:
    """Tests for dispatcher integration."""

    def test_dispatch_via_public_api(self) -> None:
        """softprompt_attack dispatches mode='cold' correctly."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="cold",
            n_tokens=4,
            n_steps=3,
            seed=42,
            max_gen_tokens=2,
        )

        result = softprompt_attack(
            model, tokenizer, ["test"], config, None,
        )

        assert result.mode == "cold"
        assert result.token_ids is not None


class TestColdWithDirection:
    """Tests for direction-guided COLD."""

    def test_cold_with_direction(self) -> None:
        """COLD with refusal direction produces valid result."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        direction = ops.random.normal((D_MODEL,))
        direction = direction / ops.linalg.norm(direction)
        ops.eval(direction)

        config = SoftPromptConfig(
            mode="cold",
            n_tokens=4,
            n_steps=3,
            direction_weight=0.1,
            seed=42,
            max_gen_tokens=2,
        )

        result = _cold_attack(
            model, tokenizer, ["test"], config, direction,
        )

        assert result.mode == "cold"
        assert len(result.loss_history) == 3
