# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for vauban.softprompt attack flows and evaluation behavior."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

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
from vauban.softprompt import softprompt_attack
from vauban.softprompt._continuous import _continuous_attack
from vauban.softprompt._egd import _egd_attack
from vauban.softprompt._gcg import _gcg_attack
from vauban.softprompt._generation import (
    _evaluate_attack,
    _evaluate_attack_with_history,
)
from vauban.softprompt._loss import _compute_loss
from vauban.softprompt._utils import _encode_targets
from vauban.types import (
    EnvironmentConfig,
    EnvironmentResult,
    GanRoundResult,
    SoftPromptConfig,
    SoftPromptResult,
    TransferEvalResult,
)

if TYPE_CHECKING:
    from pathlib import Path

    from vauban._array import Array


def _fake_rollout_score(
    _model: object,
    _tokenizer: object,
    _env_config: object,
    texts: list[str],
    losses: list[float],
) -> tuple[list[float], list[EnvironmentResult]]:
    """Stub for score_candidates_via_rollout that returns losses unchanged."""
    results = [
        EnvironmentResult(
            reward=0.0,
            target_called=False,
            target_args_match=False,
            turns=[],
            tool_calls_made=[],
            injection_payload=text,
        )
        for text in texts
    ]
    return list(losses), results


def _make_rollout_env_config(rollout_every_n: int = 1) -> EnvironmentConfig:
    """Build a minimal EnvironmentConfig for rollout wiring tests."""
    from vauban.types import EnvironmentTarget, EnvironmentTask, ToolSchema

    return EnvironmentConfig(
        system_prompt="You are a helpful assistant.",
        tools=[ToolSchema(name="noop", description="no-op", parameters={})],
        target=EnvironmentTarget(function="noop"),
        task=EnvironmentTask(content="Do something."),
        injection_surface="tool_output",
        rollout_every_n=rollout_every_n,
    )


class TestContinuousAttack:
    def test_basic_run(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="continuous",
            n_tokens=4,
            n_steps=5,
            learning_rate=0.01,
            seed=42,
            max_gen_tokens=3,
        )

        result = _continuous_attack(
            model, tokenizer, ["test prompt"], config, None,
        )

        assert isinstance(result, SoftPromptResult)
        assert result.mode == "continuous"
        assert result.n_steps == 5
        assert result.n_tokens == 4
        assert len(result.loss_history) == 5
        assert result.embeddings is not None
        assert result.embeddings.shape == (1, 4, D_MODEL)
        assert result.token_ids is None
        assert result.token_text is None
        assert len(result.eval_responses) == 1

    def test_loss_is_finite(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            n_tokens=4,
            n_steps=3,
            seed=42,
            max_gen_tokens=2,
        )

        result = _continuous_attack(
            model, tokenizer, ["hello"], config, None,
        )

        for loss in result.loss_history:
            assert not math.isnan(loss), "Loss is NaN"

    def test_loss_decreases(self) -> None:
        """Verify that optimization actually reduces loss (gradient flow works)."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="continuous",
            n_tokens=4,
            n_steps=30,
            learning_rate=0.1,
            init_scale=0.5,
            seed=42,
            max_gen_tokens=2,
        )

        result = _continuous_attack(
            model, tokenizer, ["test"], config, None,
        )

        first_loss = result.loss_history[0]
        final_loss = result.loss_history[-1]
        assert final_loss < first_loss, (
            f"Loss did not decrease: {first_loss:.4f} -> {final_loss:.4f}"
        )

    def test_gradient_nonzero(self) -> None:
        """Verify gradient w.r.t. soft embeds is nonzero."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        soft_embeds = ops.random.normal((1, 4, D_MODEL)) * 0.1
        ops.eval(soft_embeds)

        target_ids = _encode_targets(tokenizer, ["Sure"])
        ops.eval(target_ids)

        messages = [{"role": "user", "content": "test"}]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        assert isinstance(text, str)
        prompt_ids = ops.array(tokenizer.encode(text))[None, :]

        def loss_fn(embeds: Array) -> Array:
            return _compute_loss(model, embeds, prompt_ids, target_ids, 4, None, 0.0)

        loss_and_grad = ops.value_and_grad(loss_fn)
        loss_val, grad = loss_and_grad(soft_embeds)
        ops.eval(loss_val, grad)

        grad_norm = float(ops.linalg.norm(grad.reshape(-1)).item())
        assert grad_norm > 0, "Gradient is zero — no learning signal"

    def test_dispatch_via_public_api(self) -> None:
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

        result = softprompt_attack(
            model, tokenizer, ["test"], config, None,
        )
        assert result.mode == "continuous"


class TestGCGAttack:
    def test_basic_run(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="gcg",
            n_tokens=4,
            n_steps=3,
            batch_size=8,
            top_k=16,
            seed=42,
            max_gen_tokens=3,
        )

        result = _gcg_attack(
            model, tokenizer, ["test prompt"], config, None,
        )

        assert isinstance(result, SoftPromptResult)
        assert result.mode == "gcg"
        assert result.n_tokens == 4
        assert len(result.loss_history) == 3
        assert result.embeddings is None
        assert result.token_ids is not None
        assert len(result.token_ids) == 4
        assert result.token_text is not None
        assert len(result.eval_responses) == 1

    def test_token_ids_in_vocab_range(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="gcg",
            n_tokens=4,
            n_steps=2,
            batch_size=4,
            top_k=8,
            seed=42,
            max_gen_tokens=2,
        )

        result = _gcg_attack(
            model, tokenizer, ["test"], config, None,
        )

        assert result.token_ids is not None
        for tid in result.token_ids:
            assert 0 <= tid < VOCAB_SIZE

    def test_gcg_improves_loss(self) -> None:
        """Verify GCG finds at least one candidate better than random init."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="gcg",
            n_tokens=4,
            n_steps=10,
            batch_size=16,
            top_k=VOCAB_SIZE,  # search full vocab on small model
            seed=42,
            max_gen_tokens=2,
        )

        result = _gcg_attack(
            model, tokenizer, ["test"], config, None,
        )

        first_loss = result.loss_history[0]
        best_loss = result.final_loss
        assert best_loss <= first_loss, (
            f"GCG did not improve: {first_loss:.4f} -> {best_loss:.4f}"
        )

    def test_dispatch_via_public_api(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="gcg",
            n_tokens=4,
            n_steps=2,
            batch_size=4,
            top_k=8,
            seed=42,
            max_gen_tokens=2,
        )

        result = softprompt_attack(
            model, tokenizer, ["test"], config, None,
        )
        assert result.mode == "gcg"


class TestDirectionGuided:
    def test_continuous_with_direction(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        direction = ops.random.normal((D_MODEL,))
        direction = direction / ops.linalg.norm(direction)
        ops.eval(direction)

        config = SoftPromptConfig(
            mode="continuous",
            n_tokens=4,
            n_steps=3,
            direction_weight=0.1,
            seed=42,
            max_gen_tokens=2,
        )

        result = softprompt_attack(
            model, tokenizer, ["test"], config, direction,
        )

        assert result.mode == "continuous"
        assert len(result.loss_history) == 3

    def test_gcg_with_direction(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        direction = ops.random.normal((D_MODEL,))
        direction = direction / ops.linalg.norm(direction)
        ops.eval(direction)

        config = SoftPromptConfig(
            mode="gcg",
            n_tokens=4,
            n_steps=2,
            batch_size=4,
            top_k=8,
            direction_weight=0.1,
            seed=42,
            max_gen_tokens=2,
        )

        result = softprompt_attack(
            model, tokenizer, ["test"], config, direction,
        )

        assert result.mode == "gcg"
        assert len(result.loss_history) == 2


class TestEvaluateAttack:
    def test_success_rate_range(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        soft_embeds = ops.random.normal((1, 4, D_MODEL))
        ops.eval(soft_embeds)

        config = SoftPromptConfig(max_gen_tokens=3)

        success_rate, responses = _evaluate_attack(
            model, tokenizer, ["test1", "test2"], soft_embeds, config,
        )

        assert 0.0 <= success_rate <= 1.0
        assert len(responses) == 2

    def test_empty_prompts(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        soft_embeds = ops.random.normal((1, 4, D_MODEL))
        ops.eval(soft_embeds)

        config = SoftPromptConfig(max_gen_tokens=2)

        success_rate, responses = _evaluate_attack(
            model, tokenizer, [], soft_embeds, config,
        )

        assert success_rate == 0.0
        assert responses == []


class TestInvalidMode:
    def test_invalid_mode_raises(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        # Bypass frozen by constructing manually
        bad_config = SoftPromptConfig.__new__(SoftPromptConfig)
        object.__setattr__(bad_config, "mode", "invalid")
        object.__setattr__(bad_config, "n_tokens", 4)
        object.__setattr__(bad_config, "n_steps", 1)
        object.__setattr__(bad_config, "seed", None)
        object.__setattr__(bad_config, "gan_rounds", 0)
        object.__setattr__(bad_config, "injection_context", None)
        object.__setattr__(bad_config, "injection_context_template", None)
        object.__setattr__(bad_config, "paraphrase_strategies", [])
        object.__setattr__(bad_config, "token_position", "prefix")
        object.__setattr__(bad_config, "largo_reflection_rounds", 0)

        with pytest.raises(ValueError, match="Unknown soft prompt mode"):
            softprompt_attack(model, tokenizer, ["test"], bad_config, None)


class TestMultiPromptContinuous:
    def test_all_strategy(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            n_tokens=4,
            n_steps=3,
            seed=42,
            max_gen_tokens=2,
            prompt_strategy="all",
        )

        result = _continuous_attack(
            model, tokenizer, ["hello", "world"], config, None,
        )

        assert result.mode == "continuous"
        assert len(result.per_prompt_losses) == 2
        assert all(loss > 0 for loss in result.per_prompt_losses)
        assert result.accessibility_score > 0.0

    def test_cycle_strategy(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            n_tokens=4,
            n_steps=4,
            seed=42,
            max_gen_tokens=2,
            prompt_strategy="cycle",
        )

        result = _continuous_attack(
            model, tokenizer, ["hello", "world"], config, None,
        )

        assert result.mode == "continuous"
        assert len(result.per_prompt_losses) == 2
        assert len(result.loss_history) == 4


class TestEarlyStopping:
    def test_fires_with_patience(self) -> None:
        """Early stopping should fire with tiny LR and small patience."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            n_tokens=4,
            n_steps=100,
            learning_rate=1e-10,  # tiny LR -> no improvement
            seed=42,
            max_gen_tokens=2,
            patience=3,
        )

        result = _continuous_attack(
            model, tokenizer, ["test"], config, None,
        )

        assert result.early_stopped is True
        assert result.n_steps < 100

    def test_disabled_with_patience_zero(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            n_tokens=4,
            n_steps=5,
            seed=42,
            max_gen_tokens=2,
            patience=0,
        )

        result = _continuous_attack(
            model, tokenizer, ["test"], config, None,
        )

        assert result.early_stopped is False
        assert result.n_steps == 5


class TestCosineSchedule:
    def test_runs_without_nan(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            n_tokens=4,
            n_steps=10,
            learning_rate=0.01,
            seed=42,
            max_gen_tokens=2,
            lr_schedule="cosine",
        )

        result = _continuous_attack(
            model, tokenizer, ["test"], config, None,
        )

        for loss in result.loss_history:
            assert not math.isnan(loss), "Cosine schedule produced NaN loss"


class TestEmbedRegularizationIntegration:
    def test_runs_without_error(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            n_tokens=4,
            n_steps=3,
            seed=42,
            max_gen_tokens=2,
            embed_reg_weight=0.1,
        )

        result = _continuous_attack(
            model, tokenizer, ["test"], config, None,
        )

        assert result.mode == "continuous"
        for loss in result.loss_history:
            assert not math.isnan(loss)


class TestGCGMultiRestart:
    def test_correct_loss_history_length(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="gcg",
            n_tokens=4,
            n_steps=3,
            batch_size=4,
            top_k=8,
            seed=42,
            max_gen_tokens=2,
            n_restarts=2,
        )

        result = _gcg_attack(
            model, tokenizer, ["test"], config, None,
        )

        # 2 restarts * 3 steps = 6 loss history entries
        assert len(result.loss_history) == 6
        assert result.n_steps == 6

    def test_token_ids_correct_length(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="gcg",
            n_tokens=4,
            n_steps=2,
            batch_size=4,
            top_k=8,
            seed=42,
            max_gen_tokens=2,
            n_restarts=3,
        )

        result = _gcg_attack(
            model, tokenizer, ["test"], config, None,
        )

        assert result.token_ids is not None
        assert len(result.token_ids) == 4
        for tid in result.token_ids:
            assert 0 <= tid < VOCAB_SIZE


class TestEGDAttack:
    def test_basic_run(self) -> None:
        """mode='egd' returns valid SoftPromptResult with token_ids."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="egd",
            n_tokens=4,
            n_steps=5,
            learning_rate=0.1,
            seed=42,
            max_gen_tokens=2,
        )

        result = _egd_attack(
            model, tokenizer, ["test"], config, None,
        )

        assert isinstance(result, SoftPromptResult)
        assert result.mode == "egd"
        assert result.token_ids is not None
        assert len(result.token_ids) == 4
        assert result.token_text is not None
        assert result.embeddings is None
        assert len(result.loss_history) == 5
        assert len(result.eval_responses) == 1

    def test_egd_improves_loss(self) -> None:
        """Loss decreases over steps."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="egd",
            n_tokens=4,
            n_steps=30,
            learning_rate=0.5,
            seed=42,
            max_gen_tokens=2,
        )

        result = _egd_attack(
            model, tokenizer, ["test"], config, None,
        )

        first_loss = result.loss_history[0]
        best_loss = min(result.loss_history)
        assert best_loss < first_loss, (
            f"EGD did not improve: {first_loss:.4f} -> {best_loss:.4f}"
        )

    def test_egd_token_ids_in_range(self) -> None:
        """All token IDs in [0, vocab_size)."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="egd",
            n_tokens=4,
            n_steps=3,
            learning_rate=0.1,
            seed=42,
            max_gen_tokens=2,
        )

        result = _egd_attack(
            model, tokenizer, ["test"], config, None,
        )

        assert result.token_ids is not None
        for tid in result.token_ids:
            assert 0 <= tid < VOCAB_SIZE

    def test_egd_dispatch(self) -> None:
        """softprompt_attack dispatches correctly to EGD."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="egd",
            n_tokens=4,
            n_steps=3,
            learning_rate=0.1,
            seed=42,
            max_gen_tokens=2,
        )

        result = softprompt_attack(
            model, tokenizer, ["test"], config, None,
        )
        assert result.mode == "egd"


class TestWorstKIntegration:
    def test_continuous_worst_k(self) -> None:
        """Continuous mode with worst_k strategy runs."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="continuous",
            n_tokens=4,
            n_steps=3,
            seed=42,
            max_gen_tokens=2,
            prompt_strategy="worst_k",
            worst_k=2,
        )

        result = _continuous_attack(
            model, tokenizer, ["hello", "world", "test"], config, None,
        )

        assert result.mode == "continuous"
        for loss in result.loss_history:
            assert not math.isnan(loss)

    def test_gcg_worst_k(self) -> None:
        """GCG mode with worst_k strategy runs."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="gcg",
            n_tokens=4,
            n_steps=2,
            batch_size=4,
            top_k=8,
            seed=42,
            max_gen_tokens=2,
            prompt_strategy="worst_k",
            worst_k=1,
        )

        result = _gcg_attack(
            model, tokenizer, ["hello", "world"], config, None,
        )

        assert result.mode == "gcg"
        assert len(result.loss_history) == 2

    def test_egd_worst_k(self) -> None:
        """EGD mode with worst_k strategy runs."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="egd",
            n_tokens=4,
            n_steps=3,
            learning_rate=0.1,
            seed=42,
            max_gen_tokens=2,
            prompt_strategy="worst_k",
            worst_k=1,
        )

        result = _egd_attack(
            model, tokenizer, ["hello", "world"], config, None,
        )

        assert result.mode == "egd"
        for loss in result.loss_history:
            assert not math.isnan(loss)


class TestEvaluateAttackWithHistory:
    """Tests for _evaluate_attack_with_history."""

    def test_basic_evaluation(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        tok = MockTokenizer(VOCAB_SIZE)
        config = SoftPromptConfig(n_tokens=2, max_gen_tokens=5)
        soft_embeds = ops.random.normal((1, 2, D_MODEL))
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        success_rate, responses = _evaluate_attack_with_history(
            model, tok, ["Test prompt"], soft_embeds, config, history,
        )
        assert 0.0 <= success_rate <= 1.0
        assert len(responses) == 1

    def test_empty_history_works(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        tok = MockTokenizer(VOCAB_SIZE)
        config = SoftPromptConfig(n_tokens=2, max_gen_tokens=3)
        soft_embeds = ops.random.normal((1, 2, D_MODEL))
        _success_rate, responses = _evaluate_attack_with_history(
            model, tok, ["Hello"], soft_embeds, config, history=[],
        )
        assert len(responses) == 1


class TestGcgBeamSearch:
    def test_beam_width_1_is_greedy(self) -> None:
        """beam_width=1 should produce valid results (fast path)."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        tokenizer = MockTokenizer(VOCAB_SIZE)
        config = SoftPromptConfig(
            mode="gcg",
            n_tokens=4,
            n_steps=2,
            batch_size=4,
            top_k=8,
            beam_width=1,
        )
        result = _gcg_attack(model, tokenizer, ["Hello"], config, None)
        assert result.mode == "gcg"
        assert result.token_ids is not None
        assert len(result.token_ids) == 4

    def test_beam_width_gt1_produces_result(self) -> None:
        """beam_width>1 should produce valid results with beam search."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        tokenizer = MockTokenizer(VOCAB_SIZE)
        config = SoftPromptConfig(
            mode="gcg",
            n_tokens=4,
            n_steps=2,
            batch_size=8,
            top_k=8,
            beam_width=3,
        )
        result = _gcg_attack(model, tokenizer, ["Hello"], config, None)
        assert result.mode == "gcg"
        assert result.token_ids is not None
        assert len(result.token_ids) == 4

    def test_zero_steps_falls_back_to_initial_tokens(self) -> None:
        """Zero-step GCG should still return a valid initialized suffix."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        tokenizer = MockTokenizer(VOCAB_SIZE)
        config = SoftPromptConfig(
            mode="gcg",
            n_tokens=4,
            n_steps=0,
            batch_size=8,
            top_k=8,
            beam_width=3,
        )
        result = _gcg_attack(model, tokenizer, ["Hello"], config, None)
        assert result.mode == "gcg"
        assert result.token_ids is not None
        assert len(result.token_ids) == 4


class TestWriteArenaCard:
    def test_basic_card_written(self, tmp_path: Path) -> None:
        """Arena card should be written with expected sections."""
        from vauban._pipeline._helpers import write_arena_card as _write_arena_card

        result = SoftPromptResult(
            mode="gcg", success_rate=0.75, final_loss=1.2,
            loss_history=[2.0, 1.5, 1.2], n_steps=3, n_tokens=8,
            embeddings=None, token_ids=[1, 2, 3],
            token_text="adversarial suffix", eval_responses=["Sure thing"],
        )
        card_path = tmp_path / "arena_card.txt"
        _write_arena_card(card_path, result, ["Test prompt"])

        assert card_path.exists()
        content = card_path.read_text()
        assert "ARENA SUBMISSION CARD" in content
        assert "SUFFIX (copy-paste ready)" in content
        assert "adversarial suffix" in content
        assert "PER-PROMPT SUBMISSIONS" in content
        assert "Test prompt" in content
        assert "75.00%" in content

    def test_card_with_transfer_results(self, tmp_path: Path) -> None:
        """Arena card should include transfer results when present."""
        from vauban._pipeline._helpers import write_arena_card as _write_arena_card

        result = SoftPromptResult(
            mode="gcg", success_rate=0.5, final_loss=2.0,
            loss_history=[2.0], n_steps=1, n_tokens=4,
            embeddings=None, token_ids=[1, 2],
            token_text="suffix", eval_responses=["ok"],
            transfer_results=[
                TransferEvalResult(
                    model_id="test-model", success_rate=0.8,
                    eval_responses=["resp"],
                ),
            ],
        )
        card_path = tmp_path / "card.txt"
        _write_arena_card(card_path, result, ["prompt"])

        content = card_path.read_text()
        assert "TRANSFER RESULTS" in content
        assert "test-model" in content
        assert "80.00%" in content

    def test_card_with_gan_history(self, tmp_path: Path) -> None:
        """Arena card should include GAN round history."""
        from vauban._pipeline._helpers import write_arena_card as _write_arena_card

        attack = SoftPromptResult(
            mode="gcg", success_rate=0.6, final_loss=1.5,
            loss_history=[1.5], n_steps=1, n_tokens=4,
            embeddings=None, token_ids=[1, 2],
            token_text="suffix", eval_responses=["ok"],
        )
        rnd = GanRoundResult(
            round_index=0, attack_result=attack,
            defense_result=None, attacker_won=True,
            config_snapshot={},
        )
        result = SoftPromptResult(
            mode="gcg", success_rate=0.6, final_loss=1.5,
            loss_history=[1.5], n_steps=1, n_tokens=4,
            embeddings=None, token_ids=[1, 2],
            token_text="suffix", eval_responses=["ok"],
            gan_history=[rnd],
        )
        card_path = tmp_path / "card.txt"
        _write_arena_card(card_path, result, ["prompt"])

        content = card_path.read_text()
        assert "GAN ROUND HISTORY" in content
        assert "Round 0: WON" in content


class TestInjectionContextDispatch:
    """Tests that injection context wires through to GCG/EGD."""

    def test_gcg_with_injection_context(self) -> None:
        """GCG attack with injection_context produces valid result."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)
        config = SoftPromptConfig(
            mode="gcg",
            n_tokens=4,
            n_steps=2,
            batch_size=4,
            top_k=8,
            injection_context="web_page",
        )
        result = softprompt_attack(
            model, tokenizer, ["Hello"], config, None,
        )
        assert result.mode == "gcg"
        assert result.token_ids is not None
        assert len(result.token_ids) == 4

    def test_egd_with_injection_template(self) -> None:
        """EGD attack with injection_context_template works."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)
        config = SoftPromptConfig(
            mode="egd",
            n_tokens=4,
            n_steps=2,
            batch_size=4,
            top_k=8,
            injection_context_template="Doc: {payload}",
        )
        result = softprompt_attack(
            model, tokenizer, ["Hello"], config, None,
        )
        assert result.mode == "egd"
        assert result.token_ids is not None


class TestThreeFeaturesIntegration:
    def test_loss_with_all_features(self) -> None:
        """Compute loss with perplexity + infix position + paraphrased prompts."""
        from vauban.softprompt import paraphrase_prompts
        from vauban.softprompt._loss import _compute_loss

        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())

        # Paraphrase
        prompts = ["do something dangerous"]
        expanded = paraphrase_prompts(prompts, ["narrative", "technical"])
        assert len(expanded) == 3  # 1 original + 2 paraphrased

        # Infix position with perplexity
        soft_embeds = ops.random.normal((1, 4, D_MODEL)) * 0.1
        prompt_ids = ops.array([[1, 2, 3, 4, 5]])
        target_ids = ops.array([5, 6])
        suffix_ids = ops.array([[1, 2, 3, 4]])

        loss = _compute_loss(
            model, soft_embeds, prompt_ids, target_ids,
            n_tokens=4, direction=None, direction_weight=0.0,
            perplexity_weight=0.1, suffix_token_ids=suffix_ids,
            token_position="infix", infix_split=2,
        )
        ops.eval(loss)
        assert loss.shape == ()
        assert float(loss.item()) > 0.0


class TestGcgRolloutWiring:
    def test_rollout_called_during_gcg(self) -> None:
        """Verify environment rollout is wired into GCG candidate scoring."""
        from unittest.mock import patch

        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="gcg",
            n_tokens=4,
            n_steps=1,
            batch_size=4,
            top_k=8,
            seed=42,
            max_gen_tokens=2,
        )

        with patch(
            "vauban.environment.score_candidates_via_rollout",
            side_effect=_fake_rollout_score,
        ) as mock_rollout:
            _gcg_attack(
                model, tokenizer, ["test prompt"], config, None,
                environment_config=_make_rollout_env_config(),
            )
            assert mock_rollout.call_count >= 1

    def test_rollout_every_n_skips_steps(self) -> None:
        """Verify rollout_every_n=3 skips intermediate steps.

        With patience=0 (default, disabled) no early stopping can occur,
        so all 6 steps run and rollout fires at steps 0 and 3 only.
        """
        from unittest.mock import patch

        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="gcg",
            n_tokens=4,
            n_steps=6,
            batch_size=4,
            top_k=8,
            seed=42,
            max_gen_tokens=2,
        )

        with patch(
            "vauban.environment.score_candidates_via_rollout",
            side_effect=_fake_rollout_score,
        ) as mock_rollout:
            _gcg_attack(
                model, tokenizer, ["test prompt"], config, None,
                environment_config=_make_rollout_env_config(rollout_every_n=3),
            )
            # Steps 0,1,2,3,4,5 — rollout at 0,3 only = 2 calls
            assert mock_rollout.call_count == 2


class TestInfixMultiturnDispatch:
    def test_dispatch_multiturn_forwards_infix_map(self) -> None:
        """_dispatch_attack_multiturn resolves infix splits with history."""
        from vauban.softprompt._gan import _dispatch_attack_multiturn

        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="gcg",
            n_tokens=4,
            n_steps=1,
            batch_size=4,
            top_k=8,
            seed=42,
            max_gen_tokens=2,
            token_position="infix",
        )

        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        prompts = ["Tell me about {suffix} this topic"]

        result = _dispatch_attack_multiturn(
            model, tokenizer, prompts, config,
            direction=None, ref_model=None,
            history=history,
        )
        assert result.mode == "gcg"
        assert result.token_ids is not None
        assert len(result.token_ids) == 4
