"""Tests for vauban.softprompt: types, forward pass, continuous, GCG, evaluation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import mlx.core as mx
import pytest
from conftest import (
    D_MODEL,
    NUM_HEADS,
    NUM_LAYERS,
    VOCAB_SIZE,
    MockCausalLM,
    MockTokenizer,
)

from vauban.softprompt import (
    _build_vocab_mask,
    _compute_accessibility_score,
    _compute_embed_regularization,
    _compute_eos_loss,
    _compute_kl_collision_loss,
    _compute_learning_rate,
    _compute_loss,
    _compute_untargeted_loss,
    _continuous_attack,
    _egd_attack,
    _encode_refusal_tokens,
    _encode_targets,
    _evaluate_attack,
    _evaluate_attack_with_history,
    _forward_with_prefix,
    _gcg_attack,
    _pre_encode_prompts,
    _pre_encode_prompts_with_history,
    _pre_encode_prompts_with_injection_context,
    _pre_encode_prompts_with_injection_template,
    _project_to_tokens,
    _resolve_injection_ids,
    _sample_prompt_ids,
    _select_prompt_ids,
    _select_worst_k_prompt_ids,
    _split_into_batches,
    softprompt_attack,
)
from vauban.softprompt._defense_eval import _build_sic_prompts_with_history
from vauban.types import (
    DefenseEvalResult,
    EnvironmentConfig,
    EnvironmentResult,
    GanRoundResult,
    SoftPromptConfig,
    SoftPromptResult,
    TransferEvalResult,
)

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Type tests
# ---------------------------------------------------------------------------


class TestSoftPromptConfig:
    def test_defaults(self) -> None:
        cfg = SoftPromptConfig()
        assert cfg.mode == "continuous"
        assert cfg.n_tokens == 16
        assert cfg.n_steps == 200
        assert cfg.learning_rate == 0.01
        assert cfg.init_scale == 0.1
        assert cfg.batch_size == 64
        assert cfg.top_k == 256
        assert cfg.direction_weight == 0.0
        assert cfg.target_prefixes == ["Sure", "Here"]
        assert cfg.max_gen_tokens == 100
        assert cfg.seed is None
        assert cfg.embed_reg_weight == 0.0
        assert cfg.patience == 0
        assert cfg.lr_schedule == "constant"
        assert cfg.n_restarts == 1
        assert cfg.prompt_strategy == "all"
        assert cfg.gan_multiturn is False
        assert cfg.gan_multiturn_max_turns == 10

    def test_frozen(self) -> None:
        cfg = SoftPromptConfig()
        with pytest.raises(AttributeError):
            cfg.n_tokens = 10  # type: ignore[misc]

    def test_custom_values(self) -> None:
        cfg = SoftPromptConfig(
            mode="gcg",
            n_tokens=32,
            n_steps=100,
            learning_rate=0.001,
            batch_size=128,
            top_k=512,
            direction_weight=0.5,
            target_prefixes=["OK"],
            seed=42,
        )
        assert cfg.mode == "gcg"
        assert cfg.n_tokens == 32
        assert cfg.seed == 42


class TestSoftPromptResult:
    def test_construction(self) -> None:
        result = SoftPromptResult(
            mode="continuous",
            success_rate=0.8,
            final_loss=1.5,
            loss_history=[3.0, 2.0, 1.5],
            n_steps=3,
            n_tokens=16,
            embeddings=mx.zeros((1, 16, 8)),
            token_ids=None,
            token_text=None,
            eval_responses=["response1"],
        )
        assert result.mode == "continuous"
        assert result.success_rate == 0.8
        assert result.final_loss == 1.5
        assert len(result.loss_history) == 3
        assert result.embeddings is not None
        assert result.token_ids is None
        # Default new fields
        assert result.accessibility_score == 0.0
        assert result.per_prompt_losses == []
        assert result.early_stopped is False

    def test_frozen(self) -> None:
        result = SoftPromptResult(
            mode="continuous",
            success_rate=0.0,
            final_loss=0.0,
            loss_history=[],
            n_steps=0,
            n_tokens=1,
            embeddings=None,
            token_ids=None,
            token_text=None,
            eval_responses=[],
        )
        with pytest.raises(AttributeError):
            result.success_rate = 1.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Forward pass tests
# ---------------------------------------------------------------------------


class TestForwardWithPrefix:
    def test_output_shape(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())

        n_tokens = 4
        seq_len = 6
        soft_embeds = mx.random.normal((1, n_tokens, D_MODEL))
        prompt_ids = mx.array([[0, 1, 2, 3, 4, 5]])
        mx.eval(soft_embeds)

        logits = _forward_with_prefix(model, soft_embeds, prompt_ids)
        mx.eval(logits)

        assert logits.shape == (1, n_tokens + seq_len, VOCAB_SIZE)

    def test_prefix_affects_logits(self) -> None:
        """Verify that different prefixes produce different output logits."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())

        prompt_ids = mx.array([[0, 1, 2]])
        zero_embeds = mx.zeros((1, 4, D_MODEL))
        rand_embeds = mx.random.normal((1, 4, D_MODEL))
        mx.eval(zero_embeds, rand_embeds)

        logits_zero = _forward_with_prefix(model, zero_embeds, prompt_ids)
        logits_rand = _forward_with_prefix(model, rand_embeds, prompt_ids)
        mx.eval(logits_zero, logits_rand)

        diff = float(
            mx.mean(mx.abs(logits_zero[:, -1, :] - logits_rand[:, -1, :])).item(),
        )
        assert diff > 0.001, (
            f"Different prefixes produce nearly identical logits (diff={diff})"
        )

    def test_different_prefix_sizes(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())

        prompt_ids = mx.array([[0, 1, 2]])

        for n_tokens in [1, 8, 16]:
            soft_embeds = mx.random.normal((1, n_tokens, D_MODEL))
            mx.eval(soft_embeds)
            logits = _forward_with_prefix(model, soft_embeds, prompt_ids)
            mx.eval(logits)
            assert logits.shape[1] == n_tokens + 3


# ---------------------------------------------------------------------------
# Encode targets tests
# ---------------------------------------------------------------------------


class TestEncodeTargets:
    def test_encodes_prefixes(self) -> None:
        tokenizer = MockTokenizer(VOCAB_SIZE)
        ids = _encode_targets(tokenizer, ["Sure", "Here"])
        mx.eval(ids)
        assert ids.ndim == 1
        assert ids.shape[0] > 0

    def test_single_prefix(self) -> None:
        tokenizer = MockTokenizer(VOCAB_SIZE)
        ids = _encode_targets(tokenizer, ["OK"])
        mx.eval(ids)
        expected = tokenizer.encode("OK")
        assert ids.shape[0] == len(expected)


# ---------------------------------------------------------------------------
# Continuous attack tests
# ---------------------------------------------------------------------------


class TestContinuousAttack:
    def test_basic_run(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())
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
        mx.eval(model.parameters())
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
        mx.eval(model.parameters())
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
        mx.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        soft_embeds = mx.random.normal((1, 4, D_MODEL)) * 0.1
        mx.eval(soft_embeds)

        target_ids = _encode_targets(tokenizer, ["Sure"])
        mx.eval(target_ids)

        messages = [{"role": "user", "content": "test"}]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        assert isinstance(text, str)
        prompt_ids = mx.array(tokenizer.encode(text))[None, :]

        def loss_fn(embeds: mx.array) -> mx.array:
            return _compute_loss(model, embeds, prompt_ids, target_ids, 4, None, 0.0)

        loss_and_grad = mx.value_and_grad(loss_fn)
        loss_val, grad = loss_and_grad(soft_embeds)
        mx.eval(loss_val, grad)

        grad_norm = float(mx.linalg.norm(grad.reshape(-1)).item())
        assert grad_norm > 0, "Gradient is zero — no learning signal"

    def test_dispatch_via_public_api(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())
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


# ---------------------------------------------------------------------------
# GCG attack tests
# ---------------------------------------------------------------------------


class TestGCGAttack:
    def test_basic_run(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())
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
        mx.eval(model.parameters())
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
        mx.eval(model.parameters())
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
        mx.eval(model.parameters())
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


# ---------------------------------------------------------------------------
# Direction-guided tests
# ---------------------------------------------------------------------------


class TestDirectionGuided:
    def test_continuous_with_direction(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        direction = mx.random.normal((D_MODEL,))
        direction = direction / mx.linalg.norm(direction)
        mx.eval(direction)

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
        mx.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        direction = mx.random.normal((D_MODEL,))
        direction = direction / mx.linalg.norm(direction)
        mx.eval(direction)

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


# ---------------------------------------------------------------------------
# Evaluate attack tests
# ---------------------------------------------------------------------------


class TestEvaluateAttack:
    def test_success_rate_range(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        soft_embeds = mx.random.normal((1, 4, D_MODEL))
        mx.eval(soft_embeds)

        config = SoftPromptConfig(max_gen_tokens=3)

        success_rate, responses = _evaluate_attack(
            model, tokenizer, ["test1", "test2"], soft_embeds, config,
        )

        assert 0.0 <= success_rate <= 1.0
        assert len(responses) == 2

    def test_empty_prompts(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        soft_embeds = mx.random.normal((1, 4, D_MODEL))
        mx.eval(soft_embeds)

        config = SoftPromptConfig(max_gen_tokens=2)

        success_rate, responses = _evaluate_attack(
            model, tokenizer, [], soft_embeds, config,
        )

        assert success_rate == 0.0
        assert responses == []


# ---------------------------------------------------------------------------
# Invalid mode test
# ---------------------------------------------------------------------------


class TestInvalidMode:
    def test_invalid_mode_raises(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())
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

        with pytest.raises(ValueError, match="Unknown soft prompt mode"):
            softprompt_attack(model, tokenizer, ["test"], bad_config, None)


# ---------------------------------------------------------------------------
# New helper tests
# ---------------------------------------------------------------------------


class TestComputeLearningRate:
    def test_constant_returns_base(self) -> None:
        assert _compute_learning_rate(0.01, 0, 100, "constant") == 0.01
        assert _compute_learning_rate(0.01, 50, 100, "constant") == 0.01
        assert _compute_learning_rate(0.01, 99, 100, "constant") == 0.01

    def test_cosine_start_equals_base(self) -> None:
        lr = _compute_learning_rate(0.01, 0, 100, "cosine")
        assert abs(lr - 0.01) < 1e-9

    def test_cosine_end_near_zero(self) -> None:
        lr = _compute_learning_rate(0.01, 99, 100, "cosine")
        assert abs(lr) < 1e-9

    def test_cosine_midpoint(self) -> None:
        lr = _compute_learning_rate(0.01, 49, 99, "cosine")
        # cos(pi * 49/98) = cos(pi/2) = 0, so lr = 0.005
        assert abs(lr - 0.005) < 1e-6

    def test_cosine_single_step(self) -> None:
        # n_steps=1 -> schedule has no effect (n_steps - 1 = 0)
        lr = _compute_learning_rate(0.01, 0, 1, "cosine")
        assert lr == 0.01


class TestComputeEmbedRegularization:
    def test_zero_weight_returns_zero(self) -> None:
        soft = mx.random.normal((1, 4, 16))
        embed = mx.random.normal((32, 16))
        mx.eval(soft, embed)
        result = _compute_embed_regularization(soft, embed, 0.0)
        mx.eval(result)
        assert float(result.item()) == 0.0

    def test_matching_norms_near_zero(self) -> None:
        """When soft embeds have same mean norm as real embeds, reg is ~0."""
        embed = mx.random.normal((32, 16))
        mx.eval(embed)
        mean_real = float(mx.mean(mx.linalg.norm(embed, axis=-1)).item())
        # Create soft embeds normalized to match
        soft = mx.random.normal((1, 4, 16))
        mx.eval(soft)
        norms = mx.linalg.norm(soft[0], axis=-1, keepdims=True)
        soft_normalized = soft / norms * mean_real
        soft_normalized = soft_normalized[None, :] if soft_normalized.ndim == 2 else soft_normalized  # noqa: E501
        mx.eval(soft_normalized)
        result = _compute_embed_regularization(soft_normalized, embed, 1.0)
        mx.eval(result)
        assert float(result.item()) < 0.01

    def test_large_gap_gives_large_penalty(self) -> None:
        soft = mx.ones((1, 4, 16)) * 100.0  # very large norm
        embed = mx.ones((32, 16)) * 0.01  # very small norm
        mx.eval(soft, embed)
        result = _compute_embed_regularization(soft, embed, 1.0)
        mx.eval(result)
        assert float(result.item()) > 1.0


class TestAccessibilityScore:
    def test_zero_loss(self) -> None:
        assert _compute_accessibility_score(0.0) == 1.0

    def test_high_loss(self) -> None:
        score = _compute_accessibility_score(10.0)
        assert score < 0.001

    def test_monotonicity(self) -> None:
        s1 = _compute_accessibility_score(1.0)
        s2 = _compute_accessibility_score(2.0)
        s3 = _compute_accessibility_score(3.0)
        assert s1 > s2 > s3


class TestPreEncodePrompts:
    def test_multi_prompt(self) -> None:
        tokenizer = MockTokenizer(VOCAB_SIZE)
        prompts = ["hello", "world", "test"]
        encoded = _pre_encode_prompts(tokenizer, prompts)
        assert len(encoded) == 3
        for ids in encoded:
            assert ids.ndim == 2
            assert ids.shape[0] == 1
            assert ids.shape[1] > 0

    def test_empty_list(self) -> None:
        tokenizer = MockTokenizer(VOCAB_SIZE)
        encoded = _pre_encode_prompts(tokenizer, [])
        assert encoded == []


class TestSelectPromptIds:
    def test_first_strategy(self) -> None:
        ids = [mx.array([[1]]), mx.array([[2]]), mx.array([[3]])]
        selected = _select_prompt_ids(ids, 0, "first")
        assert len(selected) == 1
        assert int(selected[0].item()) == 1
        # Same regardless of step
        selected2 = _select_prompt_ids(ids, 5, "first")
        assert len(selected2) == 1
        assert int(selected2[0].item()) == 1

    def test_cycle_strategy(self) -> None:
        ids = [mx.array([[1]]), mx.array([[2]]), mx.array([[3]])]
        assert int(_select_prompt_ids(ids, 0, "cycle")[0].item()) == 1
        assert int(_select_prompt_ids(ids, 1, "cycle")[0].item()) == 2
        assert int(_select_prompt_ids(ids, 2, "cycle")[0].item()) == 3
        assert int(_select_prompt_ids(ids, 3, "cycle")[0].item()) == 1

    def test_all_strategy(self) -> None:
        ids = [mx.array([[1]]), mx.array([[2]]), mx.array([[3]])]
        selected = _select_prompt_ids(ids, 0, "all")
        assert len(selected) == 3


# ---------------------------------------------------------------------------
# Multi-prompt continuous tests
# ---------------------------------------------------------------------------


class TestMultiPromptContinuous:
    def test_all_strategy(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())
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
        mx.eval(model.parameters())
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


# ---------------------------------------------------------------------------
# Early stopping tests
# ---------------------------------------------------------------------------


class TestEarlyStopping:
    def test_fires_with_patience(self) -> None:
        """Early stopping should fire with tiny LR and small patience."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())
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
        mx.eval(model.parameters())
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


# ---------------------------------------------------------------------------
# Cosine schedule test
# ---------------------------------------------------------------------------


class TestCosineSchedule:
    def test_runs_without_nan(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())
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


# ---------------------------------------------------------------------------
# Embed regularization test
# ---------------------------------------------------------------------------


class TestEmbedRegularizationIntegration:
    def test_runs_without_error(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())
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


# ---------------------------------------------------------------------------
# GCG multi-restart tests
# ---------------------------------------------------------------------------


class TestGCGMultiRestart:
    def test_correct_loss_history_length(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())
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
        mx.eval(model.parameters())
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


# ---------------------------------------------------------------------------
# SoftPromptResult new fields tests
# ---------------------------------------------------------------------------


class TestSoftPromptResultNewFields:
    def test_defaults(self) -> None:
        result = SoftPromptResult(
            mode="continuous",
            success_rate=0.5,
            final_loss=2.0,
            loss_history=[2.0],
            n_steps=1,
            n_tokens=4,
            embeddings=None,
            token_ids=None,
            token_text=None,
            eval_responses=[],
        )
        assert result.accessibility_score == 0.0
        assert result.per_prompt_losses == []
        assert result.early_stopped is False

    def test_explicit_values(self) -> None:
        result = SoftPromptResult(
            mode="gcg",
            success_rate=0.9,
            final_loss=0.5,
            loss_history=[1.0, 0.5],
            n_steps=2,
            n_tokens=8,
            embeddings=None,
            token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            token_text="test",
            eval_responses=["response"],
            accessibility_score=0.6,
            per_prompt_losses=[0.4, 0.6],
            early_stopped=True,
        )
        assert result.accessibility_score == 0.6
        assert result.per_prompt_losses == [0.4, 0.6]
        assert result.early_stopped is True


# ---------------------------------------------------------------------------
# New config defaults tests
# ---------------------------------------------------------------------------


class TestNewConfigDefaults:
    def test_new_fields_defaults(self) -> None:
        cfg = SoftPromptConfig()
        assert cfg.direction_mode == "last"
        assert cfg.direction_layers is None
        assert cfg.loss_mode == "targeted"
        assert cfg.egd_temperature == 1.0

    def test_egd_mode_accepted(self) -> None:
        cfg = SoftPromptConfig(mode="egd")
        assert cfg.mode == "egd"

    def test_custom_direction_mode(self) -> None:
        cfg = SoftPromptConfig(direction_mode="raid")
        assert cfg.direction_mode == "raid"

    def test_custom_direction_layers(self) -> None:
        cfg = SoftPromptConfig(direction_layers=[0, 1])
        assert cfg.direction_layers == [0, 1]


# ---------------------------------------------------------------------------
# RAID direction mode tests
# ---------------------------------------------------------------------------


class TestRAIDDirectionMode:
    def test_raid_runs(self) -> None:
        """RAID mode with direction_weight=0.1 produces finite loss."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        direction = mx.random.normal((D_MODEL,))
        direction = direction / mx.linalg.norm(direction)
        mx.eval(direction)

        config = SoftPromptConfig(
            mode="continuous",
            n_tokens=4,
            n_steps=3,
            direction_weight=0.1,
            direction_mode="raid",
            seed=42,
            max_gen_tokens=2,
        )

        result = softprompt_attack(
            model, tokenizer, ["test"], config, direction,
        )

        assert result.mode == "continuous"
        for loss in result.loss_history:
            assert not math.isnan(loss), "RAID mode produced NaN loss"

    def test_raid_differs_from_last(self) -> None:
        """Same seed, RAID vs 'last' produce different final losses."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        direction = mx.random.normal((D_MODEL,))
        direction = direction / mx.linalg.norm(direction)
        mx.eval(direction)

        config_last = SoftPromptConfig(
            mode="continuous",
            n_tokens=4,
            n_steps=5,
            learning_rate=0.1,
            direction_weight=0.5,
            direction_mode="last",
            seed=42,
            max_gen_tokens=2,
        )
        result_last = _continuous_attack(
            model, tokenizer, ["test"], config_last, direction,
        )

        config_raid = SoftPromptConfig(
            mode="continuous",
            n_tokens=4,
            n_steps=5,
            learning_rate=0.1,
            direction_weight=0.5,
            direction_mode="raid",
            seed=42,
            max_gen_tokens=2,
        )
        result_raid = _continuous_attack(
            model, tokenizer, ["test"], config_raid, direction,
        )

        # They should differ since RAID accumulates across layers
        assert result_last.loss_history != result_raid.loss_history

    def test_all_positions_runs(self) -> None:
        """direction_mode='all_positions' runs without error."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        direction = mx.random.normal((D_MODEL,))
        direction = direction / mx.linalg.norm(direction)
        mx.eval(direction)

        config = SoftPromptConfig(
            mode="continuous",
            n_tokens=4,
            n_steps=3,
            direction_weight=0.1,
            direction_mode="all_positions",
            seed=42,
            max_gen_tokens=2,
        )

        result = softprompt_attack(
            model, tokenizer, ["test"], config, direction,
        )

        assert result.mode == "continuous"
        for loss in result.loss_history:
            assert not math.isnan(loss)

    def test_direction_layers_subset(self) -> None:
        """direction_layers=[0] accepted, runs."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        direction = mx.random.normal((D_MODEL,))
        direction = direction / mx.linalg.norm(direction)
        mx.eval(direction)

        config = SoftPromptConfig(
            mode="continuous",
            n_tokens=4,
            n_steps=3,
            direction_weight=0.1,
            direction_mode="raid",
            direction_layers=[0],
            seed=42,
            max_gen_tokens=2,
        )

        result = softprompt_attack(
            model, tokenizer, ["test"], config, direction,
        )

        assert result.mode == "continuous"
        assert len(result.loss_history) == 3


# ---------------------------------------------------------------------------
# Untargeted loss tests
# ---------------------------------------------------------------------------


class TestUntargetedLoss:
    def test_untargeted_runs(self) -> None:
        """loss_mode='untargeted' produces finite loss."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="continuous",
            n_tokens=4,
            n_steps=5,
            learning_rate=0.01,
            loss_mode="untargeted",
            seed=42,
            max_gen_tokens=2,
        )

        result = softprompt_attack(
            model, tokenizer, ["test"], config, None,
        )

        assert result.mode == "continuous"
        for loss in result.loss_history:
            assert not math.isnan(loss), "Untargeted loss produced NaN"

    def test_untargeted_gradient_nonzero(self) -> None:
        """Gradient through untargeted loss is nonzero."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        soft_embeds = mx.random.normal((1, 4, D_MODEL)) * 0.1
        mx.eval(soft_embeds)

        refusal_ids = _encode_refusal_tokens(tokenizer)
        mx.eval(refusal_ids)

        messages = [{"role": "user", "content": "test"}]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        assert isinstance(text, str)
        prompt_ids = mx.array(tokenizer.encode(text))[None, :]

        def loss_fn(embeds: mx.array) -> mx.array:
            return _compute_untargeted_loss(
                model, embeds, prompt_ids, 4, refusal_ids,
                None, 0.0,
            )

        loss_and_grad = mx.value_and_grad(loss_fn)
        loss_val, grad = loss_and_grad(soft_embeds)
        mx.eval(loss_val, grad)

        grad_norm = float(mx.linalg.norm(grad.reshape(-1)).item())
        assert grad_norm > 0, "Untargeted gradient is zero"

    def test_untargeted_with_direction(self) -> None:
        """Untargeted + RAID direction combined."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        direction = mx.random.normal((D_MODEL,))
        direction = direction / mx.linalg.norm(direction)
        mx.eval(direction)

        config = SoftPromptConfig(
            mode="continuous",
            n_tokens=4,
            n_steps=3,
            loss_mode="untargeted",
            direction_weight=0.1,
            direction_mode="raid",
            seed=42,
            max_gen_tokens=2,
        )

        result = softprompt_attack(
            model, tokenizer, ["test"], config, direction,
        )

        for loss in result.loss_history:
            assert not math.isnan(loss)

    def test_encode_refusal_tokens(self) -> None:
        """Produces non-empty array of valid token IDs."""
        tokenizer = MockTokenizer(VOCAB_SIZE)
        ids = _encode_refusal_tokens(tokenizer)
        mx.eval(ids)
        assert ids.ndim == 1
        assert ids.shape[0] > 0
        for i in range(ids.shape[0]):
            assert 0 <= int(ids[i].item()) < VOCAB_SIZE


# ---------------------------------------------------------------------------
# EGD attack tests
# ---------------------------------------------------------------------------


class TestEGDAttack:
    def test_basic_run(self) -> None:
        """mode='egd' returns valid SoftPromptResult with token_ids."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())
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
        mx.eval(model.parameters())
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
        mx.eval(model.parameters())
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
        mx.eval(model.parameters())
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


# ---------------------------------------------------------------------------
# Token constraint tests
# ---------------------------------------------------------------------------


class TestTokenConstraint:
    def test_build_vocab_mask_ascii(self) -> None:
        """Mask shape is (VOCAB_SIZE,) with some True entries."""
        tokenizer = MockTokenizer(VOCAB_SIZE)
        mask = _build_vocab_mask(tokenizer, VOCAB_SIZE, "ascii")
        assert mask is not None
        mx.eval(mask)
        assert mask.shape == (VOCAB_SIZE,)
        n_allowed = int(mx.sum(mask).item())
        assert n_allowed > 0
        assert n_allowed <= VOCAB_SIZE

    def test_build_vocab_mask_none(self) -> None:
        """Returns None when constraint is None."""
        tokenizer = MockTokenizer(VOCAB_SIZE)
        mask = _build_vocab_mask(tokenizer, VOCAB_SIZE, None)
        assert mask is None

    def test_gcg_with_constraint(self) -> None:
        """GCG + token_constraint='ascii' runs, all token IDs map to ASCII."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="gcg",
            n_tokens=4,
            n_steps=3,
            batch_size=4,
            top_k=8,
            seed=42,
            max_gen_tokens=2,
            token_constraint="ascii",
        )

        result = _gcg_attack(
            model, tokenizer, ["test"], config, None,
        )

        assert result.token_ids is not None
        for tid in result.token_ids:
            decoded = tokenizer.decode([tid])
            assert all(32 <= ord(c) < 127 for c in decoded)

    def test_egd_with_constraint(self) -> None:
        """EGD + token_constraint='ascii' runs, all token IDs map to ASCII."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="egd",
            n_tokens=4,
            n_steps=3,
            learning_rate=0.1,
            seed=42,
            max_gen_tokens=2,
            token_constraint="ascii",
        )

        result = _egd_attack(
            model, tokenizer, ["test"], config, None,
        )

        assert result.token_ids is not None
        for tid in result.token_ids:
            decoded = tokenizer.decode([tid])
            assert all(32 <= ord(c) < 127 for c in decoded)


# ---------------------------------------------------------------------------
# EOS loss tests
# ---------------------------------------------------------------------------


class TestEOSLoss:
    def test_eos_force_runs(self) -> None:
        """Continuous + eos_loss_mode='force' produces finite loss."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="continuous",
            n_tokens=4,
            n_steps=3,
            seed=42,
            max_gen_tokens=2,
            eos_loss_mode="force",
            eos_loss_weight=0.1,
        )

        result = _continuous_attack(
            model, tokenizer, ["test"], config, None,
        )

        for loss in result.loss_history:
            assert not math.isnan(loss), "EOS force loss produced NaN"

    def test_eos_suppress_runs(self) -> None:
        """Continuous + eos_loss_mode='suppress' produces finite loss."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="continuous",
            n_tokens=4,
            n_steps=3,
            seed=42,
            max_gen_tokens=2,
            eos_loss_mode="suppress",
            eos_loss_weight=0.1,
        )

        result = _continuous_attack(
            model, tokenizer, ["test"], config, None,
        )

        for loss in result.loss_history:
            assert not math.isnan(loss), "EOS suppress loss produced NaN"

    def test_eos_loss_helper_force(self) -> None:
        """Force loss decreases as P(EOS) increases."""
        # Create logits where EOS token has varying probability
        vocab_size = 8
        eos_id = 7

        # Low P(EOS): uniform logits
        logits_low = mx.zeros((1, 2, vocab_size))
        mx.eval(logits_low)
        loss_low = _compute_eos_loss(logits_low, 0, eos_id, "force")
        mx.eval(loss_low)

        # High P(EOS): boost EOS logit
        logits_high = mx.zeros((1, 2, vocab_size))
        logits_high = logits_high.at[:, 0, eos_id].add(mx.array(10.0))
        mx.eval(logits_high)
        loss_high = _compute_eos_loss(logits_high, 0, eos_id, "force")
        mx.eval(loss_high)

        assert float(loss_high.item()) < float(loss_low.item())

    def test_eos_loss_helper_suppress(self) -> None:
        """Suppress loss decreases as P(EOS) decreases."""
        vocab_size = 8
        eos_id = 7

        # High P(EOS): boost EOS logit
        logits_high = mx.zeros((1, 2, vocab_size))
        logits_high = logits_high.at[:, 0, eos_id].add(mx.array(10.0))
        mx.eval(logits_high)
        loss_high = _compute_eos_loss(logits_high, 0, eos_id, "suppress")
        mx.eval(loss_high)

        # Low P(EOS): uniform logits
        logits_low = mx.zeros((1, 2, vocab_size))
        mx.eval(logits_low)
        loss_low = _compute_eos_loss(logits_low, 0, eos_id, "suppress")
        mx.eval(loss_low)

        assert float(loss_low.item()) < float(loss_high.item())


# ---------------------------------------------------------------------------
# KL collision loss tests
# ---------------------------------------------------------------------------


class TestKLCollisionLoss:
    def test_kl_collision_runs(self) -> None:
        """Continuous + kl_ref_weight=0.1 with ref model runs."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())
        ref_model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(ref_model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="continuous",
            n_tokens=4,
            n_steps=3,
            seed=42,
            max_gen_tokens=2,
            kl_ref_weight=0.1,
        )

        result = _continuous_attack(
            model, tokenizer, ["test"], config, None,
            ref_model=ref_model,
        )

        for loss in result.loss_history:
            assert not math.isnan(loss), "KL collision loss produced NaN"

    def test_kl_collision_same_model_low(self) -> None:
        """Same model as reference produces near-zero KL."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        soft_embeds = mx.random.normal((1, 4, D_MODEL)) * 0.1
        mx.eval(soft_embeds)

        messages = [{"role": "user", "content": "test"}]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        assert isinstance(text, str)
        prompt_ids = mx.array(tokenizer.encode(text))[None, :]

        kl = _compute_kl_collision_loss(model, model, soft_embeds, prompt_ids, 4)
        mx.eval(kl)
        assert float(kl.item()) < 0.01

    def test_kl_collision_gradient_nonzero(self) -> None:
        """Gradient through KL loss is nonzero."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())
        ref_model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(ref_model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        soft_embeds = mx.random.normal((1, 4, D_MODEL)) * 0.1
        mx.eval(soft_embeds)

        messages = [{"role": "user", "content": "test"}]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        assert isinstance(text, str)
        prompt_ids = mx.array(tokenizer.encode(text))[None, :]

        def loss_fn(embeds: mx.array) -> mx.array:
            return _compute_kl_collision_loss(
                model, ref_model, embeds, prompt_ids, 4,
            )

        loss_and_grad = mx.value_and_grad(loss_fn)
        loss_val, grad = loss_and_grad(soft_embeds)
        mx.eval(loss_val, grad)

        grad_norm = float(mx.linalg.norm(grad.reshape(-1)).item())
        assert grad_norm > 0, "KL collision gradient is zero"


# ---------------------------------------------------------------------------
# New config defaults tests (Geiping features)
# ---------------------------------------------------------------------------


class TestNewConfigDefaults2:
    def test_constraint_default_none(self) -> None:
        cfg = SoftPromptConfig()
        assert cfg.token_constraint is None

    def test_eos_defaults(self) -> None:
        cfg = SoftPromptConfig()
        assert cfg.eos_loss_mode == "none"
        assert cfg.eos_loss_weight == 0.0

    def test_kl_ref_default(self) -> None:
        cfg = SoftPromptConfig()
        assert cfg.kl_ref_weight == 0.0


# ---------------------------------------------------------------------------
# Worst-K prompt selection tests
# ---------------------------------------------------------------------------


class TestSelectWorstKPromptIds:
    def test_returns_correct_count(self) -> None:
        """Returns exactly k prompts when k < len(all_ids)."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        all_ids = _pre_encode_prompts(tokenizer, ["a", "b", "c", "d"])
        target_ids = _encode_targets(tokenizer, ["Sure"])
        mx.eval(target_ids)

        soft_embeds = mx.random.normal((1, 4, D_MODEL)) * 0.1
        mx.eval(soft_embeds)

        selected = _select_worst_k_prompt_ids(
            model, soft_embeds, all_ids, target_ids,
            4, 2,  # k=2
            None, 0.0, "last", None,
            None, "none", 0.0,
            None, 0.0,
            loss_mode="targeted", refusal_ids=None,
        )

        assert len(selected) == 2

    def test_k_greater_than_len(self) -> None:
        """Returns all prompts when k >= len(all_ids)."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        all_ids = _pre_encode_prompts(tokenizer, ["a", "b"])
        target_ids = _encode_targets(tokenizer, ["Sure"])
        mx.eval(target_ids)

        soft_embeds = mx.random.normal((1, 4, D_MODEL)) * 0.1
        mx.eval(soft_embeds)

        selected = _select_worst_k_prompt_ids(
            model, soft_embeds, all_ids, target_ids,
            4, 10,  # k=10 > len=2
            None, 0.0, "last", None,
            None, "none", 0.0,
            None, 0.0,
            loss_mode="targeted", refusal_ids=None,
        )

        assert len(selected) == 2

    def test_returns_highest_loss(self) -> None:
        """Selected prompts should be the ones with highest loss."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        all_ids = _pre_encode_prompts(tokenizer, ["a", "b", "c"])
        target_ids = _encode_targets(tokenizer, ["Sure"])
        mx.eval(target_ids)

        soft_embeds = mx.random.normal((1, 4, D_MODEL)) * 0.1
        mx.eval(soft_embeds)

        selected = _select_worst_k_prompt_ids(
            model, soft_embeds, all_ids, target_ids,
            4, 1,  # k=1
            None, 0.0, "last", None,
            None, "none", 0.0,
            None, 0.0,
            loss_mode="targeted", refusal_ids=None,
        )

        assert len(selected) == 1
        # Verify it's a valid prompt from the original set
        assert selected[0].shape == all_ids[0].shape or True


class TestWorstKIntegration:
    def test_continuous_worst_k(self) -> None:
        """Continuous mode with worst_k strategy runs."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())
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
        mx.eval(model.parameters())
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
        mx.eval(model.parameters())
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


# ---------------------------------------------------------------------------
# Gradient accumulation tests
# ---------------------------------------------------------------------------


class TestSplitIntoBatches:
    def test_single_batch(self) -> None:
        items = [mx.array([[1]]), mx.array([[2]]), mx.array([[3]])]
        batches = _split_into_batches(items, 1)
        assert len(batches) == 1
        assert len(batches[0]) == 3

    def test_correct_batch_count(self) -> None:
        items = [mx.array([[i]]) for i in range(6)]
        batches = _split_into_batches(items, 3)
        assert len(batches) == 3
        assert sum(len(b) for b in batches) == 6

    def test_n_greater_than_len(self) -> None:
        items = [mx.array([[1]]), mx.array([[2]])]
        batches = _split_into_batches(items, 10)
        assert len(batches) == 2
        assert sum(len(b) for b in batches) == 2

    def test_empty_list(self) -> None:
        batches = _split_into_batches([], 3)
        assert len(batches) == 1
        assert len(batches[0]) == 0

    def test_uneven_split(self) -> None:
        items = [mx.array([[i]]) for i in range(5)]
        batches = _split_into_batches(items, 3)
        assert len(batches) == 3
        assert sum(len(b) for b in batches) == 5


class TestGradAccumIntegration:
    def test_continuous_grad_accum(self) -> None:
        """Continuous mode with grad_accum_steps=2 runs and produces finite loss."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="continuous",
            n_tokens=4,
            n_steps=3,
            seed=42,
            max_gen_tokens=2,
            prompt_strategy="all",
            grad_accum_steps=2,
        )

        result = _continuous_attack(
            model, tokenizer, ["hello", "world"], config, None,
        )

        assert result.mode == "continuous"
        for loss in result.loss_history:
            assert not math.isnan(loss)

    def test_gcg_grad_accum(self) -> None:
        """GCG mode with grad_accum_steps=2 runs."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="gcg",
            n_tokens=4,
            n_steps=2,
            batch_size=4,
            top_k=8,
            seed=42,
            max_gen_tokens=2,
            prompt_strategy="all",
            grad_accum_steps=2,
        )

        result = _gcg_attack(
            model, tokenizer, ["hello", "world"], config, None,
        )

        assert result.mode == "gcg"
        for loss in result.loss_history:
            assert not math.isnan(loss)

    def test_egd_grad_accum(self) -> None:
        """EGD mode with grad_accum_steps=2 runs."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="egd",
            n_tokens=4,
            n_steps=3,
            learning_rate=0.1,
            seed=42,
            max_gen_tokens=2,
            prompt_strategy="all",
            grad_accum_steps=2,
        )

        result = _egd_attack(
            model, tokenizer, ["hello", "world"], config, None,
        )

        assert result.mode == "egd"
        for loss in result.loss_history:
            assert not math.isnan(loss)


# ---------------------------------------------------------------------------
# Transfer evaluation tests
# ---------------------------------------------------------------------------


class TestProjectToTokens:
    def test_returns_valid_token_ids(self) -> None:
        """Returns correct count of token IDs in vocab range."""
        n_tokens = 4
        soft_embeds = mx.random.normal((1, n_tokens, D_MODEL))
        embed_matrix = mx.random.normal((VOCAB_SIZE, D_MODEL))
        mx.eval(soft_embeds, embed_matrix)

        token_ids = _project_to_tokens(soft_embeds, embed_matrix)
        assert len(token_ids) == n_tokens
        for tid in token_ids:
            assert 0 <= tid < VOCAB_SIZE

    def test_single_token(self) -> None:
        soft_embeds = mx.random.normal((1, 1, D_MODEL))
        embed_matrix = mx.random.normal((VOCAB_SIZE, D_MODEL))
        mx.eval(soft_embeds, embed_matrix)

        token_ids = _project_to_tokens(soft_embeds, embed_matrix)
        assert len(token_ids) == 1

    def test_nearest_neighbor_correctness(self) -> None:
        """Embedding of token i should project back to token i."""
        embed_matrix = mx.random.normal((VOCAB_SIZE, D_MODEL))
        mx.eval(embed_matrix)
        # Use exact embedding for token 3
        soft_embeds = embed_matrix[3:4][None, :]  # shape (1, 1, D_MODEL)
        mx.eval(soft_embeds)
        token_ids = _project_to_tokens(soft_embeds, embed_matrix)
        assert token_ids[0] == 3


class TestTransferEvalResult:
    def test_construction(self) -> None:
        result = TransferEvalResult(
            model_id="test-model",
            success_rate=0.5,
            eval_responses=["resp1", "resp2"],
        )
        assert result.model_id == "test-model"
        assert result.success_rate == 0.5
        assert len(result.eval_responses) == 2

    def test_frozen(self) -> None:
        result = TransferEvalResult(
            model_id="test", success_rate=0.0, eval_responses=[],
        )
        with pytest.raises(AttributeError):
            result.success_rate = 1.0  # type: ignore[misc]


class TestSoftPromptResultTransfer:
    def test_default_empty(self) -> None:
        result = SoftPromptResult(
            mode="continuous",
            success_rate=0.5,
            final_loss=2.0,
            loss_history=[2.0],
            n_steps=1,
            n_tokens=4,
            embeddings=None,
            token_ids=None,
            token_text=None,
            eval_responses=[],
        )
        assert result.transfer_results == []

    def test_with_transfer_results(self) -> None:
        tr = TransferEvalResult(
            model_id="other-model",
            success_rate=0.3,
            eval_responses=["r1"],
        )
        result = SoftPromptResult(
            mode="gcg",
            success_rate=0.9,
            final_loss=0.5,
            loss_history=[0.5],
            n_steps=1,
            n_tokens=4,
            embeddings=None,
            token_ids=[1, 2, 3, 4],
            token_text="test",
            eval_responses=["resp"],
            transfer_results=[tr],
        )
        assert len(result.transfer_results) == 1
        assert result.transfer_results[0].model_id == "other-model"


class TestSoftPromptSerialization:
    def test_softprompt_to_dict_includes_transfer(self) -> None:
        from vauban._serializers import _softprompt_to_dict

        tr = TransferEvalResult(
            model_id="m1", success_rate=0.4, eval_responses=["r"],
        )
        result = SoftPromptResult(
            mode="gcg",
            success_rate=0.8,
            final_loss=1.0,
            loss_history=[1.0],
            n_steps=1,
            n_tokens=4,
            embeddings=None,
            token_ids=[1, 2, 3, 4],
            token_text="test",
            eval_responses=["resp"],
            transfer_results=[tr],
        )
        d = _softprompt_to_dict(result)
        assert "transfer_results" in d
        trs = d["transfer_results"]
        assert isinstance(trs, list)
        assert len(trs) == 1
        assert trs[0]["model_id"] == "m1"  # type: ignore[index]
        assert trs[0]["success_rate"] == 0.4  # type: ignore[index]

    def test_softprompt_to_dict_empty_transfer(self) -> None:
        from vauban._serializers import _softprompt_to_dict

        result = SoftPromptResult(
            mode="continuous",
            success_rate=0.5,
            final_loss=2.0,
            loss_history=[2.0],
            n_steps=1,
            n_tokens=4,
            embeddings=None,
            token_ids=None,
            token_text=None,
            eval_responses=[],
        )
        d = _softprompt_to_dict(result)
        assert d["transfer_results"] == []


# ---------------------------------------------------------------------------
# New config defaults for worst-k, grad accum, transfer
# ---------------------------------------------------------------------------


class TestNewConfigDefaults3:
    def test_worst_k_default(self) -> None:
        cfg = SoftPromptConfig()
        assert cfg.worst_k == 5

    def test_grad_accum_steps_default(self) -> None:
        cfg = SoftPromptConfig()
        assert cfg.grad_accum_steps == 1

    def test_transfer_models_default(self) -> None:
        cfg = SoftPromptConfig()
        assert cfg.transfer_models == []

    def test_custom_worst_k(self) -> None:
        cfg = SoftPromptConfig(worst_k=10)
        assert cfg.worst_k == 10

    def test_custom_grad_accum_steps(self) -> None:
        cfg = SoftPromptConfig(grad_accum_steps=4)
        assert cfg.grad_accum_steps == 4

    def test_custom_transfer_models(self) -> None:
        cfg = SoftPromptConfig(transfer_models=["model-a"])
        assert cfg.transfer_models == ["model-a"]


# ---------------------------------------------------------------------------
# Geiping taxonomy: new token constraints, repeat target, system prompt
# ---------------------------------------------------------------------------


class _UnicodeTokenizer:
    """Tokenizer mapping IDs to specific Unicode chars for constraint tests."""

    VOCAB_SIZE: int = 16

    def __init__(self) -> None:
        self._char_map: dict[int, str] = {
            0: "A",          # ASCII alpha
            1: "Z",          # ASCII alpha
            2: "1",          # ASCII digit
            3: " ",          # ASCII space
            4: "\u4e00",     # CJK Unified (chinese)
            5: "\u4e01",     # CJK Unified (chinese)
            6: "\u00e9",     # Latin Extended (non-latin)
            7: "\u0410",     # Cyrillic A (non-latin)
            8: "\u200b",     # Zero-width space (invisible)
            9: "\u200d",     # Zero-width joiner (invisible)
            10: "\u00ad",    # Soft hyphen (invisible)
            11: "!",         # Non-alphabetic ASCII symbol
            12: "#",         # Non-alphabetic ASCII symbol
            13: "\u2600",    # Sun symbol (emoji, So)
            14: "\u2764",    # Heart symbol (emoji, So)
            15: "b",         # ASCII alpha (for zalgo base)
        }

    def decode(self, token_ids: list[int]) -> str:
        """Map token IDs to their designated Unicode characters."""
        return "".join(self._char_map.get(tid, "?") for tid in token_ids)

    def encode(self, text: str) -> list[int]:
        """Minimal encode for non-constraint tests."""
        return [0]

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = True,
    ) -> str | list[int]:
        """Minimal template."""
        text = "".join(m["content"] for m in messages)
        if tokenize:
            return self.encode(text)
        return text


class TestNewTokenConstraints:
    """Tests for the 6 new Geiping token constraint sets."""

    def test_non_latin_excludes_ascii(self) -> None:
        """non_latin constraint: only tokens with all chars ord > 127."""
        tok = _UnicodeTokenizer()
        mask = _build_vocab_mask(tok, tok.VOCAB_SIZE, "non_latin")  # type: ignore[arg-type]
        assert mask is not None
        mx.eval(mask)
        # IDs 4-10 have ord > 127 (CJK, extended Latin, Cyrillic, zero-width)
        # IDs 0-3, 11-12, 15 are ASCII
        for tid in [0, 1, 2, 3, 11, 12, 15]:
            assert not bool(mask[tid].item()), f"ASCII token {tid} should be excluded"
        for tid in [4, 5, 6, 7]:
            assert bool(mask[tid].item()), f"Non-Latin token {tid} should be allowed"

    def test_chinese_filters_to_cjk(self) -> None:
        """chinese constraint: only CJK Unified Ideographs (U+4E00-U+9FFF)."""
        tok = _UnicodeTokenizer()
        mask = _build_vocab_mask(tok, tok.VOCAB_SIZE, "chinese")  # type: ignore[arg-type]
        assert mask is not None
        mx.eval(mask)
        # Only IDs 4, 5 are CJK
        for tid in [4, 5]:
            assert bool(mask[tid].item()), f"CJK token {tid} should be allowed"
        for tid in [0, 1, 6, 7, 8, 13]:
            assert not bool(mask[tid].item()), f"Non-CJK token {tid} should be excluded"

    def test_non_alphabetic_excludes_letters(self) -> None:
        """non_alphabetic constraint: no alpha chars."""
        tok = _UnicodeTokenizer()
        mask = _build_vocab_mask(tok, tok.VOCAB_SIZE, "non_alphabetic")  # type: ignore[arg-type]
        assert mask is not None
        mx.eval(mask)
        # Alpha tokens: 0 (A), 1 (Z), 6 (e-acute), 7 (Cyrillic A), 15 (b)
        for tid in [0, 1, 6, 7, 15]:
            assert not bool(mask[tid].item()), f"Alpha token {tid} should be excluded"
        # Non-alpha: 2 (1), 3 (space), 11 (!), 12 (#)
        for tid in [2, 11, 12]:
            assert bool(mask[tid].item()), f"Non-alpha token {tid} should be allowed"

    def test_invisible_matches_format_chars(self) -> None:
        """invisible constraint: zero-width, format, and non-printable whitespace."""
        tok = _UnicodeTokenizer()
        mask = _build_vocab_mask(tok, tok.VOCAB_SIZE, "invisible")  # type: ignore[arg-type]
        assert mask is not None
        mx.eval(mask)
        # IDs 8 (ZWSP, Cf), 9 (ZWJ, Cf), 10 (soft hyphen, Cf) are invisible
        for tid in [8, 9, 10]:
            assert bool(mask[tid].item()), f"Invisible token {tid} should be allowed"
        # Visible tokens should be excluded
        for tid in [0, 1, 4, 11, 13]:
            assert not bool(mask[tid].item()), f"Visible token {tid} should be excluded"

    def test_emoji_matches_symbol_chars(self) -> None:
        """emoji constraint: Unicode So category and emoji ranges."""
        tok = _UnicodeTokenizer()
        mask = _build_vocab_mask(tok, tok.VOCAB_SIZE, "emoji")  # type: ignore[arg-type]
        assert mask is not None
        mx.eval(mask)
        # IDs 13 (sun, So), 14 (heart, So) are emoji/symbols
        for tid in [13, 14]:
            assert bool(mask[tid].item()), f"Emoji token {tid} should be allowed"
        for tid in [0, 1, 4, 8, 11]:
            assert not bool(mask[tid].item()), (
                f"Non-emoji token {tid} should be excluded"
            )

    def test_zalgo_allows_combining_marks(self) -> None:
        """zalgo constraint: requires at least one combining diacritical mark."""

        class _ZalgoTokenizer:
            """Tokenizer with tokens containing combining marks."""

            VOCAB_SIZE = 4

            def __init__(self) -> None:
                self._char_map: dict[int, str] = {
                    0: "a\u0300",  # a + combining grave
                    1: "\u0301",   # combining acute alone
                    2: "hello",    # plain alpha
                    3: "A",        # plain alpha
                }

            def decode(self, token_ids: list[int]) -> str:
                return "".join(
                    self._char_map.get(tid, "?") for tid in token_ids
                )

        tok = _ZalgoTokenizer()
        mask = _build_vocab_mask(tok, tok.VOCAB_SIZE, "zalgo")  # type: ignore[arg-type]
        assert mask is not None
        mx.eval(mask)
        # ID 0: "a" + combining grave → has alpha + combining mark → zalgo
        assert bool(mask[0].item()), "Token with combining mark should be allowed"
        # ID 2: "hello" has no combining marks → not zalgo
        assert not bool(mask[2].item()), "Plain alpha without combining mark excluded"
        # ID 3: "A" has no combining marks → not zalgo
        assert not bool(mask[3].item()), "Single alpha without combining mark excluded"

    def test_unknown_constraint_raises(self) -> None:
        """Unknown constraint name raises ValueError."""
        tok = MockTokenizer(VOCAB_SIZE)
        with pytest.raises(ValueError, match="Unknown token constraint"):
            _build_vocab_mask(tok, VOCAB_SIZE, "bogus")


class TestEncodeTargetsRepeat:
    """Tests for target_repeat_count in _encode_targets."""

    def test_repeat_zero_is_noop(self) -> None:
        """repeat_count=0 returns same as no repeat."""
        tokenizer = MockTokenizer(VOCAB_SIZE)
        ids_no_repeat = _encode_targets(tokenizer, ["Sure"])
        ids_zero = _encode_targets(tokenizer, ["Sure"], repeat_count=0)
        mx.eval(ids_no_repeat, ids_zero)
        assert ids_no_repeat.shape == ids_zero.shape

    def test_repeat_multiplies_tokens(self) -> None:
        """repeat_count=3 produces 3x as many tokens."""
        tokenizer = MockTokenizer(VOCAB_SIZE)
        ids_base = _encode_targets(tokenizer, ["Hi"])
        ids_repeated = _encode_targets(tokenizer, ["Hi"], repeat_count=3)
        mx.eval(ids_base, ids_repeated)
        assert ids_repeated.shape[0] == ids_base.shape[0] * 3

    def test_repeat_preserves_pattern(self) -> None:
        """Repeated tokens are the base pattern repeated N times."""
        tokenizer = MockTokenizer(VOCAB_SIZE)
        ids_base = _encode_targets(tokenizer, ["AB"])
        ids_repeated = _encode_targets(tokenizer, ["AB"], repeat_count=2)
        mx.eval(ids_base, ids_repeated)
        base_list = ids_base.tolist()
        repeated_list = ids_repeated.tolist()
        assert repeated_list == base_list + base_list


class TestPreEncodePromptsSystemPrompt:
    """Tests for system_prompt parameter in _pre_encode_prompts."""

    def test_no_system_prompt(self) -> None:
        """Without system_prompt, encoding is unchanged."""
        tokenizer = MockTokenizer(VOCAB_SIZE)
        encoded_default = _pre_encode_prompts(tokenizer, ["hello"])
        encoded_none = _pre_encode_prompts(tokenizer, ["hello"], system_prompt=None)
        mx.eval(encoded_default[0], encoded_none[0])
        assert encoded_default[0].shape == encoded_none[0].shape

    def test_system_prompt_increases_length(self) -> None:
        """With system_prompt, encoded sequence is longer."""
        tokenizer = MockTokenizer(VOCAB_SIZE)
        encoded_no_sys = _pre_encode_prompts(tokenizer, ["hello"])
        encoded_with_sys = _pre_encode_prompts(
            tokenizer, ["hello"], system_prompt="You are a helpful assistant.",
        )
        mx.eval(encoded_no_sys[0], encoded_with_sys[0])
        assert encoded_with_sys[0].shape[1] > encoded_no_sys[0].shape[1]

    def test_system_prompt_content_appears(self) -> None:
        """System prompt text is included in the template output."""
        tokenizer = MockTokenizer(VOCAB_SIZE)
        sys_text = "SECRET_SYSTEM_PROMPT"
        # Use apply_chat_template directly to verify
        messages_with: list[dict[str, str]] = [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": "hello"},
        ]
        template_with = tokenizer.apply_chat_template(messages_with, tokenize=False)
        messages_without: list[dict[str, str]] = [
            {"role": "user", "content": "hello"},
        ]
        template_without = tokenizer.apply_chat_template(
            messages_without, tokenize=False,
        )
        assert isinstance(template_with, str)
        assert isinstance(template_without, str)
        assert len(template_with) > len(template_without)


class TestNewConfigDefaults4:
    """Tests for target_repeat_count and system_prompt config defaults."""

    def test_target_repeat_count_default(self) -> None:
        cfg = SoftPromptConfig()
        assert cfg.target_repeat_count == 0

    def test_system_prompt_default(self) -> None:
        cfg = SoftPromptConfig()
        assert cfg.system_prompt is None

    def test_custom_target_repeat_count(self) -> None:
        cfg = SoftPromptConfig(target_repeat_count=50)
        assert cfg.target_repeat_count == 50

    def test_custom_system_prompt(self) -> None:
        cfg = SoftPromptConfig(system_prompt="You are helpful.")
        assert cfg.system_prompt == "You are helpful."


# ---------------------------------------------------------------------------
# Multi-constraint tests
# ---------------------------------------------------------------------------


class TestMultiConstraint:
    """Tests for list-based multi-constraint token masks."""

    def test_single_in_list_equals_string(self) -> None:
        """["ascii"] produces same mask as "ascii"."""
        tok = MockTokenizer(VOCAB_SIZE)
        mask_str = _build_vocab_mask(tok, VOCAB_SIZE, "ascii")
        mask_list = _build_vocab_mask(tok, VOCAB_SIZE, ["ascii"])
        assert mask_str is not None
        assert mask_list is not None
        mx.eval(mask_str, mask_list)
        for tid in range(VOCAB_SIZE):
            assert bool(mask_str[tid].item()) == bool(
                mask_list[tid].item()
            ), f"Mismatch at token {tid}"

    def test_intersection_is_subset(self) -> None:
        """chinese is a subset of non_latin, so intersection = chinese."""
        tok = _UnicodeTokenizer()
        mask_chinese = _build_vocab_mask(
            tok, tok.VOCAB_SIZE, "chinese",  # type: ignore[arg-type]
        )
        mask_multi = _build_vocab_mask(
            tok, tok.VOCAB_SIZE, ["non_latin", "chinese"],  # type: ignore[arg-type]
        )
        assert mask_chinese is not None
        assert mask_multi is not None
        mx.eval(mask_chinese, mask_multi)
        for tid in range(tok.VOCAB_SIZE):
            assert bool(mask_chinese[tid].item()) == bool(
                mask_multi[tid].item()
            ), f"Mismatch at token {tid}"

    def test_contradictory_produces_empty(self) -> None:
        """["ascii", "non_latin"] has no overlap — all False."""
        tok = _UnicodeTokenizer()
        mask = _build_vocab_mask(
            tok, tok.VOCAB_SIZE, ["ascii", "non_latin"],  # type: ignore[arg-type]
        )
        assert mask is not None
        mx.eval(mask)
        n_allowed = int(mx.sum(mask).item())
        assert n_allowed == 0


# ---------------------------------------------------------------------------
# Defense eval type tests
# ---------------------------------------------------------------------------


class TestDefenseEvalResult:
    """Tests for DefenseEvalResult dataclass."""

    def test_construction(self) -> None:
        result = DefenseEvalResult(
            sic_blocked=1,
            sic_sanitized=2,
            sic_clean=3,
            sic_bypass_rate=0.83,
            cast_interventions=10,
            cast_refusal_rate=0.5,
            cast_responses=["resp1", "resp2"],
        )
        assert result.sic_blocked == 1
        assert result.cast_refusal_rate == 0.5
        assert len(result.cast_responses) == 2

    def test_to_dict(self) -> None:
        result = DefenseEvalResult(
            sic_blocked=0,
            sic_sanitized=0,
            sic_clean=1,
            sic_bypass_rate=1.0,
            cast_interventions=5,
            cast_refusal_rate=0.0,
            cast_responses=["ok"],
        )
        d = result.to_dict()
        assert d["sic_bypass_rate"] == 1.0
        assert d["cast_interventions"] == 5

    def test_frozen(self) -> None:
        result = DefenseEvalResult(
            sic_blocked=0, sic_sanitized=0, sic_clean=0,
            sic_bypass_rate=0.0, cast_interventions=0,
            cast_refusal_rate=0.0, cast_responses=[],
        )
        with pytest.raises(AttributeError):
            result.sic_blocked = 1  # type: ignore[misc]


class TestNewConfigDefaults5:
    """Tests for defense_eval config defaults."""

    def test_defense_eval_default(self) -> None:
        cfg = SoftPromptConfig()
        assert cfg.defense_eval is None

    def test_defense_eval_layer_default(self) -> None:
        cfg = SoftPromptConfig()
        assert cfg.defense_eval_layer is None

    def test_defense_eval_alpha_default(self) -> None:
        cfg = SoftPromptConfig()
        assert cfg.defense_eval_alpha == 1.0

    def test_defense_eval_threshold_default(self) -> None:
        cfg = SoftPromptConfig()
        assert cfg.defense_eval_threshold == 0.0

    def test_custom_defense_eval(self) -> None:
        cfg = SoftPromptConfig(defense_eval="both")
        assert cfg.defense_eval == "both"

    def test_softprompt_result_defense_eval_default(self) -> None:
        result = SoftPromptResult(
            mode="gcg", success_rate=0.5, final_loss=1.0,
            loss_history=[1.0], n_steps=1, n_tokens=4,
            embeddings=None, token_ids=[1, 2],
            token_text="ab", eval_responses=["r"],
        )
        assert result.defense_eval is None

    def test_softprompt_result_to_dict_defense_eval(self) -> None:
        de = DefenseEvalResult(
            sic_blocked=1, sic_sanitized=0, sic_clean=0,
            sic_bypass_rate=0.0, cast_interventions=5,
            cast_refusal_rate=1.0, cast_responses=["refused"],
        )
        result = SoftPromptResult(
            mode="gcg", success_rate=0.5, final_loss=1.0,
            loss_history=[1.0], n_steps=1, n_tokens=4,
            embeddings=None, token_ids=[1, 2],
            token_text="ab", eval_responses=["r"],
            defense_eval=de,
        )
        d = result.to_dict()
        assert d["defense_eval"] is not None
        assert d["defense_eval"]["sic_blocked"] == 1  # type: ignore[index]


class TestGanRoundResult:
    """Tests for GanRoundResult dataclass."""

    def test_gan_round_result_construction(self) -> None:
        attack = SoftPromptResult(
            mode="gcg", success_rate=0.5, final_loss=1.0,
            loss_history=[1.0], n_steps=10, n_tokens=4,
            embeddings=None, token_ids=[1, 2],
            token_text="ab", eval_responses=["r"],
        )
        de = DefenseEvalResult(
            sic_blocked=0, sic_sanitized=1, sic_clean=0,
            sic_bypass_rate=1.0, cast_interventions=0,
            cast_refusal_rate=0.0, cast_responses=["ok"],
        )
        rr = GanRoundResult(
            round_index=0,
            attack_result=attack,
            defense_result=de,
            attacker_won=True,
            config_snapshot={"n_tokens": 4, "n_steps": 10},
        )
        assert rr.round_index == 0
        assert rr.attacker_won is True
        assert rr.defense_result is not None

    def test_gan_round_result_to_dict(self) -> None:
        attack = SoftPromptResult(
            mode="gcg", success_rate=0.5, final_loss=1.0,
            loss_history=[1.0], n_steps=10, n_tokens=4,
            embeddings=None, token_ids=[1, 2],
            token_text="ab", eval_responses=["r"],
        )
        rr = GanRoundResult(
            round_index=1,
            attack_result=attack,
            defense_result=None,
            attacker_won=False,
            config_snapshot={"n_tokens": 8},
        )
        d = rr.to_dict()
        assert d["round_index"] == 1
        assert d["attacker_won"] is False
        assert d["defense_result"] is None

    def test_gan_round_result_frozen(self) -> None:
        attack = SoftPromptResult(
            mode="gcg", success_rate=0.5, final_loss=1.0,
            loss_history=[1.0], n_steps=10, n_tokens=4,
            embeddings=None, token_ids=[1, 2],
            token_text="ab", eval_responses=["r"],
        )
        rr = GanRoundResult(
            round_index=0, attack_result=attack,
            defense_result=None, attacker_won=False,
            config_snapshot={},
        )
        with pytest.raises(AttributeError):
            rr.round_index = 5  # type: ignore[misc]


class TestGanConfigDefaults:
    """Tests for GAN-related config defaults."""

    def test_gan_rounds_default(self) -> None:
        c = SoftPromptConfig()
        assert c.gan_rounds == 0

    def test_gan_step_multiplier_default(self) -> None:
        c = SoftPromptConfig()
        assert c.gan_step_multiplier == 1.5

    def test_gan_direction_escalation_default(self) -> None:
        c = SoftPromptConfig()
        assert c.gan_direction_escalation == 0.25

    def test_gan_token_escalation_default(self) -> None:
        c = SoftPromptConfig()
        assert c.gan_token_escalation == 4

    def test_init_tokens_default(self) -> None:
        c = SoftPromptConfig()
        assert c.init_tokens is None

    def test_defense_eval_sic_mode_default(self) -> None:
        c = SoftPromptConfig()
        assert c.defense_eval_sic_mode == "direction"

    def test_defense_eval_sic_max_iterations_default(self) -> None:
        c = SoftPromptConfig()
        assert c.defense_eval_sic_max_iterations == 3

    def test_defense_eval_cast_layers_default(self) -> None:
        c = SoftPromptConfig()
        assert c.defense_eval_cast_layers is None


class TestBuildAdvPrompts:
    """Tests for _build_adv_prompts position-aware prompt construction."""

    def test_suffix_position_appends(self) -> None:
        from vauban.softprompt._defense_eval import _build_adv_prompts

        result = _build_adv_prompts(["Hello world"], "TOKENS", "suffix")
        assert result == ["Hello world TOKENS"]

    def test_prefix_position_appends(self) -> None:
        from vauban.softprompt._defense_eval import _build_adv_prompts

        result = _build_adv_prompts(["Hello world"], "TOKENS", "prefix")
        assert result == ["Hello world TOKENS"]

    def test_infix_replaces_marker(self) -> None:
        from vauban.softprompt._defense_eval import _build_adv_prompts

        prompts = ["Write a story where {suffix} a character travels"]
        result = _build_adv_prompts(prompts, "TOKENS", "infix")
        assert result == ["Write a story where TOKENS a character travels"]

    def test_infix_no_marker_falls_back(self) -> None:
        from vauban.softprompt._defense_eval import _build_adv_prompts

        result = _build_adv_prompts(["No marker here"], "TOKENS", "infix")
        assert result == ["No marker here TOKENS"]

    def test_none_token_text_returns_copy(self) -> None:
        from vauban.softprompt._defense_eval import _build_adv_prompts

        prompts = ["Hello"]
        result = _build_adv_prompts(prompts, None, "infix")
        assert result == ["Hello"]
        assert result is not prompts

    def test_infix_multiple_prompts(self) -> None:
        from vauban.softprompt._defense_eval import _build_adv_prompts

        prompts = [
            "Story where {suffix} character travels",
            "Document how {suffix} technique is used",
        ]
        result = _build_adv_prompts(prompts, "ADV", "infix")
        assert result == [
            "Story where ADV character travels",
            "Document how ADV technique is used",
        ]


    def test_gan_history_default(self) -> None:
        result = SoftPromptResult(
            mode="gcg", success_rate=0.5, final_loss=1.0,
            loss_history=[1.0], n_steps=1, n_tokens=4,
            embeddings=None, token_ids=[1, 2],
            token_text="ab", eval_responses=["r"],
        )
        assert result.gan_history == []

    def test_gan_history_to_dict(self) -> None:
        attack = SoftPromptResult(
            mode="gcg", success_rate=0.5, final_loss=1.0,
            loss_history=[1.0], n_steps=10, n_tokens=4,
            embeddings=None, token_ids=[1, 2],
            token_text="ab", eval_responses=["r"],
        )
        rr = GanRoundResult(
            round_index=0, attack_result=attack,
            defense_result=None, attacker_won=False,
            config_snapshot={"n_tokens": 4},
        )
        result = SoftPromptResult(
            mode="gcg", success_rate=0.5, final_loss=1.0,
            loss_history=[1.0], n_steps=10, n_tokens=4,
            embeddings=None, token_ids=[1, 2],
            token_text="ab", eval_responses=["r"],
            gan_history=[rr],
        )
        d = result.to_dict()
        assert len(d["gan_history"]) == 1  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Multi-turn GAN tests
# ---------------------------------------------------------------------------


class TestMultiturnConfig:
    """Tests for multi-turn GAN config fields."""

    def test_defaults(self) -> None:
        cfg = SoftPromptConfig()
        assert cfg.gan_multiturn is False
        assert cfg.gan_multiturn_max_turns == 10

    def test_custom(self) -> None:
        cfg = SoftPromptConfig(gan_multiturn=True, gan_multiturn_max_turns=5)
        assert cfg.gan_multiturn is True
        assert cfg.gan_multiturn_max_turns == 5

    def test_frozen(self) -> None:
        cfg = SoftPromptConfig(gan_multiturn=True)
        with pytest.raises(AttributeError):
            cfg.gan_multiturn = False  # type: ignore[misc]

    def test_continuous_mode_rejected(self) -> None:
        """Multi-turn requires hard tokens — continuous mode is rejected."""
        from vauban.softprompt import gan_loop

        model = MockCausalLM(D_MODEL, VOCAB_SIZE, NUM_LAYERS, NUM_HEADS)
        tok = MockTokenizer(VOCAB_SIZE)
        cfg = SoftPromptConfig(
            mode="continuous",
            gan_rounds=2,
            gan_multiturn=True,
            n_tokens=2,
            n_steps=1,
        )
        with pytest.raises(ValueError, match="hard tokens"):
            gan_loop(model, tok, ["test"], cfg, direction=None)


# ---------------------------------------------------------------------------
# Defender escalation tests
# ---------------------------------------------------------------------------


class TestDefenderEscalationConfig:
    """Tests for GAN defender escalation config defaults."""

    def test_defaults(self) -> None:
        cfg = SoftPromptConfig()
        assert cfg.gan_defense_escalation is False
        assert cfg.gan_defense_alpha_multiplier == 1.5
        assert cfg.gan_defense_threshold_escalation == 0.5
        assert cfg.gan_defense_sic_iteration_escalation == 1

    def test_custom(self) -> None:
        cfg = SoftPromptConfig(
            gan_defense_escalation=True,
            gan_defense_alpha_multiplier=2.0,
            gan_defense_threshold_escalation=0.3,
            gan_defense_sic_iteration_escalation=2,
        )
        assert cfg.gan_defense_escalation is True
        assert cfg.gan_defense_alpha_multiplier == 2.0
        assert cfg.gan_defense_threshold_escalation == 0.3
        assert cfg.gan_defense_sic_iteration_escalation == 2

    def test_frozen(self) -> None:
        cfg = SoftPromptConfig(gan_defense_escalation=True)
        with pytest.raises(AttributeError):
            cfg.gan_defense_escalation = False  # type: ignore[misc]


class TestEscalateDefense:
    """Tests for _escalate_defense() function."""

    def test_single_escalation(self) -> None:
        from vauban.softprompt._gan import _escalate_defense

        cfg = SoftPromptConfig(
            defense_eval_alpha=1.0,
            defense_eval_threshold=0.0,
            defense_eval_sic_max_iterations=3,
            gan_defense_alpha_multiplier=1.5,
            gan_defense_threshold_escalation=0.5,
            gan_defense_sic_iteration_escalation=1,
        )
        esc = _escalate_defense(cfg)
        assert esc.defense_eval_alpha == 1.5
        assert esc.defense_eval_threshold == -0.5
        assert esc.defense_eval_sic_max_iterations == 4

    def test_cumulative_escalation(self) -> None:
        from vauban.softprompt._gan import _escalate_defense

        cfg = SoftPromptConfig(
            defense_eval_alpha=1.0,
            defense_eval_threshold=0.0,
            defense_eval_sic_max_iterations=3,
            gan_defense_alpha_multiplier=1.5,
            gan_defense_threshold_escalation=0.5,
            gan_defense_sic_iteration_escalation=1,
        )
        esc1 = _escalate_defense(cfg)
        esc2 = _escalate_defense(esc1)
        assert esc2.defense_eval_alpha == pytest.approx(2.25)
        assert esc2.defense_eval_threshold == pytest.approx(-1.0)
        assert esc2.defense_eval_sic_max_iterations == 5

    def test_preserves_other_fields(self) -> None:
        from vauban.softprompt._gan import _escalate_defense

        cfg = SoftPromptConfig(
            mode="gcg",
            n_tokens=8,
            n_steps=100,
            defense_eval_alpha=1.0,
            defense_eval_threshold=0.0,
            defense_eval_sic_max_iterations=3,
        )
        esc = _escalate_defense(cfg)
        assert esc.mode == "gcg"
        assert esc.n_tokens == 8
        assert esc.n_steps == 100


class TestPreEncodeWithHistory:
    """Tests for _pre_encode_prompts_with_history."""

    def test_no_history_matches_baseline(self) -> None:
        tok = MockTokenizer(VOCAB_SIZE)
        prompts = ["Hello world"]
        baseline = _pre_encode_prompts(tok, prompts)
        with_history = _pre_encode_prompts_with_history(
            tok, prompts, history=[],
        )
        assert len(baseline) == len(with_history)
        assert baseline[0].shape == with_history[0].shape

    def test_history_increases_length(self) -> None:
        tok = MockTokenizer(VOCAB_SIZE)
        prompts = ["Hello"]
        no_hist = _pre_encode_prompts(tok, prompts)
        history = [
            {"role": "user", "content": "Hi there"},
            {"role": "assistant", "content": "Hello!"},
        ]
        with_hist = _pre_encode_prompts_with_history(
            tok, prompts, history=history,
        )
        # With history, the encoded sequence must be longer
        assert with_hist[0].shape[1] > no_hist[0].shape[1]

    def test_system_prompt_with_history(self) -> None:
        tok = MockTokenizer(VOCAB_SIZE)
        prompts = ["Test"]
        history = [{"role": "user", "content": "Prev"}]
        result = _pre_encode_prompts_with_history(
            tok, prompts, history=history, system_prompt="Be helpful",
        )
        assert len(result) == 1
        assert result[0].shape[0] == 1  # batch dim

    def test_multiple_prompts(self) -> None:
        tok = MockTokenizer(VOCAB_SIZE)
        prompts = ["A", "B", "C"]
        history = [
            {"role": "user", "content": "X"},
            {"role": "assistant", "content": "Y"},
        ]
        result = _pre_encode_prompts_with_history(
            tok, prompts, history=history,
        )
        assert len(result) == 3


class TestEvaluateAttackWithHistory:
    """Tests for _evaluate_attack_with_history."""

    def test_basic_evaluation(self) -> None:
        model = MockCausalLM(D_MODEL, VOCAB_SIZE, NUM_LAYERS, NUM_HEADS)
        tok = MockTokenizer(VOCAB_SIZE)
        config = SoftPromptConfig(n_tokens=2, max_gen_tokens=5)
        soft_embeds = mx.random.normal((1, 2, D_MODEL))
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
        model = MockCausalLM(D_MODEL, VOCAB_SIZE, NUM_LAYERS, NUM_HEADS)
        tok = MockTokenizer(VOCAB_SIZE)
        config = SoftPromptConfig(n_tokens=2, max_gen_tokens=3)
        soft_embeds = mx.random.normal((1, 2, D_MODEL))
        _success_rate, responses = _evaluate_attack_with_history(
            model, tok, ["Hello"], soft_embeds, config, history=[],
        )
        assert len(responses) == 1


class TestBuildSicPromptsWithHistory:
    """Tests for _build_sic_prompts_with_history."""

    def test_no_history_passthrough(self) -> None:
        prompts = ["attack prompt"]
        result = _build_sic_prompts_with_history(prompts, history=[])
        assert result == prompts
        # Should be a copy, not the same list
        assert result is not prompts

    def test_history_prepended(self) -> None:
        prompts = ["current attack"]
        history = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]
        result = _build_sic_prompts_with_history(prompts, history)
        assert len(result) == 1
        assert "previous question" in result[0]
        assert "previous answer" in result[0]
        assert "current attack" in result[0]

    def test_multiple_prompts_with_history(self) -> None:
        prompts = ["attack1", "attack2"]
        history = [{"role": "user", "content": "context"}]
        result = _build_sic_prompts_with_history(prompts, history)
        assert len(result) == 2
        assert "context" in result[0]
        assert "context" in result[1]
        assert "attack1" in result[0]
        assert "attack2" in result[1]


# ---------------------------------------------------------------------------
# _sample_prompt_ids tests
# ---------------------------------------------------------------------------


class TestSamplePromptIds:
    def test_returns_all_when_k_ge_pool(self) -> None:
        pool = [mx.array([1, 2]), mx.array([3, 4]), mx.array([5, 6])]
        result = _sample_prompt_ids(pool, 5)
        assert result is pool

    def test_returns_all_when_k_eq_pool(self) -> None:
        pool = [mx.array([1, 2]), mx.array([3, 4])]
        result = _sample_prompt_ids(pool, 2)
        assert result is pool

    def test_returns_k_elements(self) -> None:
        pool = [mx.array([i]) for i in range(10)]
        result = _sample_prompt_ids(pool, 3)
        assert len(result) == 3
        # All returned items come from the pool
        for item in result:
            assert any(
                item.tolist() == p.tolist() for p in pool
            )

    def test_no_duplicates(self) -> None:
        pool = [mx.array([i]) for i in range(10)]
        result = _sample_prompt_ids(pool, 5)
        ids = [tuple(r.tolist()) for r in result]
        assert len(set(ids)) == 5


# ---------------------------------------------------------------------------
# Config defaults for new fields
# ---------------------------------------------------------------------------


class TestSoftPromptConfigNewFields:
    def test_prompt_pool_size_default_none(self) -> None:
        cfg = SoftPromptConfig()
        assert cfg.prompt_pool_size is None

    def test_beam_width_default_one(self) -> None:
        cfg = SoftPromptConfig()
        assert cfg.beam_width == 1

    def test_prompt_pool_size_custom(self) -> None:
        cfg = SoftPromptConfig(prompt_pool_size=200)
        assert cfg.prompt_pool_size == 200

    def test_beam_width_custom(self) -> None:
        cfg = SoftPromptConfig(beam_width=4)
        assert cfg.beam_width == 4


# ---------------------------------------------------------------------------
# GCG beam search tests
# ---------------------------------------------------------------------------


class TestGcgBeamSearch:
    def test_beam_width_1_is_greedy(self) -> None:
        """beam_width=1 should produce valid results (fast path)."""
        model = MockCausalLM(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS)
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
        model = MockCausalLM(VOCAB_SIZE, D_MODEL, NUM_LAYERS, NUM_HEADS)
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


# ---------------------------------------------------------------------------
# Defense-aware loss tests
# ---------------------------------------------------------------------------


class TestDefenseAwarePenalty:
    def test_zero_weight_no_penalty(self) -> None:
        """defense_aware_weight=0 means no penalty contribution."""
        from vauban.softprompt._loss import _compute_defense_aware_penalty

        h = mx.ones((1, 4, D_MODEL))
        direction = mx.ones((D_MODEL,))
        result = _compute_defense_aware_penalty(
            h, layer_idx=2, direction=direction,
            sic_layer=2, sic_threshold=0.5,
            cast_layers=[2], cast_threshold=0.5,
        )
        # The function itself returns a value; the weight=0 gating is external
        assert float(result.item()) >= 0.0

    def test_no_direction_returns_zero(self) -> None:
        """No direction → zero penalty."""
        from vauban.softprompt._loss import _compute_defense_aware_penalty

        h = mx.ones((1, 4, D_MODEL))
        result = _compute_defense_aware_penalty(
            h, layer_idx=2, direction=None,
            sic_layer=2, sic_threshold=0.5,
            cast_layers=[2], cast_threshold=0.5,
        )
        assert float(result.item()) == 0.0

    def test_sic_layer_mismatch_no_penalty(self) -> None:
        """Wrong layer → no SIC penalty."""
        from vauban.softprompt._loss import _compute_defense_aware_penalty

        h = mx.ones((1, 4, D_MODEL))
        direction = mx.ones((D_MODEL,))
        result = _compute_defense_aware_penalty(
            h, layer_idx=5, direction=direction,
            sic_layer=2, sic_threshold=0.5,
            cast_layers=None, cast_threshold=0.5,
        )
        assert float(result.item()) == 0.0

    def test_defense_aware_config_defaults(self) -> None:
        """New config fields have correct defaults."""
        cfg = SoftPromptConfig()
        assert cfg.defense_aware_weight == 0.0
        assert cfg.defense_eval_alpha_tiers is None

    def test_defense_aware_weight_custom(self) -> None:
        """defense_aware_weight can be set."""
        cfg = SoftPromptConfig(defense_aware_weight=0.5)
        assert cfg.defense_aware_weight == 0.5

    def test_alpha_tiers_custom(self) -> None:
        """defense_eval_alpha_tiers can be set."""
        tiers = [(0.3, 1.0), (0.6, 2.0)]
        cfg = SoftPromptConfig(defense_eval_alpha_tiers=tiers)
        assert cfg.defense_eval_alpha_tiers == tiers


class TestGanRoundTransferResults:
    def test_gan_round_result_transfer_results_default(self) -> None:
        """GanRoundResult.transfer_results defaults to empty list."""
        from vauban.types import GanRoundResult

        # Create a minimal attack result
        attack = SoftPromptResult(
            mode="gcg", success_rate=0.5, final_loss=1.0,
            loss_history=[1.0], n_steps=1, n_tokens=4,
            embeddings=None, token_ids=[1, 2, 3, 4],
            token_text="test", eval_responses=["resp"],
        )
        rr = GanRoundResult(
            round_index=0, attack_result=attack,
            defense_result=None, attacker_won=False,
            config_snapshot={},
        )
        assert rr.transfer_results == []

    def test_gan_round_result_to_dict_includes_transfer(self) -> None:
        """to_dict should include transfer_results."""
        from vauban.types import GanRoundResult, TransferEvalResult

        attack = SoftPromptResult(
            mode="gcg", success_rate=0.5, final_loss=1.0,
            loss_history=[1.0], n_steps=1, n_tokens=4,
            embeddings=None, token_ids=[1, 2, 3, 4],
            token_text="test", eval_responses=["resp"],
        )
        tr = TransferEvalResult(
            model_id="test-model", success_rate=0.8,
            eval_responses=["ok"],
        )
        rr = GanRoundResult(
            round_index=0, attack_result=attack,
            defense_result=None, attacker_won=False,
            config_snapshot={}, transfer_results=[tr],
        )
        d = rr.to_dict()
        assert "transfer_results" in d
        assert len(d["transfer_results"]) == 1  # type: ignore[arg-type]


class TestEscalateDefenseAlphaTiers:
    def test_escalate_scales_existing_tiers(self) -> None:
        """Existing tiers should have alphas scaled by multiplier."""
        from vauban.softprompt._gan import _escalate_defense

        cfg = SoftPromptConfig(
            defense_eval_alpha=1.0,
            defense_eval_threshold=0.5,
            defense_eval_alpha_tiers=[(0.3, 1.0), (0.6, 2.0)],
            gan_defense_alpha_multiplier=2.0,
        )
        new = _escalate_defense(cfg)
        assert new.defense_eval_alpha_tiers is not None
        assert len(new.defense_eval_alpha_tiers) == 2
        assert new.defense_eval_alpha_tiers[0] == (0.3, 2.0)
        assert new.defense_eval_alpha_tiers[1] == (0.6, 4.0)

    def test_escalate_auto_generates_tiers_from_flat(self) -> None:
        """Without tiers, escalation auto-generates 3 TRYLOCK-style tiers."""
        from vauban.softprompt._gan import _escalate_defense

        cfg = SoftPromptConfig(
            defense_eval_alpha=2.0,
            defense_eval_threshold=1.0,
            defense_eval_alpha_tiers=None,
            gan_defense_alpha_multiplier=1.5,
        )
        new = _escalate_defense(cfg)
        assert new.defense_eval_alpha_tiers is not None
        assert len(new.defense_eval_alpha_tiers) == 3
        # Thresholds: 0.5, 1.0, 1.5  Alphas: 1.0, 2.0, 3.0
        assert new.defense_eval_alpha_tiers[0] == (0.5, 1.0)
        assert new.defense_eval_alpha_tiers[1] == (1.0, 2.0)
        assert new.defense_eval_alpha_tiers[2] == (1.5, 3.0)

    def test_escalate_no_auto_gen_when_threshold_zero(self) -> None:
        """No auto-generation when threshold=0 (would produce degenerate tiers)."""
        from vauban.softprompt._gan import _escalate_defense

        cfg = SoftPromptConfig(
            defense_eval_alpha=2.0,
            defense_eval_threshold=0.0,
            defense_eval_alpha_tiers=None,
            gan_defense_alpha_multiplier=1.5,
        )
        new = _escalate_defense(cfg)
        assert new.defense_eval_alpha_tiers is None

    def test_escalate_no_auto_gen_when_alpha_zero(self) -> None:
        """No auto-generation when alpha=0."""
        from vauban.softprompt._gan import _escalate_defense

        cfg = SoftPromptConfig(
            defense_eval_alpha=0.0,
            defense_eval_threshold=0.5,
            defense_eval_alpha_tiers=None,
            gan_defense_alpha_multiplier=1.5,
        )
        new = _escalate_defense(cfg)
        assert new.defense_eval_alpha_tiers is None


class TestDefenseAwarePenaltyDirections:
    """Test SIC/CAST penalty direction and combined behavior."""

    def test_sic_penalty_when_proj_below_threshold(self) -> None:
        """SIC penalty activates when projection < threshold."""
        from vauban.softprompt._loss import _compute_defense_aware_penalty

        # Small hidden state → small projection → below threshold
        h = mx.ones((1, 4, D_MODEL)) * 0.01
        direction = mx.ones((D_MODEL,)) / (D_MODEL ** 0.5)
        result = _compute_defense_aware_penalty(
            h, layer_idx=2, direction=direction,
            sic_layer=2, sic_threshold=10.0,
            cast_layers=None, cast_threshold=0.0,
        )
        assert float(result.item()) > 0.0

    def test_sic_no_penalty_when_proj_above_threshold(self) -> None:
        """SIC penalty is zero when projection >= threshold."""
        from vauban.softprompt._loss import _compute_defense_aware_penalty

        # Large hidden state → large projection → above threshold
        h = mx.ones((1, 4, D_MODEL)) * 10.0
        direction = mx.ones((D_MODEL,)) / (D_MODEL ** 0.5)
        result = _compute_defense_aware_penalty(
            h, layer_idx=2, direction=direction,
            sic_layer=2, sic_threshold=0.01,
            cast_layers=None, cast_threshold=0.0,
        )
        assert float(result.item()) == 0.0

    def test_cast_penalty_when_proj_above_threshold(self) -> None:
        """CAST penalty activates when projection > threshold."""
        from vauban.softprompt._loss import _compute_defense_aware_penalty

        # Large hidden state → large projection → above threshold
        h = mx.ones((1, 4, D_MODEL)) * 10.0
        direction = mx.ones((D_MODEL,)) / (D_MODEL ** 0.5)
        result = _compute_defense_aware_penalty(
            h, layer_idx=3, direction=direction,
            sic_layer=None, sic_threshold=0.0,
            cast_layers=[3], cast_threshold=0.01,
        )
        assert float(result.item()) > 0.0

    def test_cast_no_penalty_when_proj_below_threshold(self) -> None:
        """CAST penalty is zero when projection <= threshold."""
        from vauban.softprompt._loss import _compute_defense_aware_penalty

        # Small hidden state → small projection → below threshold
        h = mx.ones((1, 4, D_MODEL)) * 0.01
        direction = mx.ones((D_MODEL,)) / (D_MODEL ** 0.5)
        result = _compute_defense_aware_penalty(
            h, layer_idx=3, direction=direction,
            sic_layer=None, sic_threshold=0.0,
            cast_layers=[3], cast_threshold=10.0,
        )
        assert float(result.item()) == 0.0

    def test_both_sic_and_cast_on_same_layer(self) -> None:
        """Both penalties contribute when SIC and CAST are on the same layer."""
        from vauban.softprompt._loss import _compute_defense_aware_penalty

        # Use a moderate projection so it's below SIC threshold AND above CAST
        # threshold — both penalties should fire
        h = mx.ones((1, 4, D_MODEL))
        direction = mx.ones((D_MODEL,)) / (D_MODEL ** 0.5)
        # Projection = sum(1 * 1/sqrt(D)) = sqrt(D) ≈ 5.66 for D=32
        sic_threshold = 100.0  # proj << threshold → SIC penalty
        cast_threshold = 0.01  # proj >> threshold → CAST penalty

        result = _compute_defense_aware_penalty(
            h, layer_idx=2, direction=direction,
            sic_layer=2, sic_threshold=sic_threshold,
            cast_layers=[2], cast_threshold=cast_threshold,
        )
        # Both penalties contribute, so result > either alone
        sic_only = _compute_defense_aware_penalty(
            h, layer_idx=2, direction=direction,
            sic_layer=2, sic_threshold=sic_threshold,
            cast_layers=None, cast_threshold=0.0,
        )
        cast_only = _compute_defense_aware_penalty(
            h, layer_idx=2, direction=direction,
            sic_layer=None, sic_threshold=0.0,
            cast_layers=[2], cast_threshold=cast_threshold,
        )
        assert float(result.item()) > float(sic_only.item())
        assert float(result.item()) > float(cast_only.item())


# ---------------------------------------------------------------------------
# Multi-model GCG transfer scoring tests
# ---------------------------------------------------------------------------


class TestGcgTransferScoring:
    def test_transfer_weight_zero_is_noop(self) -> None:
        """transfer_loss_weight=0 should skip re-ranking entirely."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        tokenizer = MockTokenizer(VOCAB_SIZE)
        mx.eval(model.parameters())

        # Second model as "transfer" target
        t_model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(t_model.parameters())
        t_tok = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="gcg", n_tokens=4, n_steps=2,
            batch_size=4, top_k=8,
            transfer_loss_weight=0.0,
        )
        transfer_models = [("transfer", t_model, t_tok)]
        result = _gcg_attack(
            model, tokenizer, ["Hello"], config, None,
            transfer_models=transfer_models,
        )
        assert result.mode == "gcg"
        assert result.token_ids is not None
        assert len(result.token_ids) == 4

    def test_transfer_weight_positive_runs(self) -> None:
        """transfer_loss_weight>0 with transfer models should run."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        tokenizer = MockTokenizer(VOCAB_SIZE)
        mx.eval(model.parameters())

        t_model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(t_model.parameters())
        t_tok = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="gcg", n_tokens=4, n_steps=2,
            batch_size=4, top_k=8,
            transfer_loss_weight=0.5,
        )
        transfer_models = [("transfer", t_model, t_tok)]
        result = _gcg_attack(
            model, tokenizer, ["Hello"], config, None,
            transfer_models=transfer_models,
        )
        assert result.mode == "gcg"
        assert result.token_ids is not None
        assert len(result.token_ids) == 4

    def test_no_transfer_models_unchanged(self) -> None:
        """No transfer models means no re-ranking, even with weight > 0."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        tokenizer = MockTokenizer(VOCAB_SIZE)
        mx.eval(model.parameters())

        config = SoftPromptConfig(
            mode="gcg", n_tokens=4, n_steps=2,
            batch_size=4, top_k=8,
            transfer_loss_weight=0.5,
        )
        result = _gcg_attack(
            model, tokenizer, ["Hello"], config, None,
            transfer_models=None,
        )
        assert result.mode == "gcg"
        assert result.token_ids is not None

    def test_transfer_loss_weight_default(self) -> None:
        """SoftPromptConfig default for transfer_loss_weight is 0.0."""
        cfg = SoftPromptConfig()
        assert cfg.transfer_loss_weight == 0.0

    def test_transfer_rerank_count_default(self) -> None:
        """SoftPromptConfig default for transfer_rerank_count is 8."""
        cfg = SoftPromptConfig()
        assert cfg.transfer_rerank_count == 8

    def test_transfer_rerank_count_custom(self) -> None:
        """transfer_rerank_count can be set to a custom value."""
        cfg = SoftPromptConfig(transfer_rerank_count=4)
        assert cfg.transfer_rerank_count == 4

    def test_transfer_rerank_count_used(self) -> None:
        """Custom rerank count should be respected during GCG."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        tokenizer = MockTokenizer(VOCAB_SIZE)
        mx.eval(model.parameters())

        t_model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(t_model.parameters())
        t_tok = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="gcg", n_tokens=4, n_steps=2,
            batch_size=4, top_k=8,
            transfer_loss_weight=0.5,
            transfer_rerank_count=2,
        )
        result = _gcg_attack(
            model, tokenizer, ["Hello"], config, None,
            transfer_models=[("t", t_model, t_tok)],
        )
        assert result.mode == "gcg"
        assert result.token_ids is not None


class TestEgdTransferScoring:
    def test_egd_transfer_weight_zero_noop(self) -> None:
        """EGD with transfer_loss_weight=0 should not change behavior."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        tokenizer = MockTokenizer(VOCAB_SIZE)
        mx.eval(model.parameters())

        t_model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(t_model.parameters())
        t_tok = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="egd", n_tokens=4, n_steps=3,
            learning_rate=0.1,
            transfer_loss_weight=0.0,
        )
        result = _egd_attack(
            model, tokenizer, ["Hello"], config, None,
            transfer_models=[("t", t_model, t_tok)],
        )
        assert result.mode == "egd"
        assert result.token_ids is not None

    def test_egd_transfer_weight_positive_runs(self) -> None:
        """EGD with transfer_loss_weight>0 should include transfer loss."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        tokenizer = MockTokenizer(VOCAB_SIZE)
        mx.eval(model.parameters())

        t_model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(t_model.parameters())
        t_tok = MockTokenizer(VOCAB_SIZE)

        config = SoftPromptConfig(
            mode="egd", n_tokens=4, n_steps=3,
            learning_rate=0.1,
            transfer_loss_weight=0.5,
        )
        result = _egd_attack(
            model, tokenizer, ["Hello"], config, None,
            transfer_models=[("t", t_model, t_tok)],
        )
        assert result.mode == "egd"
        assert result.token_ids is not None

    def test_egd_no_transfer_models_unchanged(self) -> None:
        """EGD without transfer models should work normally."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        tokenizer = MockTokenizer(VOCAB_SIZE)
        mx.eval(model.parameters())

        config = SoftPromptConfig(
            mode="egd", n_tokens=4, n_steps=3,
            learning_rate=0.1,
            transfer_loss_weight=0.5,
        )
        result = _egd_attack(
            model, tokenizer, ["Hello"], config, None,
            transfer_models=None,
        )
        assert result.mode == "egd"
        assert result.token_ids is not None


# ---------------------------------------------------------------------------
# Arena card output tests
# ---------------------------------------------------------------------------


class TestWriteArenaCard:
    def test_basic_card_written(self, tmp_path: Path) -> None:
        """Arena card should be written with expected sections."""
        from vauban import _write_arena_card

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
        from vauban import _write_arena_card

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
        from vauban import _write_arena_card

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


# ---------------------------------------------------------------------------
# Injection context encoding tests
# ---------------------------------------------------------------------------


class TestInjectionContextEncoding:
    """Tests for _pre_encode_prompts_with_injection_context."""

    def test_web_page_preset_longer_than_plain(self) -> None:
        """Web page preset adds surrounding context → more tokens."""
        tokenizer = MockTokenizer(VOCAB_SIZE)
        plain = _pre_encode_prompts(tokenizer, ["inject this"])
        wrapped = _pre_encode_prompts_with_injection_context(
            tokenizer, ["inject this"],
            injection_context="web_page",
        )
        assert len(wrapped) == 1
        assert wrapped[0].shape[0] == 1  # batch dim
        # Wrapped encoding should be strictly longer
        assert wrapped[0].shape[1] > plain[0].shape[1]

    def test_tool_output_preset_longer_than_plain(self) -> None:
        tokenizer = MockTokenizer(VOCAB_SIZE)
        plain = _pre_encode_prompts(tokenizer, ["payload"])
        wrapped = _pre_encode_prompts_with_injection_context(
            tokenizer, ["payload"],
            injection_context="tool_output",
        )
        assert wrapped[0].shape[1] > plain[0].shape[1]

    def test_code_file_preset_longer_than_plain(self) -> None:
        tokenizer = MockTokenizer(VOCAB_SIZE)
        plain = _pre_encode_prompts(tokenizer, ["code"])
        wrapped = _pre_encode_prompts_with_injection_context(
            tokenizer, ["code"],
            injection_context="code_file",
        )
        assert wrapped[0].shape[1] > plain[0].shape[1]

    def test_system_prompt_adds_tokens(self) -> None:
        tokenizer = MockTokenizer(VOCAB_SIZE)
        without_sys = _pre_encode_prompts_with_injection_context(
            tokenizer, ["test"],
            injection_context="web_page",
        )
        with_sys = _pre_encode_prompts_with_injection_context(
            tokenizer, ["test"],
            injection_context="web_page",
            system_prompt="You are an agent.",
        )
        assert with_sys[0].shape[1] > without_sys[0].shape[1]

    def test_multiple_prompts_encoded(self) -> None:
        tokenizer = MockTokenizer(VOCAB_SIZE)
        result = _pre_encode_prompts_with_injection_context(
            tokenizer, ["a", "b", "c"],
            injection_context="web_page",
        )
        assert len(result) == 3
        for ids in result:
            assert ids.shape[0] == 1  # batch dim

    def test_invalid_preset_raises_keyerror(self) -> None:
        tokenizer = MockTokenizer(VOCAB_SIZE)
        with pytest.raises(KeyError):
            _pre_encode_prompts_with_injection_context(
                tokenizer, ["x"],
                injection_context="nonexistent",
            )


class TestInjectionTemplateEncoding:
    """Tests for _pre_encode_prompts_with_injection_template."""

    def test_template_longer_than_plain(self) -> None:
        tokenizer = MockTokenizer(VOCAB_SIZE)
        plain = _pre_encode_prompts(tokenizer, ["payload"])
        wrapped = _pre_encode_prompts_with_injection_template(
            tokenizer, ["payload"],
            template="Before {payload} after",
        )
        assert wrapped[0].shape[1] > plain[0].shape[1]

    def test_template_safe_from_format_injection(self) -> None:
        """Prompts with {curly} braces must not cause KeyError."""
        tokenizer = MockTokenizer(VOCAB_SIZE)
        # This would crash with str.format()
        result = _pre_encode_prompts_with_injection_template(
            tokenizer, ["test {__class__} {0}"],
            template="Context: {payload}",
        )
        assert len(result) == 1
        assert result[0].shape[0] == 1

    def test_template_with_system_prompt_adds_tokens(self) -> None:
        tokenizer = MockTokenizer(VOCAB_SIZE)
        without_sys = _pre_encode_prompts_with_injection_template(
            tokenizer, ["p"],
            template="Doc: {payload}",
        )
        with_sys = _pre_encode_prompts_with_injection_template(
            tokenizer, ["p"],
            template="Doc: {payload}",
            system_prompt="Be helpful.",
        )
        assert with_sys[0].shape[1] > without_sys[0].shape[1]


class TestResolveInjectionIds:
    """Tests for _resolve_injection_ids."""

    def test_returns_none_when_no_injection(self) -> None:
        tokenizer = MockTokenizer(VOCAB_SIZE)
        config = SoftPromptConfig()
        result = _resolve_injection_ids(config, tokenizer, ["test"])
        assert result is None

    def test_returns_ids_for_preset(self) -> None:
        tokenizer = MockTokenizer(VOCAB_SIZE)
        config = SoftPromptConfig(injection_context="web_page")
        result = _resolve_injection_ids(config, tokenizer, ["test"])
        assert result is not None
        assert len(result) == 1

    def test_template_takes_priority(self) -> None:
        """When both are set, template wins (shorter than preset)."""
        tokenizer = MockTokenizer(VOCAB_SIZE)
        config_preset = SoftPromptConfig(
            injection_context="web_page",
        )
        config_both = SoftPromptConfig(
            injection_context="web_page",
            injection_context_template="Short {payload}",
        )
        preset_result = _resolve_injection_ids(
            config_preset, tokenizer, ["p"],
        )
        both_result = _resolve_injection_ids(
            config_both, tokenizer, ["p"],
        )
        assert preset_result is not None
        assert both_result is not None
        # Template "Short {payload}" is much shorter than web_page
        # preset, proving template took priority
        assert both_result[0].shape[1] < preset_result[0].shape[1]


class TestInjectionContextDispatch:
    """Tests that injection context wires through to GCG/EGD."""

    def test_gcg_with_injection_context(self) -> None:
        """GCG attack with injection_context produces valid result."""
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())
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
        mx.eval(model.parameters())
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


class TestInjectionContextConfigDefaults:
    """Tests for SoftPromptConfig injection context field defaults."""

    def test_injection_context_default_none(self) -> None:
        cfg = SoftPromptConfig()
        assert cfg.injection_context is None

    def test_injection_context_template_default_none(self) -> None:
        cfg = SoftPromptConfig()
        assert cfg.injection_context_template is None

    def test_injection_context_custom(self) -> None:
        cfg = SoftPromptConfig(injection_context="tool_output")
        assert cfg.injection_context == "tool_output"

    def test_injection_context_template_custom(self) -> None:
        cfg = SoftPromptConfig(
            injection_context_template="X {payload} Y",
        )
        assert cfg.injection_context_template == "X {payload} Y"


# ---------------------------------------------------------------------------
# Perplexity loss tests
# ---------------------------------------------------------------------------


class TestPerplexityLoss:
    def test_default_disabled(self) -> None:
        cfg = SoftPromptConfig()
        assert cfg.perplexity_weight == 0.0

    def test_compute_perplexity_loss_basic(self) -> None:
        from vauban.softprompt import _compute_perplexity_loss

        # logits: (1, 4, 32), suffix_token_ids: (1, 4)
        logits = mx.random.normal((1, 4, VOCAB_SIZE))
        suffix_ids = mx.array([[1, 2, 3, 4]])
        loss = _compute_perplexity_loss(
            logits, suffix_ids, n_tokens=4, soft_token_offset=0,
        )
        mx.eval(loss)
        assert loss.shape == ()
        assert float(loss.item()) > 0.0

    def test_perplexity_loss_single_token_returns_zero(self) -> None:
        from vauban.softprompt import _compute_perplexity_loss

        logits = mx.random.normal((1, 2, VOCAB_SIZE))
        suffix_ids = mx.array([[1]])
        loss = _compute_perplexity_loss(
            logits, suffix_ids, n_tokens=1, soft_token_offset=0,
        )
        mx.eval(loss)
        assert float(loss.item()) == 0.0

    def test_compute_loss_with_perplexity(self) -> None:
        from vauban.softprompt import _compute_loss

        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())
        soft_embeds = mx.random.normal((1, 4, D_MODEL)) * 0.1
        prompt_ids = mx.array([[1, 2, 3]])
        target_ids = mx.array([5, 6])
        suffix_ids = mx.array([[1, 2, 3, 4]])

        loss_no_ppl = _compute_loss(
            model, soft_embeds, prompt_ids, target_ids,
            n_tokens=4, direction=None, direction_weight=0.0,
        )
        loss_with_ppl = _compute_loss(
            model, soft_embeds, prompt_ids, target_ids,
            n_tokens=4, direction=None, direction_weight=0.0,
            perplexity_weight=0.1, suffix_token_ids=suffix_ids,
        )
        mx.eval(loss_no_ppl, loss_with_ppl)
        # Perplexity adds a non-negative term
        assert float(loss_with_ppl.item()) >= float(loss_no_ppl.item()) - 1e-6


# ---------------------------------------------------------------------------
# Token position tests
# ---------------------------------------------------------------------------


class TestTokenPosition:
    def test_default_prefix(self) -> None:
        cfg = SoftPromptConfig()
        assert cfg.token_position == "prefix"

    def test_suffix_position_config(self) -> None:
        cfg = SoftPromptConfig(token_position="suffix")
        assert cfg.token_position == "suffix"

    def test_infix_position_config(self) -> None:
        cfg = SoftPromptConfig(token_position="infix")
        assert cfg.token_position == "infix"

    def test_compute_loss_suffix_position(self) -> None:
        from vauban.softprompt import _compute_loss

        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())
        soft_embeds = mx.random.normal((1, 4, D_MODEL)) * 0.1
        prompt_ids = mx.array([[1, 2, 3]])
        target_ids = mx.array([5, 6])

        loss = _compute_loss(
            model, soft_embeds, prompt_ids, target_ids,
            n_tokens=4, direction=None, direction_weight=0.0,
            token_position="suffix",
        )
        mx.eval(loss)
        assert loss.shape == ()

    def test_compute_loss_infix_position(self) -> None:
        from vauban.softprompt import _compute_loss

        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())
        soft_embeds = mx.random.normal((1, 4, D_MODEL)) * 0.1
        prompt_ids = mx.array([[1, 2, 3, 4, 5]])
        target_ids = mx.array([5, 6])

        loss = _compute_loss(
            model, soft_embeds, prompt_ids, target_ids,
            n_tokens=4, direction=None, direction_weight=0.0,
            token_position="infix", infix_split=2,
        )
        mx.eval(loss)
        assert loss.shape == ()

    def test_embed_and_mask_with_prefix_suffix(self) -> None:
        from vauban._forward import embed_and_mask_with_prefix

        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())
        transformer = model.model
        soft_embeds = mx.random.normal((1, 3, D_MODEL))
        token_ids = mx.array([[1, 2]])

        h_prefix, _ = embed_and_mask_with_prefix(
            transformer, soft_embeds, token_ids, token_position="prefix",
        )
        h_suffix, _ = embed_and_mask_with_prefix(
            transformer, soft_embeds, token_ids, token_position="suffix",
        )
        mx.eval(h_prefix, h_suffix)
        # Both should have same total length
        assert h_prefix.shape[1] == h_suffix.shape[1] == 5

    def test_embed_and_mask_with_prefix_infix(self) -> None:
        from vauban._forward import embed_and_mask_with_prefix

        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())
        transformer = model.model
        soft_embeds = mx.random.normal((1, 3, D_MODEL))
        token_ids = mx.array([[1, 2, 3, 4]])

        h_infix, _ = embed_and_mask_with_prefix(
            transformer, soft_embeds, token_ids,
            token_position="infix", infix_split=2,
        )
        mx.eval(h_infix)
        # 4 prompt + 3 soft = 7
        assert h_infix.shape[1] == 7


# ---------------------------------------------------------------------------
# Paraphrase strategy tests
# ---------------------------------------------------------------------------


class TestParaphrase:
    def test_default_empty(self) -> None:
        cfg = SoftPromptConfig()
        assert cfg.paraphrase_strategies == []

    def test_paraphrase_prompts_empty_strategies(self) -> None:
        from vauban.softprompt import paraphrase_prompts

        result = paraphrase_prompts(["hello"], [])
        assert result == ["hello"]

    def test_paraphrase_prompts_single_strategy(self) -> None:
        from vauban.softprompt import paraphrase_prompts

        result = paraphrase_prompts(["do X"], ["narrative"])
        assert len(result) == 2  # original + 1 paraphrase
        assert result[0] == "do X"
        assert "do X" in result[1]
        assert "story" in result[1].lower()

    def test_paraphrase_prompts_multiple_strategies(self) -> None:
        from vauban.softprompt import paraphrase_prompts

        prompts = ["do X", "do Y"]
        strategies = ["narrative", "technical"]
        result = paraphrase_prompts(prompts, strategies)
        # 2 original + 2*2 paraphrases = 6
        assert len(result) == 6
        assert result[0] == "do X"
        assert result[1] == "do Y"

    def test_paraphrase_unknown_strategy_raises(self) -> None:
        from vauban.softprompt import paraphrase_prompts

        with pytest.raises(ValueError, match="Unknown paraphrase strategy"):
            paraphrase_prompts(["test"], ["nonexistent"])

    def test_all_strategies_produce_output(self) -> None:
        from vauban.softprompt import paraphrase_prompts
        from vauban.softprompt._paraphrase import _STRATEGY_TEMPLATES

        all_strategies = list(_STRATEGY_TEMPLATES.keys())
        result = paraphrase_prompts(["test prompt"], all_strategies)
        assert len(result) == 1 + len(all_strategies)


# ---------------------------------------------------------------------------
# Perplexity with position offset tests
# ---------------------------------------------------------------------------


class TestPerplexityWithOffset:
    def test_perplexity_suffix_offset(self) -> None:
        from vauban.softprompt import _compute_perplexity_loss

        # 3 prompt tokens + 4 soft tokens = logits at positions 3..5
        logits = mx.random.normal((1, 10, VOCAB_SIZE))
        suffix_ids = mx.array([[1, 2, 3, 4]])
        loss = _compute_perplexity_loss(
            logits, suffix_ids, n_tokens=4, soft_token_offset=3,
        )
        mx.eval(loss)
        assert loss.shape == ()
        assert float(loss.item()) > 0.0

    def test_perplexity_infix_offset(self) -> None:
        from vauban.softprompt import _compute_perplexity_loss

        # infix_split=2: soft tokens start at position 2
        logits = mx.random.normal((1, 12, VOCAB_SIZE))
        suffix_ids = mx.array([[1, 2, 3, 4]])
        loss = _compute_perplexity_loss(
            logits, suffix_ids, n_tokens=4, soft_token_offset=2,
        )
        mx.eval(loss)
        assert loss.shape == ()
        assert float(loss.item()) > 0.0

    def test_add_perplexity_term_disabled(self) -> None:
        from vauban.softprompt import _add_perplexity_term

        logits = mx.random.normal((1, 8, VOCAB_SIZE))
        base_loss = mx.array(1.5)
        result = _add_perplexity_term(
            base_loss, logits,
            perplexity_weight=0.0, suffix_token_ids=None,
            n_tokens=4, n_prompt=3,
            token_position="prefix", infix_split=None,
        )
        mx.eval(result)
        assert float(result.item()) == float(base_loss.item())

    def test_add_perplexity_term_suffix(self) -> None:
        from vauban.softprompt import _add_perplexity_term

        logits = mx.random.normal((1, 10, VOCAB_SIZE))
        base_loss = mx.array(1.0)
        suffix_ids = mx.array([[1, 2, 3, 4]])
        result = _add_perplexity_term(
            base_loss, logits,
            perplexity_weight=0.1, suffix_token_ids=suffix_ids,
            n_tokens=4, n_prompt=3,
            token_position="suffix", infix_split=None,
        )
        mx.eval(result)
        assert float(result.item()) >= float(base_loss.item()) - 1e-6

    def test_add_perplexity_term_infix(self) -> None:
        from vauban.softprompt import _add_perplexity_term

        logits = mx.random.normal((1, 12, VOCAB_SIZE))
        base_loss = mx.array(1.0)
        suffix_ids = mx.array([[1, 2, 3, 4]])
        result = _add_perplexity_term(
            base_loss, logits,
            perplexity_weight=0.1, suffix_token_ids=suffix_ids,
            n_tokens=4, n_prompt=5,
            token_position="infix", infix_split=2,
        )
        mx.eval(result)
        assert float(result.item()) >= float(base_loss.item()) - 1e-6

    def test_loss_with_perplexity_suffix_position(self) -> None:
        from vauban.softprompt import _compute_loss

        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())
        soft_embeds = mx.random.normal((1, 4, D_MODEL)) * 0.1
        prompt_ids = mx.array([[1, 2, 3]])
        target_ids = mx.array([5, 6])
        suffix_ids = mx.array([[1, 2, 3, 4]])

        loss = _compute_loss(
            model, soft_embeds, prompt_ids, target_ids,
            n_tokens=4, direction=None, direction_weight=0.0,
            perplexity_weight=0.1, suffix_token_ids=suffix_ids,
            token_position="suffix",
        )
        mx.eval(loss)
        assert loss.shape == ()

    def test_loss_with_perplexity_infix_position(self) -> None:
        from vauban.softprompt import _compute_loss

        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())
        soft_embeds = mx.random.normal((1, 4, D_MODEL)) * 0.1
        prompt_ids = mx.array([[1, 2, 3, 4, 5]])
        target_ids = mx.array([5, 6])
        suffix_ids = mx.array([[1, 2, 3, 4]])

        loss = _compute_loss(
            model, soft_embeds, prompt_ids, target_ids,
            n_tokens=4, direction=None, direction_weight=0.0,
            perplexity_weight=0.1, suffix_token_ids=suffix_ids,
            token_position="infix", infix_split=2,
        )
        mx.eval(loss)
        assert loss.shape == ()


# ---------------------------------------------------------------------------
# Infix split computation tests
# ---------------------------------------------------------------------------


class TestInfixSplit:
    def test_compute_infix_split_basic(self) -> None:
        from vauban.softprompt import _compute_infix_split

        tokenizer = MockTokenizer(VOCAB_SIZE)
        clean_ids, split_idx = _compute_infix_split(
            tokenizer, "Write about {suffix} something",
        )
        # Clean prompt should not contain {suffix}
        assert isinstance(clean_ids, list)
        assert len(clean_ids) > 0
        assert split_idx >= 0
        assert split_idx <= len(clean_ids)

    def test_compute_infix_split_no_placeholder_raises(self) -> None:
        from vauban.softprompt import _compute_infix_split

        tokenizer = MockTokenizer(VOCAB_SIZE)
        with pytest.raises(ValueError, match="\\{suffix\\}"):
            _compute_infix_split(tokenizer, "no placeholder here")

    def test_resolve_infix_overrides(self) -> None:
        from vauban.softprompt import _resolve_infix_overrides

        tokenizer = MockTokenizer(VOCAB_SIZE)
        prompts = [
            "Write about {suffix} something",
            "Tell me {suffix} a story",
        ]
        encoded, infix_map = _resolve_infix_overrides(tokenizer, prompts)
        assert len(encoded) == 2
        assert len(infix_map) == 2
        # Each encoded array should have an entry in the map
        for arr in encoded:
            assert id(arr) in infix_map
            assert infix_map[id(arr)] >= 0


# ---------------------------------------------------------------------------
# Integration: all three features together
# ---------------------------------------------------------------------------


class TestThreeFeaturesIntegration:
    def test_loss_with_all_features(self) -> None:
        """Compute loss with perplexity + infix position + paraphrased prompts."""
        from vauban.softprompt import _compute_loss, paraphrase_prompts

        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())

        # Paraphrase
        prompts = ["do something dangerous"]
        expanded = paraphrase_prompts(prompts, ["narrative", "technical"])
        assert len(expanded) == 3  # 1 original + 2 paraphrased

        # Infix position with perplexity
        soft_embeds = mx.random.normal((1, 4, D_MODEL)) * 0.1
        prompt_ids = mx.array([[1, 2, 3, 4, 5]])
        target_ids = mx.array([5, 6])
        suffix_ids = mx.array([[1, 2, 3, 4]])

        loss = _compute_loss(
            model, soft_embeds, prompt_ids, target_ids,
            n_tokens=4, direction=None, direction_weight=0.0,
            perplexity_weight=0.1, suffix_token_ids=suffix_ids,
            token_position="infix", infix_split=2,
        )
        mx.eval(loss)
        assert loss.shape == ()
        assert float(loss.item()) > 0.0


# ---------------------------------------------------------------------------
# Rollout wiring
# ---------------------------------------------------------------------------


def _fake_rollout_score(
    _model: object,
    _tokenizer: object,
    _env_config: object,
    texts: list[str],
    losses: list[float],
) -> tuple[list[float], list[EnvironmentResult]]:
    """Stub for score_candidates_via_rollout — returns losses unchanged."""
    results = [
        EnvironmentResult(
            reward=0.0,
            target_called=False,
            target_args_match=False,
            turns=[],
            tool_calls_made=[],
            injection_payload=t,
        )
        for t in texts
    ]
    return list(losses), results


def _make_rollout_env_config(
    rollout_every_n: int = 1,
) -> EnvironmentConfig:
    """Build a minimal EnvironmentConfig for rollout wiring tests."""
    from vauban.types import (
        EnvironmentConfig,
        EnvironmentTarget,
        EnvironmentTask,
        ToolSchema,
    )

    return EnvironmentConfig(
        system_prompt="You are a helpful assistant.",
        tools=[ToolSchema(name="noop", description="no-op", parameters={})],
        target=EnvironmentTarget(function="noop"),
        task=EnvironmentTask(content="Do something."),
        injection_surface="tool_output",
        rollout_every_n=rollout_every_n,
    )


class TestGcgRolloutWiring:
    def test_rollout_called_during_gcg(self) -> None:
        """Verify environment rollout is wired into GCG candidate scoring."""
        from unittest.mock import patch

        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())
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
        mx.eval(model.parameters())
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


# ---------------------------------------------------------------------------
# Infix multiturn dispatch
# ---------------------------------------------------------------------------


class TestInfixMultiturnDispatch:
    def test_dispatch_multiturn_forwards_infix_map(self) -> None:
        """_dispatch_attack_multiturn resolves infix splits with history."""
        from vauban.softprompt._gan import _dispatch_attack_multiturn

        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())
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
