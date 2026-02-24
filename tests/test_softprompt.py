"""Tests for vauban.softprompt: types, forward pass, continuous, GCG, evaluation."""

from __future__ import annotations

import math

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
    _compute_accessibility_score,
    _compute_embed_regularization,
    _compute_learning_rate,
    _compute_loss,
    _continuous_attack,
    _encode_targets,
    _evaluate_attack,
    _forward_with_prefix,
    _gcg_attack,
    _pre_encode_prompts,
    _select_prompt_ids,
    softprompt_attack,
)
from vauban.types import SoftPromptConfig, SoftPromptResult

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
