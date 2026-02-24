"""Tests for vauban.softprompt: types, forward pass, continuous, GCG, evaluation."""

from __future__ import annotations

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
    _continuous_attack,
    _encode_targets,
    _evaluate_attack,
    _forward_with_prefix,
    _gcg_attack,
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

        import math

        for loss in result.loss_history:
            assert not math.isnan(loss), "Loss is NaN"

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
        assert result.n_steps == 3
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
