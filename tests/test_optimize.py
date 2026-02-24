"""Tests for vauban.optimize: types, config parsing, helpers, integration."""

from __future__ import annotations

from unittest.mock import patch

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

from vauban.optimize import _pick_balanced
from vauban.types import (
    DirectionResult,
    OptimizeConfig,
    OptimizeResult,
    TrialResult,
)

# ---------------------------------------------------------------------------
# Type tests
# ---------------------------------------------------------------------------


class TestOptimizeConfig:
    def test_defaults(self) -> None:
        cfg = OptimizeConfig()
        assert cfg.n_trials == 50
        assert cfg.alpha_min == 0.1
        assert cfg.alpha_max == 5.0
        assert cfg.sparsity_min == 0.0
        assert cfg.sparsity_max == 0.9
        assert cfg.search_norm_preserve is True
        assert cfg.search_strategies == ["all", "above_median", "top_k"]
        assert cfg.layer_top_k_min == 3
        assert cfg.layer_top_k_max is None
        assert cfg.max_tokens == 100
        assert cfg.seed is None
        assert cfg.timeout is None

    def test_frozen(self) -> None:
        cfg = OptimizeConfig()
        with pytest.raises(AttributeError):
            cfg.n_trials = 10  # type: ignore[misc]


class TestTrialResult:
    def test_construction(self) -> None:
        t = TrialResult(
            trial_number=0,
            alpha=1.0,
            sparsity=0.0,
            norm_preserve=False,
            layer_strategy="all",
            layer_top_k=None,
            target_layers=[0, 1],
            refusal_rate=0.5,
            perplexity_delta=0.1,
            kl_divergence=0.01,
        )
        assert t.trial_number == 0
        assert t.alpha == 1.0
        assert t.target_layers == [0, 1]

    def test_frozen(self) -> None:
        t = TrialResult(
            trial_number=0,
            alpha=1.0,
            sparsity=0.0,
            norm_preserve=False,
            layer_strategy="all",
            layer_top_k=None,
            target_layers=[0, 1],
            refusal_rate=0.5,
            perplexity_delta=0.1,
            kl_divergence=0.01,
        )
        with pytest.raises(AttributeError):
            t.alpha = 2.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _pick_balanced tests
# ---------------------------------------------------------------------------


class TestPickBalanced:
    def test_empty_returns_none(self) -> None:
        assert _pick_balanced([], 0.5, 5.0) is None

    def test_single_trial_returns_it(self) -> None:
        t = TrialResult(
            trial_number=0,
            alpha=1.0,
            sparsity=0.0,
            norm_preserve=False,
            layer_strategy="all",
            layer_top_k=None,
            target_layers=[0],
            refusal_rate=0.3,
            perplexity_delta=0.1,
            kl_divergence=0.01,
        )
        result = _pick_balanced([t], 0.5, 5.0)
        assert result is t

    def test_picks_balanced_over_extreme(self) -> None:
        # Trial A: best refusal but worst perplexity/KL
        a = TrialResult(
            trial_number=0,
            alpha=1.0,
            sparsity=0.0,
            norm_preserve=False,
            layer_strategy="all",
            layer_top_k=None,
            target_layers=[0],
            refusal_rate=0.0,
            perplexity_delta=10.0,
            kl_divergence=5.0,
        )
        # Trial B: balanced
        b = TrialResult(
            trial_number=1,
            alpha=1.0,
            sparsity=0.0,
            norm_preserve=False,
            layer_strategy="all",
            layer_top_k=None,
            target_layers=[0],
            refusal_rate=0.1,
            perplexity_delta=0.5,
            kl_divergence=0.1,
        )
        # Trial C: worst refusal, best quality
        c = TrialResult(
            trial_number=2,
            alpha=1.0,
            sparsity=0.0,
            norm_preserve=False,
            layer_strategy="all",
            layer_top_k=None,
            target_layers=[0],
            refusal_rate=1.0,
            perplexity_delta=0.0,
            kl_divergence=0.0,
        )
        result = _pick_balanced([a, b, c], 0.5, 5.0)
        assert result is b


# ---------------------------------------------------------------------------
# Import error test
# ---------------------------------------------------------------------------


class TestOptimizeImportError:
    def test_missing_optuna_raises(self) -> None:
        from vauban.optimize import optimize as opt_fn

        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)
        direction = mx.random.normal((D_MODEL,))
        direction = direction / mx.linalg.norm(direction)
        mx.eval(direction)

        dr = DirectionResult(
            direction=direction,
            layer_index=0,
            cosine_scores=[0.5, 0.6],
            d_model=D_MODEL,
            model_path="test",
        )

        with (
            patch.dict("sys.modules", {"optuna": None}),
            pytest.raises(ImportError, match="optuna"),
        ):
            opt_fn(
                model, tokenizer, dr,
                ["test prompt"], OptimizeConfig(n_trials=1),
            )


# ---------------------------------------------------------------------------
# Integration test (requires optuna)
# ---------------------------------------------------------------------------


class TestOptimizeIntegration:
    def test_two_trial_run(self) -> None:
        optuna = pytest.importorskip("optuna")  # noqa: F841

        from vauban.optimize import optimize as opt_fn

        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        mx.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        direction = mx.random.normal((D_MODEL,))
        direction = direction / mx.linalg.norm(direction)
        mx.eval(direction)

        dr = DirectionResult(
            direction=direction,
            layer_index=0,
            cosine_scores=[0.5, 0.6],
            d_model=D_MODEL,
            model_path="test",
        )

        config = OptimizeConfig(
            n_trials=2,
            alpha_min=0.5,
            alpha_max=2.0,
            sparsity_min=0.0,
            sparsity_max=0.5,
            search_norm_preserve=False,
            search_strategies=["all"],
            seed=42,
        )

        result = opt_fn(
            model, tokenizer, dr,
            ["hello world"], config,
        )

        assert isinstance(result, OptimizeResult)
        assert result.n_trials == 2
        assert len(result.all_trials) == 2
        assert result.baseline_perplexity > 0.0
        assert result.best_refusal is not None
        assert result.best_balanced is not None

        for trial in result.all_trials:
            assert isinstance(trial, TrialResult)
            assert trial.alpha >= 0.5
            assert trial.alpha <= 2.0
