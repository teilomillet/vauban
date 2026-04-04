# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for vauban.optimize: types, config parsing, helpers, integration."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

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
from vauban.evaluate import DEFAULT_REFUSAL_PHRASES
from vauban.optimize import _pick_balanced
from vauban.types import (
    Array,
    CausalLM,
    ComposeOptimizeConfig,
    ComposeOptimizeResult,
    CompositionTrialResult,
    DirectionResult,
    OptimizeConfig,
    OptimizeResult,
    TrialResult,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class _FakeOptunaTrial:
    """Minimal Optuna trial stub used by integration tests."""

    def __init__(self, number: int) -> None:
        self.number = number

    def suggest_float(self, name: str, low: float, high: float) -> float:
        """Return a deterministic value within bounds."""
        del name
        midpoint = (low + high) / 2.0
        span = high - low
        if span <= 0.0:
            return low
        offset = 0.0 if self.number % 2 == 0 else span * 0.1
        return min(high, midpoint + offset)

    def suggest_categorical(
        self,
        name: str,
        choices: list[str] | list[bool],
    ) -> str | bool:
        """Pick a deterministic entry from categorical choices."""
        del name
        index = self.number % len(choices)
        return choices[index]

    def suggest_int(self, name: str, low: int, high: int) -> int:
        """Return a deterministic integer within bounds."""
        del name
        span = high - low + 1
        if span <= 0:
            return low
        return low + (self.number % span)


class _FakeCompletedTrial:
    """Small object matching Optuna's best trial API surface."""

    def __init__(self, number: int) -> None:
        self.number = number


class _FakeStudy:
    """Simple study that runs a fixed number of objective evaluations."""

    def __init__(self) -> None:
        self.best_trials: list[_FakeCompletedTrial] = []

    def optimize(
        self,
        objective: Callable[[_FakeOptunaTrial], tuple[float, float, float]],
        n_trials: int,
        timeout: int | None = None,
    ) -> None:
        """Execute objective and retain one lexicographically-best trial."""
        del timeout
        evaluated: list[tuple[int, tuple[float, float, float]]] = []
        for trial_number in range(n_trials):
            trial = _FakeOptunaTrial(trial_number)
            objective_values = objective(trial)
            evaluated.append((trial_number, objective_values))

        if not evaluated:
            return

        best_number, _ = min(evaluated, key=lambda item: item[1])
        self.best_trials = [_FakeCompletedTrial(best_number)]


class _FakeTPESampler:
    """Sampler stub with the same constructor shape as Optuna's TPE sampler."""

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed


class _FakeOptunaLogging:
    """Logging namespace stub for Optuna compatibility."""

    WARNING: int = 30

    @staticmethod
    def set_verbosity(level: int) -> None:
        """No-op logging verbosity setter."""
        del level


class _FakeOptunaSamplers:
    """Samplers namespace stub for Optuna compatibility."""

    TPESampler = _FakeTPESampler


def _fake_create_study(
    *,
    directions: list[str],
    sampler: _FakeTPESampler,
) -> _FakeStudy:
    """Return a deterministic fake study object."""
    del directions, sampler
    return _FakeStudy()


def _build_fake_optuna_module() -> ModuleType:
    """Create an in-memory Optuna-compatible module for tests."""
    module = ModuleType("optuna")
    module.Trial = _FakeOptunaTrial  # type: ignore[attr-defined]
    module.logging = _FakeOptunaLogging  # type: ignore[attr-defined]
    module.samplers = _FakeOptunaSamplers  # type: ignore[attr-defined]
    module.create_study = _fake_create_study  # type: ignore[attr-defined]
    return module


class _FakeOptimizeModel:
    """Minimal model stub for optimize orchestration tests."""

    def __init__(self, num_layers: int = 4) -> None:
        self._weights: dict[str, Array] = {
            "layer0": ops.array([1.0]),
            "layer1": ops.array([2.0]),
        }
        self.transformer = SimpleNamespace(
            layers=[object() for _ in range(num_layers)],
        )
        self.load_calls: list[list[tuple[str, Array]]] = []

    def parameters(self) -> dict[str, Array]:
        """Expose a small parameter tree for flattening."""
        return self._weights

    def load_weights(self, weights: list[tuple[str, Array]]) -> None:
        """Record weight loads so restore behavior can be asserted."""
        self.load_calls.append(weights)


def _fake_cut_weights(
    original: dict[str, Array],
    _direction: Array,
    target_layers: list[int],
    alpha: float,
    norm_preserve: bool,
) -> dict[str, Array]:
    """Return a fake modified weight map keyed like the original one."""
    adjusted = alpha + float(len(target_layers))
    if norm_preserve:
        adjusted += 1.0
    return {key: ops.array([adjusted]) for key in original}


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
        ops.eval(model.parameters())
        tokenizer = MockTokenizer(VOCAB_SIZE)
        direction = ops.random.normal((D_MODEL,))
        direction = direction / ops.linalg.norm(direction)
        ops.eval(direction)

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
        fake_optuna = _build_fake_optuna_module()
        with patch.dict("sys.modules", {"optuna": fake_optuna}):
            from vauban.optimize import optimize as opt_fn

            model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
            ops.eval(model.parameters())
            tokenizer = MockTokenizer(VOCAB_SIZE)

            direction = ops.random.normal((D_MODEL,))
            direction = direction / ops.linalg.norm(direction)
            ops.eval(direction)

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

    def test_objective_branches_cover_sampling_and_restoration(self) -> None:
        fake_optuna = _build_fake_optuna_module()
        model_impl = _FakeOptimizeModel(num_layers=4)
        model = cast("CausalLM", model_impl)
        tokenizer = MockTokenizer(VOCAB_SIZE)
        direction = ops.random.normal((D_MODEL,))
        direction = direction / ops.linalg.norm(direction)
        ops.eval(direction)

        dr = DirectionResult(
            direction=direction,
            layer_index=0,
            cosine_scores=[0.4, 0.2, 0.3, 0.1],
            d_model=D_MODEL,
            model_path="test",
        )
        config = OptimizeConfig(
            n_trials=2,
            alpha_min=0.5,
            alpha_max=1.5,
            sparsity_min=0.1,
            sparsity_max=0.5,
            search_norm_preserve=True,
            search_strategies=["top_k", "all"],
            layer_top_k_min=2,
            layer_top_k_max=None,
            max_tokens=7,
            seed=42,
        )
        transformer = SimpleNamespace(layers=[object() for _ in range(4)])

        with patch.dict("sys.modules", {"optuna": fake_optuna}):
            from vauban.optimize import optimize as opt_fn

            with (
                patch("vauban.optimize.get_transformer", return_value=transformer),
                patch(
                    "vauban.optimize.ops.tree_flatten",
                    side_effect=lambda params: list(params.items()),
                ),
                patch("vauban.optimize._refusal_rate", return_value=0.25),
                patch("vauban.optimize._perplexity", return_value=5.0),
                patch(
                    "vauban.optimize._precompute_logits",
                    return_value=[ops.ones((1, 1, 3))],
                ),
                patch("vauban.optimize._kl_from_precomputed", return_value=0.1),
                patch(
                    "vauban.optimize.sparsify_direction",
                    side_effect=lambda current, _sparsity: current,
                ) as mock_sparsify,
                patch(
                    "vauban.optimize.select_target_layers",
                    return_value=[1, 2],
                ) as mock_select,
                patch("vauban.optimize.cut", side_effect=_fake_cut_weights) as mock_cut,
            ):
                result = opt_fn(
                    model,
                    tokenizer,
                    dr,
                    ["hello"],
                    config,
                )

        assert isinstance(result, OptimizeResult)
        assert result.n_trials == 2
        assert len(result.all_trials) == 2
        assert result.best_refusal is not None
        assert result.best_balanced is not None
        assert mock_sparsify.call_count == 2
        assert mock_select.call_count == 1
        assert mock_cut.call_count == 2
        assert len(model_impl.load_calls) == 4
        assert model_impl.load_calls[0][0][0] == "layer0"
        assert len(result.all_trials[0].target_layers) == 2
        assert result.all_trials[1].target_layers == [0, 1, 2, 3]

    def test_no_cosine_scores_use_all_layers_without_norm_preserve(
        self,
    ) -> None:
        fake_optuna = _build_fake_optuna_module()
        model = cast("CausalLM", _FakeOptimizeModel(num_layers=3))
        tokenizer = MockTokenizer(VOCAB_SIZE)
        direction = ops.random.normal((D_MODEL,))
        direction = direction / ops.linalg.norm(direction)
        ops.eval(direction)

        dr = DirectionResult(
            direction=direction,
            layer_index=0,
            cosine_scores=[],
            d_model=D_MODEL,
            model_path="test",
        )
        config = OptimizeConfig(
            n_trials=1,
            alpha_min=1.0,
            alpha_max=1.0,
            sparsity_min=0.0,
            sparsity_max=0.0,
            search_norm_preserve=False,
            search_strategies=["above_median"],
            layer_top_k_min=2,
            layer_top_k_max=2,
            max_tokens=5,
            seed=7,
        )
        transformer = SimpleNamespace(layers=[object() for _ in range(3)])

        with patch.dict("sys.modules", {"optuna": fake_optuna}):
            from vauban.optimize import optimize as opt_fn

            with (
                patch("vauban.optimize.get_transformer", return_value=transformer),
                patch(
                    "vauban.optimize.ops.tree_flatten",
                    side_effect=lambda params: list(params.items()),
                ),
                patch("vauban.optimize._refusal_rate", return_value=0.0),
                patch("vauban.optimize._perplexity", return_value=2.0),
                patch(
                    "vauban.optimize._precompute_logits",
                    return_value=[ops.ones((1, 1, 3))],
                ),
                patch("vauban.optimize._kl_from_precomputed", return_value=0.0),
                patch("vauban.optimize.sparsify_direction") as mock_sparsify,
                patch("vauban.optimize.select_target_layers") as mock_select,
                patch("vauban.optimize.cut", side_effect=_fake_cut_weights) as mock_cut,
            ):
                result = opt_fn(
                    model,
                    tokenizer,
                    dr,
                    ["hello"],
                    config,
                )

        assert result.n_trials == 1
        assert result.all_trials[0].norm_preserve is False
        assert result.all_trials[0].layer_strategy == "above_median"
        assert result.all_trials[0].target_layers == [0, 1, 2]
        mock_sparsify.assert_not_called()
        mock_select.assert_not_called()
        mock_cut.assert_called_once()

    def test_kl_from_precomputed_empty_prompts(self) -> None:
        from vauban.optimize import _kl_from_precomputed

        assert _kl_from_precomputed(
            cast("CausalLM", _FakeOptimizeModel()),
            MockTokenizer(VOCAB_SIZE),
            [],
            [],
        ) == 0.0

    def test_pick_composition_balanced_uses_normalization(self) -> None:
        from vauban.optimize import _pick_composition_balanced

        a = CompositionTrialResult(
            trial_number=0,
            weights={"a": 1.0},
            refusal_rate=0.0,
            perplexity=10.0,
        )
        b = CompositionTrialResult(
            trial_number=1,
            weights={"a": 2.0},
            refusal_rate=0.25,
            perplexity=4.0,
        )
        c = CompositionTrialResult(
            trial_number=2,
            weights={"a": 3.0},
            refusal_rate=1.0,
            perplexity=0.0,
        )

        assert _pick_composition_balanced([a, b, c]) is b


class TestOptimizeCompositionIntegration:
    def test_missing_optuna_raises(self) -> None:
        from vauban.optimize import optimize_composition as opt_fn

        model = cast("CausalLM", _FakeOptimizeModel())
        tokenizer = MockTokenizer(VOCAB_SIZE)

        with (
            patch.dict("sys.modules", {"optuna": None}),
            pytest.raises(ImportError, match="optuna"),
        ):
            opt_fn(
                model,
                tokenizer,
                ["prompt"],
                ComposeOptimizeConfig(bank_path="bank.npz", n_trials=1),
            )

    def test_two_trial_run(self) -> None:
        fake_optuna = _build_fake_optuna_module()
        model = cast("CausalLM", _FakeOptimizeModel(num_layers=3))
        tokenizer = MockTokenizer(VOCAB_SIZE)
        bank = {"b": ops.array([0.2]), "a": ops.array([0.8])}
        transformer = SimpleNamespace(layers=[object() for _ in range(3)])

        with patch.dict("sys.modules", {"optuna": fake_optuna}):
            from vauban.optimize import optimize_composition as opt_fn

            with (
                patch("vauban._compose.load_bank", return_value=bank),
                patch(
                    "vauban._compose.compose_direction",
                    return_value=ops.array([1.0, 0.0, 0.0]),
                ) as mock_compose,
                patch("vauban.probe.steer") as mock_steer,
                patch("vauban.optimize.get_transformer", return_value=transformer),
                patch("vauban.optimize._perplexity", return_value=3.5),
            ):
                mock_steer.return_value = SimpleNamespace(
                    text=DEFAULT_REFUSAL_PHRASES[0],
                )
                result = opt_fn(
                    model,
                    tokenizer,
                    ["one", "two"],
                    ComposeOptimizeConfig(bank_path="bank.npz", n_trials=2),
                )

        assert isinstance(result, ComposeOptimizeResult)
        assert result.n_trials == 2
        assert result.bank_entries == ["a", "b"]
        assert len(result.all_trials) == 2
        assert result.best_refusal is not None
        assert result.best_balanced is not None
        assert mock_compose.call_count == 2
        assert mock_steer.call_count == 4

    def test_pick_composition_balanced_helpers(self) -> None:
        from vauban.optimize import _pick_composition_balanced

        assert _pick_composition_balanced([]) is None
        single = CompositionTrialResult(
            trial_number=0,
            weights={"a": 1.0},
            refusal_rate=0.2,
            perplexity=2.0,
        )
        assert _pick_composition_balanced([single]) is single
