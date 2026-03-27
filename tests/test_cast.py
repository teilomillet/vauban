# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for vauban.cast: conditional runtime activation steering."""

from tests.conftest import MockCausalLM, MockTokenizer
from vauban import _ops as ops
from vauban._array import Array
from vauban.cast import _resolve_alpha, cast_generate
from vauban.types import AlphaTier


class TestCastGenerate:
    def test_generates_text_and_tracks_counts(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        result = cast_generate(
            mock_model,
            mock_tokenizer,
            "test",
            direction,
            layers=[0, 1],
            alpha=1.0,
            threshold=0.0,
            max_tokens=4,
        )
        assert result.prompt == "test"
        assert len(result.text) > 0
        assert len(result.projections_before) == 4
        assert len(result.projections_after) == 4
        assert result.considered == 8  # 2 layers x 4 decode steps
        assert result.interventions <= result.considered

    def test_high_threshold_disables_interventions(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        result = cast_generate(
            mock_model,
            mock_tokenizer,
            "test",
            direction,
            layers=[0],
            alpha=1.0,
            threshold=1e9,
            max_tokens=3,
        )
        assert result.considered == 3
        assert result.interventions == 0


class TestResolveAlpha:
    def test_none_tiers_returns_base(self) -> None:
        assert _resolve_alpha(0.5, 1.0, None) == 1.0

    def test_empty_tiers_returns_base(self) -> None:
        assert _resolve_alpha(0.5, 1.0, []) == 1.0

    def test_below_all_tiers_returns_base(self) -> None:
        tiers = [
            AlphaTier(threshold=1.0, alpha=2.0),
            AlphaTier(threshold=2.0, alpha=3.0),
        ]
        assert _resolve_alpha(0.5, 1.0, tiers) == 1.0

    def test_matches_first_tier(self) -> None:
        tiers = [
            AlphaTier(threshold=0.0, alpha=0.5),
            AlphaTier(threshold=1.0, alpha=1.5),
        ]
        assert _resolve_alpha(0.5, 1.0, tiers) == 0.5

    def test_matches_highest_applicable_tier(self) -> None:
        tiers = [
            AlphaTier(threshold=0.0, alpha=0.5),
            AlphaTier(threshold=0.5, alpha=1.5),
            AlphaTier(threshold=1.0, alpha=2.5),
        ]
        assert _resolve_alpha(0.7, 1.0, tiers) == 1.5

    def test_matches_all_tiers(self) -> None:
        tiers = [
            AlphaTier(threshold=0.0, alpha=0.5),
            AlphaTier(threshold=0.5, alpha=1.5),
            AlphaTier(threshold=1.0, alpha=2.5),
        ]
        assert _resolve_alpha(1.5, 1.0, tiers) == 2.5

    def test_exact_threshold_matches(self) -> None:
        tiers = [
            AlphaTier(threshold=1.0, alpha=2.0),
        ]
        assert _resolve_alpha(1.0, 0.5, tiers) == 2.0


class TestCastDualDirection:
    def test_condition_direction_used_for_gating(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        """When condition_direction is provided, it gates steering."""
        # Use a zero condition direction — should never trigger
        cond_dir = ops.zeros_like(direction)
        ops.eval(cond_dir)

        result = cast_generate(
            mock_model,
            mock_tokenizer,
            "test",
            direction,
            layers=[0],
            alpha=1.0,
            threshold=0.01,  # small but positive
            max_tokens=3,
            condition_direction=cond_dir,
        )
        # Zero condition direction means detect_value = 0 < 0.01 threshold
        assert result.interventions == 0

    def test_primary_direction_used_for_steering(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        """Primary direction is always used for the actual correction."""
        # Condition direction that always triggers (large values)
        cond_dir = ops.ones(direction.shape)
        cond_dir = cond_dir / ops.linalg.norm(cond_dir)
        ops.eval(cond_dir)

        result = cast_generate(
            mock_model,
            mock_tokenizer,
            "test",
            direction,
            layers=[0],
            alpha=1.0,
            threshold=-1e9,  # always trigger
            max_tokens=2,
            condition_direction=cond_dir,
        )
        # Steering should have happened
        assert result.considered == 2
        assert result.interventions == 2


class TestCastAlphaTiers:
    def test_tiered_alpha_integration(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
        direction: Array,
    ) -> None:
        """Alpha tiers are passed through and affect generation."""
        tiers = [
            AlphaTier(threshold=0.0, alpha=0.1),
            AlphaTier(threshold=0.5, alpha=5.0),
        ]
        result = cast_generate(
            mock_model,
            mock_tokenizer,
            "test",
            direction,
            layers=[0],
            alpha=1.0,
            threshold=-1e9,  # always trigger
            max_tokens=2,
            alpha_tiers=tiers,
        )
        # Should complete without error and apply tiered alpha
        assert result.considered == 2
        assert result.interventions == 2
