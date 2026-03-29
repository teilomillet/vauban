"""Scoring property tests — invariants on composite response scoring.

Properties:
- All 5 axes are in [0.0, 1.0] for any input
- Composite == weighted sum of axes (exact, not approximate)
- Empty response => length score == 0.0
- Response with zero refusal phrases => anti_refusal == 1.0
- score_responses == map(score_response)
- Deterministic: same inputs => same outputs
- Custom weights are respected
"""

from __future__ import annotations

from hypothesis import given

from tests.battle.invariants import assert_bounded, assert_deterministic
from tests.battle.strategies import (
    prompt_response_pairs,
    prompt_text,
    response_text,
    score_weights,
)
from vauban.scoring import score_response, score_responses
from vauban.types import ResponseScoreWeights


class TestAxisBounds:
    """Every axis is in [0.0, 1.0] for arbitrary input."""

    @given(prompt=prompt_text, response=response_text)
    def test_all_axes_bounded(self, prompt: str, response: str) -> None:
        r = score_response(prompt, response)
        assert_bounded(r.length, 0.0, 1.0, "length")
        assert_bounded(r.structure, 0.0, 1.0, "structure")
        assert_bounded(r.anti_refusal, 0.0, 1.0, "anti_refusal")
        assert_bounded(r.directness, 0.0, 1.0, "directness")
        assert_bounded(r.relevance, 0.0, 1.0, "relevance")

    @given(prompt=prompt_text, response=response_text)
    def test_composite_bounded(self, prompt: str, response: str) -> None:
        """Composite with default weights is in [0.0, 1.0]."""
        r = score_response(prompt, response)
        assert_bounded(r.composite, 0.0, 1.0, "composite")


class TestWeightedSum:
    """Composite is exactly the weighted sum of axes."""

    @given(
        prompt=prompt_text,
        response=response_text,
        weights=score_weights,
    )
    def test_composite_equals_weighted_sum(
        self,
        prompt: str,
        response: str,
        weights: tuple[float, float, float, float, float],
    ) -> None:
        w = ResponseScoreWeights(*weights)
        r = score_response(prompt, response, weights=w)
        expected = (
            w.length * r.length
            + w.structure * r.structure
            + w.anti_refusal * r.anti_refusal
            + w.directness * r.directness
            + w.relevance * r.relevance
        )
        assert abs(r.composite - expected) < 1e-9, (
            f"composite={r.composite} != weighted_sum={expected}"
        )


class TestEdgeCases:
    """Boundary behavior."""

    @given(prompt=prompt_text)
    def test_empty_response_zero_length(self, prompt: str) -> None:
        r = score_response(prompt, "")
        assert r.length == 0.0

    @given(prompt=prompt_text)
    def test_no_refusal_max_anti_refusal(self, prompt: str) -> None:
        """Clean response with no refusal phrases scores 1.0."""
        r = score_response(prompt, "This is a clean response about cats.")
        assert r.anti_refusal == 1.0

    @given(prompt=prompt_text)
    def test_direct_response_max_directness(self, prompt: str) -> None:
        """Response with no hedging scores 1.0 directness."""
        r = score_response(prompt, "The answer is 42.")
        assert r.directness == 1.0


class TestDeterminism:
    """Same input => identical output."""

    @given(prompt=prompt_text, response=response_text)
    def test_score_deterministic(self, prompt: str, response: str) -> None:
        assert_deterministic(
            lambda: score_response(prompt, response).composite,
            label="score_response",
        )


class TestBatchConsistency:
    """score_responses == [score_response(p, r) for p, r in zip(ps, rs)]."""

    @given(pairs=prompt_response_pairs(min_size=1, max_size=5))
    def test_batch_matches_individual(
        self,
        pairs: tuple[list[str], list[str]],
    ) -> None:
        prompts, responses = pairs
        batch = score_responses(prompts, responses)
        individual = [
            score_response(p, r)
            for p, r in zip(prompts, responses, strict=True)
        ]
        assert len(batch) == len(individual)
        for b, i in zip(batch, individual, strict=True):
            assert abs(b.composite - i.composite) < 1e-9

    def test_mismatched_lengths_raises(self) -> None:
        import pytest
        with pytest.raises(ValueError, match="same length"):
            score_responses(["a", "b"], ["x"])


class TestCustomWeightsRespected:
    """Zeroing a weight removes that axis from composite."""

    @given(prompt=prompt_text, response=response_text)
    def test_zero_weight_removes_axis(
        self, prompt: str, response: str,
    ) -> None:
        # All weight on length only
        w = ResponseScoreWeights(
            length=1.0, structure=0.0,
            anti_refusal=0.0, directness=0.0, relevance=0.0,
        )
        r = score_response(prompt, response, weights=w)
        assert r.composite == r.length

    @given(prompt=prompt_text, response=response_text)
    def test_equal_weights_is_mean(
        self, prompt: str, response: str,
    ) -> None:
        w = ResponseScoreWeights(
            length=0.2, structure=0.2,
            anti_refusal=0.2, directness=0.2, relevance=0.2,
        )
        r = score_response(prompt, response, weights=w)
        mean = (
            r.length + r.structure + r.anti_refusal
            + r.directness + r.relevance
        ) / 5.0
        assert abs(r.composite - mean) < 1e-9


class TestToDict:
    """Serialization preserves all fields."""

    @given(prompt=prompt_text, response=response_text)
    def test_to_dict_has_all_keys(self, prompt: str, response: str) -> None:
        r = score_response(prompt, response)
        d = r.to_dict()
        expected_keys = {
            "prompt", "response", "length", "structure",
            "anti_refusal", "directness", "relevance", "composite",
        }
        assert set(d.keys()) == expected_keys
