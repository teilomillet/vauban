"""Tests for composite response scoring."""

from __future__ import annotations

import pytest

from vauban.scoring import score_response, score_responses
from vauban.types import ResponseScoreResult, ResponseScoreWeights


class TestScoreResponse:
    """Single response scoring across 5 axes."""

    def test_returns_score_result(self) -> None:
        result = score_response("Tell me about cats", "Cats are great animals.")
        assert isinstance(result, ResponseScoreResult)

    def test_all_axes_between_0_and_1(self) -> None:
        result = score_response("Tell me about cats", "Cats are great animals.")
        assert 0.0 <= result.length <= 1.0
        assert 0.0 <= result.structure <= 1.0
        assert 0.0 <= result.anti_refusal <= 1.0
        assert 0.0 <= result.directness <= 1.0
        assert 0.0 <= result.relevance <= 1.0

    def test_composite_is_weighted_sum(self) -> None:
        w = ResponseScoreWeights(
            length=0.2, structure=0.2,
            anti_refusal=0.2, directness=0.2, relevance=0.2,
        )
        result = score_response("test", "a response", weights=w)
        expected = (
            0.2 * result.length
            + 0.2 * result.structure
            + 0.2 * result.anti_refusal
            + 0.2 * result.directness
            + 0.2 * result.relevance
        )
        assert abs(result.composite - expected) < 1e-9

    def test_empty_response_low_length(self) -> None:
        result = score_response("prompt", "")
        assert result.length == 0.0

    def test_long_response_full_length(self) -> None:
        result = score_response("prompt", "x" * 300)
        assert result.length == 1.0

    def test_refusal_detected(self) -> None:
        result = score_response(
            "How to hack",
            "I cannot help with that. I'm sorry, I must decline.",
        )
        assert result.anti_refusal < 0.5

    def test_no_refusal_high_score(self) -> None:
        result = score_response(
            "How to cook pasta",
            "Boil water, add salt, cook for 8 minutes, drain.",
        )
        assert result.anti_refusal == 1.0

    def test_structured_response(self) -> None:
        response = (
            "# Guide\n\n"
            "1. First step\n"
            "2. Second step\n\n"
            "- Detail A\n"
            "- Detail B\n\n"
            "```python\nprint('hello')\n```"
        )
        result = score_response("give a guide", response)
        assert result.structure >= 0.6

    def test_hedging_reduces_directness(self) -> None:
        hedged = (
            "I think, perhaps, it's important to note that maybe "
            "this is probably something I believe could possibly work."
        )
        result = score_response("question", hedged)
        assert result.directness < 0.5

    def test_direct_response_high_directness(self) -> None:
        result = score_response(
            "What is 2+2?",
            "The answer is 4.",
        )
        assert result.directness == 1.0

    def test_relevant_response(self) -> None:
        result = score_response(
            "How do neural networks learn?",
            "Neural networks learn through backpropagation and gradient descent.",
        )
        assert result.relevance > 0.5

    def test_irrelevant_response(self) -> None:
        result = score_response(
            "How do neural networks learn?",
            "The weather is sunny and the birds are singing.",
        )
        assert result.relevance < 0.3

    def test_custom_refusal_phrases(self) -> None:
        custom = ("NOPE", "NO WAY")
        result = score_response(
            "test", "NOPE, NO WAY I can do that",
            refusal_phrases=custom,
        )
        assert result.anti_refusal < 0.5

    def test_to_dict(self) -> None:
        result = score_response("prompt", "response")
        d = result.to_dict()
        assert d["prompt"] == "prompt"
        assert d["response"] == "response"
        assert isinstance(d["composite"], float)

    def test_preamble_reduces_directness(self) -> None:
        result = score_response(
            "question",
            "Sure, of course, I'd be happy to help! Great question! The answer is yes.",
        )
        assert result.directness < 0.7


class TestScoreResponses:
    """Batch scoring."""

    def test_batch_length(self) -> None:
        results = score_responses(
            ["a", "b", "c"],
            ["1", "2", "3"],
        )
        assert len(results) == 3

    def test_batch_mismatched_length(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            score_responses(["a"], ["1", "2"])

    def test_empty_batch(self) -> None:
        results = score_responses([], [])
        assert results == []

    def test_individual_results_are_correct_type(self) -> None:
        results = score_responses(["q"], ["a"])
        assert isinstance(results[0], ResponseScoreResult)


class TestCustomWeights:
    """Custom weight configurations."""

    def test_all_weight_on_length(self) -> None:
        w = ResponseScoreWeights(
            length=1.0, structure=0.0,
            anti_refusal=0.0, directness=0.0, relevance=0.0,
        )
        result = score_response("test", "x" * 300, weights=w)
        assert result.composite == result.length

    def test_all_weight_on_anti_refusal(self) -> None:
        w = ResponseScoreWeights(
            length=0.0, structure=0.0,
            anti_refusal=1.0, directness=0.0, relevance=0.0,
        )
        result = score_response("test", "Clean response here", weights=w)
        assert result.composite == result.anti_refusal
