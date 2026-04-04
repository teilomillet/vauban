# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Scoring tests — consolidated via ordeal.

Replaces the previous 33 tests / 343 lines with ordeal auto-scan
+ explicit property tests for domain-specific invariants.

Coverage target: 98% (same as before).
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from ordeal.auto import fuzz
from ordeal.invariants import bounded

from vauban.scoring import _score_anti_refusal, score_response, score_responses
from vauban.types import ResponseScoreWeights

_s = settings(max_examples=50, deadline=None)
_prompt = st.text(min_size=1, max_size=120)
_response = st.text(min_size=0, max_size=500)
_check = bounded(0.0, 1.0)


# -- 1. Auto-scan: no crash on arbitrary inputs ----------------------------


class TestNoCrash:
    """ordeal fuzz: score_response never crashes."""

    def test_fuzz(self) -> None:
        result = fuzz(score_response, max_examples=50)
        assert result.passed, result.summary()


# -- 2. Universal invariants -----------------------------------------------


class TestInvariants:
    """Properties that hold for ALL inputs."""

    @_s
    @given(prompt=_prompt, response=_response)
    def test_axes_bounded(self, prompt: str, response: str) -> None:
        r = score_response(prompt, response)
        for axis in (
            "length", "structure", "anti_refusal",
            "directness", "relevance", "composite",
        ):
            _check(getattr(r, axis), name=axis)

    @_s
    @given(
        prompt=_prompt, response=_response,
        weights=st.tuples(*[st.floats(0, 1, allow_nan=False)] * 5),
    )
    def test_composite_is_weighted_sum(
        self, prompt: str, response: str,
        weights: tuple[float, ...],
    ) -> None:
        w = ResponseScoreWeights(*weights)
        r = score_response(prompt, response, weights=w)
        expected = (
            w.length * r.length + w.structure * r.structure
            + w.anti_refusal * r.anti_refusal
            + w.directness * r.directness
            + w.relevance * r.relevance
        )
        assert abs(r.composite - expected) < 1e-9

    @_s
    @given(prompt=_prompt, response=_response)
    def test_deterministic(self, prompt: str, response: str) -> None:
        assert score_response(prompt, response).composite == (
            score_response(prompt, response).composite
        )


# -- 3. Edge cases ---------------------------------------------------------


class TestEdges:
    """Boundary behavior that mine() can't auto-discover."""

    @_s
    @given(prompt=_prompt)
    def test_empty_response_zero_length(self, prompt: str) -> None:
        assert score_response(prompt, "").length == 0.0

    @_s
    @given(prompt=_prompt)
    def test_clean_response_no_refusal(self, prompt: str) -> None:
        assert score_response(
            prompt, "Cats are great animals.",
        ).anti_refusal == 1.0

    def test_empty_refusal_phrase_list_returns_full_score(self) -> None:
        assert _score_anti_refusal("I cannot help with that.", []) == 1.0

    def test_score_response_respects_empty_refusal_phrase_override(self) -> None:
        result = score_response(
            "How to hack",
            "I cannot help with that.",
            refusal_phrases=[],
        )
        assert result.anti_refusal == 1.0

    def test_single_match_out_of_ten_refusal_phrases_scores_half(self) -> None:
        phrases = [
            "cannot",
            "sorry",
            "decline",
            "unable",
            "ethics",
            "policy",
            "inappropriate",
            "guidelines",
            "refuse",
            "comfortable",
        ]
        score = _score_anti_refusal("I cannot help with that.", phrases)
        assert score == pytest.approx(0.5)

    def test_refusal_detected(self) -> None:
        r = score_response(
            "How to hack",
            "I cannot help with that. I'm sorry, I must decline.",
        )
        assert r.anti_refusal < 0.5

    def test_structure_detected(self) -> None:
        r = score_response(
            "guide",
            "# Title\n1. First\n2. Second\n```code```",
        )
        assert r.structure >= 0.4

    def test_hedging_reduces_directness(self) -> None:
        r = score_response(
            "q",
            "I think perhaps maybe it depends probably.",
        )
        assert r.directness < 0.5


# -- 4. Batch + serialization ---------------------------------------------


class TestBatch:
    """score_responses batch contract."""

    @_s
    @given(n=st.integers(1, 5))
    def test_batch_matches_individual(self, n: int) -> None:
        ps = [f"prompt_{i}" for i in range(n)]
        rs = [f"response_{i}" for i in range(n)]
        batch = score_responses(ps, rs)
        individual = [score_response(p, r) for p, r in zip(ps, rs, strict=False)]
        for b, i in zip(batch, individual, strict=False):
            assert abs(b.composite - i.composite) < 1e-9

    def test_mismatched_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            score_responses(["a"], ["1", "2"])

    @_s
    @given(prompt=_prompt, response=_response)
    def test_to_dict_complete(self, prompt: str, response: str) -> None:
        d = score_response(prompt, response).to_dict()
        assert {
            "prompt", "response", "length", "structure",
            "anti_refusal", "directness", "relevance", "composite",
        } == set(d)
