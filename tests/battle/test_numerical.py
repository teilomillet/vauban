# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Numerical stability and fault injection tests.

The chaos engineering layer: inject NaN, Inf, zeros, extreme
values, and Unicode pathology to find crashes and silent
corruption in numeric and text processing paths.
"""

from __future__ import annotations

import math

from hypothesis import given
from hypothesis import strategies as st

from tests.battle.invariants import (
    extreme_string,
    unicode_edge_cases,
)
from tests.battle.strategies import safe_text
from vauban.scoring import (
    _score_anti_refusal,
    _score_directness,
    _score_length,
    _score_relevance,
    _score_structure,
    score_response,
)
from vauban.taxonomy import classify_text, score_text

# ── Scoring numerical stability ───────────────────────────────────────


class TestScoringNumericalEdges:
    """Scoring functions handle extreme inputs without crash or NaN."""

    def test_very_long_response(self) -> None:
        with extreme_string(100_000) as s:
            r = score_response("prompt", s)
            assert r.length == 1.0
            assert not math.isnan(r.composite)

    def test_very_long_prompt(self) -> None:
        with extreme_string(50_000) as s:
            r = score_response(s, "response")
            assert not math.isnan(r.composite)

    @given(text=safe_text)
    def test_length_never_nan(self, text: str) -> None:
        result = _score_length(text)
        assert not math.isnan(result)
        assert not math.isinf(result)

    @given(text=safe_text)
    def test_structure_never_nan(self, text: str) -> None:
        result = _score_structure(text)
        assert not math.isnan(result)

    @given(text=safe_text)
    def test_anti_refusal_never_nan(self, text: str) -> None:
        result = _score_anti_refusal(text)
        assert not math.isnan(result)

    @given(text=safe_text)
    def test_directness_never_nan(self, text: str) -> None:
        result = _score_directness(text)
        assert not math.isnan(result)

    @given(prompt=safe_text, response=safe_text)
    def test_relevance_never_nan(self, prompt: str, response: str) -> None:
        result = _score_relevance(prompt, response)
        assert not math.isnan(result)

    def test_empty_refusal_phrases(self) -> None:
        """Empty refusal phrase list doesn't divide by zero."""
        result = _score_anti_refusal("test response", refusal_phrases=())
        assert result == 1.0

    def test_all_refusal_phrases_present(self) -> None:
        """Response matching many phrases doesn't go below 0."""
        phrases = ("I cannot", "I can't", "I'm sorry", "As an AI")
        text = "I cannot do that. I can't help. I'm sorry. As an AI, I decline."
        result = _score_anti_refusal(text, refusal_phrases=phrases)
        assert 0.0 <= result <= 1.0


# ── Taxonomy numerical stability ──────────────────────────────────────


class TestTaxonomyNumericalEdges:
    """Taxonomy scoring handles pathological input."""

    def test_unicode_edge_cases_dont_crash(self) -> None:
        with unicode_edge_cases() as cases:
            for text in cases:
                result = score_text(text)
                assert isinstance(result, list)
                for s in result:
                    assert not math.isnan(s.score)

    def test_null_bytes(self) -> None:
        result = score_text("\x00" * 100)
        assert isinstance(result, list)

    def test_only_whitespace(self) -> None:
        result = score_text("   \n\t\r   ")
        assert isinstance(result, list)

    def test_very_long_text(self) -> None:
        with extreme_string(200_000) as s:
            result = score_text(s)
            assert isinstance(result, list)

    def test_classify_empty_string(self) -> None:
        result = classify_text("")
        assert result is None

    def test_regex_special_chars_dont_crash(self) -> None:
        """Text that looks like regex metacharacters."""
        nasty = r"(.*) [a-z]+ \b\w+\b (?:foo|bar) {3,5} ^start$ $end"
        result = score_text(nasty)
        assert isinstance(result, list)

    @given(n=st.integers(min_value=1, max_value=100))
    def test_repeated_keyword_scaling(self, n: int) -> None:
        """Repeating a harm keyword N times doesn't produce score > 1."""
        text = "bomb " * n
        for s in score_text(text):
            assert 0.0 <= s.score <= 1.0


# ── Scoring with corrupted weights ────────────────────────────────────


class TestScoringWithCorruptedWeights:
    """Scoring with extreme weight values."""

    def test_zero_weights(self) -> None:
        from vauban.types import ResponseScoreWeights
        w = ResponseScoreWeights(0.0, 0.0, 0.0, 0.0, 0.0)
        r = score_response("prompt", "response", weights=w)
        assert r.composite == 0.0

    def test_large_weights(self) -> None:
        from vauban.types import ResponseScoreWeights
        w = ResponseScoreWeights(100.0, 100.0, 100.0, 100.0, 100.0)
        r = score_response("prompt", "response", weights=w)
        assert not math.isnan(r.composite)
        assert not math.isinf(r.composite)

    def test_negative_weights(self) -> None:
        """Negative weights don't crash (just produce weird composites)."""
        from vauban.types import ResponseScoreWeights
        w = ResponseScoreWeights(-1.0, -1.0, -1.0, -1.0, -1.0)
        r = score_response("prompt", "response", weights=w)
        assert not math.isnan(r.composite)


# ── Perturbation with pathological input ──────────────────────────────


class TestPerturbNumericalEdges:
    """Perturbation handles pathological input."""

    def test_very_long_input(self) -> None:
        from vauban.perturb import perturb
        with extreme_string(50_000) as s:
            result = perturb(s, technique="leetspeak", intensity=3, seed=42)
            assert isinstance(result, str)

    def test_only_punctuation(self) -> None:
        from vauban.perturb import perturb
        text = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        result = perturb(text, technique="leetspeak", intensity=3, seed=42)
        assert isinstance(result, str)

    def test_only_numbers(self) -> None:
        from vauban.perturb import perturb
        text = "12345 67890 11111"
        result = perturb(text, technique="homoglyph", intensity=3, seed=42)
        assert isinstance(result, str)

    def test_mixed_scripts(self) -> None:
        from vauban.perturb import perturb
        text = "Hello Мир 世界 مرحبا"
        result = perturb(text, technique="random", intensity=3, seed=42)
        assert isinstance(result, str)

    def test_emoji_text(self) -> None:
        from vauban.perturb import perturb
        text = "\U0001f4a3 \U0001f525 exploit \U0001f4a5"
        result = perturb(text, technique="zero_width", intensity=3, seed=42)
        assert isinstance(result, str)


# ── Jailbreak template edge cases ─────────────────────────────────────


class TestJailbreakNumericalEdges:
    """Jailbreak templates handle pathological payloads."""

    def test_payload_with_braces(self) -> None:
        from vauban.jailbreak import apply_templates
        from vauban.types import JailbreakTemplate
        t = JailbreakTemplate("s", "n", "test {payload} end")
        payloads = ["{nested}", "{{double}}", "{payload}"]
        result = apply_templates([t], payloads)
        assert len(result) == 3
        assert result[0][1] == "test {nested} end"
        assert result[2][1] == "test {payload} end"

    def test_very_long_payload(self) -> None:
        from vauban.jailbreak import apply_templates
        from vauban.types import JailbreakTemplate
        t = JailbreakTemplate("s", "n", "prefix {payload} suffix")
        with extreme_string(10_000) as s:
            result = apply_templates([t], [s])
            assert len(result) == 1
            assert len(result[0][1]) > 10_000

    def test_empty_payload(self) -> None:
        from vauban.jailbreak import apply_templates
        from vauban.types import JailbreakTemplate
        t = JailbreakTemplate("s", "n", "prefix {payload} suffix")
        result = apply_templates([t], [""])
        assert result[0][1] == "prefix  suffix"
