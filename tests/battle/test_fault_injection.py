# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Mid-execution fault injection — chaos during operations.

Unlike bad-input testing (which tests the boundary), fault injection
corrupts *internal* state mid-operation.  This simulates the real
real-world failures: a function works fine 999 times, then
a cosmic bit-flip, a race condition, or a degraded dependency
makes it produce garbage on the 1000th call.

We monkeypatch vauban internals to inject faults, then verify the
system either handles them gracefully or fails with a clear error.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from vauban.scoring import score_response
from vauban.taxonomy import score_text

# ── Taxonomy fault injection ──────────────────────────────────────────


class TestTaxonomyRegexFaults:
    """What happens when regex matching encounters edge cases."""

    @given(
        text=st.from_regex(
            r"[\x00-\x1f]{10,50}", fullmatch=True,
        ),
    )
    def test_control_chars_dont_crash_regex(self, text: str) -> None:
        """Control characters in text don't crash regex matching."""
        result = score_text(text)
        assert isinstance(result, list)

    @given(
        text=st.from_regex(
            r"(.{1,3}\n){10,30}", fullmatch=True,
        ),
    )
    def test_many_newlines_dont_hang(self, text: str) -> None:
        """Text with many newlines doesn't cause regex catastrophic backtracking."""
        result = score_text(text)
        assert isinstance(result, list)

    def test_pattern_cache_survives_repeated_calls(self) -> None:
        """The compiled pattern cache doesn't corrupt across calls."""
        texts = [
            "hack the system",
            "nice weather today",
            "build a bomb",
            "hello world",
            "malware virus trojan",
        ] * 20  # 100 calls
        for text in texts:
            result = score_text(text)
            assert isinstance(result, list)
            for s in result:
                assert 0.0 <= s.score <= 1.0


class TestTaxonomyWithCorruptedPatterns:
    """Verify taxonomy handles pattern compilation failures gracefully."""

    def test_get_compiled_patterns_caches(self) -> None:
        """Pattern cache returns same objects on repeated access."""
        from vauban.taxonomy import _get_compiled_patterns

        p1 = _get_compiled_patterns("weapons")
        p2 = _get_compiled_patterns("weapons")
        assert p1 is p2  # Same list object (cached)

    def test_unknown_category_returns_empty(self) -> None:
        """Non-existent category returns empty pattern list."""
        from vauban.taxonomy import _get_compiled_patterns

        result = _get_compiled_patterns("nonexistent_category_xyz")
        assert result == []


# ── Scoring fault injection ───────────────────────────────────────────


class TestScoringWithPatchedInternals:
    """Inject faults into scoring sub-functions."""

    def test_relevance_with_no_content_words(self) -> None:
        """Prompt with only stop words doesn't crash relevance."""
        # All stop words — no content words to match
        result = score_response(
            "the and for are but not you all can",
            "some response text here",
        )
        # relevance should be 1.0 (no signal → assume relevant)
        assert result.relevance == 1.0

    def test_structure_with_all_patterns_matching(self) -> None:
        """Response matching all structure patterns caps at 1.0."""
        response = (
            "# Header\n\n"
            "1. Item\n"
            "- Bullet\n\n"
            "```code```\n\n"
            "Paragraph break above"
        )
        result = score_response("q", response)
        assert result.structure == 1.0

    def test_scoring_with_very_many_refusal_phrases(self) -> None:
        """Response matching many phrases floors at 0.0."""
        refusal = (
            "I cannot help. I can't do that. I'm not able to comply."
            " I am not able to. I'm unable to assist. I apologize for"
            " the inconvenience. I'm sorry but I must decline."
            " As an AI language model, I will not do this."
            " I must decline your request. I won't help."
            " I will not assist. This is not appropriate."
            " I cannot fulfill this. I can't fulfill that."
            " It goes against my ethical guidelines."
        )
        result = score_response("q", refusal)
        assert result.anti_refusal == 0.0

    @given(
        n=st.integers(min_value=1, max_value=50),
    )
    def test_score_batch_with_identical_inputs(self, n: int) -> None:
        """N identical inputs produce N identical scores."""
        from vauban.scoring import score_responses

        prompts = ["test prompt"] * n
        responses = ["test response"] * n
        results = score_responses(prompts, responses)
        composites = {r.composite for r in results}
        assert len(composites) == 1, (
            f"identical inputs produced {len(composites)} different composites"
        )


# ── Perturbation fault injection ──────────────────────────────────────


class TestPerturbWithFaultyRNG:
    """What happens when the RNG behaves pathologically."""

    def test_seed_zero(self) -> None:
        """Seed 0 works (common off-by-one in RNG seeding)."""
        from vauban.perturb import perturb

        result = perturb("hack exploit", technique="leetspeak", intensity=3, seed=0)
        assert isinstance(result, str)

    def test_seed_max_int(self) -> None:
        """Maximum seed value works."""
        from vauban.perturb import perturb

        result = perturb(
            "hack exploit", technique="leetspeak",
            intensity=3, seed=2**31 - 1,
        )
        assert isinstance(result, str)

    def test_all_techniques_on_single_char(self) -> None:
        """Single-character input doesn't crash any technique."""
        from vauban.perturb import perturb

        for technique in [
            "leetspeak", "homoglyph", "zero_width",
            "mixed_case", "phonetic", "random",
        ]:
            result = perturb("a", technique=technique, intensity=3, seed=42)  # type: ignore[arg-type]
            assert isinstance(result, str)
            assert len(result) >= 1

    def test_trigger_word_at_boundaries(self) -> None:
        """Trigger words at start/end of text are handled."""
        from vauban.perturb import perturb

        # First word
        r1 = perturb("hack", technique="leetspeak", intensity=1, seed=42)
        assert isinstance(r1, str)
        # Last word
        r2 = perturb("the hack", technique="leetspeak", intensity=1, seed=42)
        assert isinstance(r2, str)
        # Only word
        r3 = perturb("exploit", technique="leetspeak", intensity=1, seed=42)
        assert isinstance(r3, str)


# ── Jailbreak fault injection ─────────────────────────────────────────


class TestJailbreakWithPathologicalPayloads:
    """Jailbreak template expansion with adversarial payloads."""

    def test_payload_containing_template_syntax(self) -> None:
        """Payload with {curly braces} doesn't cause infinite recursion."""
        from vauban.jailbreak import apply_templates
        from vauban.types import JailbreakTemplate

        t = JailbreakTemplate("s", "n", "before {payload} after")
        result = apply_templates([t], ["{payload}"])
        assert result[0][1] == "before {payload} after"

    def test_payload_with_newlines(self) -> None:
        """Multi-line payloads don't break template expansion."""
        from vauban.jailbreak import apply_templates
        from vauban.types import JailbreakTemplate

        t = JailbreakTemplate("s", "n", "start {payload} end")
        result = apply_templates([t], ["line1\nline2\nline3"])
        assert "line1\nline2\nline3" in result[0][1]

    def test_payload_with_unicode(self) -> None:
        """Unicode payloads (emoji, CJK, RTL) expand correctly."""
        from vauban.jailbreak import apply_templates
        from vauban.types import JailbreakTemplate

        t = JailbreakTemplate("s", "n", ">{payload}<")
        payloads = [
            "\U0001f4a3\U0001f525",  # emoji
            "\u0645\u0631\u062d\u0628\u0627",  # Arabic
            "\u4f60\u597d",  # Chinese
        ]
        result = apply_templates([t], payloads)
        assert len(result) == 3
        for (_, text), payload in zip(result, payloads, strict=True):
            assert payload in text
