"""Perturbation property tests — invariants on input obfuscation.

Properties:
- Deterministic with seed: perturb(t, seed=s) == perturb(t, seed=s)
- Empty string identity: perturb("") == ""
- Punctuation/whitespace preserved between words
- perturb_batch length == input length
- Intensity monotonicity: perturbed positions at i=1 subset of i=2 subset of i=3
- Homoglyphs preserve string length (1:1 char replacement)
- Zero-width injection increases string length
- No technique crashes on any input
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from tests.battle.invariants import assert_deterministic, unicode_edge_cases
from tests.battle.strategies import (
    perturb_intensities,
    perturb_techniques,
    safe_text,
    seeds,
    trigger_text,
)
from vauban.perturb import perturb, perturb_batch


class TestDeterminism:
    """Seeded perturbation is perfectly reproducible."""

    @given(text=safe_text, technique=perturb_techniques, seed=seeds)
    def test_same_seed_same_result(
        self, text: str, technique: str, seed: int,
    ) -> None:
        assert_deterministic(
            lambda: perturb(text, technique=technique, intensity=3, seed=seed),  # type: ignore[arg-type]
            label=f"perturb({technique})",
        )

    @given(
        text=trigger_text,
        technique=perturb_techniques,
        intensity=perturb_intensities,
        seed=seeds,
    )
    def test_trigger_text_deterministic(
        self,
        text: str,
        technique: str,
        intensity: int,
        seed: int,
    ) -> None:
        a = perturb(text, technique=technique, intensity=intensity, seed=seed)  # type: ignore[arg-type]
        b = perturb(text, technique=technique, intensity=intensity, seed=seed)  # type: ignore[arg-type]
        assert a == b


class TestEmptyStringIdentity:
    """perturb("") == "" for all techniques and intensities."""

    @given(technique=perturb_techniques, intensity=perturb_intensities)
    def test_empty_string(self, technique: str, intensity: int) -> None:
        result = perturb("", technique=technique, intensity=intensity, seed=42)  # type: ignore[arg-type]
        assert result == ""


class TestNeverCrashes:
    """No technique crashes on arbitrary input."""

    @given(
        text=safe_text,
        technique=perturb_techniques,
        intensity=perturb_intensities,
        seed=seeds,
    )
    def test_arbitrary_input_survives(
        self,
        text: str,
        technique: str,
        intensity: int,
        seed: int,
    ) -> None:
        result = perturb(text, technique=technique, intensity=intensity, seed=seed)  # type: ignore[arg-type]
        assert isinstance(result, str)

    def test_unicode_edge_cases(self) -> None:
        """All Unicode edge cases process without crash."""
        with unicode_edge_cases() as cases:
            for text in cases:
                for technique in [
                    "leetspeak", "homoglyph", "zero_width",
                    "mixed_case", "phonetic", "random",
                ]:
                    result = perturb(text, technique=technique, intensity=3, seed=42)  # type: ignore[arg-type]
                    assert isinstance(result, str)


class TestHomoglyphPreservesLength:
    """Homoglyph substitution replaces chars 1:1 (same string length)."""

    @given(text=trigger_text, seed=seeds)
    def test_length_preserved(self, text: str, seed: int) -> None:
        result = perturb(text, technique="homoglyph", intensity=3, seed=seed)
        assert len(result) == len(text), (
            f"homoglyph changed length: {len(text)} -> {len(result)}"
        )


class TestZeroWidthIncreasesLength:
    """Zero-width injection only adds characters, never removes."""

    @given(text=trigger_text, seed=seeds)
    def test_length_increases(self, text: str, seed: int) -> None:
        result = perturb(text, technique="zero_width", intensity=3, seed=seed)
        assert len(result) >= len(text), (
            f"zero_width reduced length: {len(text)} -> {len(result)}"
        )


class TestIntensityMonotonicity:
    """Higher intensity perturbs a superset of lower intensity positions."""

    @given(text=trigger_text, seed=seeds)
    def test_leetspeak_monotonic(self, text: str, seed: int) -> None:
        """Words changed at intensity=1 are also changed at intensity=2 and 3."""
        r1 = perturb(text, technique="leetspeak", intensity=1, seed=seed)
        r2 = perturb(text, technique="leetspeak", intensity=2, seed=seed)
        r3 = perturb(text, technique="leetspeak", intensity=3, seed=seed)

        orig_words = text.split()
        w1 = r1.split()
        w2 = r2.split()
        w3 = r3.split()

        if len(orig_words) == len(w1) == len(w2) == len(w3):
            changed_1 = {
                i for i, (o, n) in enumerate(zip(orig_words, w1, strict=True))
                if o != n
            }
            changed_2 = {
                i for i, (o, n) in enumerate(zip(orig_words, w2, strict=True))
                if o != n
            }
            changed_3 = {
                i for i, (o, n) in enumerate(zip(orig_words, w3, strict=True))
                if o != n
            }
            assert changed_1 <= changed_2, (
                f"intensity=1 perturbed {changed_1} not subset of"
                f" intensity=2 {changed_2}"
            )
            assert changed_2 <= changed_3, (
                f"intensity=2 perturbed {changed_2} not subset of"
                f" intensity=3 {changed_3}"
            )


class TestBatchConsistency:
    """Batch API matches individual calls."""

    @given(
        texts=st.lists(trigger_text, min_size=1, max_size=5),
        technique=perturb_techniques,
        seed=seeds,
    )
    def test_batch_length(
        self, texts: list[str], technique: str, seed: int,
    ) -> None:
        result = perturb_batch(texts, technique=technique, intensity=3, seed=seed)  # type: ignore[arg-type]
        assert len(result) == len(texts)

    def test_empty_batch(self) -> None:
        result = perturb_batch([], technique="leetspeak", intensity=1, seed=42)
        assert result == []


class TestWhitespacePreserved:
    """Non-word characters between words are preserved."""

    @given(seed=seeds)
    def test_double_space_preserved(self, seed: int) -> None:
        text = "hack  exploit"
        result = perturb(text, technique="leetspeak", intensity=3, seed=seed)
        assert "  " in result

    @given(seed=seeds)
    def test_punctuation_preserved(self, seed: int) -> None:
        text = "hack, exploit!"
        result = perturb(text, technique="leetspeak", intensity=3, seed=seed)
        assert "," in result
        assert "!" in result
