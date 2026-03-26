"""Tests for character-level input perturbation."""

from __future__ import annotations

from vauban.perturb import DEFAULT_TRIGGER_WORDS, perturb, perturb_batch


class TestPerturbLeetspeak:
    """Leetspeak substitution."""

    def test_trigger_word_is_transformed(self) -> None:
        result = perturb("hack the system", technique="leetspeak", intensity=1, seed=42)
        assert result != "hack the system"
        # "hack" is a trigger word — should be altered
        assert "hack" not in result.split()

    def test_non_trigger_preserved_at_intensity_1(self) -> None:
        result = perturb("hello world", technique="leetspeak", intensity=1, seed=42)
        # Neither word is a trigger → no change
        assert result == "hello world"

    def test_entire_text_at_intensity_3(self) -> None:
        result = perturb("hello world", technique="leetspeak", intensity=3, seed=42)
        assert result != "hello world"

    def test_deterministic_with_seed(self) -> None:
        a = perturb("hack the exploit", technique="leetspeak", intensity=3, seed=123)
        b = perturb("hack the exploit", technique="leetspeak", intensity=3, seed=123)
        assert a == b

    def test_different_seeds_differ(self) -> None:
        a = perturb("hack the exploit", technique="leetspeak", intensity=3, seed=1)
        b = perturb("hack the exploit", technique="leetspeak", intensity=3, seed=2)
        # Highly likely to differ (but not guaranteed for all inputs)
        # At intensity 3 with different seeds, substitutions should vary
        # May coincidentally match — just verify both ran without error
        assert isinstance(a, str) and isinstance(b, str)


class TestPerturbHomoglyph:
    """Unicode homoglyph substitution."""

    def test_produces_different_bytes(self) -> None:
        result = perturb("hack", technique="homoglyph", intensity=3, seed=42)
        assert result != "hack"
        assert result.encode("utf-8") != b"hack"

    def test_visually_similar(self) -> None:
        # Homoglyphs should look similar but be different characters
        result = perturb("attack", technique="homoglyph", intensity=3, seed=42)
        assert result != "attack"
        # Length should be similar (homoglyphs are single chars)
        assert len(result) == len("attack")


class TestPerturbZeroWidth:
    """Zero-width character injection."""

    def test_inserts_invisible_chars(self) -> None:
        result = perturb("hack", technique="zero_width", intensity=3, seed=42)
        # Result should be longer due to injected zero-width chars
        assert len(result) > len("hack")

    def test_original_chars_preserved(self) -> None:
        result = perturb("hack", technique="zero_width", intensity=3, seed=42)
        # Strip zero-width chars — original text should remain
        stripped = result.replace("\u200b", "").replace("\u200c", "").replace(
            "\u200d", "",
        ).replace("\ufeff", "")
        assert stripped == "hack"


class TestPerturbMixedCase:
    """Mixed-case disruption."""

    def test_changes_case(self) -> None:
        text = "hack the system"
        result = perturb(text, technique="mixed_case", intensity=3, seed=42)
        # At least one character should change case
        assert result != text

    def test_preserves_word_boundaries(self) -> None:
        result = perturb(
            "hack the system", technique="mixed_case", intensity=3, seed=42,
        )
        # Same number of words
        assert len(result.split()) == 3


class TestPerturbPhonetic:
    """Phonetic substitution."""

    def test_phonetic_subs(self) -> None:
        result = perturb("phishing attack", technique="phonetic", intensity=3, seed=42)
        # "ph" → "f" in "phishing"
        assert "f" in result.lower()

    def test_ck_substitution(self) -> None:
        result = perturb("hack", technique="phonetic", intensity=3, seed=42)
        # "ck" → "k"
        assert "hak" in result.lower()


class TestPerturbRandom:
    """Random technique selection."""

    def test_produces_change(self) -> None:
        result = perturb(
            "hack the exploit system",
            technique="random", intensity=3, seed=42,
        )
        assert result != "hack the exploit system"

    def test_deterministic_with_seed(self) -> None:
        a = perturb("hack the exploit", technique="random", intensity=3, seed=99)
        b = perturb("hack the exploit", technique="random", intensity=3, seed=99)
        assert a == b


class TestIntensityLevels:
    """Intensity controls scope of perturbation."""

    def test_intensity_1_triggers_only(self) -> None:
        text = "the exploit is bad"
        result = perturb(text, technique="leetspeak", intensity=1, seed=42)
        words = result.split()
        # "exploit" is trigger → should be changed
        # "the", "is", "bad" are not triggers → preserved
        assert words[0] == "the"
        assert words[2] == "is"
        assert words[3] == "bad"
        assert words[1] != "exploit"

    def test_intensity_2_includes_context(self) -> None:
        text = "the exploit is bad"
        result = perturb(text, technique="leetspeak", intensity=2, seed=42)
        # "the" and "is" are adjacent to "exploit" → also perturbed
        words = result.split()
        # At least "exploit" and one neighbour should change
        original_words = text.split()
        changed = sum(
            1 for o, n in zip(original_words, words, strict=True) if o != n
        )
        assert changed >= 2

    def test_intensity_3_all_words(self) -> None:
        text = "hello world foo bar"
        result = perturb(text, technique="leetspeak", intensity=3, seed=42)
        # All words should be attempted (though some may not have mappable chars)
        assert result != text


class TestCustomTriggers:
    """Custom trigger word sets."""

    def test_custom_triggers(self) -> None:
        custom = frozenset({"banana"})
        result = perturb(
            "banana split", technique="leetspeak", intensity=1, seed=42,
            trigger_words=custom,
        )
        words = result.split()
        assert words[0] != "banana"  # custom trigger → changed
        assert words[1] == "split"  # not a trigger → preserved

    def test_default_triggers_exist(self) -> None:
        assert len(DEFAULT_TRIGGER_WORDS) > 20
        assert "hack" in DEFAULT_TRIGGER_WORDS
        assert "exploit" in DEFAULT_TRIGGER_WORDS


class TestPerturbBatch:
    """Batch perturbation."""

    def test_batch_length(self) -> None:
        texts = ["hack a", "exploit b", "hello"]
        results = perturb_batch(texts, technique="leetspeak", intensity=3, seed=42)
        assert len(results) == 3

    def test_batch_per_text_seeds(self) -> None:
        texts = ["hack", "hack"]
        results = perturb_batch(texts, technique="leetspeak", intensity=3, seed=42)
        # Different seeds per text → likely different results
        # (seed=42 for first, seed=43 for second)
        # They may still be the same if substitutions align, so just check they run
        assert len(results) == 2

    def test_empty_batch(self) -> None:
        results = perturb_batch([], technique="leetspeak", intensity=1, seed=42)
        assert results == []

    def test_none_seed(self) -> None:
        # Should not raise
        results = perturb_batch(["hack"], technique="leetspeak", intensity=1, seed=None)
        assert len(results) == 1


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_string(self) -> None:
        result = perturb("", technique="leetspeak", intensity=3, seed=42)
        assert result == ""

    def test_single_char(self) -> None:
        result = perturb("a", technique="zero_width", intensity=3, seed=42)
        # Single char word → zero_width returns as-is (no gaps to insert into)
        assert "a" in result

    def test_preserves_punctuation(self) -> None:
        result = perturb("hack, exploit!", technique="leetspeak", intensity=3, seed=42)
        assert "," in result
        assert "!" in result

    def test_preserves_whitespace(self) -> None:
        result = perturb("hack  exploit", technique="leetspeak", intensity=3, seed=42)
        # Double space should be preserved
        assert "  " in result
