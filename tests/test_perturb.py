# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Perturbation tests — consolidated via ordeal.

Coverage target: 100% (same as before).
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st
from ordeal.auto import fuzz

from vauban.perturb import DEFAULT_TRIGGER_WORDS, perturb, perturb_batch

_s = settings(max_examples=50, deadline=None)
_techniques = st.sampled_from([
    "leetspeak", "homoglyph", "zero_width",
    "mixed_case", "phonetic", "random",
])
_trigger = st.builds(
    lambda pre, tw, post: f"{pre} {tw} {post}",
    pre=st.text(min_size=2, max_size=15, alphabet="abcdefgh "),
    tw=st.sampled_from(["hack", "exploit", "malware", "phishing"]),
    post=st.text(min_size=2, max_size=15, alphabet="abcdefgh "),
)
_seeds = st.integers(min_value=0, max_value=2**31 - 1)


# -- 1. No crash on arbitrary inputs ---------------------------------------


class TestNoCrash:
    def test_fuzz(self) -> None:
        result = fuzz(perturb, max_examples=50)
        assert result.passed, result.summary()

    def test_unicode_edge_cases(self) -> None:
        cases = [
            "", "\x00", "\u200b\u200c\u200d", "\U0001f4a3" * 5,
            " " * 100, "a" * 10_000,
        ]
        for text in cases:
            for tech in ["leetspeak", "homoglyph", "zero_width",
                         "mixed_case", "phonetic", "random"]:
                assert isinstance(
                    perturb(text, technique=tech, intensity=3, seed=42),
                    str,
                )


# -- 2. Universal invariants -----------------------------------------------


class TestInvariants:
    @_s
    @given(text=_trigger, technique=_techniques, seed=_seeds)
    def test_deterministic(
        self, text: str, technique: str, seed: int,
    ) -> None:
        a = perturb(text, technique=technique, intensity=3, seed=seed)
        b = perturb(text, technique=technique, intensity=3, seed=seed)
        assert a == b

    @_s
    @given(technique=_techniques, intensity=st.sampled_from([1, 2, 3]))
    def test_empty_string_identity(
        self, technique: str, intensity: int,
    ) -> None:
        assert perturb("", technique=technique, intensity=intensity, seed=42) == ""

    @_s
    @given(text=_trigger, seed=_seeds)
    def test_homoglyph_preserves_length(
        self, text: str, seed: int,
    ) -> None:
        result = perturb(text, technique="homoglyph", intensity=3, seed=seed)
        assert len(result) == len(text)

    @_s
    @given(text=_trigger, seed=_seeds)
    def test_zero_width_never_shortens(
        self, text: str, seed: int,
    ) -> None:
        result = perturb(text, technique="zero_width", intensity=3, seed=seed)
        assert len(result) >= len(text)


# -- 3. Technique-specific behavior ----------------------------------------


class TestTechniques:
    def test_leetspeak_trigger_changed(self) -> None:
        result = perturb("hack the system", technique="leetspeak", intensity=1, seed=42)
        assert "hack" not in result.split()

    def test_leetspeak_non_trigger_preserved_at_1(self) -> None:
        result = perturb("hello world", technique="leetspeak", intensity=1, seed=42)
        assert result == "hello world"

    def test_homoglyph_different_bytes(self) -> None:
        result = perturb("hack", technique="homoglyph", intensity=3, seed=42)
        assert result.encode("utf-8") != b"hack"

    def test_zero_width_chars_strippable(self) -> None:
        result = perturb("hack", technique="zero_width", intensity=3, seed=42)
        stripped = result.replace("\u200b", "").replace(
            "\u200c", "",
        ).replace("\u200d", "").replace("\ufeff", "")
        assert stripped == "hack"

    def test_phonetic_substitutes(self) -> None:
        result = perturb("phishing", technique="phonetic", intensity=3, seed=42)
        assert "f" in result.lower()

    def test_random_changes_text(self) -> None:
        result = perturb("hack the exploit", technique="random", intensity=3, seed=42)
        assert result != "hack the exploit"


# -- 4. Intensity monotonicity --------------------------------------------


class TestIntensity:
    @_s
    @given(text=_trigger, seed=_seeds)
    def test_monotonic_word_coverage(self, text: str, seed: int) -> None:
        r1 = perturb(text, technique="leetspeak", intensity=1, seed=seed)
        r2 = perturb(text, technique="leetspeak", intensity=2, seed=seed)
        r3 = perturb(text, technique="leetspeak", intensity=3, seed=seed)
        orig = text.split()
        w1, w2, w3 = r1.split(), r2.split(), r3.split()
        if len(orig) == len(w1) == len(w2) == len(w3):
            c1 = {i for i, (o, n) in enumerate(zip(orig, w1, strict=False)) if o != n}
            c2 = {i for i, (o, n) in enumerate(zip(orig, w2, strict=False)) if o != n}
            c3 = {i for i, (o, n) in enumerate(zip(orig, w3, strict=False)) if o != n}
            assert c1 <= c2 <= c3


# -- 5. Batch + defaults ---------------------------------------------------


class TestBatch:
    @_s
    @given(
        texts=st.lists(_trigger, min_size=1, max_size=5),
        technique=_techniques, seed=_seeds,
    )
    def test_batch_length(
        self, texts: list[str], technique: str, seed: int,
    ) -> None:
        assert len(perturb_batch(
            texts, technique=technique, intensity=3, seed=seed,
        )) == len(texts)

    def test_empty_batch(self) -> None:
        assert perturb_batch([], technique="leetspeak", intensity=1, seed=42) == []

    def test_default_triggers_exist(self) -> None:
        assert len(DEFAULT_TRIGGER_WORDS) > 20
        assert "hack" in DEFAULT_TRIGGER_WORDS
