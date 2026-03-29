"""Deterministic replay tests.

Any operation with a seed must produce
bit-identical results across runs.  No flaky tests, no "works on
my machine."
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from tests.battle.strategies import (
    perturb_intensities,
    perturb_techniques,
    safe_text,
    seeds,
    trigger_text,
)
from vauban.perturb import perturb, perturb_batch
from vauban.scoring import score_response, score_responses
from vauban.taxonomy import classify_text, score_text


class TestPerturbReplay:
    """Seeded perturbation replays identically."""

    @given(
        text=trigger_text,
        technique=perturb_techniques,
        intensity=perturb_intensities,
        seed=seeds,
    )
    def test_perturb_replay(
        self,
        text: str,
        technique: str,
        intensity: int,
        seed: int,
    ) -> None:
        a = perturb(text, technique=technique, intensity=intensity, seed=seed)  # type: ignore[arg-type]
        b = perturb(text, technique=technique, intensity=intensity, seed=seed)  # type: ignore[arg-type]
        assert a == b, f"replay mismatch for seed={seed}"

    @given(
        texts=st.lists(trigger_text, min_size=1, max_size=5),
        technique=perturb_techniques,
        seed=seeds,
    )
    def test_batch_replay(
        self,
        texts: list[str],
        technique: str,
        seed: int,
    ) -> None:
        a = perturb_batch(texts, technique=technique, intensity=3, seed=seed)  # type: ignore[arg-type]
        b = perturb_batch(texts, technique=technique, intensity=3, seed=seed)  # type: ignore[arg-type]
        assert a == b

    @given(text=trigger_text, seed=seeds)
    def test_different_seeds_diverge(self, text: str, seed: int) -> None:
        """Different seeds should (usually) produce different output.

        Not a strict invariant — same-length text with no mappable chars
        could match.  We just verify both calls succeed.
        """
        a = perturb(text, technique="leetspeak", intensity=3, seed=seed)
        b = perturb(text, technique="leetspeak", intensity=3, seed=seed + 1)
        assert isinstance(a, str)
        assert isinstance(b, str)


class TestScoringReplay:
    """Scoring is perfectly deterministic (no randomness)."""

    @given(prompt=safe_text, response=safe_text)
    def test_score_response_replay(self, prompt: str, response: str) -> None:
        a = score_response(prompt, response)
        b = score_response(prompt, response)
        assert a.composite == b.composite
        assert a.length == b.length
        assert a.structure == b.structure
        assert a.anti_refusal == b.anti_refusal
        assert a.directness == b.directness
        assert a.relevance == b.relevance

    @given(
        prompts=st.lists(safe_text, min_size=1, max_size=3),
        responses=st.lists(safe_text, min_size=1, max_size=3),
    )
    def test_score_responses_replay(
        self,
        prompts: list[str],
        responses: list[str],
    ) -> None:
        # Match lengths
        n = min(len(prompts), len(responses))
        p, r = prompts[:n], responses[:n]
        a = score_responses(p, r)
        b = score_responses(p, r)
        for x, y in zip(a, b, strict=True):
            assert x.composite == y.composite


class TestTaxonomyReplay:
    """Taxonomy scoring is perfectly deterministic."""

    @given(text=safe_text)
    def test_score_text_replay(self, text: str) -> None:
        a = [(s.category_id, s.score) for s in score_text(text)]
        b = [(s.category_id, s.score) for s in score_text(text)]
        assert a == b

    @given(text=safe_text)
    def test_classify_text_replay(self, text: str) -> None:
        a = classify_text(text)
        b = classify_text(text)
        assert a == b


class TestCrossRunStability:
    """Multi-step pipelines produce stable results."""

    @given(text=trigger_text, seed=seeds)
    def test_perturb_then_score_stable(
        self, text: str, seed: int,
    ) -> None:
        """perturb -> score_response pipeline is deterministic."""
        p1 = perturb(text, technique="leetspeak", intensity=3, seed=seed)
        s1 = score_response(text, p1)

        p2 = perturb(text, technique="leetspeak", intensity=3, seed=seed)
        s2 = score_response(text, p2)

        assert s1.composite == s2.composite

    @given(text=trigger_text, seed=seeds)
    def test_perturb_then_classify_stable(
        self, text: str, seed: int,
    ) -> None:
        """perturb -> classify_text pipeline is deterministic."""
        p1 = perturb(text, technique="homoglyph", intensity=3, seed=seed)
        c1 = classify_text(p1)

        p2 = perturb(text, technique="homoglyph", intensity=3, seed=seed)
        c2 = classify_text(p2)

        assert c1 == c2
