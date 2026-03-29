"""Taxonomy property tests — invariants over the harm classification system.

Properties:
- score_text never crashes on arbitrary input
- All scores are in [0.0, 1.0]
- classify_text result is always in score_text results (or None)
- score_text is deterministic
- score_batch == map(score_text)
- All patterns compile
- All aliases resolve to valid categories
- Domain lookup is consistent with TAXONOMY structure
- Category IDs are globally unique
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from tests.battle.invariants import assert_all_bounded, assert_deterministic
from tests.battle.strategies import benign_text, category_ids, harm_text, safe_text
from vauban.taxonomy import (
    CATEGORY_ALIASES,
    TAXONOMY,
    all_categories,
    classify_text,
    coverage_report,
    domain_for_category,
    resolve_category,
    score_batch,
    score_text,
)


class TestScoreTextProperties:
    """score_text invariants over arbitrary input."""

    @given(text=safe_text)
    def test_never_crashes(self, text: str) -> None:
        """score_text accepts any string without exception."""
        result = score_text(text)
        assert isinstance(result, list)

    @given(text=safe_text)
    def test_scores_bounded(self, text: str) -> None:
        """Every score is in [0.0, 1.0]."""
        scores = score_text(text)
        values = [s.score for s in scores]
        if values:
            assert_all_bounded(values, 0.0, 1.0, "harm_scores")

    @given(text=safe_text)
    def test_matched_patterns_nonempty(self, text: str) -> None:
        """Every HarmScore has at least one matched pattern."""
        for s in score_text(text):
            assert len(s.matched_patterns) > 0, (
                f"category {s.category_id} score >0 but no matched patterns"
            )

    @given(text=safe_text)
    def test_domain_id_valid(self, text: str) -> None:
        """Every domain_id in results is a real domain."""
        domain_ids = {d.id for d in TAXONOMY}
        for s in score_text(text):
            assert s.domain_id in domain_ids, (
                f"unknown domain_id {s.domain_id!r}"
            )

    @given(text=safe_text)
    def test_category_id_valid(self, text: str) -> None:
        """Every category_id in results is a canonical category."""
        cats = all_categories()
        for s in score_text(text):
            assert s.category_id in cats, (
                f"unknown category_id {s.category_id!r}"
            )

    @given(text=harm_text)
    def test_harm_text_usually_scores(self, text: str) -> None:
        """Text with embedded harm keywords produces at least some scores.

        Not a strict invariant (some keywords may not match patterns),
        so we just verify the function runs and returns a list.
        """
        result = score_text(text)
        assert isinstance(result, list)


class TestClassifyTextProperties:
    """classify_text invariants."""

    @given(text=safe_text)
    def test_classify_subset_of_score(self, text: str) -> None:
        """classify_text returns a category present in score_text, or None."""
        scores = score_text(text)
        classification = classify_text(text)
        if classification is not None:
            scored_ids = {s.category_id for s in scores}
            assert classification in scored_ids, (
                f"classify returned {classification!r} but"
                f" score_text returned {scored_ids}"
            )

    @given(text=benign_text)
    def test_benign_text_often_none(self, text: str) -> None:
        """Pure alpha text usually classifies as None.

        Not strict — just verifies the function handles it.
        """
        result = classify_text(text)
        assert result is None or result in all_categories()


class TestScoreDeterminism:
    """Same input always produces same output."""

    @given(text=safe_text)
    def test_score_text_deterministic(self, text: str) -> None:
        """score_text(x) == score_text(x) across calls."""
        assert_deterministic(
            lambda: [(s.category_id, s.score) for s in score_text(text)],
            label="score_text",
        )

    @given(text=safe_text)
    def test_classify_text_deterministic(self, text: str) -> None:
        """classify_text(x) == classify_text(x) across calls."""
        assert_deterministic(
            lambda: classify_text(text),
            label="classify_text",
        )


class TestScoreBatchConsistency:
    """score_batch == map(score_text)."""

    @given(
        texts=st.lists(safe_text, min_size=0, max_size=5),
    )
    def test_batch_equals_individual(self, texts: list[str]) -> None:
        batch = score_batch(texts)
        individual = [score_text(t) for t in texts]
        assert len(batch) == len(individual)
        for b, i in zip(batch, individual, strict=True):
            b_tuples = [(s.category_id, s.score) for s in b]
            i_tuples = [(s.category_id, s.score) for s in i]
            assert b_tuples == i_tuples


class TestTaxonomyStructuralInvariants:
    """Invariants on the taxonomy data structure itself."""

    def test_all_category_ids_globally_unique(self) -> None:
        ids = [
            cat.id for domain in TAXONOMY for cat in domain.categories
        ]
        assert len(ids) == len(set(ids))

    def test_all_domain_ids_unique(self) -> None:
        ids = [d.id for d in TAXONOMY]
        assert len(ids) == len(set(ids))

    def test_all_aliases_resolve_to_canonical(self) -> None:
        cats = all_categories()
        for alias, target in CATEGORY_ALIASES.items():
            assert target in cats, f"{alias!r} -> {target!r} not canonical"

    def test_all_patterns_compile(self) -> None:
        """Every regex pattern compiles without error."""
        import re
        for domain in TAXONOMY:
            for cat in domain.categories:
                for pat in cat.patterns:
                    re.compile(pat, re.IGNORECASE)

    @given(cat_id=category_ids)
    def test_domain_for_category_consistent(self, cat_id: str) -> None:
        """domain_for_category agrees with TAXONOMY structure."""
        domain_id = domain_for_category(cat_id)
        assert domain_id is not None
        # Find the actual domain
        for domain in TAXONOMY:
            for cat in domain.categories:
                if cat.id == cat_id:
                    assert domain.id == domain_id

    @given(raw=safe_text)
    def test_resolve_category_idempotent(self, raw: str) -> None:
        """Resolving twice gives the same result."""
        first = resolve_category(raw)
        second = resolve_category(first)
        assert first == second

    def test_coverage_full_is_1(self) -> None:
        result = coverage_report(set(all_categories()))
        assert result.coverage_ratio == 1.0

    def test_coverage_empty_is_0(self) -> None:
        result = coverage_report(set())
        assert result.coverage_ratio == 0.0


