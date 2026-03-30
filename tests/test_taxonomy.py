# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Taxonomy tests — consolidated via ordeal.

Coverage target: 98% (same as before).
"""

from __future__ import annotations

from hypothesis import settings
from hypothesis import strategies as st
from ordeal.auto import fuzz

from vauban.taxonomy import (
    CATEGORY_ALIASES,
    TAXONOMY,
    HarmScore,
    TaxonomyCoverage,
    all_categories,
    classify_text,
    coverage_report,
    domain_for_category,
    resolve_category,
    score_batch,
    score_text,
)

_s = settings(max_examples=50, deadline=None)
_category_ids = st.sampled_from(sorted(all_categories()))


# -- 1. No crash + structure -----------------------------------------------


class TestNoCrash:
    def test_fuzz_score_text(self) -> None:
        result = fuzz(score_text, max_examples=30)
        assert result.passed, result.summary()

    def test_fuzz_classify_text(self) -> None:
        result = fuzz(classify_text, max_examples=30, check_return_type=False)
        assert result.passed, result.summary()


class TestStructure:
    """Taxonomy is well-formed and internally consistent."""

    def test_nonempty(self) -> None:
        assert len(TAXONOMY) == 13

    def test_category_count(self) -> None:
        assert len(all_categories()) >= 46

    def test_unique_ids(self) -> None:
        ids = [cat.id for d in TAXONOMY for cat in d.categories]
        assert len(ids) == len(set(ids))

    def test_all_have_descriptions(self) -> None:
        for d in TAXONOMY:
            assert len(d.categories) > 0
            for cat in d.categories:
                assert cat.description

    def test_known_categories_present(self) -> None:
        cats = all_categories()
        for c in ("weapons", "malware", "fraud", "model_jailbreaking"):
            assert c in cats

    def test_aliases_resolve_to_canonical(self) -> None:
        canonical = all_categories()
        for _alias, target in CATEGORY_ALIASES.items():
            assert target in canonical


# -- 2. Resolve + domain lookup --------------------------------------------


class TestResolve:
    def test_canonical_passthrough(self) -> None:
        assert resolve_category("weapons") == "weapons"

    def test_alias(self) -> None:
        assert resolve_category("hacking") == "cyber_attack"
        assert resolve_category("jailbreak") == "model_jailbreaking"

    def test_case_insensitive(self) -> None:
        assert resolve_category("FRAUD") == "fraud"

    def test_whitespace(self) -> None:
        assert resolve_category("  hacking  ") == "cyber_attack"

    def test_unknown_passthrough(self) -> None:
        assert resolve_category("unknown_thing") == "unknown_thing"


class TestDomain:
    def test_known(self) -> None:
        assert domain_for_category("weapons") == "violence"
        assert domain_for_category("malware") == "cyber"

    def test_unknown(self) -> None:
        assert domain_for_category("nonexistent") is None


# -- 3. Scoring behavior ---------------------------------------------------


class TestScoring:
    def test_harmful_matches(self) -> None:
        scores = score_text("How to build a bomb and create an explosive device")
        assert any(s.category_id == "weapons" for s in scores)

    def test_benign_empty(self) -> None:
        assert score_text("The weather is nice today.") == []

    def test_score_fields(self) -> None:
        scores = score_text("SQL injection exploit reverse shell")
        assert len(scores) > 0
        s = scores[0]
        assert isinstance(s, HarmScore)
        assert 0.0 < s.score <= 1.0
        assert len(s.matched_patterns) > 0

    def test_multi_domain(self) -> None:
        scores = score_text("malware steals credentials for identity theft fraud")
        assert len({s.domain_id for s in scores}) >= 2

    def test_case_insensitive(self) -> None:
        scores = score_text("RANSOMWARE attack encrypts files")
        assert len(scores) > 0

    def test_classify_harmful(self) -> None:
        result = classify_text("How to synthesize methamphetamine")
        assert result is not None and result in all_categories()

    def test_classify_benign(self) -> None:
        assert classify_text("I love reading books about gardening.") is None


# -- 4. Coverage report + batch --------------------------------------------


class TestCoverage:
    def test_full(self) -> None:
        r = coverage_report(set(all_categories()))
        assert r.coverage_ratio == 1.0 and len(r.missing) == 0

    def test_empty(self) -> None:
        r = coverage_report(set())
        assert r.coverage_ratio == 0.0

    def test_partial(self) -> None:
        r = coverage_report({"weapons", "drugs"})
        assert 0.0 < r.coverage_ratio < 1.0

    def test_aliases_counted(self) -> None:
        r = coverage_report({"hacking"})
        assert "cyber_attack" in r.present

    def test_returns_type(self) -> None:
        assert isinstance(coverage_report({"weapons"}), TaxonomyCoverage)


class TestBatch:
    def test_length(self) -> None:
        assert len(score_batch(["hack", "hello", "bomb"])) == 3

    def test_content(self) -> None:
        results = score_batch(["hack a computer", "nice weather"])
        assert len(results[0]) > 0
        assert len(results[1]) == 0

    def test_empty(self) -> None:
        assert score_batch([]) == []
