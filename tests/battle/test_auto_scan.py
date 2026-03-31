# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Auto-scan tests via ordeal.auto — zero-boilerplate coverage.

Uses ordeal's scan_module, fuzz, and chaos_for to auto-test
vauban modules without hand-written test logic. Fixtures are
provided for domain-specific types; ordeal handles the rest.
"""

from __future__ import annotations

import hypothesis.strategies as st
from ordeal.auto import chaos_for, fuzz, scan_module
from ordeal.invariants import bounded

# ---------------------------------------------------------------------------
# Fixtures for vauban-specific parameter names
# ---------------------------------------------------------------------------

# Shared fixture strategies that match common parameter names across modules
_TEXT_FIXTURES: dict[str, st.SearchStrategy[object]] = {
    "text": st.text(min_size=1, max_size=200),
    "prompt": st.text(min_size=1, max_size=200),
    "response": st.text(min_size=1, max_size=500),
    "name_or_id": st.sampled_from([
        "violence", "cyber", "sexual", "fraud",
        "hate", "self_harm", "weapons", "drugs",
    ]),
    "category_id": st.sampled_from([
        "violence", "cyber", "sexual", "fraud",
        "hate", "self_harm", "weapons", "drugs",
    ]),
    "technique": st.sampled_from([
        "leetspeak", "homoglyph", "zero_width",
        "mixed_case", "phonetic", "random",
    ]),
    "intensity": st.sampled_from([1, 2, 3]),
    "seed": st.integers(min_value=0, max_value=2**31 - 1),
}


# ---------------------------------------------------------------------------
# 1. scan_module — smoke-test public APIs
# ---------------------------------------------------------------------------


class TestScanPerturb:
    """ordeal auto-scans vauban.perturb — should pass clean."""

    def test_scan(self) -> None:
        result = scan_module(
            "vauban.perturb",
            max_examples=20,
            fixtures=_TEXT_FIXTURES,
        )
        for f in result.functions:
            assert f.passed, f"perturb.{f.name}: {f.error}"


class TestScanScoring:
    """ordeal auto-scans vauban.scoring."""

    def test_score_response_no_crash(self) -> None:
        result = fuzz(
            _get_fn("vauban.scoring", "score_response"),
            max_examples=30,
            prompt=st.text(min_size=1, max_size=100),
            response=st.text(min_size=0, max_size=300),
        )
        assert result.passed, result.summary()

    def test_score_response_axes_bounded(self) -> None:
        """Every axis of score_response is in [0, 1]."""
        from vauban.scoring import score_response

        check = bounded(0.0, 1.0)

        @st.composite
        def inputs(draw: st.DrawFn) -> tuple[str, str]:
            p = draw(st.text(min_size=1, max_size=100))
            r = draw(st.text(min_size=0, max_size=300))
            return p, r

        from hypothesis import given, settings

        @given(data=inputs())
        @settings(max_examples=30, deadline=None)
        def _test(data: tuple[str, str]) -> None:
            p, r = data
            result = score_response(p, r)
            check(result.composite, name="composite")
            check(result.length, name="length")
            check(result.structure, name="structure")
            check(result.anti_refusal, name="anti_refusal")
            check(result.directness, name="directness")
            check(result.relevance, name="relevance")

        _test()


class TestScanTaxonomy:
    """ordeal auto-scans vauban.taxonomy."""

    def test_score_text_no_crash(self) -> None:
        result = fuzz(
            _get_fn("vauban.taxonomy", "score_text"),
            max_examples=30,
            text=st.text(min_size=0, max_size=500),
        )
        assert result.passed, result.summary()

    def test_classify_text_no_crash(self) -> None:
        result = fuzz(
            _get_fn("vauban.taxonomy", "classify_text"),
            max_examples=30,
            text=st.text(min_size=0, max_size=500),
        )
        assert result.passed, result.summary()

    def test_all_categories_stable(self) -> None:
        from vauban.taxonomy import all_categories

        cats = all_categories()
        assert len(cats) > 0
        # Deterministic
        assert all_categories() == cats


# ---------------------------------------------------------------------------
# 2. fuzz — deep-fuzz individual functions
# ---------------------------------------------------------------------------


class TestFuzzPerturb:
    """Deep-fuzz vauban.perturb.perturb with boundary inputs."""

    def test_fuzz_all_techniques(self) -> None:
        from vauban.perturb import perturb

        result = fuzz(
            perturb,
            max_examples=100,
            text=st.text(min_size=1, max_size=200),
            technique=st.sampled_from([
                "leetspeak", "homoglyph", "zero_width",
                "mixed_case", "phonetic", "random",
            ]),
            intensity=st.sampled_from([1, 2, 3]),
            seed=st.integers(min_value=0, max_value=2**31 - 1),
        )
        assert result.passed, result.summary()


class TestFuzzScoring:
    """Deep-fuzz scoring with adversarial inputs."""

    def test_fuzz_score_response(self) -> None:
        from vauban.scoring import score_response

        result = fuzz(
            score_response,
            max_examples=200,
            prompt=st.text(min_size=0, max_size=500),
            response=st.text(min_size=0, max_size=1000),
        )
        assert result.passed, result.summary()


# ---------------------------------------------------------------------------
# 3. chaos_for — auto-generated stateful tests
# ---------------------------------------------------------------------------


class TestChaosScoring:
    """Deep-fuzz score_response (the single-pair API)."""

    def test_scoring_fuzz_deep(self) -> None:
        from vauban.scoring import score_response

        result = fuzz(
            score_response,
            max_examples=100,
            prompt=st.text(min_size=1, max_size=200),
            response=st.text(min_size=0, max_size=500),
        )
        assert result.passed, result.summary()


class TestChaosTaxonomy:
    """ChaosTest auto-generated from vauban.taxonomy."""

    def test_taxonomy_chaos(self) -> None:
        test_case = chaos_for(
            "vauban.taxonomy",
            fixtures={
                "text": st.text(min_size=0, max_size=200),
                "texts": st.lists(
                    st.text(min_size=1, max_size=100),
                    min_size=1, max_size=5,
                ),
                "name_or_id": st.sampled_from([
                    "violence", "cyber", "sexual", "fraud",
                ]),
                "category_id": st.sampled_from([
                    "violence", "cyber", "sexual",
                ]),
            },
            max_examples=10,
            stateful_step_count=10,
        )
        t = test_case("runTest")
        t.runTest()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_fn(module: str, name: str) -> object:
    """Import and return a function by dotted path."""
    import importlib

    mod = importlib.import_module(module)
    return getattr(mod, name)
