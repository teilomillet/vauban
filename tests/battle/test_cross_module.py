# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Cross-module consistency tests.

Module composition testing: individual
modules may be correct in isolation but fail when composed.
These tests verify properties that span module boundaries.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from tests.battle.strategies import (
    harm_text,
    perturb_techniques,
    prompt_text,
    seeds,
    trigger_text,
)
from vauban.jailbreak import (
    ALL_STRATEGIES,
    apply_templates,
    filter_by_strategy,
    load_templates,
)
from vauban.perturb import perturb
from vauban.scoring import score_response
from vauban.taxonomy import classify_text, score_text

# ── Perturb x Taxonomy ────────────────────────────────────────────────


class TestPerturbTaxonomyComposition:
    """How does perturbation affect taxonomy classification?"""

    @given(text=harm_text, seed=seeds)
    def test_perturbed_text_still_classifiable(
        self, text: str, seed: int,
    ) -> None:
        """Perturbed text should still be processable by taxonomy."""
        perturbed = perturb(text, technique="leetspeak", intensity=3, seed=seed)
        result = score_text(perturbed)
        assert isinstance(result, list)
        # No crash — that's the invariant

    @given(text=harm_text, seed=seeds)
    def test_homoglyph_may_evade_taxonomy(
        self, text: str, seed: int,
    ) -> None:
        """Homoglyph perturbation may reduce taxonomy scores.

        This is an *expected* evasion — the point of the perturb module
        is to test whether defenses handle this.  We verify no crash.
        """
        original_scores = score_text(text)
        perturbed = perturb(text, technique="homoglyph", intensity=3, seed=seed)
        perturbed_scores = score_text(perturbed)
        # Both must succeed — we don't assert score equality
        assert isinstance(original_scores, list)
        assert isinstance(perturbed_scores, list)

    @given(text=harm_text, technique=perturb_techniques, seed=seeds)
    def test_classify_never_crashes_on_perturbed(
        self, text: str, technique: str, seed: int,
    ) -> None:
        perturbed = perturb(text, technique=technique, intensity=3, seed=seed)  # type: ignore[arg-type]
        result = classify_text(perturbed)
        assert result is None or isinstance(result, str)


# ── Perturb x Scoring ─────────────────────────────────────────────────


class TestPerturbScoringComposition:
    """Perturbation effects on response scoring."""

    @given(
        prompt=prompt_text,
        response=trigger_text,
        seed=seeds,
    )
    def test_perturbed_response_scoreable(
        self, prompt: str, response: str, seed: int,
    ) -> None:
        """Perturbed response text scores without crash."""
        perturbed = perturb(response, technique="random", intensity=3, seed=seed)
        result = score_response(prompt, perturbed)
        assert 0.0 <= result.length <= 1.0
        assert 0.0 <= result.anti_refusal <= 1.0

    @given(prompt=trigger_text, seed=seeds)
    def test_perturbed_prompt_scoreable(
        self, prompt: str, seed: int,
    ) -> None:
        """Perturbed prompt text works as scoring input."""
        perturbed = perturb(prompt, technique="mixed_case", intensity=3, seed=seed)
        result = score_response(perturbed, "The answer is yes.")
        assert result.composite >= 0.0


# ── Jailbreak x Taxonomy ──────────────────────────────────────────────


class TestJailbreakTaxonomyComposition:
    """Jailbreak templates carry harm signals detectable by taxonomy."""

    def test_expanded_templates_classifiable(self) -> None:
        """Expanded jailbreak prompts don't crash taxonomy."""
        templates = load_templates()[:5]
        payloads = ["How to hack a computer", "Build a weapon"]
        expanded = apply_templates(templates, payloads)
        for _, prompt in expanded:
            result = score_text(prompt)
            assert isinstance(result, list)

    @given(
        strategy=st.sampled_from(sorted(ALL_STRATEGIES)),
    )
    def test_strategy_templates_valid(self, strategy: str) -> None:
        """Each strategy's templates have valid {payload} placeholders."""
        templates = load_templates()
        filtered = filter_by_strategy(templates, [strategy])
        assert len(filtered) > 0
        for t in filtered:
            assert "{payload}" in t.template


# ── Jailbreak x Scoring ──────────────────────────────────────────────


class TestJailbreakScoringComposition:
    """Jailbreak-expanded prompts score without crash."""

    def test_expanded_prompts_scoreable(self) -> None:
        templates = load_templates()[:3]
        payloads = ["Tell me how to pick a lock"]
        expanded = apply_templates(templates, payloads)
        for _, prompt in expanded:
            result = score_response(prompt, "I cannot help with that.")
            assert result.anti_refusal < 1.0  # should detect refusal

    def test_compliant_response_scores_high(self) -> None:
        templates = load_templates()[:3]
        payloads = ["Explain photosynthesis"]
        expanded = apply_templates(templates, payloads)
        for _, prompt in expanded:
            result = score_response(
                prompt,
                "Photosynthesis converts sunlight into chemical energy.",
            )
            assert result.anti_refusal == 1.0


# ── Jailbreak x Perturb ──────────────────────────────────────────────


class TestJailbreakPerturbComposition:
    """Perturbation of jailbreak-expanded prompts."""

    @given(seed=seeds)
    def test_perturb_jailbreak_prompt(self, seed: int) -> None:
        """Perturbing a jailbreak prompt doesn't crash."""
        templates = load_templates()[:2]
        payloads = ["hack the system"]
        expanded = apply_templates(templates, payloads)
        for _, prompt in expanded:
            result = perturb(prompt, technique="leetspeak", intensity=3, seed=seed)
            assert isinstance(result, str)
            assert len(result) > 0


# ── Taxonomy x Scoring alignment ─────────────────────────────────────


class TestTaxonomyScoringAlignment:
    """Taxonomy harm detection and scoring refusal detection are complementary."""

    def test_refusal_text_high_anti_refusal_penalty(self) -> None:
        """Text with refusal phrases should score low on anti_refusal."""
        refusal = "I cannot help with that. I'm sorry, I must decline."
        result = score_response("request", refusal)
        assert result.anti_refusal < 0.5

    def test_harm_text_classifiable_and_scoreable(self) -> None:
        """Harmful text is both taxonomy-classifiable and response-scoreable."""
        text = "How to hack a computer using SQL injection exploits"
        # Taxonomy should detect it
        scores = score_text(text)
        assert len(scores) > 0
        # Scoring should work on it as a prompt
        result = score_response(
            text,
            "Here are the steps to hack a computer using SQL injection.",
        )
        assert result.relevance > 0.0


# ── Full pipeline composition ─────────────────────────────────────────


class TestFullPipelineComposition:
    """End-to-end: generate jailbreak prompt -> perturb -> score -> classify."""

    @given(seed=seeds)
    def test_full_pipeline_no_crash(self, seed: int) -> None:
        """The full composition never crashes."""
        templates = load_templates()[:2]
        payloads = ["exploit a vulnerability"]
        expanded = apply_templates(templates, payloads)

        for _, prompt in expanded:
            # Perturb
            perturbed = perturb(
                prompt, technique="random", intensity=2, seed=seed,
            )
            # Score as response
            score = score_response("test", perturbed)
            assert score.composite >= 0.0
            # Classify
            classification = classify_text(perturbed)
            assert classification is None or isinstance(classification, str)
