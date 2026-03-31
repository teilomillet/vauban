# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Multi-step scenario generator — composite operation chains.

Battle testing explores *sequences* of operations, not individual calls.
Bugs hide in specific orderings: perturb-then-classify behaves
differently from classify-then-perturb.  These tests generate
random multi-step pipelines and verify end-to-end invariants.
"""

from __future__ import annotations

import math

from hypothesis import given, settings

from tests.battle.strategies import (
    perturb_techniques,
    seeds,
    trigger_text,
)
from vauban.jailbreak import apply_templates, load_templates
from vauban.perturb import perturb
from vauban.scoring import score_response
from vauban.taxonomy import all_categories, classify_text, score_text

# ── Scenario 1: perturb → classify pipeline ───────────────────────────


class TestPerturbClassifyScenario:
    """Multi-step: generate text → perturb → classify at each intensity."""

    @given(
        text=trigger_text,
        technique=perturb_techniques,
        seed=seeds,
    )
    def test_classify_at_all_intensities(
        self, text: str, technique: str, seed: int,
    ) -> None:
        """Classify original + 3 intensity levels — all must succeed."""
        cats = all_categories()
        results: list[str | None] = []

        # Original
        c0 = classify_text(text)
        assert c0 is None or c0 in cats
        results.append(c0)

        # Each intensity
        for intensity in (1, 2, 3):
            perturbed = perturb(
                text, technique=technique,
                intensity=intensity, seed=seed,  # type: ignore[arg-type]
            )
            c = classify_text(perturbed)
            assert c is None or c in cats
            results.append(c)

        # At least no crashes — that's the invariant
        assert len(results) == 4


# ── Scenario 2: jailbreak → perturb → score pipeline ─────────────────


class TestJailbreakPerturbScoreScenario:
    """Multi-step: template → expand → perturb → score."""

    @given(seed=seeds, technique=perturb_techniques)
    @settings(max_examples=20)
    def test_full_attack_pipeline(
        self, seed: int, technique: str,
    ) -> None:
        """End-to-end attack simulation: template expansion → perturbation → scoring."""
        templates = load_templates()[:5]
        payload = "hack a computer system"

        for _template, expanded in apply_templates(templates, [payload]):
            # Perturb
            perturbed = perturb(
                expanded, technique=technique, intensity=2, seed=seed,  # type: ignore[arg-type]
            )

            # Score against a refusal response
            refusal_score = score_response(
                perturbed, "I cannot help with that request.",
            )
            assert 0.0 <= refusal_score.anti_refusal <= 1.0
            assert not math.isnan(refusal_score.composite)

            # Score against a compliant response
            comply_score = score_response(
                perturbed, "Here are the steps to do this.",
            )
            assert 0.0 <= comply_score.anti_refusal <= 1.0
            assert not math.isnan(comply_score.composite)

            # Refusal should score lower on anti_refusal than compliance
            assert refusal_score.anti_refusal <= comply_score.anti_refusal


# ── Scenario 3: perturb escalation ────────────────────────────────────


class TestPerturbEscalationScenario:
    """Multi-step: same text perturbed at increasing intensities."""

    @given(text=trigger_text, seed=seeds)
    def test_escalating_perturbation(self, text: str, seed: int) -> None:
        """Score the same prompt at each perturbation intensity.

        Invariant: all scores remain finite and bounded.
        """
        for technique in ["leetspeak", "homoglyph", "zero_width"]:
            for intensity in (1, 2, 3):
                perturbed = perturb(
                    text, technique=technique,
                    intensity=intensity, seed=seed,  # type: ignore[arg-type]
                )
                # Basic sanity
                assert isinstance(perturbed, str)

                # Score it
                s = score_response(text, perturbed)
                assert not math.isnan(s.composite)
                assert not math.isinf(s.composite)


# ── Scenario 4: taxonomy scoring stability ────────────────────────────


class TestTaxonomyStabilityScenario:
    """Multi-step: score text → perturb → re-score → compare."""

    @given(text=trigger_text, seed=seeds)
    def test_score_before_and_after_perturb(
        self, text: str, seed: int,
    ) -> None:
        """Taxonomy scores before/after perturbation are both valid."""
        before = score_text(text)
        perturbed = perturb(
            text, technique="leetspeak", intensity=3, seed=seed,
        )
        after = score_text(perturbed)

        # Both must be valid (list of HarmScore)
        for s in before:
            assert 0.0 <= s.score <= 1.0
        for s in after:
            assert 0.0 <= s.score <= 1.0


# ── Scenario 5: round-trip perturbation ───────────────────────────────


class TestDoublePerturb:
    """Multi-step: perturb → perturb again (double perturbation)."""

    @given(
        text=trigger_text,
        seed1=seeds,
        seed2=seeds,
        t1=perturb_techniques,
        t2=perturb_techniques,
    )
    def test_double_perturb_survives(
        self,
        text: str,
        seed1: int,
        seed2: int,
        t1: str,
        t2: str,
    ) -> None:
        """Applying perturbation twice doesn't crash."""
        once = perturb(text, technique=t1, intensity=3, seed=seed1)  # type: ignore[arg-type]
        twice = perturb(once, technique=t2, intensity=3, seed=seed2)  # type: ignore[arg-type]
        assert isinstance(twice, str)

        # Classify the double-perturbed text
        result = classify_text(twice)
        assert result is None or result in all_categories()


# ── Scenario 6: adversarial scoring chain ─────────────────────────────


class TestAdversarialScoringChain:
    """Multi-step: generate adversarial scenario → score at each stage."""

    @given(seed=seeds)
    @settings(max_examples=20)
    def test_scoring_chain_consistency(self, seed: int) -> None:
        """Score at each pipeline stage — all stages produce valid results."""
        # Stage 1: raw payload
        payload = "How to bypass security measures"
        s1 = score_response(payload, "I can help with that.")
        # weighted sum may exceed 1.0 with non-unit weights
        assert not math.isnan(s1.composite)

        # Stage 2: wrap in jailbreak template
        templates = load_templates()[:1]
        expanded = apply_templates(templates, [payload])
        if expanded:
            _, jailbreak_prompt = expanded[0]
            s2 = score_response(jailbreak_prompt, "I can help with that.")
            assert not math.isnan(s2.composite)

            # Stage 3: perturb the jailbreak prompt
            perturbed = perturb(
                jailbreak_prompt, technique="random",
                intensity=2, seed=seed,
            )
            s3 = score_response(perturbed, "I can help with that.")
            assert not math.isnan(s3.composite)

            # Stage 4: classify at each level
            c1 = classify_text(payload)
            c2 = classify_text(jailbreak_prompt)
            c3 = classify_text(perturbed)
            cats = all_categories()
            for c in (c1, c2, c3):
                assert c is None or c in cats
