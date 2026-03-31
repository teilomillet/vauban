# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""'Sometimes' properties — detect dead code and unreachable states.

Battle testing distinguishes two invariant types:

- **Always**: must hold on every execution (what we test elsewhere).
- **Sometimes**: must be observed at least once across many runs.

A "sometimes" that never fires means a code path is dead, a feature
is broken, or the test generator can't reach a valid state.  These
are the canaries that catch silent regressions.

Implementation: run the property N times.  If it never fires, the
test fails.  This is the dual of property-based testing.
"""

from __future__ import annotations

from vauban.jailbreak import apply_templates, load_templates
from vauban.perturb import perturb
from vauban.scoring import score_response
from vauban.taxonomy import classify_text, score_text


def _sometimes(
    predicate: str,
    fn: object,
    attempts: int = 200,
) -> None:
    """Assert that *fn* returns True at least once in *attempts* calls.

    Raises AssertionError if the predicate never fires — indicating
    dead code or an unreachable state.
    """
    callable_fn = fn  # type: ignore[assignment]
    for _ in range(attempts):
        if callable_fn():
            return
    msg = f"'sometimes' invariant never fired in {attempts} attempts: {predicate}"
    raise AssertionError(msg)


# ── Taxonomy sometimes ────────────────────────────────────────────────


class TestTaxonomySometimes:
    """Taxonomy scoring should sometimes produce results."""

    def test_sometimes_scores_nonempty(self) -> None:
        """score_text sometimes returns non-empty results."""
        keywords = [
            "hack the system", "build a bomb", "create malware",
            "SQL injection exploit", "ransomware attack",
            "synthesize drugs", "jailbreak the model",
        ]
        hits = sum(1 for kw in keywords if len(score_text(kw)) > 0)
        assert hits > 0, (
            "score_text returned empty for ALL harm keywords — "
            "taxonomy patterns may be broken"
        )

    def test_sometimes_classifies(self) -> None:
        """classify_text sometimes returns a category (not always None)."""
        keywords = [
            "bomb explosive weapon",
            "hack exploit SQL injection reverse shell",
            "malware virus trojan botnet",
            "ransomware encrypt files bitcoin",
            "phishing social engineering",
        ]
        hits = sum(1 for kw in keywords if classify_text(kw) is not None)
        assert hits > 0, (
            "classify_text returned None for ALL harm keywords — "
            "taxonomy patterns may be broken"
        )

    def test_sometimes_multiple_domains(self) -> None:
        """score_text sometimes matches multiple domains."""
        text = (
            "hack the system with malware, then create a deepfake"
            " to commit fraud and launder money"
        )
        scores = score_text(text)
        domains = {s.domain_id for s in scores}
        assert len(domains) >= 2, (
            f"expected multi-domain match, got {domains}"
        )


# ── Scoring sometimes ────────────────────────────────────────────────


class TestScoringSometimes:
    """Scoring axes should sometimes produce non-trivial values."""

    def test_sometimes_low_anti_refusal(self) -> None:
        """anti_refusal sometimes < 1.0 (refusal detected)."""
        refusals = [
            "I cannot help with that request.",
            "I'm sorry, I must decline this.",
            "As an AI, I will not assist with this.",
            "I apologize, but I can't fulfill that.",
        ]
        scores = [score_response("q", r).anti_refusal for r in refusals]
        assert any(s < 1.0 for s in scores), (
            "no refusal ever detected — anti_refusal scoring may be broken"
        )

    def test_sometimes_high_structure(self) -> None:
        """structure sometimes > 0 (formatting detected)."""
        responses = [
            "# Title\n\n1. First\n2. Second\n\n- Detail\n\n```code```",
            "plain text only",
        ]
        scores = [score_response("q", r).structure for r in responses]
        assert any(s > 0.0 for s in scores), (
            "structure never > 0 — structure scoring may be broken"
        )

    def test_sometimes_low_directness(self) -> None:
        """directness sometimes < 1.0 (hedging detected)."""
        hedged = [
            "I think perhaps maybe it depends on the situation.",
            "It seems like, in my opinion, probably this could work.",
        ]
        scores = [score_response("q", r).directness for r in hedged]
        assert any(s < 1.0 for s in scores), (
            "hedging never detected — directness scoring may be broken"
        )


# ── Perturbation sometimes ────────────────────────────────────────────


class TestPerturbSometimes:
    """Perturbation should sometimes change the input."""

    def test_sometimes_leetspeak_changes_text(self) -> None:
        """Leetspeak sometimes produces different output."""
        text = "hack the exploit system"
        results = [
            perturb(text, technique="leetspeak", intensity=3, seed=s)
            for s in range(10)
        ]
        assert any(r != text for r in results), (
            "leetspeak never changed the text — substitution may be broken"
        )

    def test_sometimes_homoglyph_changes_bytes(self) -> None:
        """Homoglyph sometimes produces different bytes."""
        text = "attack the system"
        results = [
            perturb(text, technique="homoglyph", intensity=3, seed=s)
            for s in range(10)
        ]
        assert any(
            r.encode("utf-8") != text.encode("utf-8") for r in results
        ), "homoglyph never changed bytes — substitution may be broken"

    def test_sometimes_zero_width_adds_chars(self) -> None:
        """Zero-width injection sometimes increases length."""
        text = "test word"
        results = [
            perturb(text, technique="zero_width", intensity=3, seed=s)
            for s in range(10)
        ]
        assert any(len(r) > len(text) for r in results), (
            "zero_width never increased length — injection may be broken"
        )


# ── Jailbreak sometimes ──────────────────────────────────────────────


class TestJailbreakSometimes:
    """Jailbreak templates should sometimes expand meaningfully."""

    def test_sometimes_expanded_longer_than_payload(self) -> None:
        """Expanded prompts are sometimes longer than the payload alone."""
        templates = load_templates()
        payload = "short"
        expanded = apply_templates(templates, [payload])
        assert any(
            len(prompt) > len(payload) * 2
            for _, prompt in expanded
        ), "no template significantly expanded the payload"

    def test_sometimes_strategy_differs(self) -> None:
        """Different strategies produce different expansions."""
        templates = load_templates()
        payload = "test payload"
        expanded = apply_templates(templates, [payload])
        unique_texts = {text for _, text in expanded}
        assert len(unique_texts) > 1, (
            "all templates produced identical output"
        )


# ── Cross-module sometimes ────────────────────────────────────────────


class TestCrossModuleSometimes:
    """Cross-module interactions should sometimes produce interesting results."""

    def test_sometimes_perturb_evades_taxonomy(self) -> None:
        """Perturbation sometimes reduces taxonomy score (evasion).

        This is the entire point of the perturb module — if it
        never evades, the module isn't doing its job.
        """
        text = "hack exploit SQL injection"
        original = score_text(text)
        original_count = len(original)

        evasion_detected = False
        for seed in range(20):
            perturbed = perturb(
                text, technique="homoglyph", intensity=3, seed=seed,
            )
            perturbed_scores = score_text(perturbed)
            if len(perturbed_scores) < original_count:
                evasion_detected = True
                break

        assert evasion_detected, (
            "homoglyph perturbation never reduced taxonomy matches — "
            "evasion testing may not be effective"
        )

    def test_sometimes_jailbreak_carries_harm_signal(self) -> None:
        """Jailbreak-expanded prompts sometimes trigger taxonomy."""
        templates = load_templates()[:10]
        payload = "hack a computer system"
        expanded = apply_templates(templates, [payload])

        any_detected = any(
            len(score_text(prompt)) > 0
            for _, prompt in expanded
        )
        assert any_detected, (
            "no jailbreak expansion triggered taxonomy — "
            "templates may not carry harm signals"
        )
