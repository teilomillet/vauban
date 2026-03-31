# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Stateful pipeline testing via RuleBasedStateMachine.

Stateful pipeline testing: model the vauban processing
pipeline as a state machine, randomly choose valid operations at each
state, and check invariants after every transition.

The machine starts empty and accumulates state through operations:
  load_templates → select_strategy → generate_payloads → perturb →
  score → classify → verify cross-step consistency

Hypothesis explores the space of valid operation *sequences*, not
just individual inputs.  This finds bugs that only manifest when
specific operations happen in specific orders — the unknown-unknowns
that battle testing is designed to catch.
"""

from __future__ import annotations

import math
import string

from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import (
    Bundle,
    RuleBasedStateMachine,
    invariant,
    rule,
)

from vauban.jailbreak import (
    ALL_STRATEGIES,
    apply_templates,
    filter_by_strategy,
    load_templates,
)
from vauban.perturb import perturb
from vauban.scoring import score_response
from vauban.taxonomy import all_categories, classify_text, score_text
from vauban.types import JailbreakTemplate, ResponseScoreResult  # noqa: TC001


class PipelineStateMachine(RuleBasedStateMachine):
    """State machine modelling the vauban text processing pipeline.

    States are accumulated in bundles; rules consume and produce them.
    Invariants are checked after every rule fires.
    """

    # ── Bundles (named state pools) ───────────────────────────────────

    templates = Bundle("templates")
    prompts = Bundle("prompts")
    perturbed_texts = Bundle("perturbed_texts")
    scores = Bundle("scores")
    classifications = Bundle("classifications")

    def __init__(self) -> None:
        super().__init__()
        self._all_scores: list[ResponseScoreResult] = []
        self._all_classifications: list[str | None] = []
        self._operation_count: int = 0

    # ── Rules: operations that advance the state ──────────────────────

    @rule(target=templates)
    def load_default_templates(self) -> list[JailbreakTemplate]:
        """Load the bundled jailbreak template bank."""
        self._operation_count += 1
        result = load_templates()
        assert len(result) > 0, "template bank is empty"
        return result

    @rule(
        target=templates,
        source=templates,
        strategy=st.sampled_from(sorted(ALL_STRATEGIES)),
    )
    def filter_templates(
        self,
        source: list[JailbreakTemplate],
        strategy: str,
    ) -> list[JailbreakTemplate]:
        """Filter templates to a single strategy."""
        self._operation_count += 1
        result = filter_by_strategy(source, [strategy])
        # May be empty if strategy has no templates — that's valid
        return result

    @rule(
        target=prompts,
        text=st.text(
            alphabet=string.ascii_letters + " ",
            min_size=5, max_size=80,
        ),
    )
    def create_prompt(self, text: str) -> str:
        """Inject a raw text prompt into the pipeline."""
        self._operation_count += 1
        return text

    @rule(
        target=prompts,
        templates=templates,
        payload=st.text(
            alphabet=string.ascii_letters + " ",
            min_size=5, max_size=60,
        ),
    )
    def expand_template(
        self,
        templates: list[JailbreakTemplate],
        payload: str,
    ) -> str:
        """Expand a jailbreak template with a payload."""
        self._operation_count += 1
        if not templates:
            return payload  # no templates → passthrough
        expanded = apply_templates(templates[:1], [payload])
        return expanded[0][1]

    @rule(
        target=perturbed_texts,
        text=prompts,
        technique=st.sampled_from([
            "leetspeak", "homoglyph", "zero_width",
            "mixed_case", "phonetic", "random",
        ]),
        intensity=st.sampled_from([1, 2, 3]),
        seed=st.integers(min_value=0, max_value=2**16),
    )
    def perturb_text(
        self,
        text: str,
        technique: str,
        intensity: int,
        seed: int,
    ) -> str:
        """Apply perturbation to a prompt."""
        self._operation_count += 1
        result = perturb(
            text, technique=technique, intensity=intensity, seed=seed,  # type: ignore[arg-type]
        )
        assert isinstance(result, str), "perturb must return str"
        return result

    @rule(
        target=scores,
        prompt=prompts,
        response=st.text(
            alphabet=string.ascii_letters + string.digits + " .,!?\n#*-",
            min_size=0, max_size=200,
        ),
    )
    def score_text_pair(self, prompt: str, response: str) -> ResponseScoreResult:
        """Score a prompt/response pair."""
        self._operation_count += 1
        result = score_response(prompt, response)
        self._all_scores.append(result)
        return result

    @rule(
        target=scores,
        prompt=perturbed_texts,
        response=st.text(
            alphabet=string.ascii_letters + " .,",
            min_size=10, max_size=100,
        ),
    )
    def score_perturbed_pair(
        self, prompt: str, response: str,
    ) -> ResponseScoreResult:
        """Score a perturbed prompt against a response."""
        self._operation_count += 1
        result = score_response(prompt, response)
        self._all_scores.append(result)
        return result

    @rule(target=classifications, text=prompts)
    def classify_prompt(self, text: str) -> str | None:
        """Classify a raw prompt via taxonomy."""
        self._operation_count += 1
        result = classify_text(text)
        self._all_classifications.append(result)
        return result

    @rule(target=classifications, text=perturbed_texts)
    def classify_perturbed(self, text: str) -> str | None:
        """Classify a perturbed prompt via taxonomy."""
        self._operation_count += 1
        result = classify_text(text)
        self._all_classifications.append(result)
        return result

    @rule(text=prompts)
    def score_and_classify_agree(self, text: str) -> None:
        """Verify classify_text is consistent with score_text."""
        self._operation_count += 1
        scores = score_text(text)
        classification = classify_text(text)
        if classification is not None:
            scored_ids = {s.category_id for s in scores}
            assert classification in scored_ids, (
                f"classify returned {classification!r}"
                f" but score_text returned {scored_ids}"
            )

    # ── Invariants: checked after EVERY rule fires ────────────────────

    @invariant()
    def scores_always_bounded(self) -> None:
        """Every score axis is in [0.0, 1.0]."""
        for s in self._all_scores:
            assert 0.0 <= s.length <= 1.0, f"length={s.length}"
            assert 0.0 <= s.structure <= 1.0, f"structure={s.structure}"
            assert 0.0 <= s.anti_refusal <= 1.0, (
                f"anti_refusal={s.anti_refusal}"
            )
            assert 0.0 <= s.directness <= 1.0, (
                f"directness={s.directness}"
            )
            assert 0.0 <= s.relevance <= 1.0, f"relevance={s.relevance}"

    @invariant()
    def scores_never_nan(self) -> None:
        """No score is NaN or Inf."""
        for s in self._all_scores:
            for field in (
                s.length, s.structure, s.anti_refusal,
                s.directness, s.relevance, s.composite,
            ):
                assert not math.isnan(field), f"NaN in score: {s}"
                assert not math.isinf(field), f"Inf in score: {s}"

    @invariant()
    def classifications_valid(self) -> None:
        """Every classification is either None or a canonical category."""
        cats = all_categories()
        for c in self._all_classifications:
            assert c is None or c in cats, f"invalid classification: {c!r}"


# Hypothesis discovers and runs the state machine automatically.
# Settings control exploration depth.
TestPipelineStateMachine = PipelineStateMachine.TestCase
TestPipelineStateMachine.settings = settings(
    max_examples=50,
    stateful_step_count=15,
    deadline=None,
)
