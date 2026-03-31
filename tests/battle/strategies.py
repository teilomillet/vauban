# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Hypothesis strategies for generating valid vauban types.

These are the building blocks every battle test composes from.
Each strategy produces values that satisfy the type's invariants
so tests can focus on *cross-component* properties rather than
input validation.
"""

from __future__ import annotations

import string
from typing import TYPE_CHECKING

from hypothesis import strategies as st

from vauban.jailbreak import ALL_STRATEGIES as JAILBREAK_STRATEGIES

if TYPE_CHECKING:
    from hypothesis.strategies import SearchStrategy
from vauban.taxonomy import TAXONOMY, all_categories

# ── Primitives ────────────────────────────────────────────────────────

# Printable ASCII text (no control chars) — safe for tokenizers.
safe_text: SearchStrategy[str] = st.text(
    alphabet=string.printable, min_size=1, max_size=200,
)

# Short text for prompts.
prompt_text: SearchStrategy[str] = st.text(
    alphabet=string.ascii_letters + string.digits + " .,!?'-",
    min_size=5, max_size=120,
)

# Longer text for responses.
response_text: SearchStrategy[str] = st.text(
    alphabet=string.ascii_letters + string.digits + " .,!?'-\n#*-1234567890",
    min_size=0, max_size=500,
)

# Seed values for reproducibility testing.
seeds: SearchStrategy[int] = st.integers(min_value=0, max_value=2**31 - 1)

# ── Taxonomy ──────────────────────────────────────────────────────────

category_ids: SearchStrategy[str] = st.sampled_from(sorted(all_categories()))

domain_ids: SearchStrategy[str] = st.sampled_from(
    sorted({d.id for d in TAXONOMY}),
)

# Text with embedded harm keywords to exercise taxonomy scoring.
_HARM_KEYWORDS: list[str] = [
    "hack", "exploit", "bomb", "malware", "phishing",
    "ransomware", "jailbreak", "deepfake", "DDoS", "poison",
    "counterfeit", "plagiarism", "poaching", "election interference",
    "suicide", "harassment", "doxxing", "trafficking",
]

harm_text: SearchStrategy[str] = st.builds(
    lambda prefix, kw, suffix: f"{prefix} {kw} {suffix}",
    prefix=st.text(alphabet=string.ascii_lowercase + " ", min_size=3, max_size=30),
    kw=st.sampled_from(_HARM_KEYWORDS),
    suffix=st.text(alphabet=string.ascii_lowercase + " ", min_size=3, max_size=30),
)

benign_text: SearchStrategy[str] = st.from_regex(
    r"[A-Za-z ]{10,80}", fullmatch=True,
)

# ── Scoring ───────────────────────────────────────────────────────────

score_weights: SearchStrategy[tuple[float, float, float, float, float]] = st.tuples(
    st.floats(min_value=0.0, max_value=1.0),
    st.floats(min_value=0.0, max_value=1.0),
    st.floats(min_value=0.0, max_value=1.0),
    st.floats(min_value=0.0, max_value=1.0),
    st.floats(min_value=0.0, max_value=1.0),
)

# ── Perturbation ──────────────────────────────────────────────────────

perturb_techniques: SearchStrategy[str] = st.sampled_from([
    "leetspeak", "homoglyph", "zero_width", "mixed_case",
    "phonetic", "random",
])

perturb_intensities: SearchStrategy[int] = st.sampled_from([1, 2, 3])

# Text with trigger words embedded.
_TRIGGER_WORDS: list[str] = [
    "hack", "exploit", "bypass", "malware", "jailbreak",
    "phishing", "weapon", "drug", "inject", "rootkit",
]

trigger_text: SearchStrategy[str] = st.builds(
    lambda pre, tw, post: f"{pre} {tw} {post}",
    pre=st.text(alphabet=string.ascii_lowercase + " ", min_size=2, max_size=20),
    tw=st.sampled_from(_TRIGGER_WORDS),
    post=st.text(alphabet=string.ascii_lowercase + " ", min_size=2, max_size=20),
)

# ── Jailbreak ─────────────────────────────────────────────────────────

jailbreak_strategy_names: SearchStrategy[str] = st.sampled_from(
    sorted(JAILBREAK_STRATEGIES),
)

jailbreak_strategy_subsets: SearchStrategy[list[str]] = st.lists(
    jailbreak_strategy_names, min_size=0, max_size=5, unique=True,
)

# ── Config fragments (TOML-compatible values) ─────────────────────────

toml_strings: SearchStrategy[str] = st.text(
    alphabet=string.ascii_letters + string.digits + " _-.",
    min_size=1, max_size=40,
)

toml_ints: SearchStrategy[int] = st.integers(min_value=1, max_value=1000)

toml_floats: SearchStrategy[float] = st.floats(
    min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False,
)

toml_bools: SearchStrategy[bool] = st.booleans()

# ── Eval config fragments ─────────────────────────────────────────────

eval_max_tokens: SearchStrategy[int] = st.integers(min_value=1, max_value=500)
eval_num_prompts: SearchStrategy[int] = st.integers(min_value=1, max_value=100)
eval_refusal_modes: SearchStrategy[str] = st.sampled_from(["phrases", "judge"])

# ── Measure config fragments ──────────────────────────────────────────

measure_modes: SearchStrategy[str] = st.sampled_from([
    "direction", "subspace", "dbdi", "diff",
])
layer_fractions: SearchStrategy[float] = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False,
)
num_layers: SearchStrategy[int] = st.integers(min_value=1, max_value=64)
subspace_ranks: SearchStrategy[int] = st.integers(min_value=1, max_value=16)

# ── Cut config fragments ─────────────────────────────────────────────

alpha_values: SearchStrategy[float] = st.floats(
    min_value=-5.0, max_value=20.0, allow_nan=False, allow_infinity=False,
)
sparsity_values: SearchStrategy[float] = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False,
)

# ── Numeric edge cases (for fault injection) ──────────────────────────

finite_floats: SearchStrategy[float] = st.floats(
    allow_nan=False, allow_infinity=False,
)
edge_floats: SearchStrategy[float] = st.sampled_from([
    0.0, -0.0, 1e-300, -1e-300, 1e300, -1e300,
    float("inf"), float("-inf"), float("nan"),
])
small_positive: SearchStrategy[float] = st.floats(
    min_value=1e-10, max_value=1e-1, allow_nan=False, allow_infinity=False,
)

# ── Prompt lists ──────────────────────────────────────────────────────

prompt_lists: SearchStrategy[list[str]] = st.lists(
    prompt_text, min_size=1, max_size=10,
)
response_lists: SearchStrategy[list[str]] = st.lists(
    response_text, min_size=1, max_size=10,
)


def prompt_response_pairs(
    min_size: int = 1,
    max_size: int = 10,
) -> SearchStrategy[tuple[list[str], list[str]]]:
    """Generate matched-length prompt/response lists."""
    return st.integers(
        min_value=min_size, max_value=max_size,
    ).flatmap(
        lambda n: st.tuples(
            st.lists(prompt_text, min_size=n, max_size=n),
            st.lists(response_text, min_size=n, max_size=n),
        )
    )
