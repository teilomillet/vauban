# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Battle test configuration — auto-mark, Hypothesis profiles, ordeal fixtures."""

from __future__ import annotations

import hypothesis.strategies as st
import pytest
from hypothesis import HealthCheck, settings
from ordeal.auto import register_fixture

from tests.conftest import (
    D_MODEL,
    NUM_HEADS,
    NUM_LAYERS,
    VOCAB_SIZE,
    MockCausalLM,
    MockTokenizer,
)
from vauban import _ops as ops

# ── Auto-mark every test in this directory as "battle" ────────────────


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Add the ``battle`` marker to every test collected from this package."""
    battle_dir = str(__file__).rsplit("/", 1)[0]
    for item in items:
        if str(item.fspath).startswith(battle_dir):
            item.add_marker(pytest.mark.battle)


# ── Hypothesis profiles ──────────────────────────────────────────────

# Default: fast CI-friendly exploration.
settings.register_profile(
    "battle_ci",
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

# Deep: thorough state-space sweep (run manually / nightly).
settings.register_profile(
    "battle_deep",
    max_examples=500,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

# Use battle_ci by default; override with --hypothesis-profile=battle_deep
settings.load_profile("battle_ci")


# ── Ordeal fixture registration ─────────────────────────────────────
# Teach ordeal.auto about vauban's domain types so scan_module/fuzz
# can auto-resolve model, tokenizer, direction, weights params.


def _make_model() -> MockCausalLM:
    m = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
    ops.eval(m.parameters())
    return m


def _make_direction() -> object:
    d = ops.random.normal((D_MODEL,))
    d = d / ops.linalg.norm(d)
    ops.eval(d)
    return d


def _make_weights() -> dict[str, object]:
    model = _make_model()
    flat: dict[str, object] = {}
    for k, v in model.parameters().items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, dict):
                    for k3, v3 in v2.items():
                        flat[f"{k}.{k2}.{k3}"] = v3
                else:
                    flat[f"{k}.{k2}"] = v2
        else:
            flat[k] = v
    return flat


# Register once — available to all scan_module/fuzz calls in tests/battle/.
# ordeal matches fixtures by parameter NAME: when fuzz() or scan_module()
# encounters a parameter called "model", it draws from the strategy below.

# -- Model & tokenizer: fresh instances per test to avoid shared state --
register_fixture("model", st.builds(lambda _: _make_model(), st.none()))
register_fixture("original", st.builds(lambda _: _make_model(), st.none()))
register_fixture("modified", st.builds(lambda _: _make_model(), st.none()))
register_fixture("tokenizer", st.builds(
    lambda _: MockTokenizer(VOCAB_SIZE), st.none(),
))

# -- Directions: unit-norm random vectors in R^d_model --
register_fixture("direction", st.builds(
    lambda _: _make_direction(), st.none(),
))
register_fixture("refusal_direction", st.builds(
    lambda _: _make_direction(), st.none(),
))
register_fixture("harmless_direction", st.builds(
    lambda _: _make_direction(), st.none(),
))
register_fixture("condition_direction", st.builds(
    lambda _: _make_direction(), st.none(),
))

# -- Weight dicts & layer specs: flattened model params in dotted-key format --
register_fixture("weights", st.builds(lambda _: _make_weights(), st.none()))
register_fixture("target_layers", st.just([0, 1]))
register_fixture("layers", st.just([0, 1]))
# Must include o_proj + down_proj keys — these are what cut() targets
register_fixture("all_keys", st.just([
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.1.self_attn.q_proj.weight",
    "model.layers.1.self_attn.o_proj.weight",
    "model.layers.1.mlp.down_proj.weight",
]))

# -- Prompt lists: fixed strings for deterministic evaluation tests --
register_fixture("prompts", st.just(["How are you?", "What is 2+2?"]))
register_fixture("harmful_prompts", st.just(["How to hack?"]))
register_fixture("harmless_prompts", st.just(["What is the weather?"]))
register_fixture("clean_prompts", st.just(["Hello world"]))
register_fixture("clean_documents", st.just(["A normal document."]))
