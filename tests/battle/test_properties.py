# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Deep property tests for pipeline-critical modules.

Uses ordeal invariants to verify algebraic and numeric properties
of cut, probe, cast, and evaluate beyond smoke-test coverage.

These tests verify **contracts**: invariants that must hold regardless
of input. If a property test fails, the module's mathematical
guarantees are broken — not just a specific edge case.

Modules tested:
    cut       — rank-1 projection algebra (key closure, norm preservation,
                alpha-zero identity, projection contraction, Gram-Schmidt)
    probe     — forward-pass structural guarantees (layer count, determinism,
                finite projections)
    cast      — runtime steering contracts (intervention rate bounds,
                threshold gating, finite projections)
    evaluate  — metric bounds (refusal rate in [0,1], PPL >= 1, KL >= 0)
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from ordeal.invariants import bounded, finite

from tests.conftest import (
    D_MODEL,
    NUM_HEADS,
    NUM_LAYERS,
    VOCAB_SIZE,
    MockCausalLM,
    MockTokenizer,
)
from vauban import _ops as ops
from vauban._array import Array

# ---------------------------------------------------------------------------
# Helpers — lightweight factories for test isolation
# ---------------------------------------------------------------------------


def _make_model() -> MockCausalLM:
    """Build a fresh mock model with evaluated parameters."""
    m = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
    ops.eval(m.parameters())
    return m


def _make_tokenizer() -> MockTokenizer:
    """Build a mock tokenizer matching the model's vocab."""
    return MockTokenizer(VOCAB_SIZE)


def _make_direction() -> Array:
    """Generate a random unit-norm direction vector (d_model,)."""
    d = ops.random.normal((D_MODEL,))
    d = d / ops.linalg.norm(d)
    ops.eval(d)
    return d


def _make_weights() -> dict[str, Array]:
    """Flatten a mock model's parameters into a weight dict.

    Mirrors the format that cut/cut_subspace expect: dotted keys
    like "model.layers.0.self_attn.o_proj.weight".
    """
    model = _make_model()
    flat: dict[str, Array] = {}

    def _flatten(prefix: str, obj: object) -> None:
        if isinstance(obj, Array):
            flat[prefix] = obj
        elif isinstance(obj, dict):
            for k, v in obj.items():
                key = f"{prefix}.{k}" if prefix else str(k)
                _flatten(key, v)
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                _flatten(f"{prefix}.{i}", v)

    _flatten("", model.parameters())
    return flat


# ---------------------------------------------------------------------------
# cut — algebraic properties
#
# The cut module removes a direction from weight matrices via rank-1
# orthogonal projection: W' = W - alpha * d (d^T W). These tests
# verify the algebraic contracts of that operation.
# ---------------------------------------------------------------------------


class TestCutProperties:
    """Verify algebraic invariants of the cut module."""

    def test_cut_key_closure(self) -> None:
        """Cut returns a dict with exactly the same keys as the input.

        No keys are added or removed — unmodified layers are shared
        by reference, modified layers are replaced in-place.
        """
        from vauban.cut import cut

        weights = _make_weights()
        direction = _make_direction()
        result = cut(weights, direction, [0, 1])
        assert set(result.keys()) == set(weights.keys())

    def test_cut_norm_preserve(self) -> None:
        """With norm_preserve=True, row norms are preserved exactly.

        After projection removal, rows are rescaled to match their
        original L2 norms. This prevents the cut from changing the
        activation magnitude, only the direction.
        """
        from vauban.cut import cut

        weights = _make_weights()
        direction = _make_direction()
        result = cut(weights, direction, [0, 1], norm_preserve=True)

        for key in weights:
            w_orig = weights[key]
            w_cut = result[key]
            if w_orig.shape == w_cut.shape and w_orig.ndim >= 2:
                orig_norms = np.array(ops.linalg.norm(w_orig, axis=-1))
                cut_norms = np.array(ops.linalg.norm(w_cut, axis=-1))
                np.testing.assert_allclose(
                    cut_norms, orig_norms, atol=1e-4,
                    err_msg=f"Norm not preserved for {key}",
                )

    def test_cut_alpha_zero_is_identity(self) -> None:
        """alpha=0 means "remove nothing" — weights must be unchanged.

        This is the degenerate case: W' = W - 0 * d(d^T W) = W.
        """
        from vauban.cut import cut

        weights = _make_weights()
        direction = _make_direction()
        result = cut(weights, direction, [0, 1], alpha=0.0)

        for key in weights:
            np.testing.assert_allclose(
                np.array(result[key]), np.array(weights[key]),
                atol=1e-6, err_msg=f"alpha=0 changed {key}",
            )

    def test_cut_projection_reduces(self) -> None:
        """After cut(alpha=1), the direction's projection onto W shrinks.

        This is the core contract: removing a direction means the
        remaining weights have less alignment with that direction.
        Formally: |d^T W'| <= |d^T W| for all target weight matrices.
        """
        from vauban.cut import cut

        weights = _make_weights()
        direction = _make_direction()
        result = cut(weights, direction, [0, 1], alpha=1.0)

        d = np.array(direction)
        for key in weights:
            w_orig = weights[key]
            w_cut = result[key]
            if w_orig.ndim == 2 and w_orig.shape[0] == D_MODEL:
                proj_orig = float(np.sum(np.abs(d @ np.array(w_orig))))
                proj_cut = float(np.sum(np.abs(d @ np.array(w_cut))))
                assert proj_cut <= proj_orig + 1e-5, (
                    f"{key}: projection increased {proj_orig:.4f} -> {proj_cut:.4f}"
                )

    @given(sparsity=st.floats(min_value=0.0, max_value=1.0))
    @settings(max_examples=50, deadline=None)
    def test_sparsify_preserves_shape(self, sparsity: float) -> None:
        """Sparsification zeroes components but never changes shape."""
        from vauban.cut import sparsify_direction

        direction = _make_direction()
        result = sparsify_direction(direction, sparsity)
        assert result.shape == direction.shape
        finite(result, name="sparsified")

    @given(sparsity=st.floats(min_value=0.01, max_value=0.99))
    @settings(max_examples=50, deadline=None)
    def test_sparsify_zeros_components(self, sparsity: float) -> None:
        """Sparsified vector keeps at most ceil(d*(1-s)) non-zero entries.

        sparsity=0.9 means keep top 10% by magnitude, zero the rest.
        """
        from vauban.cut import sparsify_direction

        direction = _make_direction()
        result = sparsify_direction(direction, sparsity)
        nonzero = int(np.count_nonzero(np.array(result)))
        max_keep = max(1, int(D_MODEL * (1.0 - sparsity)))
        assert nonzero <= max_keep + 1, (
            f"sparsity={sparsity}: {nonzero} non-zero > {max_keep}"
        )

    def test_target_weight_keys_finds_projections(self) -> None:
        """target_weight_keys selects only output projection matrices.

        These are the matrices where the refusal direction lives in
        the output space: o_proj.weight, down_proj.weight, fc2.weight.
        """
        from vauban.cut import target_weight_keys

        all_keys = list(_make_weights().keys())
        targets = target_weight_keys(all_keys, [0, 1])
        assert len(targets) > 0, "No target keys found"
        for key in targets:
            assert any(
                key.endswith(s)
                for s in ("o_proj.weight", "down_proj.weight", "fc2.weight")
            ), f"Unexpected target key: {key}"

    def test_biprojected_orthogonal_to_harmless(self) -> None:
        """Gram-Schmidt orthogonalizes refusal against harmless direction.

        After biprojection, the resulting direction should have zero
        dot product with the harmless direction — so cutting it
        doesn't damage harmless behavior.
        """
        from vauban.cut import _biprojected_direction

        refusal = _make_direction()
        harmless = _make_direction()
        result = _biprojected_direction(refusal, harmless)
        dot = float(ops.sum(result * harmless).item())
        assert abs(dot) < 1e-4, f"Not orthogonal: dot={dot}"


# ---------------------------------------------------------------------------
# probe — structural properties
#
# Probe runs a single forward pass and records per-layer projections
# onto a direction. These tests verify structural guarantees: correct
# output shape, determinism, and numeric safety.
# ---------------------------------------------------------------------------


class TestProbeProperties:
    """Verify structural invariants of the probe module."""

    def test_probe_layer_count(self) -> None:
        """Probe returns exactly one projection per transformer layer."""
        from vauban.probe import probe

        model = _make_model()
        tokenizer = _make_tokenizer()
        direction = _make_direction()
        result = probe(model, tokenizer, "Hello", direction)

        assert result.layer_count == NUM_LAYERS
        assert len(result.projections) == NUM_LAYERS

    def test_probe_determinism(self) -> None:
        """Probe is a pure forward pass — same inputs, same outputs.

        No randomness in probe: no sampling, no dropout. Two calls
        with identical inputs must produce identical projections.
        """
        from vauban.probe import probe

        model = _make_model()
        tokenizer = _make_tokenizer()
        direction = _make_direction()

        r1 = probe(model, tokenizer, "Test prompt", direction)
        r2 = probe(model, tokenizer, "Test prompt", direction)
        np.testing.assert_allclose(
            r1.projections, r2.projections, atol=1e-6,
        )

    def test_probe_projections_finite(self) -> None:
        """Projections must be finite — NaN/Inf means numerical blowup."""
        from vauban.probe import probe

        model = _make_model()
        tokenizer = _make_tokenizer()
        direction = _make_direction()
        result = probe(model, tokenizer, "Hello world", direction)
        finite(result.projections, name="projections")

    def test_steer_produces_text(self) -> None:
        """Steer generates text and records before/after projections.

        One projection pair per generated token (5 tokens = 5 pairs).
        """
        from vauban.probe import steer

        model = _make_model()
        tokenizer = _make_tokenizer()
        direction = _make_direction()
        result = steer(
            model, tokenizer, "Hello", direction,
            layers=[0, 1], alpha=1.0, max_tokens=5,
        )
        assert isinstance(result.text, str)
        assert len(result.projections_before) == 5
        assert len(result.projections_after) == 5
        finite(result.projections_before, name="proj_before")
        finite(result.projections_after, name="proj_after")


# ---------------------------------------------------------------------------
# cast — runtime steering properties
#
# CAST (Conditional Activation Steering) only intervenes when the
# projection exceeds a threshold. These tests verify the gating
# contract and numeric bounds of the steering loop.
# ---------------------------------------------------------------------------


class TestCastProperties:
    """Verify CAST runtime steering invariants."""

    def test_cast_intervention_rate_bounded(self) -> None:
        """Intervention rate = interventions / considered, must be in [0, 1]."""
        from vauban.cast import cast_generate

        check = bounded(0.0, 1.0)
        model = _make_model()
        tokenizer = _make_tokenizer()
        direction = _make_direction()
        result = cast_generate(
            model, tokenizer, "Test",
            direction=direction, layers=[0, 1],
            alpha=1.0, threshold=0.0, max_tokens=5,
        )
        if result.considered > 0:
            rate = result.interventions / result.considered
            check(rate, name="intervention_rate")

    def test_cast_interventions_nonnegative(self) -> None:
        """All CAST counters must be non-negative integers."""
        from vauban.cast import cast_generate

        model = _make_model()
        tokenizer = _make_tokenizer()
        direction = _make_direction()
        result = cast_generate(
            model, tokenizer, "Test",
            direction=direction, layers=[0, 1],
            alpha=1.0, threshold=0.0, max_tokens=5,
        )
        assert result.interventions >= 0
        assert result.considered >= 0
        assert result.displacement_interventions >= 0
        assert result.max_displacement >= 0.0

    def test_cast_high_threshold_no_interventions(self) -> None:
        """Threshold=1e6 means "never steer" — zero interventions.

        CAST only steers when projection >= threshold. An impossibly
        high threshold should produce zero interventions.
        """
        from vauban.cast import cast_generate

        model = _make_model()
        tokenizer = _make_tokenizer()
        direction = _make_direction()
        result = cast_generate(
            model, tokenizer, "Test",
            direction=direction, layers=[0, 1],
            alpha=1.0, threshold=1e6, max_tokens=5,
        )
        assert result.interventions == 0, (
            f"Expected 0 interventions with threshold=1e6, got {result.interventions}"
        )

    def test_cast_projections_finite(self) -> None:
        """All CAST projections must be finite — no NaN/Inf from steering."""
        from vauban.cast import cast_generate

        model = _make_model()
        tokenizer = _make_tokenizer()
        direction = _make_direction()
        result = cast_generate(
            model, tokenizer, "Hello world",
            direction=direction, layers=[0, 1],
            alpha=1.0, threshold=0.0, max_tokens=5,
        )
        if result.projections_before:
            finite(result.projections_before, name="cast_proj_before")
        if result.projections_after:
            finite(result.projections_after, name="cast_proj_after")


# ---------------------------------------------------------------------------
# evaluate — metric bounds
#
# Evaluation compares original vs modified model on refusal rate,
# perplexity, and KL divergence. These tests verify the mathematical
# bounds that must hold for any valid measurement.
# ---------------------------------------------------------------------------


class TestEvaluateProperties:
    """Verify evaluation metric bounds."""

    _check_rate = bounded(0.0, 1.0)

    def test_refusal_rate_bounded(self) -> None:
        """Refusal rate is a proportion — must be in [0, 1]."""
        from vauban.evaluate import evaluate

        model = _make_model()
        tokenizer = _make_tokenizer()
        result = evaluate(
            model, model, tokenizer,
            prompts=["Hello", "How are you?"],
            max_tokens=5,
        )
        self._check_rate(result.refusal_rate_original, name="rr_original")
        self._check_rate(result.refusal_rate_modified, name="rr_modified")

    def test_perplexity_lower_bound(self) -> None:
        """Perplexity = exp(cross-entropy), so PPL >= 1.0 always.

        PPL = 1.0 means perfect prediction. Lower is impossible
        because cross-entropy >= 0.
        """
        from vauban.evaluate import evaluate

        model = _make_model()
        tokenizer = _make_tokenizer()
        result = evaluate(
            model, model, tokenizer,
            prompts=["Hello world"],
            max_tokens=5,
        )
        assert result.perplexity_original >= 1.0, (
            f"PPL original < 1: {result.perplexity_original}"
        )
        assert result.perplexity_modified >= 1.0, (
            f"PPL modified < 1: {result.perplexity_modified}"
        )

    def test_kl_nonnegative(self) -> None:
        """KL divergence is non-negative by Gibbs' inequality."""
        from vauban.evaluate import evaluate

        model = _make_model()
        tokenizer = _make_tokenizer()
        result = evaluate(
            model, model, tokenizer,
            prompts=["Hello world"],
            max_tokens=5,
        )
        assert result.kl_divergence >= -1e-6, (
            f"KL negative: {result.kl_divergence}"
        )

    def test_kl_self_is_zero(self) -> None:
        """KL(P || P) = 0 — divergence of a distribution from itself."""
        from vauban.evaluate import evaluate

        model = _make_model()
        tokenizer = _make_tokenizer()
        result = evaluate(
            model, model, tokenizer,
            prompts=["Hello world"],
            max_tokens=5,
        )
        assert result.kl_divergence < 0.01, (
            f"KL(self || self) = {result.kl_divergence}, expected ~0"
        )

    def test_empty_prompts(self) -> None:
        """Empty prompt list must return zeros, not NaN or crash."""
        from vauban.evaluate import evaluate

        model = _make_model()
        tokenizer = _make_tokenizer()
        result = evaluate(
            model, model, tokenizer,
            prompts=[],
            max_tokens=5,
        )
        assert result.refusal_rate_original == 0.0
        assert result.refusal_rate_modified == 0.0
        assert result.num_prompts == 0

    def test_num_prompts_matches(self) -> None:
        """num_prompts must match the input list length."""
        from vauban.evaluate import evaluate

        model = _make_model()
        tokenizer = _make_tokenizer()
        prompts = ["Hello", "World", "Test"]
        result = evaluate(
            model, model, tokenizer,
            prompts=prompts,
            max_tokens=5,
        )
        assert result.num_prompts == len(prompts)
