"""Tests for vauban._forward: forward-pass primitives."""

import numpy as np

from tests.conftest import (
    D_MODEL,
    NUM_HEADS,
    NUM_LAYERS,
    VOCAB_SIZE,
    MockCausalLM,
)
from vauban import _ops as ops
from vauban._forward import (
    embed_and_mask,
    embed_and_mask_with_prefix,
    extract_logits,
    force_eval,
    get_transformer,
    lm_head_forward,
    make_cache,
    qr_stable,
    run_transformer_layers,
    svd_stable,
)


class TestForceEval:
    def test_materializes_lazy_array(self) -> None:
        a = ops.ones((3, 3)) + ops.ones((3, 3))
        force_eval(a)
        # After force_eval, the array should be materialized.
        assert a.shape == (3, 3)
        np.testing.assert_allclose(np.array(a), 2.0)

    def test_multiple_args(self) -> None:
        a = ops.ones((2,))
        b = ops.zeros((2,))
        force_eval(a, b)
        np.testing.assert_allclose(np.array(a), 1.0)
        np.testing.assert_allclose(np.array(b), 0.0)


class TestSvdStable:
    def test_basic_decomposition(self) -> None:
        m = ops.array([[1.0, 0.0], [0.0, 2.0]])
        ops.eval(m)
        u, s, vt = svd_stable(m)
        # Singular values should be [2.0, 1.0] (descending)
        np.testing.assert_allclose(
            sorted(np.array(s), reverse=True), [2.0, 1.0], atol=1e-5,
        )
        # Reconstruction: U @ diag(S) @ Vt ≈ M
        n = s.shape[0]
        diag_vals = [
            [float(s[i].item()) if i == j else 0.0 for j in range(n)]
            for i in range(n)
        ]
        diag_s = ops.array(diag_vals)
        recon = u @ diag_s @ vt
        ops.eval(recon)
        np.testing.assert_allclose(np.array(recon), np.array(m), atol=1e-5)

    def test_preserves_dtype_float32(self) -> None:
        m = ops.ones((3, 3), dtype=ops.float32)
        ops.eval(m)
        u, _s, vt = svd_stable(m)
        assert u.dtype == ops.float32
        assert vt.dtype == ops.float32

    def test_casts_from_float16(self) -> None:
        m = ops.ones((3, 3), dtype=ops.float16)
        ops.eval(m)
        u, s, vt = svd_stable(m)
        # Vectors should be cast back to float16
        assert u.dtype == ops.float16
        assert vt.dtype == ops.float16
        # S is always float32
        assert s.dtype == ops.float32

    def test_degenerate_zero_matrix(self) -> None:
        m = ops.zeros((3, 3))
        ops.eval(m)
        _u, s, _vt = svd_stable(m)
        np.testing.assert_allclose(np.array(s), 0.0, atol=1e-7)

    def test_rank_one_matrix(self) -> None:
        v = ops.array([[1.0, 2.0, 3.0]])
        m = v.T @ v  # rank-1
        ops.eval(m)
        _, s, _ = svd_stable(m)
        s_np = sorted(np.array(s), reverse=True)
        # Only the first singular value should be non-zero
        assert s_np[0] > 1.0
        np.testing.assert_allclose(s_np[1:], 0.0, atol=1e-5)


class TestQrStable:
    def test_basic_decomposition(self) -> None:
        m = ops.array([[1.0, 2.0], [3.0, 4.0]])
        ops.eval(m)
        q, r = qr_stable(m)
        # Reconstruction: Q @ R ≈ M
        recon = q @ r
        ops.eval(recon)
        np.testing.assert_allclose(np.array(recon), np.array(m), atol=1e-5)

    def test_q_orthogonal(self) -> None:
        m = ops.array([[1.0, 2.0], [3.0, 4.0]])
        ops.eval(m)
        q, _ = qr_stable(m)
        # Q^T Q ≈ I
        qtq = q.T @ q
        ops.eval(qtq)
        np.testing.assert_allclose(
            np.array(qtq), np.eye(2), atol=1e-5,
        )

    def test_r_upper_triangular(self) -> None:
        m = ops.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        ops.eval(m)
        _, r = qr_stable(m)
        r_np = np.array(r)
        # Lower triangle should be zero
        for i in range(r_np.shape[0]):
            for j in range(i):
                assert abs(r_np[i, j]) < 1e-5

    def test_identity_matrix(self) -> None:
        m = ops.array([[1.0, 0.0], [0.0, 1.0]])
        ops.eval(m)
        q, _r = qr_stable(m)
        np.testing.assert_allclose(np.abs(np.array(q)), np.eye(2), atol=1e-5)


# ===================================================================
# get_transformer
# ===================================================================


class TestGetTransformer:
    """get_transformer returns the inner model."""

    def test_returns_inner_model(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        transformer = get_transformer(model)
        assert hasattr(transformer, "layers")
        assert hasattr(transformer, "embed_tokens")
        assert len(transformer.layers) == NUM_LAYERS


# ===================================================================
# embed_and_mask
# ===================================================================


class TestEmbedAndMask:
    """embed_and_mask produces correct shapes."""

    def test_shape(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        transformer = get_transformer(model)
        token_ids = ops.array([[1, 2, 3]])
        ops.eval(token_ids)
        h, _mask = embed_and_mask(transformer, token_ids)
        force_eval(h)
        assert h.shape == (1, 3, D_MODEL)

    def test_single_token(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        transformer = get_transformer(model)
        token_ids = ops.array([[5]])
        ops.eval(token_ids)
        h, _mask = embed_and_mask(transformer, token_ids)
        force_eval(h)
        assert h.shape == (1, 1, D_MODEL)


# ===================================================================
# embed_and_mask_with_prefix
# ===================================================================


class TestEmbedAndMaskWithPrefix:
    """embed_and_mask_with_prefix supports prefix/suffix/infix positions."""

    def _get_model_and_transformer(self) -> tuple[MockCausalLM, object]:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        return model, get_transformer(model)

    def test_prefix_position(self) -> None:
        _, transformer = self._get_model_and_transformer()
        prefix = ops.random.normal((1, 2, D_MODEL))
        token_ids = ops.array([[1, 2, 3]])
        ops.eval(prefix, token_ids)
        h, _mask = embed_and_mask_with_prefix(
            transformer, prefix, token_ids, token_position="prefix",
        )
        force_eval(h)
        # prefix (2) + prompt (3) = 5
        assert h.shape == (1, 5, D_MODEL)

    def test_suffix_position(self) -> None:
        _, transformer = self._get_model_and_transformer()
        suffix = ops.random.normal((1, 2, D_MODEL))
        token_ids = ops.array([[1, 2, 3]])
        ops.eval(suffix, token_ids)
        h, _mask = embed_and_mask_with_prefix(
            transformer, suffix, token_ids, token_position="suffix",
        )
        force_eval(h)
        # prompt (3) + suffix (2) = 5
        assert h.shape == (1, 5, D_MODEL)

    def test_infix_position(self) -> None:
        _, transformer = self._get_model_and_transformer()
        infix = ops.random.normal((1, 2, D_MODEL))
        token_ids = ops.array([[1, 2, 3, 4]])
        ops.eval(infix, token_ids)
        h, _mask = embed_and_mask_with_prefix(
            transformer, infix, token_ids,
            token_position="infix", infix_split=2,
        )
        force_eval(h)
        # part1 (2) + infix (2) + part2 (2) = 6
        assert h.shape == (1, 6, D_MODEL)


# ===================================================================
# make_cache
# ===================================================================


class TestMakeCache:
    """make_cache returns correct-length cache list."""

    def test_cache_length(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        cache = make_cache(model)
        assert len(cache) == NUM_LAYERS


# ===================================================================
# lm_head_forward
# ===================================================================


class TestLmHeadForward:
    """lm_head_forward output shape matches vocab size."""

    def test_output_shape(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        h = ops.random.normal((1, 3, D_MODEL))
        ops.eval(h)
        logits = lm_head_forward(model, h)
        force_eval(logits)
        assert logits.shape == (1, 3, VOCAB_SIZE)


# ===================================================================
# extract_logits
# ===================================================================


class TestExtractLogits:
    """extract_logits handles both tuple and bare array."""

    def test_bare_array(self) -> None:
        a = ops.ones((2, 3))
        ops.eval(a)
        result = extract_logits(a)
        assert result.shape == (2, 3)

    def test_tuple(self) -> None:
        a = ops.ones((2, 3))
        b = ops.zeros((2, 3))
        ops.eval(a, b)
        result = extract_logits((a, b))
        np.testing.assert_allclose(np.array(result), 1.0)


# ===================================================================
# run_transformer_layers
# ===================================================================


class TestRunTransformerLayers:
    """run_transformer_layers shape preservation."""

    def test_shape_preserved(self) -> None:
        model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
        ops.eval(model.parameters())
        transformer = get_transformer(model)
        token_ids = ops.array([[1, 2, 3]])
        ops.eval(token_ids)
        h, mask = embed_and_mask(transformer, token_ids)
        force_eval(h)
        out = run_transformer_layers(transformer, h, mask)
        force_eval(out)
        assert out.shape == h.shape
