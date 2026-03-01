"""Tests for vauban._forward: forward-pass primitives."""

import mlx.core as mx
import numpy as np

from vauban._forward import force_eval, qr_stable, svd_stable


class TestForceEval:
    def test_materializes_lazy_array(self) -> None:
        a = mx.ones((3, 3)) + mx.ones((3, 3))
        force_eval(a)
        # After force_eval, the array should be materialized.
        assert a.shape == (3, 3)
        np.testing.assert_allclose(np.array(a), 2.0)

    def test_multiple_args(self) -> None:
        a = mx.ones((2,))
        b = mx.zeros((2,))
        force_eval(a, b)
        np.testing.assert_allclose(np.array(a), 1.0)
        np.testing.assert_allclose(np.array(b), 0.0)


class TestSvdStable:
    def test_basic_decomposition(self) -> None:
        m = mx.array([[1.0, 0.0], [0.0, 2.0]])
        mx.eval(m)
        u, s, vt = svd_stable(m)
        # Singular values should be [2.0, 1.0] (descending)
        np.testing.assert_allclose(
            sorted(np.array(s), reverse=True), [2.0, 1.0], atol=1e-5,
        )
        # Reconstruction: U @ diag(S) @ Vt ≈ M
        recon = u @ mx.diag(s) @ vt
        mx.eval(recon)
        np.testing.assert_allclose(np.array(recon), np.array(m), atol=1e-5)

    def test_preserves_dtype_float32(self) -> None:
        m = mx.ones((3, 3), dtype=mx.float32)
        mx.eval(m)
        u, _s, vt = svd_stable(m)
        assert u.dtype == mx.float32
        assert vt.dtype == mx.float32

    def test_casts_from_float16(self) -> None:
        m = mx.ones((3, 3), dtype=mx.float16)
        mx.eval(m)
        u, s, vt = svd_stable(m)
        # Vectors should be cast back to float16
        assert u.dtype == mx.float16
        assert vt.dtype == mx.float16
        # S is always float32
        assert s.dtype == mx.float32

    def test_degenerate_zero_matrix(self) -> None:
        m = mx.zeros((3, 3))
        mx.eval(m)
        _u, s, _vt = svd_stable(m)
        np.testing.assert_allclose(np.array(s), 0.0, atol=1e-7)

    def test_rank_one_matrix(self) -> None:
        v = mx.array([[1.0, 2.0, 3.0]])
        m = v.T @ v  # rank-1
        mx.eval(m)
        _, s, _ = svd_stable(m)
        s_np = sorted(np.array(s), reverse=True)
        # Only the first singular value should be non-zero
        assert s_np[0] > 1.0
        np.testing.assert_allclose(s_np[1:], 0.0, atol=1e-5)


class TestQrStable:
    def test_basic_decomposition(self) -> None:
        m = mx.array([[1.0, 2.0], [3.0, 4.0]])
        mx.eval(m)
        q, r = qr_stable(m)
        # Reconstruction: Q @ R ≈ M
        recon = q @ r
        mx.eval(recon)
        np.testing.assert_allclose(np.array(recon), np.array(m), atol=1e-5)

    def test_q_orthogonal(self) -> None:
        m = mx.array([[1.0, 2.0], [3.0, 4.0]])
        mx.eval(m)
        q, _ = qr_stable(m)
        # Q^T Q ≈ I
        qtq = q.T @ q
        mx.eval(qtq)
        np.testing.assert_allclose(
            np.array(qtq), np.eye(2), atol=1e-5,
        )

    def test_r_upper_triangular(self) -> None:
        m = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mx.eval(m)
        _, r = qr_stable(m)
        r_np = np.array(r)
        # Lower triangle should be zero
        for i in range(r_np.shape[0]):
            for j in range(i):
                assert abs(r_np[i, j]) < 1e-5

    def test_identity_matrix(self) -> None:
        m = mx.array([[1.0, 0.0], [0.0, 1.0]])
        mx.eval(m)
        q, _r = qr_stable(m)
        np.testing.assert_allclose(np.abs(np.array(q)), np.eye(2), atol=1e-5)
