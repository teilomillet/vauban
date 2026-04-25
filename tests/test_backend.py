# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for vauban._backend, _ops, _ops_mlx, _nn, and _array."""

import numpy as np
import pytest

from vauban._backend import SUPPORTED_BACKENDS, get_backend
from vauban._nn_contract import NN_CONTRACT
from vauban._ops_contract import OPS_CONTRACT

mx = pytest.importorskip("mlx.core")

# ── _backend ─────────────────────────────────────────────────────────


class TestBackend:
    def test_get_backend_returns_mlx(self) -> None:
        assert get_backend() == "mlx"

    def test_supported_backends(self) -> None:
        assert "mlx" in SUPPORTED_BACKENDS
        assert "torch" in SUPPORTED_BACKENDS
        assert len(SUPPORTED_BACKENDS) == 2


# ── _ops contract compliance ─────────────────────────────────────────


class TestOpsContract:
    def test_all_contract_symbols_accessible(self) -> None:
        """Every symbol in OPS_CONTRACT must be importable from _ops."""
        from vauban import _ops as ops

        for name in OPS_CONTRACT:
            assert hasattr(ops, name), f"Missing ops symbol: {name}"

    def test_ops_dir_matches_contract(self) -> None:
        from vauban import _ops as ops

        ops_dir = set(dir(ops))
        for name in OPS_CONTRACT:
            assert name in ops_dir, f"Missing from __dir__: {name}"

    def test_unknown_attr_raises(self) -> None:
        from vauban import _ops as ops

        with pytest.raises(AttributeError, match="does not export"):
            _ = ops.totally_nonexistent_symbol  # type: ignore[attr-defined]


# ── _ops_mlx implementations ────────────────────────────────────────


class TestOpsMlx:
    def test_array_creation(self) -> None:
        from vauban import _ops as ops

        a = ops.array([1.0, 2.0, 3.0])
        mx.eval(a)
        assert a.shape == (3,)
        np.testing.assert_allclose(np.array(a), [1.0, 2.0, 3.0])

    def test_zeros_and_ones(self) -> None:
        from vauban import _ops as ops

        z = ops.zeros((2, 3))
        o = ops.ones((2, 3))
        mx.eval(z, o)
        np.testing.assert_allclose(np.array(z), 0.0)
        np.testing.assert_allclose(np.array(o), 1.0)

    def test_zeros_like(self) -> None:
        from vauban import _ops as ops

        a = ops.ones((4,))
        z = ops.zeros_like(a)
        mx.eval(z)
        np.testing.assert_allclose(np.array(z), 0.0)
        assert z.shape == (4,)

    def test_arange(self) -> None:
        from vauban import _ops as ops

        a = ops.arange(5)
        mx.eval(a)
        np.testing.assert_allclose(np.array(a), [0, 1, 2, 3, 4])

    def test_full(self) -> None:
        from vauban import _ops as ops

        a = ops.full((3,), 7.0)
        mx.eval(a)
        np.testing.assert_allclose(np.array(a), 7.0)

    def test_sum_and_mean(self) -> None:
        from vauban import _ops as ops

        a = ops.array([1.0, 2.0, 3.0])
        s = ops.sum(a)
        m = ops.mean(a)
        mx.eval(s, m)
        assert float(s) == pytest.approx(6.0)
        assert float(m) == pytest.approx(2.0)

    def test_elementwise_math(self) -> None:
        from vauban import _ops as ops

        a = ops.array([4.0])
        mx.eval(a)
        sq = ops.sqrt(a)
        mx.eval(sq)
        assert float(sq) == pytest.approx(2.0)
        e = ops.exp(ops.array([0.0]))
        mx.eval(e)
        assert float(e) == pytest.approx(1.0)
        lg = ops.log(ops.array([1.0]))
        mx.eval(lg)
        assert float(lg) == pytest.approx(0.0)

    def test_abs(self) -> None:
        from vauban import _ops as ops

        a = ops.array([-3.0, 2.0, -1.0])
        r = ops.abs(a)
        mx.eval(r)
        np.testing.assert_allclose(np.array(r), [3.0, 2.0, 1.0])

    def test_maximum_minimum(self) -> None:
        from vauban import _ops as ops

        a = ops.array([1.0, 3.0])
        b = ops.array([2.0, 1.0])
        mx_max = ops.maximum(a, b)
        mx_min = ops.minimum(a, b)
        mx.eval(mx_max, mx_min)
        np.testing.assert_allclose(np.array(mx_max), [2.0, 3.0])
        np.testing.assert_allclose(np.array(mx_min), [1.0, 1.0])

    def test_clip(self) -> None:
        from vauban import _ops as ops

        a = ops.array([0.0, 5.0, 10.0])
        c = ops.clip(a, 1.0, 8.0)
        mx.eval(c)
        np.testing.assert_allclose(np.array(c), [1.0, 5.0, 8.0])

    def test_where(self) -> None:
        from vauban import _ops as ops

        cond = ops.array([True, False, True])
        a = ops.array([1.0, 2.0, 3.0])
        b = ops.array([4.0, 5.0, 6.0])
        r = ops.where(cond, a, b)
        mx.eval(r)
        np.testing.assert_allclose(np.array(r), [1.0, 5.0, 3.0])

    def test_argmax(self) -> None:
        from vauban import _ops as ops

        a = ops.array([1.0, 3.0, 2.0])
        idx = ops.argmax(a)
        mx.eval(idx)
        assert int(idx) == 1

    def test_softmax(self) -> None:
        from vauban import _ops as ops

        a = ops.array([1.0, 2.0, 3.0])
        s = ops.softmax(a)
        mx.eval(s)
        s_np = np.array(s)
        assert s_np.sum() == pytest.approx(1.0, abs=1e-5)
        assert s_np[2] > s_np[1] > s_np[0]

    def test_matmul(self) -> None:
        from vauban import _ops as ops

        a = ops.array([[1.0, 0.0], [0.0, 1.0]])
        b = ops.array([[3.0], [4.0]])
        r = ops.matmul(a, b)
        mx.eval(r)
        np.testing.assert_allclose(np.array(r), [[3.0], [4.0]])

    def test_concatenate_and_stack(self) -> None:
        from vauban import _ops as ops

        a = ops.array([1.0, 2.0])
        b = ops.array([3.0, 4.0])
        c = ops.concatenate([a, b])
        s = ops.stack([a, b])
        mx.eval(c, s)
        np.testing.assert_allclose(np.array(c), [1.0, 2.0, 3.0, 4.0])
        assert np.array(s).shape == (2, 2)

    def test_reshape_and_expand_dims(self) -> None:
        from vauban import _ops as ops

        a = ops.array([1.0, 2.0, 3.0, 4.0])
        r = ops.reshape(a, (2, 2))
        mx.eval(r)
        assert np.array(r).shape == (2, 2)
        e = ops.expand_dims(a, axis=0)
        mx.eval(e)
        assert np.array(e).shape == (1, 4)

    def test_outer(self) -> None:
        from vauban import _ops as ops

        a = ops.array([1.0, 2.0])
        b = ops.array([3.0, 4.0])
        r = ops.outer(a, b)
        mx.eval(r)
        np.testing.assert_allclose(np.array(r), [[3.0, 4.0], [6.0, 8.0]])

    def test_dtypes_exist(self) -> None:
        from vauban import _ops as ops

        assert ops.float32 is not None
        assert ops.float16 is not None
        assert ops.bfloat16 is not None
        assert ops.int32 is not None
        assert ops.uint32 is not None
        assert ops.bool_ is not None

    def test_linalg_namespace(self) -> None:
        from vauban import _ops as ops

        assert hasattr(ops.linalg, "norm")
        assert hasattr(ops.linalg, "svd")
        assert hasattr(ops.linalg, "qr")

    def test_random_namespace(self) -> None:
        from vauban import _ops as ops

        assert hasattr(ops.random, "normal")
        assert hasattr(ops.random, "seed")

    def test_stop_gradient(self) -> None:
        from vauban import _ops as ops

        a = ops.array([1.0, 2.0])
        b = ops.stop_gradient(a)
        mx.eval(b)
        np.testing.assert_allclose(np.array(b), [1.0, 2.0])

    def test_sort_and_argsort(self) -> None:
        from vauban import _ops as ops

        a = ops.array([3.0, 1.0, 2.0])
        s = ops.sort(a)
        idx = ops.argsort(a)
        mx.eval(s, idx)
        np.testing.assert_allclose(np.array(s), [1.0, 2.0, 3.0])
        np.testing.assert_allclose(np.array(idx), [1, 2, 0])

    def test_tree_flatten(self) -> None:
        from vauban import _ops as ops

        d = {"a": mx.ones((2,)), "b": {"c": mx.zeros((3,))}}
        flat = ops.tree_flatten(d)
        assert isinstance(flat, list)
        assert len(flat) == 2


# ── _nn contract compliance ──────────────────────────────────────────


class TestNnContract:
    def test_nn_dir_matches_contract(self) -> None:
        from vauban import _nn as nn_ops

        nn_dir = set(dir(nn_ops))
        for name in NN_CONTRACT:
            assert name in nn_dir, f"Missing from __dir__: {name}"

    def test_all_nn_symbols_accessible(self) -> None:
        from vauban import _nn as nn_ops

        for name in NN_CONTRACT:
            assert hasattr(nn_ops, name), f"Missing nn symbol: {name}"

    def test_create_additive_causal_mask(self) -> None:
        from vauban import _nn as nn_ops

        mask = nn_ops.create_additive_causal_mask(4)
        mx.eval(mask)
        m_np = np.array(mask)
        assert m_np.shape == (4, 4)
        # Upper triangle (future tokens) should be very large negative
        assert m_np[0, 1] < -1e30
        # Diagonal and below should be 0
        assert m_np[0, 0] == 0.0
        assert m_np[1, 0] == 0.0

    def test_cross_entropy(self) -> None:
        from vauban import _nn as nn_ops

        logits = mx.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        targets = mx.array([0, 1])
        mx.eval(logits, targets)
        loss = nn_ops.cross_entropy(logits, targets)
        mx.eval(loss)
        # With confident predictions, loss should be very small
        assert float(loss) < 0.1

    def test_unknown_nn_attr_raises(self) -> None:
        from vauban import _nn as nn_ops

        with pytest.raises(AttributeError, match="does not export"):
            _ = nn_ops.totally_nonexistent_symbol  # type: ignore[attr-defined]


# ── _array ───────────────────────────────────────────────────────────


class TestArray:
    def test_array_alias_is_mx_array(self) -> None:
        from vauban._array import Array

        assert Array is mx.array
