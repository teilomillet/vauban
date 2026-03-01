"""Backend compliance tests — run for each backend."""

import importlib

import pytest

from vauban._nn_contract import NN_CONTRACT
from vauban._ops_contract import OPS_CONTRACT


def _try_import(name: str) -> object:
    """Import backend module, skip test if dependency missing."""
    try:
        return importlib.import_module(f"vauban.{name}")
    except ImportError as exc:
        pytest.skip(f"Backend dependency unavailable: {exc}")
        return None  # unreachable, but keeps type checkers happy


@pytest.fixture(params=["mlx", "torch"])
def backend_ops(request: pytest.FixtureRequest) -> object:
    """Load backend ops module."""
    return _try_import(f"_ops_{request.param}")


@pytest.fixture(params=["mlx", "torch"])
def backend_nn(request: pytest.FixtureRequest) -> object:
    """Load backend nn module."""
    return _try_import(f"_nn_{request.param}")


# ====================================================================
# OPS contract tests
# ====================================================================


def test_exports_all_contract_symbols(backend_ops: object) -> None:
    """Every backend must export every symbol in OPS_CONTRACT."""
    missing = [name for name in OPS_CONTRACT if not hasattr(backend_ops, name)]
    assert not missing, f"Missing symbols: {missing}"


def test_no_extra_public_symbols(backend_ops: object) -> None:
    """Backend must not leak internal imports as public symbols."""
    extras = [
        name
        for name in dir(backend_ops)
        if not name.startswith("_") and name not in OPS_CONTRACT
    ]
    assert not extras, f"Extra public symbols (not in contract): {extras}"


def test_sum_semantics(backend_ops: object) -> None:
    """Basic sum reduction."""
    x = backend_ops.array([1.0, 2.0, 3.0])  # type: ignore[attr-defined]
    assert float(backend_ops.sum(x).item()) == pytest.approx(6.0)  # type: ignore[attr-defined]


def test_sum_axis(backend_ops: object) -> None:
    """Sum along axis=0."""
    x = backend_ops.array([[1.0, 2.0], [3.0, 4.0]])  # type: ignore[attr-defined]
    result = backend_ops.sum(x, axis=0)  # type: ignore[attr-defined]
    assert float(result[0].item()) == pytest.approx(4.0)
    assert float(result[1].item()) == pytest.approx(6.0)


def test_concatenate(backend_ops: object) -> None:
    """Concatenate two arrays."""
    a = backend_ops.array([1.0, 2.0])  # type: ignore[attr-defined]
    b = backend_ops.array([3.0, 4.0])  # type: ignore[attr-defined]
    c = backend_ops.concatenate([a, b], axis=0)  # type: ignore[attr-defined]
    assert c.shape[0] == 4


def test_linalg_norm(backend_ops: object) -> None:
    """Linalg norm of a simple vector."""
    x = backend_ops.array([3.0, 4.0])  # type: ignore[attr-defined]
    assert float(backend_ops.linalg.norm(x).item()) == pytest.approx(5.0)  # type: ignore[attr-defined]


def test_dtypes_exist(backend_ops: object) -> None:
    """All declared dtype symbols must exist."""
    for dtype_name in ("float32", "float16", "bfloat16", "int32", "uint32", "bool_"):
        assert hasattr(backend_ops, dtype_name), f"Missing dtype: {dtype_name}"


def test_linalg_svd(backend_ops: object) -> None:
    """SVD produces three arrays."""
    x = backend_ops.array([[1.0, 0.0], [0.0, 2.0]])  # type: ignore[attr-defined]
    u, s, _vt = backend_ops.linalg.svd(x, stream=backend_ops.cpu)  # type: ignore[attr-defined]
    assert u.shape[0] == 2
    assert s.shape[0] == 2


def test_eye(backend_ops: object) -> None:
    """Identity matrix creation."""
    e = backend_ops.eye(3)  # type: ignore[attr-defined]
    assert e.shape == (3, 3)
    assert float(e[0, 0].item()) == pytest.approx(1.0)
    assert float(e[0, 1].item()) == pytest.approx(0.0)


def test_allclose(backend_ops: object) -> None:
    """Approximate equality check."""
    a = backend_ops.array([1.0, 2.0, 3.0])  # type: ignore[attr-defined]
    b = backend_ops.array([1.0, 2.0, 3.0 + 1e-9])  # type: ignore[attr-defined]
    assert backend_ops.allclose(a, b)  # type: ignore[attr-defined]
    c = backend_ops.array([1.0, 2.0, 4.0])  # type: ignore[attr-defined]
    assert not backend_ops.allclose(a, c)  # type: ignore[attr-defined]


def test_array_equal(backend_ops: object) -> None:
    """Exact equality check."""
    a = backend_ops.array([1.0, 2.0, 3.0])  # type: ignore[attr-defined]
    b = backend_ops.array([1.0, 2.0, 3.0])  # type: ignore[attr-defined]
    assert backend_ops.array_equal(a, b)  # type: ignore[attr-defined]
    c = backend_ops.array([1.0, 2.0, 4.0])  # type: ignore[attr-defined]
    assert not backend_ops.array_equal(a, c)  # type: ignore[attr-defined]


def test_tree_flatten(backend_ops: object) -> None:
    """tree_flatten produces (key, leaf) pairs."""
    tree = {"a": 1, "b": {"c": 2, "d": 3}}
    flat = backend_ops.tree_flatten(tree)  # type: ignore[attr-defined]
    keys = [k for k, _ in flat]
    assert "a" in keys
    assert "b.c" in keys
    assert "b.d" in keys


# ====================================================================
# NN contract tests
# ====================================================================


def test_nn_exports_all_contract_symbols(backend_nn: object) -> None:
    """Every NN backend must export every symbol in NN_CONTRACT."""
    missing = [name for name in NN_CONTRACT if not hasattr(backend_nn, name)]
    assert not missing, f"Missing NN symbols: {missing}"


def test_causal_mask_shape(backend_nn: object) -> None:
    """Causal mask has correct shape."""
    mask = backend_nn.create_additive_causal_mask(4)  # type: ignore[attr-defined]
    assert mask.shape == (4, 4)
