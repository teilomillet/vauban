# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Fake-module tests for ``vauban._ops_torch`` and ``vauban._nn_torch``."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


@pytest.fixture(autouse=True)
def _reload_real_torch_backend_after_fake_tests() -> Iterator[None]:
    """Drop fake-backed modules so later tests import the selected backend."""
    yield
    sys.modules.pop("vauban._ops_torch", None)
    sys.modules.pop("vauban._nn_torch", None)


class FakeTensor:
    """Tiny tensor wrapper backed by ``numpy`` arrays."""

    def __init__(
        self,
        data: object,
        *,
        dtype: object | None = None,
    ) -> None:
        np_dtype = dtype if isinstance(dtype, type | np.dtype) else None
        self.data = np.array(data, dtype=np_dtype)
        self.grad: FakeTensor | None = None
        self.requires_grad = False
        self._backward_callback: Callable[[], None] | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        """Expose the tensor shape."""
        return tuple(int(dim) for dim in self.data.shape)

    def unsqueeze(self, axis: int) -> FakeTensor:
        """Insert a singleton axis."""
        return FakeTensor(np.expand_dims(self.data, axis))

    def detach(self) -> FakeTensor:
        """Return a detached copy."""
        return FakeTensor(self.data.copy())

    def requires_grad_(self, required: bool) -> FakeTensor:
        """Toggle gradient tracking in-place."""
        self.requires_grad = required
        return self

    def clone(self) -> FakeTensor:
        """Return a cloned tensor."""
        return FakeTensor(self.data.copy())

    def backward(self) -> None:
        """Run the stored backward callback when present."""
        if self._backward_callback is not None:
            self._backward_callback()


def _as_array(value: FakeTensor | object) -> np.ndarray:
    """Convert fake tensors and scalars into ``numpy`` arrays."""
    if isinstance(value, FakeTensor):
        return value.data
    return np.array(value)


def _wrap_array(value: FakeTensor | object) -> FakeTensor:
    """Wrap raw values into ``FakeTensor`` instances."""
    if isinstance(value, FakeTensor):
        return value
    return FakeTensor(value)


class FakeTorchLinalgModule(ModuleType):
    """Typed fake ``torch.linalg`` namespace."""

    def __init__(self) -> None:
        super().__init__("torch.linalg")

    def norm(
        self,
        matrix: FakeTensor,
        *,
        dim: int | tuple[int, ...] | None = None,
        keepdim: bool = False,
    ) -> FakeTensor:
        """Compute an L2 norm."""
        return FakeTensor(
            np.linalg.norm(_as_array(matrix), axis=dim, keepdims=keepdim),
        )

    def svd(self, matrix: FakeTensor) -> tuple[FakeTensor, FakeTensor, FakeTensor]:
        """Compute an SVD."""
        u, s, vt = np.linalg.svd(_as_array(matrix), full_matrices=True)
        return FakeTensor(u), FakeTensor(s), FakeTensor(vt)

    def qr(self, matrix: FakeTensor) -> tuple[FakeTensor, FakeTensor]:
        """Compute a QR decomposition."""
        q, r = np.linalg.qr(_as_array(matrix))
        return FakeTensor(q), FakeTensor(r)


class FakeTorchModule(ModuleType):
    """Typed fake ``torch`` module."""

    Tensor = FakeTensor

    def __init__(self) -> None:
        super().__init__("torch")
        self.float32 = np.float32
        self.float16 = np.float16
        self.bfloat16 = np.float16
        self.int32 = np.int32
        self.bool = np.bool_
        self.last_seed: int | None = None
        self.lingalg_marker = object()
        self.linalg = FakeTorchLinalgModule()

    def tensor(self, data: object, dtype: object | None = None) -> FakeTensor:
        """Create a tensor."""
        return FakeTensor(data, dtype=dtype)

    def zeros(self, shape: tuple[int, ...]) -> FakeTensor:
        """Create zeros."""
        return FakeTensor(np.zeros(shape))

    def zeros_like(self, value: FakeTensor) -> FakeTensor:
        """Create zeros like another tensor."""
        return FakeTensor(np.zeros_like(_as_array(value)))

    def ones(self, shape: tuple[int, ...], dtype: object | None = None) -> FakeTensor:
        """Create ones."""
        return FakeTensor(np.ones(shape), dtype=dtype)

    def arange(self, stop: int) -> FakeTensor:
        """Create a range tensor."""
        return FakeTensor(np.arange(stop))

    def full(self, shape: tuple[int, ...], fill_value: float) -> FakeTensor:
        """Create a filled tensor."""
        return FakeTensor(np.full(shape, fill_value, dtype=np.float32))

    def abs(self, value: FakeTensor) -> FakeTensor:
        """Elementwise absolute value."""
        return FakeTensor(np.abs(_as_array(value)))

    def exp(self, value: FakeTensor) -> FakeTensor:
        """Elementwise exponent."""
        return FakeTensor(np.exp(_as_array(value)))

    def log(self, value: FakeTensor) -> FakeTensor:
        """Elementwise logarithm."""
        return FakeTensor(np.log(_as_array(value)))

    def sqrt(self, value: FakeTensor) -> FakeTensor:
        """Elementwise square root."""
        return FakeTensor(np.sqrt(_as_array(value)))

    def maximum(self, left: FakeTensor, right: FakeTensor) -> FakeTensor:
        """Elementwise maximum."""
        return FakeTensor(np.maximum(_as_array(left), _as_array(right)))

    def minimum(self, left: FakeTensor, right: FakeTensor) -> FakeTensor:
        """Elementwise minimum."""
        return FakeTensor(np.minimum(_as_array(left), _as_array(right)))

    def outer(self, left: FakeTensor, right: FakeTensor) -> FakeTensor:
        """Outer product."""
        return FakeTensor(np.outer(_as_array(left), _as_array(right)))

    def arccos(self, value: FakeTensor) -> FakeTensor:
        """Elementwise arccos."""
        return FakeTensor(np.arccos(_as_array(value)))

    def cos(self, value: FakeTensor) -> FakeTensor:
        """Elementwise cosine."""
        return FakeTensor(np.cos(_as_array(value)))

    def matmul(self, left: FakeTensor, right: FakeTensor) -> FakeTensor:
        """Matrix multiply."""
        return FakeTensor(np.matmul(_as_array(left), _as_array(right)))

    def reshape(self, value: FakeTensor, shape: tuple[int, ...]) -> FakeTensor:
        """Reshape a tensor."""
        return FakeTensor(np.reshape(_as_array(value), shape))

    def where(
        self,
        condition: FakeTensor,
        left: FakeTensor,
        right: FakeTensor,
    ) -> FakeTensor:
        """Select values by condition."""
        return FakeTensor(
            np.where(
                _as_array(condition),
                _as_array(left),
                _as_array(right),
            ),
        )

    def eye(self, size: int) -> FakeTensor:
        """Identity matrix."""
        return FakeTensor(np.eye(size))

    def allclose(
        self,
        left: FakeTensor,
        right: FakeTensor,
        *,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> bool:
        """Approximate tensor equality."""
        return bool(
            np.allclose(
                _as_array(left),
                _as_array(right),
                rtol=rtol,
                atol=atol,
            ),
        )

    def equal(self, left: FakeTensor, right: FakeTensor) -> bool:
        """Exact tensor equality."""
        return bool(np.array_equal(_as_array(left), _as_array(right)))

    def sum(
        self,
        value: FakeTensor,
        *,
        dim: int | tuple[int, ...] | None = None,
        keepdim: bool = False,
    ) -> FakeTensor:
        """Reduce with sum."""
        return FakeTensor(np.sum(_as_array(value), axis=dim, keepdims=keepdim))

    def mean(
        self,
        value: FakeTensor,
        *,
        dim: int | tuple[int, ...] | None = None,
        keepdim: bool = False,
    ) -> FakeTensor:
        """Reduce with mean."""
        return FakeTensor(np.mean(_as_array(value), axis=dim, keepdims=keepdim))

    def argmax(
        self,
        value: FakeTensor,
        *,
        dim: int | None = None,
    ) -> FakeTensor:
        """Argmax reduction."""
        return FakeTensor(np.argmax(_as_array(value), axis=dim))

    def softmax(self, value: FakeTensor, *, dim: int) -> FakeTensor:
        """Compute softmax along one axis."""
        array = _as_array(value)
        shifted = array - np.max(array, axis=dim, keepdims=True)
        probs = np.exp(shifted)
        return FakeTensor(probs / np.sum(probs, axis=dim, keepdims=True))

    def cat(self, values: list[FakeTensor], *, dim: int = 0) -> FakeTensor:
        """Concatenate tensors."""
        return FakeTensor(np.concatenate([_as_array(v) for v in values], axis=dim))

    def clamp(
        self,
        value: FakeTensor,
        *,
        min: float | None = None,
        max: float | None = None,
    ) -> FakeTensor:
        """Clamp values."""
        lower = -np.inf if min is None else min
        upper = np.inf if max is None else max
        return FakeTensor(np.clip(_as_array(value), lower, upper))

    def sort(self, value: FakeTensor, *, dim: int = -1) -> FakeTensor:
        """Sort values along one axis."""
        return FakeTensor(np.sort(_as_array(value), axis=dim))

    def argsort(self, value: FakeTensor, *, dim: int = -1) -> FakeTensor:
        """Argsort values along one axis."""
        return FakeTensor(np.argsort(_as_array(value), axis=dim))

    def stack(self, values: list[FakeTensor], *, dim: int = 0) -> FakeTensor:
        """Stack tensors."""
        return FakeTensor(np.stack([_as_array(v) for v in values], axis=dim))

    def randn(self, shape: tuple[int, ...]) -> FakeTensor:
        """Return deterministic pseudo-random data."""
        return FakeTensor(np.arange(np.prod(shape), dtype=float).reshape(shape))

    def manual_seed(self, seed: int) -> None:
        """Record the seed."""
        self.last_seed = seed

    def multinomial(self, probs: FakeTensor, num_samples: int) -> FakeTensor:
        """Pick the argmax index repeatedly."""
        indices = np.argmax(_as_array(probs), axis=-1)
        expanded = np.expand_dims(indices, axis=-1)
        if num_samples > 1:
            expanded = np.repeat(expanded, num_samples, axis=-1)
        return FakeTensor(expanded)

    def triu(self, value: FakeTensor, diagonal: int = 0) -> FakeTensor:
        """Upper-triangular view."""
        return FakeTensor(np.triu(_as_array(value), k=diagonal))


class FakeTorchFunctionalModule(ModuleType):
    """Typed fake ``torch.nn.functional`` module."""

    def __init__(self) -> None:
        super().__init__("torch.nn.functional")
        self.cross_entropy_calls: list[tuple[FakeTensor, FakeTensor, str]] = []

    def cross_entropy(
        self,
        logits: FakeTensor,
        targets: FakeTensor,
        *,
        reduction: str = "mean",
    ) -> FakeTensor:
        """Record calls and return a simple scalar tensor."""
        self.cross_entropy_calls.append((logits, targets, reduction))
        value = float(np.sum(_as_array(logits)) + np.sum(_as_array(targets)))
        return FakeTensor(np.array(value))


class FakeTorchNNModule(ModuleType):
    """Typed fake ``torch.nn`` module."""

    functional: FakeTorchFunctionalModule


class FakeSafetensorsTorchModule(ModuleType):
    """Typed fake ``safetensors.torch`` module."""

    def __init__(self) -> None:
        super().__init__("safetensors.torch")
        self.saved: list[tuple[dict[str, FakeTensor], str]] = []
        self.loaded: dict[str, FakeTensor] = {"loaded": FakeTensor([1.0])}

    def save_file(self, weights: dict[str, FakeTensor], path: str) -> None:
        """Record saved weights."""
        self.saved.append((weights, path))

    def load_file(self, path: str) -> dict[str, FakeTensor]:
        """Return a deterministic weight dict."""
        del path
        return dict(self.loaded)


class FakeSafetensorsModule(ModuleType):
    """Typed fake top-level ``safetensors`` module."""

    torch: FakeSafetensorsTorchModule


def _load_torch_backend_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    FakeTorchModule,
    FakeTorchFunctionalModule,
    FakeSafetensorsTorchModule,
    ModuleType,
    ModuleType,
]:
    """Install fake torch modules and import the backend modules."""
    fake_torch = FakeTorchModule()
    fake_functional = FakeTorchFunctionalModule()
    fake_nn = FakeTorchNNModule("torch.nn")
    fake_nn.functional = fake_functional
    fake_safetensors_torch = FakeSafetensorsTorchModule()
    fake_safetensors = FakeSafetensorsModule("safetensors")
    fake_safetensors.torch = fake_safetensors_torch

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.nn", fake_nn)
    monkeypatch.setitem(sys.modules, "torch.nn.functional", fake_functional)
    monkeypatch.setitem(sys.modules, "safetensors", fake_safetensors)
    monkeypatch.setitem(sys.modules, "safetensors.torch", fake_safetensors_torch)
    sys.modules.pop("vauban._ops_torch", None)
    sys.modules.pop("vauban._nn_torch", None)

    ops_module = importlib.import_module("vauban._ops_torch")
    nn_module = importlib.import_module("vauban._nn_torch")
    return (
        fake_torch,
        fake_functional,
        fake_safetensors_torch,
        ops_module,
        nn_module,
    )


class TestTorchBackendFake:
    """Coverage tests for the torch backend shims."""

    def test_wrapped_ops_translate_kwargs_and_aliases(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_torch, _functional, _safetensors, ops_module, _nn_module = (
            _load_torch_backend_modules(monkeypatch)
        )

        tensor = ops_module.array([[1.0, -2.0], [3.0, -4.0]])
        summed = ops_module.sum(tensor, axis=1, keepdims=True)
        meaned = ops_module.mean(tensor, axis=0)
        argmaxed = ops_module.argmax(tensor, axis=1)
        softmaxed = ops_module.softmax(
            ops_module.array([[0.0, 1.0]]),
            axis=1,
        )
        concatenated = ops_module.concatenate(
            [ops_module.array([[1.0]]), ops_module.array([[2.0]])],
            axis=0,
        )
        clipped = ops_module.clip(
            ops_module.array([-1.0, 0.5, 2.0]),
            min=0.0,
            max=1.0,
        )
        normed = ops_module.linalg.norm(
            ops_module.array([[3.0, 4.0]]),
            axis=1,
            keepdims=True,
        )
        sorted_values = ops_module.sort(ops_module.array([3.0, 1.0, 2.0]))
        sorted_indices = ops_module.argsort(ops_module.array([3.0, 1.0, 2.0]))
        stacked = ops_module.stack(
            [ops_module.array([1.0]), ops_module.array([2.0])],
            axis=0,
        )

        assert ops_module.array_type is fake_torch.Tensor
        assert np.array_equal(_as_array(summed), np.array([[-1.0], [-1.0]]))
        assert np.array_equal(_as_array(meaned), np.array([2.0, -3.0]))
        assert np.array_equal(_as_array(argmaxed), np.array([0, 0]))
        assert np.allclose(np.sum(_as_array(softmaxed), axis=1), np.array([1.0]))
        assert np.array_equal(_as_array(concatenated), np.array([[1.0], [2.0]]))
        assert np.array_equal(_as_array(clipped), np.array([0.0, 0.5, 1.0]))
        assert np.array_equal(_as_array(normed), np.array([[5.0]]))
        assert np.array_equal(_as_array(sorted_values), np.array([1.0, 2.0, 3.0]))
        assert np.array_equal(_as_array(sorted_indices), np.array([1, 2, 0]))
        assert np.array_equal(_as_array(stacked), np.array([[1.0], [2.0]]))

    def test_direct_aliases_and_comparisons(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _fake_torch, _functional, _safetensors, ops_module, _nn_module = (
            _load_torch_backend_modules(monkeypatch)
        )

        zeros = ops_module.zeros((2, 2))
        zeros_like = ops_module.zeros_like(ops_module.array([1.0, 2.0]))
        ones = ops_module.ones((2, 2))
        arange = ops_module.arange(3)
        full = ops_module.full((2, 2), 7.0)
        absolute = ops_module.abs(ops_module.array([-2.0, 3.0]))
        logged = ops_module.log(ops_module.exp(ops_module.array([1.0])))
        rooted = ops_module.sqrt(ops_module.array([9.0]))
        maximum = ops_module.maximum(
            ops_module.array([1.0, 5.0]),
            ops_module.array([2.0, 4.0]),
        )
        minimum = ops_module.minimum(
            ops_module.array([1.0, 5.0]),
            ops_module.array([2.0, 4.0]),
        )
        outer = ops_module.outer(
            ops_module.array([1.0, 2.0]),
            ops_module.array([3.0, 4.0]),
        )
        arccos = ops_module.arccos(ops_module.array([1.0]))
        cosine = ops_module.cos(ops_module.array([0.0]))
        matmul = ops_module.matmul(
            ops_module.array([[1.0, 2.0]]),
            ops_module.array([[3.0], [4.0]]),
        )
        reshaped = ops_module.reshape(ops_module.array([1.0, 2.0, 3.0, 4.0]), (2, 2))
        selected = ops_module.where(
            ops_module.array([True, False]),
            ops_module.array([1.0, 2.0]),
            ops_module.array([3.0, 4.0]),
        )
        eye = ops_module.eye(2)

        assert np.array_equal(_as_array(zeros), np.zeros((2, 2)))
        assert np.array_equal(_as_array(zeros_like), np.zeros(2))
        assert np.array_equal(_as_array(ones), np.ones((2, 2)))
        assert np.array_equal(_as_array(arange), np.array([0, 1, 2]))
        assert np.array_equal(_as_array(full), np.full((2, 2), 7.0))
        assert np.array_equal(_as_array(absolute), np.array([2.0, 3.0]))
        assert np.allclose(_as_array(logged), np.array([1.0]))
        assert np.array_equal(_as_array(rooted), np.array([3.0]))
        assert np.array_equal(_as_array(maximum), np.array([2.0, 5.0]))
        assert np.array_equal(_as_array(minimum), np.array([1.0, 4.0]))
        assert np.array_equal(_as_array(outer), np.array([[3.0, 4.0], [6.0, 8.0]]))
        assert np.array_equal(_as_array(arccos), np.array([0.0]))
        assert np.array_equal(_as_array(cosine), np.array([1.0]))
        assert np.array_equal(_as_array(matmul), np.array([[11.0]]))
        assert np.array_equal(_as_array(reshaped), np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert np.array_equal(_as_array(selected), np.array([1.0, 4.0]))
        assert np.array_equal(_as_array(eye), np.eye(2))
        assert ops_module.allclose(
            ops_module.array([1.0, 2.0]),
            ops_module.array([1.0, 2.0 + 1e-7]),
        )
        assert ops_module.array_equal(
            ops_module.array([1.0, 2.0]),
            ops_module.array([1.0, 2.0]),
        )

    def test_custom_ops_save_load_and_grad(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _fake_torch, _functional, fake_safetensors, ops_module, _nn_module = (
            _load_torch_backend_modules(monkeypatch)
        )

        expanded = ops_module.expand_dims(ops_module.array([1.0, 2.0]), axis=0)
        partitioned = ops_module.argpartition(ops_module.array([3.0, 1.0, 2.0]), kth=1)
        ops_module.save_safetensors(
            "weights.safetensors",
            {"w": ops_module.array([1.0])},
        )
        loaded = ops_module.load("weights.safetensors")

        def _loss_fn(x: FakeTensor) -> FakeTensor:
            loss = FakeTensor(np.array(np.sum(x.data)))

            def _backward() -> None:
                x.grad = FakeTensor(np.full_like(x.data, 2.0))

            loss._backward_callback = _backward
            return loss

        loss_value, grad_value = ops_module.value_and_grad(_loss_fn)(
            ops_module.array([1.0, 2.0]),
        )

        def _constant_fn(x: FakeTensor) -> FakeTensor:
            del x
            return FakeTensor(np.array(0.0))

        with np.testing.assert_raises_regex(
            RuntimeError,
            "loss did not depend on the input tensor",
        ):
            ops_module.value_and_grad(_constant_fn)(ops_module.array([1.0]))

        stopped = ops_module.stop_gradient(ops_module.array([1.0, 2.0]))
        assert np.array_equal(_as_array(expanded), np.array([[1.0, 2.0]]))
        assert np.array_equal(_as_array(partitioned), np.array([1, 2, 0]))
        assert fake_safetensors.saved[0][1] == "weights.safetensors"
        assert "loaded" in loaded
        assert np.array_equal(_as_array(loss_value), np.array(3.0))
        assert np.array_equal(_as_array(grad_value), np.array([2.0, 2.0]))
        assert np.array_equal(_as_array(stopped), np.array([1.0, 2.0]))
        assert ops_module.eval(ops_module.array([1.0])) is None

    def test_linalg_random_tree_flatten_and_nn_backend(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_torch, functional, _safetensors, ops_module, nn_module = (
            _load_torch_backend_modules(monkeypatch)
        )

        u, s, vt = ops_module.linalg.svd(
            ops_module.array([[3.0, 0.0], [0.0, 4.0]]),
            stream=object(),
        )
        q, r = ops_module.linalg.qr(
            ops_module.array([[1.0, 0.0], [0.0, 1.0]]),
            stream=object(),
        )
        normal = ops_module.random.normal((2, 2))
        ops_module.random.seed(123)
        categorical = ops_module.random.categorical(
            ops_module.array([[0.1, 2.0, 0.2]]),
            num_samples=2,
        )
        flattened = ops_module.tree_flatten(
            {"a": [1, {"b": 2}], "c": (3, 4)},
        )
        mask = nn_module.create_additive_causal_mask(3)
        loss = nn_module.cross_entropy(
            ops_module.array([[1.0, 2.0], [3.0, 4.0]]),
            ops_module.array([1, 0]),
            reduction="sum",
        )

        assert _as_array(u).shape == (2, 2)
        assert _as_array(s).shape == (2,)
        assert _as_array(vt).shape == (2, 2)
        assert np.array_equal(_as_array(q), np.eye(2))
        assert np.array_equal(_as_array(r), np.eye(2))
        assert np.array_equal(_as_array(normal), np.array([[0.0, 1.0], [2.0, 3.0]]))
        assert fake_torch.last_seed == 123
        assert np.array_equal(_as_array(categorical), np.array([[1, 1]]))
        assert flattened == [
            ("a.0", 1),
            ("a.1.b", 2),
            ("c.0", 3),
            ("c.1", 4),
        ]
        assert np.isneginf(_as_array(mask)[0, 1])
        assert _as_array(mask)[1, 0] == 0.0
        assert functional.cross_entropy_calls[0][2] == "sum"
        assert np.array_equal(_as_array(loss), np.array(11.0))
