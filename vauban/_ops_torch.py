# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""PyTorch tensor operations backend.

API normalization is data-driven via translation tables.
Ops with fundamentally different semantics get custom implementations.
Ops without true PyTorch equivalents raise ``NotImplementedError``.
"""

import functools as _functools
from collections.abc import Callable as _Callable
from typing import cast as _cast

import torch as _torch

_Array = _torch.Tensor
type _GradTarget = _Array | list[_Array] | tuple[_Array, ...]
type _GradOutput = _GradTarget | tuple[_GradTarget, ...]


def _tensor_astype(self: _Array, dtype: object) -> _Array:
    """Torch tensor adapter for MLX-style ``array.astype(dtype)`` calls."""
    return self.to(dtype=_cast("_torch.dtype", dtype))


if not hasattr(_torch.Tensor, "astype"):
    _torch.Tensor.astype = _tensor_astype  # type: ignore[attr-defined]

# ====================================================================
# Translation tables (single source of truth for API differences)
# ====================================================================

_KWARG_MAP: dict[str, str] = {
    "axis": "dim",
    "keepdims": "keepdim",
}

_FUNC_RENAMES: dict[str, str] = {
    "concatenate": "cat",
    "clip": "clamp",
}


def _wrap(mlx_name: str) -> _Callable[..., _Array]:
    """Auto-wrap a torch function to accept MLX-style kwargs."""
    torch_name = _FUNC_RENAMES.get(mlx_name, mlx_name)
    target: object = _torch
    for part in torch_name.split("."):
        target = getattr(target, part)
    fn = _cast("_Callable[..., _Array]", target)

    @_functools.wraps(fn)
    def wrapper(*args: object, **kwargs: object) -> _Array:
        return fn(*args, **{_KWARG_MAP.get(k, k): v for k, v in kwargs.items()})

    return wrapper


# ====================================================================
# Auto-wrapped ops (translation table handles axis->dim etc.)
# ====================================================================

sum = _wrap("sum")
mean = _wrap("mean")
argmax = _wrap("argmax")
concatenate = _wrap("concatenate")
clip = _wrap("clip")
argsort = _wrap("argsort")
stack = _wrap("stack")

# ====================================================================
# Direct aliases (identical API)
# ====================================================================

array = _torch.tensor
array_type: type = _torch.Tensor  # for isinstance checks (array is a constructor)
zeros = _torch.zeros
zeros_like = _torch.zeros_like
ones = _torch.ones
arange = _torch.arange
full = _torch.full
abs = _torch.abs
exp = _torch.exp
log = _torch.log
sqrt = _torch.sqrt
outer = _torch.outer
arccos = _torch.arccos
cos = _torch.cos
matmul = _torch.matmul
reshape = _torch.reshape
eye = _torch.eye


# ====================================================================
# Comparison
# ====================================================================


def allclose(
    a: _Array, b: _Array, rtol: float = 1e-5, atol: float = 1e-8,
) -> bool:
    """Element-wise approximate equality, matching MLX ``allclose`` semantics."""
    return bool(_torch.allclose(a, b, rtol=rtol, atol=atol))


def array_equal(a: _Array, b: _Array) -> bool:
    """Element-wise exact equality, matching MLX ``array_equal`` semantics."""
    return bool(_torch.equal(a, b))


# ====================================================================
# Custom implementations (different semantics)
# ====================================================================


def expand_dims(x: _Array, axis: int) -> _Array:
    """Unsqueeze a dimension (MLX ``expand_dims`` -> torch ``unsqueeze``)."""
    return x.unsqueeze(axis)


def to_device_like(x: _Array, reference: _Array) -> _Array:
    """Move ``x`` to the same torch device as ``reference``."""
    device = getattr(reference, "device", None)
    if not isinstance(device, (_torch.device, str, int)):
        return x
    return x.to(device=device)


def _align_scalar_devices(left: _Array, right: _Array) -> tuple[_Array, _Array]:
    """Move scalar tensors across devices to match non-scalar operands."""
    left_device = getattr(left, "device", None)
    right_device = getattr(right, "device", None)
    if left_device == right_device:
        return left, right
    left_ndim = getattr(left, "ndim", None)
    right_ndim = getattr(right, "ndim", None)
    if right_ndim == 0:
        return left, right.to(device=left_device)
    if left_ndim == 0:
        return left.to(device=right_device), right
    return left, right


def maximum(left: _Array, right: _Array) -> _Array:
    """Element-wise maximum with MLX-friendly scalar device alignment."""
    left_aligned, right_aligned = _align_scalar_devices(left, right)
    return _torch.maximum(left_aligned, right_aligned)


def minimum(left: _Array, right: _Array) -> _Array:
    """Element-wise minimum with MLX-friendly scalar device alignment."""
    left_aligned, right_aligned = _align_scalar_devices(left, right)
    return _torch.minimum(left_aligned, right_aligned)


def where(condition: _Array, left: _Array, right: _Array) -> _Array:
    """Select values with scalar branch tensors moved to the value device."""
    left_aligned, right_aligned = _align_scalar_devices(left, right)
    condition_device = getattr(condition, "device", None)
    value_device = getattr(left_aligned, "device", None)
    if condition_device != value_device and hasattr(condition, "to"):
        condition = condition.to(device=value_device)
    return _torch.where(condition, left_aligned, right_aligned)


def argpartition(x: _Array, kth: int) -> _Array:
    """Partial sort by index.

    PyTorch lacks native ``argpartition``. This uses ``argsort`` for correct
    semantics (O(n log n) instead of O(n), but exact).

    Vauban usage: ``argpartition(x, kth=n-k)[n-k:]`` to get top-k indices.
    """
    return _torch.argsort(x)


def softmax(x: _Array, axis: int = -1) -> _Array:
    """Softmax with MLX-compatible default axis semantics."""
    return _torch.softmax(x, dim=axis)


def sort(x: _Array, axis: int = -1) -> _Array:
    """Return sorted values only, matching MLX ``sort`` semantics."""
    result = _torch.sort(x, dim=axis)
    values = getattr(result, "values", None)
    if values is not None:
        return _cast("_Array", values)
    if isinstance(result, tuple):
        return result[0]
    return _cast("_Array", result)


def save_safetensors(path: str, weights: dict[str, _Array]) -> None:
    """Save weight dict via safetensors library."""
    from safetensors.torch import save_file

    save_file(weights, path)


def load(path: str) -> dict[str, _Array]:
    """Load safetensors file."""
    from safetensors.torch import load_file

    return load_file(str(path))


def value_and_grad(
    fn: _Callable[..., _Array],
    argnums: int | tuple[int, ...] = 0,
) -> _Callable[..., tuple[_Array, _GradOutput]]:
    """MLX-compatible ``value_and_grad``.

    Contract:
    - ``fn`` must accept Array args and return a scalar Array (loss).
    - Returns ``(loss, grad)`` or ``(loss, tuple(grads))``.
    - selected args may be tensors or flat tensor lists/tuples.
    - ``loss`` is detached (no grad graph).
    - each grad is a fresh tensor (not a view into autograd state).
    """

    return _value_and_grad_impl(fn, argnums)


def _value_and_grad_impl(
    fn: _Callable[..., _Array],
    argnums: int | tuple[int, ...],
) -> _Callable[..., tuple[_Array, _GradOutput]]:
    """Return a value-and-gradient wrapper for selected positional args."""
    selected = (argnums,) if isinstance(argnums, int) else argnums

    def wrapper(*args: object, **kwargs: object) -> tuple[_Array, _GradOutput]:
        call_args = list(args)
        grad_targets: list[tuple[object, list[_Array]]] = []
        for index in selected:
            replacement, leaves = _prepare_grad_target(args[index])
            call_args[index] = replacement
            grad_targets.append((args[index], leaves))
        loss = fn(*call_args, **kwargs)
        loss.backward()
        grads = [
            _collect_grad_target(original, leaves)
            for original, leaves in grad_targets
        ]
        if isinstance(argnums, int):
            return loss.detach(), grads[0]
        return loss.detach(), tuple(grads)

    return wrapper


def _prepare_grad_target(target: object) -> tuple[object, list[_Array]]:
    """Clone selected tensors into autograd leaves."""
    if isinstance(target, _torch.Tensor):
        tensor = target.detach().clone().requires_grad_(True)
        return tensor, [tensor]
    if isinstance(target, list):
        tensors = [_prepare_grad_leaf(item) for item in target]
        return tensors, tensors
    if isinstance(target, tuple):
        tensors = [_prepare_grad_leaf(item) for item in target]
        return tuple(tensors), tensors
    msg = (
        "value_and_grad selected args must be tensors or flat "
        f"tensor lists/tuples, got {type(target).__name__}"
    )
    raise TypeError(msg)


def _prepare_grad_leaf(value: object) -> _Array:
    """Clone a tensor parameter into an autograd leaf."""
    if not isinstance(value, _torch.Tensor):
        msg = (
            "value_and_grad parameter lists must contain tensors, "
            f"got {type(value).__name__}"
        )
        raise TypeError(msg)
    return value.detach().clone().requires_grad_(True)


def _collect_grad_target(original: object, leaves: list[_Array]) -> _GradTarget:
    """Return gradients with the same flat container shape as the target."""
    grads = [_gradient_for_leaf(leaf) for leaf in leaves]
    if isinstance(original, _torch.Tensor):
        return grads[0]
    if isinstance(original, list):
        return grads
    if isinstance(original, tuple):
        return tuple(grads)
    msg = f"unexpected value_and_grad target type: {type(original).__name__}"
    raise TypeError(msg)


def _gradient_for_leaf(tensor: _Array) -> _Array:
    """Return a cloned gradient for an autograd leaf."""
    grad = tensor.grad
    if grad is None:
        msg = "value_and_grad: loss did not depend on the input tensor"
        raise RuntimeError(msg)
    return grad.clone()


def stop_gradient(x: _Array) -> _Array:
    """Detach tensor from computation graph."""
    return x.detach()


def eval(*args: _Array) -> None:
    """No-op — PyTorch is eager, no lazy evaluation."""


# ====================================================================
# Types / dtypes
# ====================================================================

float32 = _torch.float32
float16 = _torch.float16
bfloat16 = _torch.bfloat16
int32 = _torch.int32
bool_ = _torch.bool

# uint32: PyTorch lacks native uint32. We use int32 with a guard.
# Consumer code uses uint32 only for dequantization scale detection.
uint32 = _torch.int32  # KNOWN LIMITATION: documented in _ops_contract


# ====================================================================
# Sub-namespaces
# ====================================================================


class linalg:  # noqa: N801 — lowercase to match MLX API
    """Linear algebra sub-namespace."""

    norm = _wrap("linalg.norm")

    @staticmethod
    def svd(
        matrix: _Array, stream: object = None,
    ) -> tuple[_Array, _Array, _Array]:
        """SVD. ``stream`` parameter is ignored (MLX-specific)."""
        return _torch.linalg.svd(matrix)

    @staticmethod
    def qr(matrix: _Array, stream: object = None) -> tuple[_Array, _Array]:
        """QR decomposition. ``stream`` parameter is ignored."""
        return _torch.linalg.qr(matrix)


class random:  # noqa: N801 — lowercase to match MLX API
    """Random number generation sub-namespace."""

    @staticmethod
    def normal(shape: tuple[int, ...]) -> _Array:
        """Sample from standard normal. Shape convention matches MLX."""
        return _torch.randn(shape)

    @staticmethod
    def uniform(shape: tuple[int, ...]) -> _Array:
        """Sample from uniform ``[0, 1)``. Shape convention matches MLX."""
        return _torch.rand(shape)

    @staticmethod
    def randint(low: int, high: int, shape: tuple[int, ...]) -> _Array:
        """Sample integer tensor with MLX-style argument names."""
        return _torch.randint(low=low, high=high, size=shape)

    @staticmethod
    def seed(s: int) -> None:
        """Set global RNG seed."""
        _torch.manual_seed(s)

    @staticmethod
    def categorical(logits: _Array, num_samples: int = 1) -> _Array:
        """Sample from categorical distribution given logits."""
        probs = _torch.softmax(logits, dim=-1)
        return _torch.multinomial(probs, num_samples)


# Stream sentinel (ignored in torch; MLX uses mx.cpu for CPU-stream ops)
cpu = None


# ====================================================================
# tree_flatten
# ====================================================================


def tree_flatten(
    tree: dict[str, object] | list[object] | object,
) -> list[tuple[str, object]]:
    """Flatten nested dict/list to ``[(key_path, leaf)]`` pairs.

    Matches ``mlx.utils.tree_flatten`` return format: list of ``(str, leaf)``.
    """
    items: list[tuple[str, object]] = []

    def _recurse(obj: object, prefix: str) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = str(k)
                _recurse(v, f"{prefix}.{key}" if prefix else key)
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                _recurse(v, f"{prefix}.{i}" if prefix else str(i))
        else:
            items.append((prefix, obj))

    _recurse(tree, "")
    return items
