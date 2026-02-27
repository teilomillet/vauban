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
softmax = _wrap("softmax")
concatenate = _wrap("concatenate")
clip = _wrap("clip")
sort = _wrap("sort")
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
maximum = _torch.maximum
minimum = _torch.minimum
outer = _torch.outer
arccos = _torch.arccos
cos = _torch.cos
matmul = _torch.matmul
reshape = _torch.reshape
where = _torch.where


# ====================================================================
# Custom implementations (different semantics)
# ====================================================================


def expand_dims(x: _Array, axis: int) -> _Array:
    """Unsqueeze a dimension (MLX ``expand_dims`` -> torch ``unsqueeze``)."""
    return x.unsqueeze(axis)


def argpartition(x: _Array, kth: int) -> _Array:
    """Partial sort by index.

    PyTorch lacks native ``argpartition``. This uses ``argsort`` for correct
    semantics (O(n log n) instead of O(n), but exact).

    Vauban usage: ``argpartition(x, kth=n-k)[n-k:]`` to get top-k indices.
    """
    return _torch.argsort(x)


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
) -> _Callable[..., tuple[_Array, _Array]]:
    """MLX-compatible ``value_and_grad``.

    Contract:
    - ``fn`` must accept an Array as first arg and return a scalar Array (loss).
    - Returns ``(loss, grad)`` where grad has same shape as first arg.
    - ``loss`` is detached (no grad graph).
    - ``grad`` is a fresh tensor (not a view into autograd state).
    """

    def wrapper(x: _Array, *args: object, **kwargs: object) -> tuple[_Array, _Array]:
        x_grad = x.detach().requires_grad_(True)
        loss = fn(x_grad, *args, **kwargs)
        loss.backward()
        grad = x_grad.grad
        if grad is None:
            msg = "value_and_grad: loss did not depend on the input tensor"
            raise RuntimeError(msg)
        return loss.detach(), grad.clone()

    return wrapper


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
