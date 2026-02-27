"""Canonical tensor operations contract.

Every symbol listed in OPS_CONTRACT must be exported by each backend module.
Protocol is used for static type checking; OPS_CONTRACT list is used for
runtime compliance validation.
"""

from typing import Protocol

from vauban._array import Array


class LinalgNamespace(Protocol):
    """Linalg sub-namespace contract."""

    def norm(
        self, x: Array, axis: int | None = None, keepdims: bool = False,
    ) -> Array: ...
    def svd(
        self, matrix: Array, stream: object = None,
    ) -> tuple[Array, Array, Array]: ...
    def qr(
        self, matrix: Array, stream: object = None,
    ) -> tuple[Array, Array]: ...


class RandomNamespace(Protocol):
    """Random sub-namespace contract."""

    def normal(self, shape: tuple[int, ...]) -> Array: ...
    def seed(self, s: int) -> None: ...
    def categorical(self, logits: Array, num_samples: int = 1) -> Array: ...


# The complete list of symbols every backend module MUST export.
# Used by compliance tests to verify no missing/extra symbols.
OPS_CONTRACT: list[str] = [
    # Array creation
    "array",
    "array_type",
    "zeros",
    "zeros_like",
    "ones",
    "arange",
    "full",
    # Reductions
    "sum",
    "mean",
    # Element-wise math
    "abs",
    "exp",
    "log",
    "sqrt",
    "maximum",
    "minimum",
    "outer",
    "arccos",
    "cos",
    "clip",
    "where",
    # Selection / sorting
    "argmax",
    "argpartition",
    "sort",
    "argsort",
    "softmax",
    "matmul",
    # Manipulation
    "concatenate",
    "stack",
    "reshape",
    "expand_dims",
    # I/O
    "load",
    "save_safetensors",
    # Gradient
    "value_and_grad",
    "stop_gradient",
    # Evaluation
    "eval",
    # Types / dtypes
    "float32",
    "float16",
    "bfloat16",
    "int32",
    "uint32",
    "bool_",
    # Sub-namespaces
    "linalg",
    "random",
    # Stream
    "cpu",
    # Utilities
    "tree_flatten",
]
