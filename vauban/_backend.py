"""Runtime backend selection — follows the Keras KERAS_BACKEND pattern.

Set ``VAUBAN_BACKEND`` environment variable before importing vauban,
or call ``set_backend()`` before any model/tensor operations.
"""

import os
from typing import Literal

BackendName = Literal["mlx", "torch"]

SUPPORTED_BACKENDS: frozenset[str] = frozenset({"mlx", "torch"})

_BACKEND: BackendName = os.environ.get("VAUBAN_BACKEND", "mlx")  # type: ignore[assignment]


def get_backend() -> BackendName:
    """Return the currently active backend name."""
    return _BACKEND


def set_backend(name: BackendName) -> None:
    """Set the active backend (must be called before importing tensor modules)."""
    global _BACKEND
    if name not in SUPPORTED_BACKENDS:
        msg = f"Unknown backend {name!r}. Choose from: {sorted(SUPPORTED_BACKENDS)}"
        raise ValueError(msg)
    _BACKEND = name
