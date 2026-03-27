# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Runtime backend selection — follows the Keras KERAS_BACKEND pattern.

Set ``VAUBAN_BACKEND`` environment variable before importing vauban,
or call ``set_backend()`` before any model/tensor operations.
"""

import os
from typing import Literal

BackendName = Literal["mlx", "torch"]

SUPPORTED_BACKENDS: frozenset[str] = frozenset({"mlx", "torch"})

_BACKEND: BackendName = os.environ.get("VAUBAN_BACKEND", "mlx")  # type: ignore[assignment]
_LOCKED: bool = False


def get_backend() -> BackendName:
    """Return the currently active backend name."""
    return _BACKEND


def set_backend(name: BackendName) -> None:
    """Set the active backend (must be called before importing tensor modules)."""
    global _BACKEND
    if _LOCKED:
        msg = "Cannot change backend after ops module has been loaded"
        raise RuntimeError(msg)
    if name not in SUPPORTED_BACKENDS:
        msg = f"Unknown backend {name!r}. Choose from: {sorted(SUPPORTED_BACKENDS)}"
        raise ValueError(msg)
    _BACKEND = name


def _lock() -> None:
    """Lock the backend — called by _ops.py on first load.

    Prevents ``set_backend()`` from changing the backend after ops are loaded.
    """
    global _LOCKED
    _LOCKED = True
