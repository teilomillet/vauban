# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Runtime backend selection — follows the Keras KERAS_BACKEND pattern.

Set ``VAUBAN_BACKEND`` environment variable before importing vauban,
or call ``set_backend()`` before any model/tensor operations.
"""

import os
from typing import Literal, cast

BackendName = Literal["mlx", "torch"]

SUPPORTED_BACKENDS: frozenset[str] = frozenset({"mlx", "torch"})
DEFAULT_BACKEND: BackendName = "mlx"


def resolve_backend(name: str | None = None) -> BackendName:
    """Resolve and validate a backend name.

    If *name* is ``None``, ``VAUBAN_BACKEND`` is used when set, otherwise the
    project default is returned. This keeps config files backend-agnostic while
    allowing environment managers such as Pixi to select the active runtime.
    """
    raw = (
        name
        if name is not None
        else os.environ.get("VAUBAN_BACKEND", DEFAULT_BACKEND)
    )
    if raw not in SUPPORTED_BACKENDS:
        msg = (
            f"Unknown backend {raw!r}; "
            f"backend must be one of {sorted(SUPPORTED_BACKENDS)}"
        )
        raise ValueError(msg)
    return cast("BackendName", raw)


_BACKEND: BackendName = resolve_backend()
_LOCKED: bool = False


def get_backend() -> BackendName:
    """Return the currently active backend name."""
    return _BACKEND


def set_backend(name: str) -> None:
    """Set the active backend (must be called before importing tensor modules)."""
    global _BACKEND
    if _LOCKED:
        msg = "Cannot change backend after ops module has been loaded"
        raise RuntimeError(msg)
    _BACKEND = resolve_backend(name)


def _lock() -> None:
    """Lock the backend — called by _ops.py on first load.

    Prevents ``set_backend()`` from changing the backend after ops are loaded.
    """
    global _LOCKED
    _LOCKED = True
