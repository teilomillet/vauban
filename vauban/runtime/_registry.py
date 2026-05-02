# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Runtime backend selection and construction."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from vauban._backend import resolve_backend
from vauban.runtime._capabilities import declared_capabilities

if TYPE_CHECKING:
    from vauban.runtime._protocols import ModelRuntime
    from vauban.runtime._types import BackendCapabilities, RuntimeBackendName

SUPPORTED_RUNTIME_BACKENDS: tuple[RuntimeBackendName, ...] = ("torch", "mlx", "max")


def available_runtime_backends() -> tuple[RuntimeBackendName, ...]:
    """Return runtime backend names known to the registry."""
    return SUPPORTED_RUNTIME_BACKENDS


def runtime_capabilities(name: str | None = None) -> BackendCapabilities:
    """Return declared capabilities for a runtime backend."""
    backend = _runtime_backend_name(name)
    return declared_capabilities(backend)


def create_runtime(name: str | None = None) -> ModelRuntime:
    """Construct a concrete runtime implementation."""
    backend = _runtime_backend_name(name)
    if backend == "torch":
        from vauban.runtime._torch import TorchRuntime

        return TorchRuntime()
    if backend == "mlx":
        from vauban.runtime._mlx import MlxRuntime

        return MlxRuntime()
    msg = (
        f"Runtime backend {backend!r} has declared capabilities but no "
        "execution adapter yet"
    )
    raise NotImplementedError(msg)


def _runtime_backend_name(name: str | None) -> RuntimeBackendName:
    """Resolve a runtime backend name without accepting unknown strings."""
    if name is None:
        return cast("RuntimeBackendName", resolve_backend())
    if name in SUPPORTED_RUNTIME_BACKENDS:
        return cast("RuntimeBackendName", name)
    msg = f"Unknown runtime backend: {name!r}"
    raise ValueError(msg)
