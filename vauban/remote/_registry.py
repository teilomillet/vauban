# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Backend registry for remote inference providers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from vauban.types import RemoteBackend

    # Factory signature: (api_key: str) -> RemoteBackend
    type BackendFactory = Callable[[str], RemoteBackend]


# name -> factory function
_REGISTRY: dict[str, BackendFactory] = {}


def register_backend(name: str, factory: BackendFactory) -> None:
    """Register a backend factory.

    Args:
        name: Backend name (must match ``[remote].backend`` in TOML).
        factory: Callable that takes an API key and returns a ``RemoteBackend``.
    """
    _REGISTRY[name] = factory


def get_backend(name: str, api_key: str) -> RemoteBackend:
    """Resolve a backend by name and instantiate it.

    Args:
        name: Backend name from config.
        api_key: API key to pass to the factory.

    Returns:
        Instantiated backend.

    Raises:
        ValueError: If the backend name is not registered.
    """
    factory = _REGISTRY.get(name)
    if factory is None:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        msg = f"Unknown remote backend {name!r}. Available: {available}"
        raise ValueError(msg)
    return factory(api_key)


def _register_builtins() -> None:
    """Register built-in backends (called at import time)."""
    from vauban.remote._jsinfer import create_jsinfer_backend

    register_backend("jsinfer", create_jsinfer_backend)


_register_builtins()
