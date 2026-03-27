# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Backend-dispatched tensor operations (PEP 562 + backend locking).

Consumer code::

    from vauban import _ops as ops
    x = ops.array([1, 2, 3])
    y = ops.sum(x)

Backend is resolved on first attribute access and locked — cannot change
mid-process. Subsequent accesses are cached (zero overhead).
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vauban._ops_mlx import *  # noqa: F403
else:
    import importlib
    from types import ModuleType

    from vauban._ops_contract import OPS_CONTRACT

    _backend_mod: ModuleType | None = None
    _backend_name: str | None = None

    def _ensure_loaded() -> ModuleType:
        global _backend_mod, _backend_name
        if _backend_mod is None:
            from vauban._backend import _lock, get_backend

            _backend_name = get_backend()
            _backend_mod = importlib.import_module(
                f"vauban._ops_{_backend_name}",
            )
            _lock()
        return _backend_mod

    def __getattr__(name: str) -> object:
        mod = _ensure_loaded()
        try:
            attr = getattr(mod, name)
        except AttributeError:
            msg = (
                f"Backend '{_backend_name}' does not export '{name}'. "
                f"Check _ops_contract.OPS_CONTRACT for the required API."
            )
            raise AttributeError(msg) from None
        globals()[name] = attr  # cache for zero-overhead next access
        return attr

    def __dir__() -> list[str]:
        return list(OPS_CONTRACT)

    __all__ = list(OPS_CONTRACT)
