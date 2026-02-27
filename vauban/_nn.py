"""Backend-dispatched neural network utilities (PEP 562 + backend locking).

Consumer code::

    from vauban import _nn as nn_ops
    mask = nn_ops.create_additive_causal_mask(seq_len)
    loss = nn_ops.cross_entropy(logits, targets)

Backend is resolved on first attribute access and locked.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vauban._nn_mlx import *  # noqa: F403
else:
    import importlib
    from types import ModuleType

    from vauban._nn_contract import NN_CONTRACT

    _backend_mod: ModuleType | None = None
    _backend_name: str | None = None

    def _ensure_loaded() -> ModuleType:
        global _backend_mod, _backend_name
        if _backend_mod is None:
            from vauban._backend import _lock, get_backend

            _backend_name = get_backend()
            _backend_mod = importlib.import_module(
                f"vauban._nn_{_backend_name}",
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
                f"Check _nn_contract.NN_CONTRACT for the required API."
            )
            raise AttributeError(msg) from None
        globals()[name] = attr
        return attr

    def __dir__() -> list[str]:
        return list(NN_CONTRACT)

    __all__ = list(NN_CONTRACT)
