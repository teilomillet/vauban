# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Array type alias — single point of change for tensor backend."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.core as mx

    Array = mx.array
    """Tensor type used throughout vauban. Runtime selects MLX or Torch."""
else:
    from vauban._backend import get_backend

    _BACKEND = get_backend()

    if _BACKEND == "mlx":
        import mlx.core as mx

        Array = mx.array
    elif _BACKEND == "torch":
        import torch

        Array = torch.Tensor
    else:
        msg = f"Unknown backend: {_BACKEND!r}"
        raise ValueError(msg)
