# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""MLX neural network utilities backend."""

from typing import Literal as _Literal

import mlx.nn as _nn

from vauban._array import Array as _Array


def create_additive_causal_mask(seq_len: int) -> _Array:
    """Additive causal mask: 0 for allowed, -inf for masked."""
    return _nn.MultiHeadAttention.create_additive_causal_mask(seq_len)


def cross_entropy(
    logits: _Array,
    targets: _Array,
    reduction: _Literal["none", "mean", "sum"] = "mean",
) -> _Array:
    """Cross-entropy loss."""
    return _nn.losses.cross_entropy(logits, targets, reduction=reduction)
