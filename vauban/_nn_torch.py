"""PyTorch neural network utilities backend."""

import torch as _torch
import torch.nn.functional as _f

_Array = _torch.Tensor


def create_additive_causal_mask(seq_len: int) -> _Array:
    """Additive causal mask: 0 for allowed, -inf for masked.

    Matches MLX convention: upper-triangular ``-inf``, diagonal and below ``0``.
    Shape: ``(seq_len, seq_len)``, dtype: ``float32``.
    """
    mask = _torch.full((seq_len, seq_len), float("-inf"))
    return _torch.triu(mask, diagonal=1)


def cross_entropy(
    logits: _Array, targets: _Array, reduction: str = "mean",
) -> _Array:
    """Cross-entropy loss matching MLX ``nn.losses.cross_entropy`` semantics."""
    return _f.cross_entropy(logits, targets, reduction=reduction)
