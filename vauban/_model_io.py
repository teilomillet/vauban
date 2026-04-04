# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Backend-dispatched model loading.

Centralizes all model loading behind a small backend switch so callers can
load a model without knowing which runtime is active.
"""

from typing import cast

from vauban._backend import get_backend
from vauban.types import CausalLM, Tokenizer

_BACKEND = get_backend()


def _load_model_mlx(model_path: str) -> tuple[CausalLM, Tokenizer]:
    """Load model + tokenizer using ``mlx_lm``.

    Args:
        model_path: HuggingFace model ID or local path.

    Returns:
        ``(model, tokenizer)`` tuple.
    """
    import mlx_lm

    model, tokenizer, *_ = mlx_lm.load(model_path)
    return model, tokenizer  # type: ignore[return-value]


def _load_model_torch(model_path: str) -> tuple[CausalLM, Tokenizer]:
    """Load model + tokenizer using HuggingFace Transformers.

    Args:
        model_path: HuggingFace model ID or local path.

    Returns:
        ``(model, tokenizer)`` tuple.
    """
    import torch  # ty: ignore[unresolved-import]
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from vauban._model_torch import TorchCausalLMWrapper

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, device_map="auto",
    )
    hf_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return (
        cast("CausalLM", TorchCausalLMWrapper(hf_model)),
        cast("Tokenizer", tokenizer),
    )


def load_model(model_path: str) -> tuple[CausalLM, Tokenizer]:
    """Load model + tokenizer using the active backend.

    Args:
        model_path: HuggingFace model ID or local path.

    Returns:
        ``(model, tokenizer)`` tuple.

    Raises:
        ValueError: If the configured backend is not supported.
    """
    if _BACKEND == "mlx":
        return _load_model_mlx(model_path)
    if _BACKEND == "torch":
        return _load_model_torch(model_path)

    msg = f"Unknown backend: {_BACKEND!r}"
    raise ValueError(msg)
