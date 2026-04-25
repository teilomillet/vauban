# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Backend-dispatched model loading.

Centralizes all model loading behind a small backend switch so callers can
load a model without knowing which runtime is active.
"""

import importlib
from typing import Protocol, cast

from vauban._backend import get_backend
from vauban.types import CausalLM, Tokenizer

_BACKEND = get_backend()


class _MlxLmModule(Protocol):
    """Subset of ``mlx_lm`` needed for model loading."""

    def load(
        self, model_path: str,
    ) -> tuple[object, object, *tuple[object, ...]]:
        """Load an MLX model and tokenizer."""


class _AutoModelForCausalLM(Protocol):
    """Subset of Transformers AutoModelForCausalLM used here."""

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs: object) -> "_LoadedHfModel":
        """Load a HuggingFace causal LM."""


class _AutoTokenizer(Protocol):
    """Subset of Transformers AutoTokenizer used here."""

    @classmethod
    def from_pretrained(cls, model_path: str) -> object:
        """Load a HuggingFace tokenizer."""


class _TorchModule(Protocol):
    """Subset of torch needed for model loading."""

    float16: object


class _LoadedHfModel(Protocol):
    """Subset of a loaded HuggingFace model needed before wrapping."""

    def eval(self) -> object:
        """Switch the model to evaluation mode."""


class _TransformersModule(Protocol):
    """Subset of the Transformers module used here."""

    AutoModelForCausalLM: type[_AutoModelForCausalLM]
    AutoTokenizer: type[_AutoTokenizer]


def _load_model_mlx(model_path: str) -> tuple[CausalLM, Tokenizer]:
    """Load model + tokenizer using ``mlx_lm``.

    Args:
        model_path: HuggingFace model ID or local path.

    Returns:
        ``(model, tokenizer)`` tuple.
    """
    mlx_lm = cast("_MlxLmModule", importlib.import_module("mlx_lm"))

    model, tokenizer, *_ = mlx_lm.load(model_path)
    return cast("CausalLM", model), cast("Tokenizer", tokenizer)


def _load_model_torch(model_path: str) -> tuple[CausalLM, Tokenizer]:
    """Load model + tokenizer using HuggingFace Transformers.

    Args:
        model_path: HuggingFace model ID or local path.

    Returns:
        ``(model, tokenizer)`` tuple.
    """
    torch = cast("_TorchModule", importlib.import_module("torch"))
    transformers = cast(
        "_TransformersModule",
        importlib.import_module("transformers"),
    )
    from vauban._model_torch import TorchCausalLMWrapper

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, device_map="auto",
    )
    hf_model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
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
