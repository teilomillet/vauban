"""Backend-dispatched model loading.

Centralizes all model loading behind a single function that delegates
to the active backend (MLX or PyTorch). The backend is resolved once
at import time — no per-call dispatch.
"""

from typing import TYPE_CHECKING

from vauban.types import CausalLM, Tokenizer

if TYPE_CHECKING:
    from vauban._backend import get_backend as _get_backend

    _BACKEND = _get_backend()
else:
    from vauban._backend import get_backend

    _BACKEND = get_backend()


if TYPE_CHECKING or _BACKEND == "mlx":
    def load_model(model_path: str) -> tuple[CausalLM, Tokenizer]:
        """Load model + tokenizer using mlx-lm.

        Args:
            model_path: HuggingFace model ID or local path.

        Returns:
            (model, tokenizer) tuple.
        """
        import mlx_lm

        model, tokenizer, *_ = mlx_lm.load(model_path)
        return model, tokenizer  # type: ignore[return-value]

elif _BACKEND == "torch":
    def load_model(model_path: str) -> tuple[CausalLM, Tokenizer]:
        """Load model + tokenizer using HuggingFace Transformers.

        Args:
            model_path: HuggingFace model ID or local path.

        Returns:
            (model, tokenizer) tuple.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from vauban._model_torch import TorchCausalLMWrapper

        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.float16, device_map="auto",
        )
        hf_model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return TorchCausalLMWrapper(hf_model), tokenizer

else:
    msg = f"Unknown backend: {_BACKEND!r}"
    raise ValueError(msg)
