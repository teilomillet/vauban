# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""PyTorch implementation of Vauban runtime primitives."""

from __future__ import annotations

import importlib
from typing import Protocol, cast

from vauban.runtime._capabilities import torch_capabilities
from vauban.runtime._profiling import profile_stage
from vauban.runtime._types import (
    BackendCapabilities,
    DeviceRef,
    ForwardRequest,
    ForwardTrace,
    InterventionRecord,
    LoadedModel,
    ModelRef,
    TensorLike,
    TokenizedPrompt,
    TokenizeRequest,
)


class _TorchModule(Protocol):
    """Subset of torch needed by the runtime adapter."""

    long: object

    def tensor(
        self,
        data: object,
        *,
        dtype: object | None = None,
        device: object | None = None,
    ) -> TensorLike:
        """Create a tensor."""

    def log_softmax(self, tensor: TensorLike, dim: int) -> TensorLike:
        """Compute log probabilities."""


class _Tokenizer(Protocol):
    """Tokenizer surface used by the Torch runtime."""

    def encode(self, text: str) -> list[int]:
        """Encode text."""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = True,
    ) -> str | list[int]:
        """Render a chat template."""


class _Embedding(Protocol):
    """Embedding surface used by the Torch runtime."""

    weight: object

    def __call__(self, token_ids: TensorLike) -> TensorLike:
        """Embed token IDs."""


class _Norm(Protocol):
    """Norm layer surface used by the Torch runtime."""

    def __call__(self, h: TensorLike) -> TensorLike:
        """Normalize hidden states."""


class _Layer(Protocol):
    """Decoder layer surface used by the Torch runtime."""

    def __call__(
        self,
        h: TensorLike,
        mask: object | None = None,
        cache: object | None = None,
    ) -> TensorLike:
        """Run a decoder layer."""


class _Head(Protocol):
    """Language-model head surface."""

    def __call__(self, h: TensorLike) -> TensorLike:
        """Project hidden states to logits."""


class _Transformer(Protocol):
    """Transformer surface used by the Torch runtime."""

    embed_tokens: _Embedding
    layers: list[_Layer]
    norm: _Norm


class _TorchRuntimeModel(Protocol):
    """Loaded model surface used by the Torch runtime."""

    model: _Transformer


class _PretrainedModelLoader(Protocol):
    """HuggingFace-style model loader surface."""

    def from_pretrained(self, model_path: str, **kwargs: object) -> object:
        """Load a pretrained model."""
        ...


class _PretrainedTokenizerLoader(Protocol):
    """HuggingFace-style tokenizer loader surface."""

    def from_pretrained(self, model_path: str) -> object:
        """Load a pretrained tokenizer."""
        ...


class _TorchLmModule(Protocol):
    """Subset of Transformers used by the runtime loader."""

    AutoModelForCausalLM: _PretrainedModelLoader
    AutoTokenizer: _PretrainedTokenizerLoader


class TorchRuntime:
    """Runtime implementation for PyTorch-backed models."""

    @property
    def capabilities(self) -> BackendCapabilities:
        """Declared PyTorch runtime support."""
        return torch_capabilities()

    def load_model(self, ref: ModelRef) -> LoadedModel:
        """Load a HuggingFace model and tokenizer through the Torch wrapper."""
        with profile_stage(
            "load_model",
            device=DeviceRef(kind="cpu", label="torch"),
            metadata={"model_path": ref.model_path},
        ) as timer:
            torch = cast("_TorchModule", importlib.import_module("torch"))
            transformers = cast(
                "_TorchLmModule",
                importlib.import_module("transformers"),
            )
            from vauban._model_torch import TorchCausalLMWrapper

            raw_model = transformers.AutoModelForCausalLM.from_pretrained(
                ref.model_path,
                dtype=getattr(torch, "float16", None),
                device_map="auto",
            )
            eval_fn = getattr(raw_model, "eval", None)
            if callable(eval_fn):
                eval_fn()
            tokenizer = transformers.AutoTokenizer.from_pretrained(ref.model_path)
            model = TorchCausalLMWrapper(raw_model)
        return LoadedModel(
            ref=ref,
            backend="torch",
            capabilities=self.capabilities,
            model=model,
            tokenizer=tokenizer,
            metadata={"load_duration_s": timer.profile.duration_s},
        )

    def tokenize(
        self,
        loaded: LoadedModel,
        request: TokenizeRequest,
    ) -> TokenizedPrompt:
        """Tokenize text with a loaded HuggingFace tokenizer."""
        tokenizer = _require_tokenizer(loaded)
        with profile_stage("tokenize", device=_torch_device(loaded.model)) as timer:
            if request.apply_chat_template:
                rendered = tokenizer.apply_chat_template(
                    [{"role": "user", "content": request.text}],
                    tokenize=False,
                )
                if not isinstance(rendered, str):
                    msg = "apply_chat_template must return str when tokenize=False"
                    raise TypeError(msg)
                text = rendered
            else:
                text = request.text
            token_ids = tuple(int(token_id) for token_id in tokenizer.encode(text))
        return TokenizedPrompt(
            token_ids=token_ids,
            text=text,
            profile=(timer.profile,),
        )

    def forward(
        self,
        loaded: LoadedModel,
        request: ForwardRequest,
    ) -> ForwardTrace:
        """Run a Torch forward pass with optional activation collection."""
        torch = cast("_TorchModule", importlib.import_module("torch"))
        model = _require_model(loaded)
        device = _torch_device(model)
        profiles = []
        with profile_stage(
            "prepare_batch",
            device=device,
            metadata={"tokens": len(request.prompt_ids)},
        ) as prepare_timer:
            token_ids = torch.tensor(
                [list(request.prompt_ids)],
                dtype=torch.long,
                device=_torch_model_device(model),
            )
            transformer = model.model
            h = transformer.embed_tokens(token_ids)
        profiles.append(prepare_timer.profile)

        collect_layers = frozenset(request.collect_layers)
        activations: dict[int, TensorLike] = {}
        intervention_records: list[InterventionRecord] = []
        with profile_stage(
            "forward",
            device=device,
            metadata={"collect_layers": list(request.collect_layers)},
        ) as forward_timer:
            for layer_index, layer in enumerate(transformer.layers):
                h = layer(h, None)
                for intervention in request.interventions:
                    if intervention.layer_index == layer_index:
                        h = intervention.apply(h)
                        intervention_records.append(
                            InterventionRecord(
                                name=intervention.name,
                                layer_index=layer_index,
                            ),
                        )
                if layer_index in collect_layers:
                    activations[layer_index] = h
            h = transformer.norm(h)
        profiles.append(forward_timer.profile)

        logits: TensorLike | None = None
        logprobs: TensorLike | None = None
        if request.return_logits:
            with profile_stage("lm_head", device=device) as head_timer:
                head = _require_lm_head(model)
                logits = head(h)
                if request.return_logprobs:
                    logprobs = torch.log_softmax(logits, dim=-1)
            profiles.append(head_timer.profile)

        return ForwardTrace(
            logits=logits,
            logprobs=logprobs,
            activations=activations,
            device=device,
            interventions=tuple(intervention_records),
            profile=tuple(profiles),
        )


def _require_model(loaded: LoadedModel) -> _TorchRuntimeModel:
    """Return the loaded Torch model or fail with a precise error."""
    if loaded.backend != "torch":
        msg = f"TorchRuntime cannot run backend {loaded.backend!r}"
        raise ValueError(msg)
    return cast("_TorchRuntimeModel", loaded.model)


def _require_tokenizer(loaded: LoadedModel) -> _Tokenizer:
    """Return the loaded tokenizer or fail with a precise error."""
    if loaded.tokenizer is None:
        msg = "loaded model has no tokenizer"
        raise ValueError(msg)
    return cast("_Tokenizer", loaded.tokenizer)


def _require_lm_head(model: _TorchRuntimeModel) -> _Head:
    """Return the model LM head or fail rather than silently guessing."""
    head = getattr(model, "lm_head", None)
    if head is None:
        msg = "TorchRuntime requires a model with an lm_head"
        raise NotImplementedError(msg)
    return cast("_Head", head)


def _torch_model_device(model: object) -> object | None:
    """Return a Torch model device when exposed by the wrapper."""
    return getattr(model, "device", None)


def _torch_device(model: object) -> DeviceRef:
    """Return device metadata for a Torch model."""
    raw_device = _torch_model_device(model)
    label = str(raw_device) if raw_device is not None else "torch"
    kind = "cuda" if label.startswith("cuda") else "cpu"
    return DeviceRef(kind=kind, label=label)
