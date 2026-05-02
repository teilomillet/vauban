# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""PyTorch implementation of Vauban runtime primitives."""

from __future__ import annotations

import importlib
from typing import Protocol, cast

from vauban.runtime._activation_primitive import primitive_metadata_for_intervention
from vauban.runtime._capabilities import torch_capabilities
from vauban.runtime._execution import run_runtime_trace
from vauban.runtime._profiling import profile_stage
from vauban.runtime._types import (
    BackendCapabilities,
    DeviceRef,
    ForwardRequest,
    ForwardTrace,
    InterventionRecord,
    LoadedModel,
    ModelRef,
    RuntimeTraceResult,
    StageProfile,
    TensorLike,
    TokenizedPrompt,
    TokenizeRequest,
    TraceRequest,
)


class _TorchModule(Protocol):
    """Subset of torch needed by the runtime adapter."""

    cuda: _CudaModule
    mps: _MpsModule
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


class _CudaModule(Protocol):
    """Subset of torch.cuda used for honest profiling sync points."""

    def is_available(self) -> bool:
        """Return whether CUDA is available."""

    def memory_allocated(self, index: int | None = None) -> int:
        """Return currently allocated CUDA bytes."""

    def synchronize(self) -> None:
        """Synchronize pending CUDA work."""


class _MpsModule(Protocol):
    """Subset of torch.mps used for MPS profiling sync points."""

    def is_available(self) -> bool:
        """Return whether MPS is available."""

    def current_allocated_memory(self) -> int:
        """Return currently allocated MPS bytes."""

    def synchronize(self) -> None:
        """Synchronize pending MPS work."""


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
        with profile_stage("tokenize", device=_tokenizer_device()) as timer:
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
        sync_points = _torch_sync_points(device)
        profiles = []
        with profile_stage(
            "prepare_batch",
            device=device,
            batch_size=1,
            token_count=len(request.prompt_ids),
            host_device_copies=1,
            sync_points=sync_points,
            metadata={"tokens": len(request.prompt_ids)},
        ) as prepare_timer:
            token_ids = torch.tensor(
                [list(request.prompt_ids)],
                dtype=torch.long,
                device=_torch_model_device(model),
            )
            transformer = model.model
            h = transformer.embed_tokens(token_ids)
            _sync_if_accelerator(torch, device)
        profiles.append(
            _observed_profile(
                prepare_timer.profile,
                memory_bytes=_torch_memory_bytes(torch, device),
                input_bytes=_tensor_nbytes(token_ids),
                output_bytes=_tensor_nbytes(h),
            ),
        )

        collect_layers = frozenset(request.collect_layers)
        activations: dict[int, TensorLike] = {}
        intervention_records: list[InterventionRecord] = []
        with profile_stage(
            "forward",
            device=device,
            batch_size=1,
            token_count=len(request.prompt_ids),
            sync_points=sync_points,
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
                                metadata=(
                                    primitive_metadata_for_intervention(
                                        intervention,
                                    )
                                    or {}
                                ),
                            ),
                        )
                if layer_index in collect_layers:
                    activations[layer_index] = h
            h = transformer.norm(h)
            _sync_if_accelerator(torch, device)
        profiles.append(
            _observed_profile(
                forward_timer.profile,
                memory_bytes=_torch_memory_bytes(torch, device),
                output_bytes=_tensor_nbytes(h),
            ),
        )

        logits: TensorLike | None = None
        logprobs: TensorLike | None = None
        if request.return_logits:
            with profile_stage(
                "lm_head",
                device=device,
                batch_size=1,
                token_count=len(request.prompt_ids),
                sync_points=sync_points,
            ) as head_timer:
                head = _require_lm_head(model)
                logits = head(h)
                if request.return_logprobs:
                    logprobs = torch.log_softmax(logits, dim=-1)
                _sync_if_accelerator(torch, device)
            profiles.append(
                _observed_profile(
                    head_timer.profile,
                    memory_bytes=_torch_memory_bytes(torch, device),
                    input_bytes=_tensor_nbytes(h),
                    output_bytes=_tensor_nbytes(
                        logprobs if logprobs is not None else logits,
                    ),
                ),
            )

        return ForwardTrace(
            logits=logits,
            logprobs=logprobs,
            activations=activations,
            device=device,
            interventions=tuple(intervention_records),
            profile=tuple(profiles),
        )

    def trace(
        self,
        loaded: LoadedModel,
        request: TraceRequest,
    ) -> RuntimeTraceResult:
        """Run tokenization and forward as one trace-first Torch execution."""
        return run_runtime_trace(self, loaded, request)


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
    if label.startswith("cuda"):
        kind = "cuda"
    elif label.startswith("mps"):
        kind = "mps"
    else:
        kind = "cpu"
    return DeviceRef(kind=kind, label=label)


def _tokenizer_device() -> DeviceRef:
    """Return device metadata for CPU tokenizer work."""
    return DeviceRef(kind="cpu", label="tokenizer")


def _torch_sync_points(device: DeviceRef) -> int:
    """Return sync point count for profiled accelerator stages."""
    return 1 if device.kind in ("cuda", "mps") else 0


def _sync_if_accelerator(torch: _TorchModule, device: DeviceRef) -> None:
    """Synchronize accelerator-backed Torch work for profile honesty."""
    if device.kind == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    if device.kind == "mps" and torch.mps.is_available():
        torch.mps.synchronize()


def _torch_memory_bytes(torch: _TorchModule, device: DeviceRef) -> int | None:
    """Return observed accelerator memory for a device-backed profile span."""
    if device.kind == "cuda" and torch.cuda.is_available():
        return int(torch.cuda.memory_allocated())
    if device.kind == "mps" and torch.mps.is_available():
        return int(torch.mps.current_allocated_memory())
    return None


def _tensor_nbytes(tensor: object | None) -> int | None:
    """Return tensor byte size when the backend object exposes it."""
    if tensor is None:
        return None
    nbytes = getattr(tensor, "nbytes", None)
    if isinstance(nbytes, int):
        return nbytes
    numel = getattr(tensor, "numel", None)
    element_size = getattr(tensor, "element_size", None)
    if not callable(numel) or not callable(element_size):
        return None
    return int(numel()) * int(element_size())


def _observed_profile(
    profile: StageProfile,
    *,
    memory_bytes: int | None = None,
    input_bytes: int | None = None,
    output_bytes: int | None = None,
) -> StageProfile:
    """Return a profile updated with post-stage tensor/device observations."""
    return StageProfile(
        name=profile.name,
        duration_s=profile.duration_s,
        device=profile.device,
        memory_bytes=(
            profile.memory_bytes if memory_bytes is None else memory_bytes
        ),
        metadata=dict(profile.metadata),
        batch_size=profile.batch_size,
        token_count=profile.token_count,
        input_bytes=profile.input_bytes if input_bytes is None else input_bytes,
        output_bytes=(
            profile.output_bytes if output_bytes is None else output_bytes
        ),
        host_device_copies=profile.host_device_copies,
        sync_points=profile.sync_points,
        queue_depth=profile.queue_depth,
    )
