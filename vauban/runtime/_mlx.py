# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""MLX implementation of Vauban runtime primitives."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Protocol, cast

import mlx.core as mx

from vauban._forward import (
    LayerModule,
    embed_and_mask,
    force_eval,
    get_transformer,
    lm_head_forward,
    make_ssm_mask,
    select_mask,
)
from vauban.runtime._capabilities import mlx_capabilities
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
    TensorLike,
    TokenizedPrompt,
    TokenizeRequest,
    TraceRequest,
)

if TYPE_CHECKING:
    from vauban._array import Array
    from vauban.types import CausalLM, Tokenizer


class _MlxLmModule(Protocol):
    """Subset of mlx-lm needed by the runtime loader."""

    def load(
        self,
        model_path: str,
    ) -> tuple[object, object, *tuple[object, ...]]:
        """Load an MLX model and tokenizer."""
        ...


class MlxRuntime:
    """Reference runtime implementation for MLX-backed models."""

    @property
    def capabilities(self) -> BackendCapabilities:
        """Declared MLX runtime support."""
        return mlx_capabilities()

    def load_model(self, ref: ModelRef) -> LoadedModel:
        """Load an MLX model and tokenizer."""
        with profile_stage(
            "load_model",
            device=_mlx_device(),
            metadata={"model_path": ref.model_path},
        ) as timer:
            mlx_lm = cast("_MlxLmModule", importlib.import_module("mlx_lm"))
            model, tokenizer, *_ = mlx_lm.load(ref.model_path)
        return LoadedModel(
            ref=ref,
            backend="mlx",
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
        """Tokenize text with an MLX tokenizer."""
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
        """Run an MLX forward pass with optional activation collection."""
        model = _require_model(loaded)
        device = _mlx_device()
        profiles = []
        with profile_stage(
            "prepare_batch",
            device=device,
            batch_size=1,
            token_count=len(request.prompt_ids),
            host_device_copies=1,
            sync_points=1,
            metadata={"tokens": len(request.prompt_ids)},
        ) as prepare_timer:
            token_ids = mx.array(list(request.prompt_ids))[None, :]
            transformer = get_transformer(model)
            h, mask = embed_and_mask(transformer, token_ids)
            force_eval(h, mask)
        profiles.append(prepare_timer.profile)

        collect_layers = frozenset(request.collect_layers)
        activations: dict[int, TensorLike] = {}
        with profile_stage(
            "forward",
            device=device,
            batch_size=1,
            token_count=len(request.prompt_ids),
            sync_points=1,
            metadata={"collect_layers": list(request.collect_layers)},
        ) as forward_timer:
            ssm_mask = make_ssm_mask(transformer, h)
            intervention_records: list[InterventionRecord] = []
            for layer_index, layer in enumerate(transformer.layers):
                typed_layer = cast("LayerModule", layer)
                h = typed_layer(h, select_mask(layer, mask, ssm_mask))
                for intervention in request.interventions:
                    if intervention.layer_index == layer_index:
                        h = cast(
                            "Array",
                            intervention.apply(cast("TensorLike", h)),
                        )
                        force_eval(h)
                        intervention_records.append(
                            InterventionRecord(
                                name=intervention.name,
                                layer_index=layer_index,
                            ),
                        )
                if layer_index in collect_layers:
                    force_eval(h)
                    activations[layer_index] = cast("TensorLike", h)
            h = transformer.norm(h)
            force_eval(h)
        profiles.append(forward_timer.profile)

        logits: TensorLike | None = None
        logprobs: TensorLike | None = None
        if request.return_logits:
            with profile_stage(
                "lm_head",
                device=device,
                batch_size=1,
                token_count=len(request.prompt_ids),
                sync_points=1,
            ) as head_timer:
                raw_logits = lm_head_forward(model, h)
                force_eval(raw_logits)
                logits = cast("TensorLike", raw_logits)
                if request.return_logprobs:
                    raw_logprobs = mx.log(mx.softmax(raw_logits, axis=-1))
                    force_eval(raw_logprobs)
                    logprobs = cast("TensorLike", raw_logprobs)
            profiles.append(head_timer.profile)

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
        """Run tokenization and forward as one trace-first MLX execution."""
        return run_runtime_trace(self, loaded, request)


def _mlx_device() -> DeviceRef:
    """Return current MLX device metadata for trace records."""
    device = mx.default_device()
    label = str(device)
    kind = "cpu" if "cpu" in label.lower() else "gpu"
    return DeviceRef(kind=kind, label=label)


def _tokenizer_device() -> DeviceRef:
    """Return device metadata for CPU tokenizer work."""
    return DeviceRef(kind="cpu", label="tokenizer")


def _require_model(loaded: LoadedModel) -> CausalLM:
    """Return the loaded MLX model or fail with a precise error."""
    if loaded.backend != "mlx":
        msg = f"MlxRuntime cannot run backend {loaded.backend!r}"
        raise ValueError(msg)
    return cast("CausalLM", loaded.model)


def _require_tokenizer(loaded: LoadedModel) -> Tokenizer:
    """Return the loaded MLX tokenizer or fail with a precise error."""
    if loaded.tokenizer is None:
        msg = "loaded model has no tokenizer"
        raise ValueError(msg)
    return cast("Tokenizer", loaded.tokenizer)
