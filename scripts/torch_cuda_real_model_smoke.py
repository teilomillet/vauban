# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Run a real HuggingFace CausalLM through Vauban's torch CUDA path.

This is intentionally a smoke and profiling command, not a benchmark. It checks
that a real HF model can be loaded, tokenized, forwarded, activation-collected,
and greedily decoded through Vauban's torch wrapper on CUDA.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from vauban import _ops as ops
from vauban._forward import (
    embed_and_mask,
    extract_logits,
    force_eval,
    get_transformer,
    make_ssm_mask,
    select_mask,
    to_model_device,
)
from vauban.evaluate import _generate
from vauban.runtime import (
    LoadedModel,
    ModelRef,
    TorchActivationPrimitiveRequest,
    TorchActivationTensor,
    TraceRequest,
    create_runtime,
    forward_trace_summary,
    run_torch_activation_primitive,
)
from vauban.runtime._profiling import profile_stage
from vauban.runtime._types import DeviceRef, RuntimeValue, StageProfile

if TYPE_CHECKING:
    from vauban._array import Array
    from vauban.types import CausalLM, Tokenizer

DEFAULT_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


class _CudaProperties(Protocol):
    """Subset of CUDA device properties used by the smoke."""

    total_memory: int


class _CudaModule(Protocol):
    """Subset of ``torch.cuda`` used by the smoke."""

    def is_available(self) -> bool:
        """Return whether CUDA is available."""

    def get_device_name(self, index: int) -> str:
        """Return CUDA device name."""

    def get_device_properties(self, index: int) -> _CudaProperties:
        """Return CUDA device properties."""

    def memory_allocated(self, index: int | None = None) -> int:
        """Return currently allocated CUDA bytes."""

    def max_memory_allocated(self, index: int | None = None) -> int:
        """Return peak allocated CUDA bytes."""

    def reset_peak_memory_stats(self, index: int | None = None) -> None:
        """Reset peak memory counters."""

    def synchronize(self) -> None:
        """Synchronize CUDA work."""


class _TorchVersion(Protocol):
    """Subset of ``torch.version`` used by the smoke."""

    cuda: str | None


class _TorchModule(Protocol):
    """Subset of torch used by the smoke."""

    __version__: str
    cuda: _CudaModule
    version: _TorchVersion
    float16: object
    long: object

    def tensor(
        self,
        data: object,
        *,
        dtype: object | None = None,
        device: object | None = None,
    ) -> Array:
        """Create a tensor."""


class _LoadedHfModel(Protocol):
    """Subset of a loaded HF model before wrapping."""

    def eval(self) -> object:
        """Switch to eval mode."""


class _PretrainedModelLoader(Protocol):
    """Subset of ``AutoModelForCausalLM``."""

    def from_pretrained(self, model_path: str, **kwargs: object) -> _LoadedHfModel:
        """Load a pretrained model."""


class _PretrainedTokenizerLoader(Protocol):
    """Subset of ``AutoTokenizer``."""

    def from_pretrained(self, model_path: str, **kwargs: object) -> object:
        """Load a pretrained tokenizer."""


class _TransformersModule(Protocol):
    """Subset of transformers used by the smoke."""

    AutoModelForCausalLM: _PretrainedModelLoader
    AutoTokenizer: _PretrainedTokenizerLoader


class _RawTokenizer(Protocol):
    """Tokenizer operations used by the adapter."""

    eos_token_id: int | None

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs."""


class _ChatTemplateTokenizer(_RawTokenizer, Protocol):
    """Tokenizer with an optional chat template."""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = True,
    ) -> str | list[int]:
        """Render or tokenize a chat prompt."""


class _TokenizerAdapter:
    """Tokenizer adapter that provides a stable chat-template fallback."""

    def __init__(self, tokenizer: object) -> None:
        self._tokenizer = cast("_RawTokenizer", tokenizer)
        self.eos_token_id = getattr(tokenizer, "eos_token_id", None)

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        return [int(token_id) for token_id in self._tokenizer.encode(text)]

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs."""
        return str(self._tokenizer.decode(token_ids))

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = True,
    ) -> str | list[int]:
        """Use the tokenizer chat template or a plain-text fallback."""
        apply_chat_template = getattr(
            self._tokenizer,
            "apply_chat_template",
            None,
        )
        if callable(apply_chat_template):
            try:
                rendered = cast("_ChatTemplateTokenizer", self._tokenizer)
                value = rendered.apply_chat_template(messages, tokenize=tokenize)
                if isinstance(value, str | list):
                    return value
            except (ValueError, TypeError, AttributeError):
                pass
        text = _fallback_chat_text(messages)
        return self.encode(text) if tokenize else text


def _fallback_chat_text(messages: list[dict[str, str]]) -> str:
    """Render messages for tokenizers without a chat template."""
    lines: list[str] = []
    for message in messages:
        role = message.get("role", "user").strip() or "user"
        content = message.get("content", "")
        lines.append(f"{role}: {content}")
    lines.append("assistant:")
    return "\n".join(lines)


def _sync_cuda(torch: _TorchModule) -> None:
    """Synchronize CUDA if it is available."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _shape_list(tensor: object) -> list[RuntimeValue]:
    """Return a JSON-friendly tensor shape."""
    shape = getattr(tensor, "shape", ())
    result: list[RuntimeValue] = []
    for dim in shape:
        result.append(int(dim))
    return result


def _stage_profile_to_json(profile: StageProfile) -> dict[str, RuntimeValue]:
    """Serialize one stage profile without importing private helpers."""
    return {
        "name": profile.name,
        "duration_s": profile.duration_s,
        "device": (
            {
                "kind": profile.device.kind,
                "label": profile.device.label,
            }
            if profile.device is not None
            else None
        ),
        "batch_size": profile.batch_size,
        "token_count": profile.token_count,
        "host_device_copies": profile.host_device_copies,
        "sync_points": profile.sync_points,
        "metadata": dict(profile.metadata),
    }


def _load_torch_model(
    model_path: str,
    *,
    local_files_only: bool,
) -> tuple[CausalLM, Tokenizer, _TorchModule]:
    """Load a real HF model and wrap it with Vauban's torch adapter."""
    torch = cast("_TorchModule", importlib.import_module("torch"))
    transformers = cast(
        "_TransformersModule",
        importlib.import_module("transformers"),
    )
    from vauban._model_torch import TorchCausalLMWrapper

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="auto",
        local_files_only=local_files_only,
    )
    hf_model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=local_files_only,
    )
    return (
        cast("CausalLM", TorchCausalLMWrapper(hf_model)),
        cast("Tokenizer", _TokenizerAdapter(tokenizer)),
        torch,
    )


def _forward_logits_smoke(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompt: str,
    torch: _TorchModule,
) -> tuple[list[int], list[RuntimeValue], StageProfile]:
    """Run one wrapped full-model forward pass and return token/logit shapes."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
    )
    if not isinstance(rendered, str):
        msg = "tokenizer apply_chat_template(tokenize=False) must return str"
        raise TypeError(msg)
    prompt_ids = tokenizer.encode(rendered)
    if not prompt_ids:
        msg = "tokenizer produced no prompt tokens"
        raise ValueError(msg)
    token_ids = ops.array(prompt_ids)[None, :]
    token_ids = to_model_device(model, token_ids)
    device = DeviceRef(kind="cuda", label=str(getattr(token_ids, "device", "cuda")))

    with profile_stage(
        "wrapped_forward",
        device=device,
        batch_size=1,
        token_count=len(prompt_ids),
        host_device_copies=1,
        sync_points=1,
    ) as timer:
        logits = extract_logits(model(token_ids))  # type: ignore[call-non-callable]
        force_eval(logits)
        _sync_cuda(torch)
    return prompt_ids, _shape_list(logits), timer.profile


def _activation_smoke(
    model: CausalLM,
    prompt_ids: list[int],
    collect_layer: int,
    torch: _TorchModule,
) -> tuple[int, list[RuntimeValue], dict[str, RuntimeValue], StageProfile]:
    """Collect one activation from a real transformer layer."""
    token_ids = to_model_device(model, ops.array(prompt_ids)[None, :])
    transformer = get_transformer(model)
    layer_count = len(transformer.layers)
    if layer_count == 0:
        msg = "model has no transformer layers"
        raise ValueError(msg)
    layer_index = min(max(collect_layer, 0), layer_count - 1)
    device = DeviceRef(kind="cuda", label=str(getattr(token_ids, "device", "cuda")))

    with profile_stage(
        "activation_collect",
        device=device,
        batch_size=1,
        token_count=len(prompt_ids),
        sync_points=1,
        metadata={"layer": layer_index},
    ) as timer:
        h, mask = embed_and_mask(transformer, token_ids)
        ssm_mask = make_ssm_mask(transformer, h)
        for current_layer, layer in enumerate(transformer.layers):
            h = layer(h, select_mask(layer, mask, ssm_mask))
            if current_layer == layer_index:
                break
        force_eval(h)
        _sync_cuda(torch)
        primitive_metadata = _activation_primitive_smoke(
            cast("TorchActivationTensor", h),
            layer_index,
            torch,
        )
    return layer_index, _shape_list(h), primitive_metadata, timer.profile


def _activation_primitive_smoke(
    activation: TorchActivationTensor,
    layer_index: int,
    torch: _TorchModule,
) -> dict[str, RuntimeValue]:
    """Run rank-1 and subspace activation primitives on real activations."""
    d_model = int(activation.shape[-1])
    direction_values = [0.0 for _ in range(d_model)]
    direction_values[0] = 1.0
    direction = cast("TorchActivationTensor", torch.tensor(direction_values))
    rank1 = run_torch_activation_primitive(
        TorchActivationPrimitiveRequest(
            activation=activation,
            direction=direction,
            layer_index=layer_index,
            name="cuda_smoke.first_axis_projection",
        ),
    )
    basis = cast("TorchActivationTensor", torch.tensor(_first_axis_basis(d_model)))
    subspace = run_torch_activation_primitive(
        TorchActivationPrimitiveRequest(
            activation=activation,
            direction=basis,
            layer_index=layer_index,
            mode="subspace_project",
            name="cuda_smoke.first_axes_subspace",
        ),
    )
    metadata: dict[str, RuntimeValue] = {}
    metadata["rank1"] = rank1.artifact_metadata()
    metadata["subspace"] = subspace.artifact_metadata()
    return metadata


def _first_axis_basis(d_model: int) -> list[list[float]]:
    """Return a one- or two-axis basis for a real activation width."""
    first = [0.0 for _ in range(d_model)]
    first[0] = 1.0
    if d_model == 1:
        return [first]
    second = [0.0 for _ in range(d_model)]
    second[1] = 1.0
    return [first, second]


def _generate_smoke(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    torch: _TorchModule,
) -> tuple[str, StageProfile]:
    """Run Vauban's greedy generation helper."""
    with profile_stage(
        "greedy_generate",
        device=DeviceRef(kind="cuda", label=str(getattr(model, "device", "cuda"))),
        batch_size=1,
        token_count=max_new_tokens,
        sync_points=1,
    ) as timer:
        generated = _generate(model, tokenizer, prompt, max_new_tokens)
        _sync_cuda(torch)
    return generated, timer.profile


def _runtime_trace_smoke(
    model: CausalLM,
    tokenizer: Tokenizer,
    model_path: str,
    prompt: str,
    collect_layer: int,
    torch: _TorchModule,
) -> tuple[dict[str, RuntimeValue], StageProfile]:
    """Run the real model through the trace-first Torch runtime primitive."""
    runtime = create_runtime("torch")
    loaded = LoadedModel(
        ref=ModelRef(model_path),
        backend="torch",
        capabilities=runtime.capabilities,
        model=model,
        tokenizer=tokenizer,
    )
    with profile_stage(
        "runtime_trace",
        device=DeviceRef(kind="cuda", label=str(getattr(model, "device", "cuda"))),
        batch_size=1,
        sync_points=1,
        metadata={"model": model_path},
    ) as timer:
        result = runtime.trace(
            loaded,
            TraceRequest(
                trace_id="cuda_smoke.runtime_trace",
                input_text=prompt,
                requested_artifacts=(
                    "tokens",
                    "logits",
                    "logprobs",
                    "activation",
                    "text",
                ),
                collect_layers=(collect_layer,),
                apply_chat_template=True,
                return_logits=True,
                return_logprobs=True,
            ),
        )
        _sync_cuda(torch)

    trace_summary = forward_trace_summary(
        result.forward,
        runtime_trace=result.trace,
    )
    trace_summary["trace_id"] = result.trace.trace_id
    trace_summary["trace_metadata"] = dict(result.trace.metadata)
    return trace_summary, timer.profile


def _cuda_report(torch: _TorchModule) -> dict[str, RuntimeValue]:
    """Return CUDA device and memory facts."""
    if not torch.cuda.is_available():
        return {"available": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "device_name": torch.cuda.get_device_name(0),
        "total_memory_bytes": int(props.total_memory),
        "allocated_bytes": int(torch.cuda.memory_allocated(0)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
    }


def _write_or_print(report: dict[str, RuntimeValue], output: Path | None) -> None:
    """Write the smoke report to disk or stdout."""
    text = json.dumps(report, indent=2, sort_keys=True)
    if output is None:
        print(text)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n")
    print(f"wrote smoke report: {output}")


def main() -> int:
    """Run the real-model CUDA smoke."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default="What changes when models change?")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--collect-layer", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    model: CausalLM
    tokenizer: Tokenizer
    profiles: list[StageProfile] = []

    with profile_stage(
        "load_model",
        device=DeviceRef(kind="cpu", label="host"),
        sync_points=1,
        metadata={"model": args.model},
    ) as load_timer:
        model, tokenizer, torch = _load_torch_model(
            str(args.model),
            local_files_only=bool(args.local_files_only),
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(0)
        _sync_cuda(torch)
    profiles.append(load_timer.profile)

    cuda_available = torch.cuda.is_available()
    if not cuda_available and not args.allow_cpu_fallback:
        print("error: torch cannot see CUDA; pass --allow-cpu-fallback to continue")
        return 3

    prompt_ids, logits_shape, forward_profile = _forward_logits_smoke(
        model,
        tokenizer,
        str(args.prompt),
        torch,
    )
    profiles.append(forward_profile)

    (
        layer_index,
        activation_shape,
        activation_primitive,
        activation_profile,
    ) = _activation_smoke(
        model,
        prompt_ids,
        int(args.collect_layer),
        torch,
    )
    profiles.append(activation_profile)

    generated, generation_profile = _generate_smoke(
        model,
        tokenizer,
        str(args.prompt),
        int(args.max_new_tokens),
        torch,
    )
    profiles.append(generation_profile)

    runtime_trace, runtime_trace_profile = _runtime_trace_smoke(
        model,
        tokenizer,
        str(args.model),
        str(args.prompt),
        layer_index,
        torch,
    )
    profiles.append(runtime_trace_profile)

    total_duration = sum(profile.duration_s for profile in profiles)
    runtime_duration = sum(profile.duration_s for profile in profiles[1:])
    total_tokens = len(prompt_ids) + int(args.max_new_tokens)
    tokens_per_second = (
        total_tokens / total_duration if total_duration > 0.0 else None
    )
    runtime_tokens_per_second = (
        total_tokens / runtime_duration if runtime_duration > 0.0 else None
    )
    forward_tokens_per_second = (
        len(prompt_ids) / forward_profile.duration_s
        if forward_profile.duration_s > 0.0
        else None
    )
    generation_tokens_per_second = (
        int(args.max_new_tokens) / generation_profile.duration_s
        if generation_profile.duration_s > 0.0
        else None
    )
    profile_values: list[RuntimeValue] = [
        _stage_profile_to_json(profile) for profile in profiles
    ]
    report: dict[str, RuntimeValue] = {}
    report["backend"] = "torch"
    report["model"] = str(args.model)
    report["python"] = sys.version.split()[0]
    report["platform"] = platform.platform()
    report["torch_version"] = torch.__version__
    report["torch_cuda_version"] = torch.version.cuda
    report["cuda"] = _cuda_report(torch)
    report["prompt_token_count"] = len(prompt_ids)
    report["max_new_tokens"] = int(args.max_new_tokens)
    report["total_profiled_duration_s"] = total_duration
    report["runtime_profiled_duration_s"] = runtime_duration
    report["tokens_per_second_including_load"] = tokens_per_second
    report["tokens_per_second_excluding_load"] = runtime_tokens_per_second
    report["forward_tokens_per_second"] = forward_tokens_per_second
    report["generation_tokens_per_second"] = generation_tokens_per_second
    report["logits_shape"] = logits_shape
    report["activation_layer"] = layer_index
    report["activation_shape"] = activation_shape
    report["activation_primitive"] = activation_primitive
    report["generated_text"] = generated
    report["runtime_trace"] = runtime_trace
    report["profiles"] = profile_values
    _write_or_print(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
