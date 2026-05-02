# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Run a controlled real-model Torch CUDA trace sweep.

This is a smoke-grade scaling check. It records trace/profile evidence across
several prompt lengths and emits a USL-ready sweep summary, but it does not fit
or claim a Universal Scaling Law model.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from scripts.torch_cuda_real_model_smoke import (
    DEFAULT_MODEL,
    _cuda_report,
    _load_torch_model,
    _sync_cuda,
)
from vauban.runtime import (
    LoadedModel,
    ModelRef,
    ModelRuntime,
    RuntimeTraceResult,
    RuntimeValue,
    Trace,
    TraceRequest,
    create_runtime,
    forward_trace_summary,
    summarize_trace_profile_sweep,
)


def _parse_counts(raw: str) -> tuple[int, ...]:
    """Parse comma-separated positive repeat counts."""
    counts: list[int] = []
    for item in raw.split(","):
        value = int(item.strip())
        if value < 1:
            msg = "repeat counts must be positive"
            raise ValueError(msg)
        counts.append(value)
    if not counts:
        msg = "at least one repeat count is required"
        raise ValueError(msg)
    return tuple(counts)


def _prompt(base_prompt: str, repeat_count: int) -> str:
    """Return a prompt with controlled text growth."""
    return " ".join(base_prompt for _ in range(repeat_count))


def _run_trace(
    runtime: ModelRuntime,
    loaded: LoadedModel,
    *,
    trace_id: str,
    prompt: str,
    collect_layer: int,
    repeat_count: int,
) -> RuntimeTraceResult:
    """Run one trace through a runtime-like object."""
    return runtime.trace(
        loaded,
        TraceRequest(
            trace_id=trace_id,
            input_text=prompt,
            requested_artifacts=(
                "tokens",
                "logits",
                "logprobs",
                "activation",
            ),
            collect_layers=(collect_layer,),
            apply_chat_template=True,
            return_logits=True,
            return_logprobs=True,
            metadata={"repeat_count": repeat_count},
        ),
    )


def _write_or_print(report: dict[str, RuntimeValue], output: Path | None) -> None:
    """Write the sweep report to disk or stdout."""
    text = json.dumps(report, indent=2, sort_keys=True)
    if output is None:
        print(text)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n")
    print(f"wrote sweep report: {output}")


def main() -> int:
    """Run the controlled Torch CUDA trace sweep."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default="What changes when models change?")
    parser.add_argument("--repeat-counts", default="1,2,4")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--collect-layer", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    repeat_counts = _parse_counts(str(args.repeat_counts))
    samples = int(args.samples)
    warmup = int(args.warmup)
    if samples < 1:
        print("error: --samples must be >= 1")
        return 2
    if warmup < 0:
        print("error: --warmup must be >= 0")
        return 2
    model, tokenizer, torch = _load_torch_model(
        str(args.model),
        local_files_only=bool(args.local_files_only),
    )
    cuda_available = torch.cuda.is_available()
    if not cuda_available and not args.allow_cpu_fallback:
        print("error: torch cannot see CUDA; pass --allow-cpu-fallback to continue")
        return 3
    if cuda_available:
        torch.cuda.reset_peak_memory_stats(0)

    runtime = create_runtime("torch")
    loaded = LoadedModel(
        ref=ModelRef(str(args.model)),
        backend="torch",
        capabilities=runtime.capabilities,
        model=model,
        tokenizer=tokenizer,
    )

    for warmup_index in range(warmup):
        for repeat_count in repeat_counts:
            _run_trace(
                runtime,
                loaded,
                trace_id=f"cuda_sweep.warmup_{warmup_index}.repeat_{repeat_count}",
                prompt=_prompt(str(args.prompt), repeat_count),
                collect_layer=int(args.collect_layer),
                repeat_count=repeat_count,
            )
            _sync_cuda(torch)
    if cuda_available:
        torch.cuda.reset_peak_memory_stats(0)

    trace_summaries: list[RuntimeValue] = []
    traces: list[Trace] = []
    for repeat_count in repeat_counts:
        for sample_index in range(samples):
            result = _run_trace(
                runtime,
                loaded,
                trace_id=(
                    f"cuda_sweep.repeat_{repeat_count}.sample_{sample_index}"
                ),
                prompt=_prompt(str(args.prompt), repeat_count),
                collect_layer=int(args.collect_layer),
                repeat_count=repeat_count,
            )
            _sync_cuda(torch)
            traces.append(result.trace)
            summary = forward_trace_summary(
                result.forward,
                runtime_trace=result.trace,
            )
            summary["trace_id"] = result.trace.trace_id
            summary["trace_metadata"] = dict(result.trace.metadata)
            trace_summaries.append(summary)

    sweep = summarize_trace_profile_sweep(
        tuple(traces),
        sweep_id="cuda_sweep.token_count",
    )
    report: dict[str, RuntimeValue] = {
        "backend": "torch",
        "model": str(args.model),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda": _cuda_report(torch),
        "repeat_counts": list(repeat_counts),
        "samples_per_count": samples,
        "warmup_per_count": warmup,
        "collect_layer": int(args.collect_layer),
        "sweep": sweep.to_dict(),
        "traces": trace_summaries,
    }
    _write_or_print(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
