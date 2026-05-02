# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Main pipeline entry point: run()."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from vauban._pipeline._context import log, write_experiment_log
from vauban._pipeline._modes import dispatch_early_mode
from vauban._pipeline._run_cut import run_cut_phase
from vauban._pipeline._run_eval import run_eval_phase
from vauban._pipeline._run_measure import run_measure_phase
from vauban._pipeline._run_state import RunState
from vauban._pipeline._run_surface import (
    finalize_surface_phase,
    prepare_surface_phase,
)

if TYPE_CHECKING:
    from pathlib import Path


def run(config_path: str | Path) -> None:
    """Run the full measure -> cut -> evaluate pipeline from a TOML config."""
    from vauban._model_io import load_model
    from vauban._pipeline._context import EarlyModeContext as _EarlyModeContext
    from vauban.config import load_config
    from vauban.config._mode_registry import active_early_mode_for_phase
    from vauban.dequantize import dequantize_model, is_quantized

    config = load_config(config_path)
    t0 = time.monotonic()

    # Standalone modes: no local model needed.
    standalone_ctx = _EarlyModeContext(
        config_path=config_path,
        config=config,
        model=None,
        tokenizer=None,
        t0=t0,
    )
    if dispatch_early_mode("standalone", standalone_ctx):
        return
    if (
        config.behavior_trace is not None
        and config.behavior_trace.runtime_backend == "api"
    ):
        from vauban._pipeline._mode_behavior_trace import _run_behavior_trace_mode

        _run_behavior_trace_mode(standalone_ctx)
        return

    log(
        f"Loading model {config.model_path}",
        verbose=config.verbose,
        elapsed=time.monotonic() - t0,
    )
    model, tokenizer = load_model(config.model_path)

    early_spec = active_early_mode_for_phase(config, "before_prompts")
    if (
        early_spec is not None
        and early_spec.mode == "behavior_trace"
        and config.lora_load is None
    ):
        # Behavior traces only need generation/runtime summaries. Keep quantized
        # MLX models quantized so large 4-bit audit targets remain runnable.
        trace_ctx = _EarlyModeContext(
            config_path=config_path,
            config=config,
            model=model,
            tokenizer=tokenizer,
            t0=t0,
        )
        if dispatch_early_mode("before_prompts", trace_ctx):
            return

    if is_quantized(model):
        log(
            "Dequantizing model weights",
            verbose=config.verbose,
            elapsed=time.monotonic() - t0,
        )
        dequantize_model(model)

    if config.lora_load is not None:
        from vauban.lora import load_and_apply_adapter, load_and_merge_adapters

        lora = config.lora_load
        if lora.adapter_path is not None:
            log(
                f"Loading LoRA adapter from {lora.adapter_path}",
                verbose=config.verbose,
                elapsed=time.monotonic() - t0,
            )
            load_and_apply_adapter(model, lora.adapter_path)
        elif lora.adapter_paths is not None:
            log(
                f"Merging {len(lora.adapter_paths)} LoRA adapters",
                verbose=config.verbose,
                elapsed=time.monotonic() - t0,
            )
            load_and_merge_adapters(model, lora.adapter_paths, lora.weights)

    state = RunState(
        config_path=config_path,
        config=config,
        model=model,
        tokenizer=tokenizer,
        t0=t0,
        verbose=config.verbose,
    )
    if dispatch_early_mode("before_prompts", state.early_mode_context()):
        return
    if run_measure_phase(state):
        return
    if dispatch_early_mode("after_measure", state.early_mode_context()):
        return

    prepare_surface_phase(state)
    run_cut_phase(state)
    finalize_surface_phase(state)
    run_eval_phase(state)

    log(
        f"Done — output written to {config.output_dir}",
        verbose=state.verbose,
        elapsed=state.elapsed(),
    )
    write_experiment_log(
        config_path,
        config,
        "default",
        state.report_files,
        state.metrics,
        state.elapsed(),
    )
