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
    from vauban.config import load_config
    from vauban.dequantize import dequantize_model, is_quantized

    config = load_config(config_path)
    t0 = time.monotonic()
    log(
        f"Loading model {config.model_path}",
        verbose=config.verbose,
        elapsed=time.monotonic() - t0,
    )
    model, tokenizer = load_model(config.model_path)
    if is_quantized(model):
        log(
            "Dequantizing model weights",
            verbose=config.verbose,
            elapsed=time.monotonic() - t0,
        )
        dequantize_model(model)

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
