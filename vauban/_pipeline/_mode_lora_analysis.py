# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""LoRA analysis early-mode runner."""

from __future__ import annotations

import time

from vauban._pipeline._context import EarlyModeContext, log
from vauban._pipeline._mode_common import ModeReport, finish_mode_run, write_mode_report


def _run_lora_analysis_mode(context: EarlyModeContext) -> None:
    """Run [lora_analysis] mode: decompose adapters via SVD."""
    config = context.config
    if config.lora_analysis is None:
        msg = "lora_analysis config is required for lora_analysis mode"
        raise ValueError(msg)

    v = config.verbose
    log(
        "Analyzing LoRA adapter structure",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )

    from vauban.lora import analyze_adapter

    analysis_config = config.lora_analysis

    # Extract direction if available and requested
    direction = None
    if analysis_config.align_with_direction and context.direction_result is not None:
        direction = context.direction_result.direction

    # Collect adapter paths to analyze
    paths: list[str] = []
    if analysis_config.adapter_path is not None:
        paths.append(analysis_config.adapter_path)
    elif analysis_config.adapter_paths is not None:
        paths.extend(analysis_config.adapter_paths)

    all_results: list[dict[str, object]] = []
    total_layers = 0
    total_params = 0

    for adapter_path in paths:
        log(
            f"Analyzing adapter: {adapter_path}",
            verbose=v,
            elapsed=time.monotonic() - context.t0,
        )
        result = analyze_adapter(
            adapter_path,
            variance_threshold=analysis_config.variance_threshold,
            direction=direction,
        )
        all_results.append(result.to_dict())
        total_layers += len(result.layers)
        total_params += result.total_params

    payload: dict[str, object] = {
        "adapters": all_results,
        "n_adapters": len(paths),
        "total_layers": total_layers,
        "total_params": total_params,
    }

    report_path = write_mode_report(
        config.output_dir,
        ModeReport("lora_analysis_report.json", payload),
    )
    log(
        f"Done — analyzed {len(paths)} adapter(s), {total_layers} weight pairs",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    log(
        f"Report: {report_path}",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    finish_mode_run(
        context,
        "lora_analysis",
        ["lora_analysis_report.json"],
        {"n_adapters": len(paths), "total_layers": total_layers},
    )
