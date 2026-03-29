# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Guard early-mode runner."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING
from typing import cast as _cast

from vauban._pipeline._context import EarlyModeContext, log
from vauban._pipeline._mode_common import ModeReport, finish_mode_run, write_mode_report
from vauban._serializers import _guard_to_dict

if TYPE_CHECKING:
    from vauban._array import Array
    from vauban.types import CausalLM, Tokenizer


def _run_guard_mode(context: EarlyModeContext) -> None:
    """Run [guard] early-return mode and write its report."""
    import numpy as np

    from vauban import _ops as ops
    from vauban._forward import get_transformer as _get_transformer
    from vauban.guard import calibrate_guard_thresholds, guard_generate

    config = context.config
    if config.guard is None:
        msg = "guard config is required for guard mode"
        raise ValueError(msg)
    v = config.verbose
    model = _cast("CausalLM", context.model)
    tokenizer = _cast("Tokenizer", context.tokenizer)

    log(
        "Running guard circuit breaker",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )

    n_layers = len(_get_transformer(model).layers)
    guard_layers = config.guard.layers or list(range(n_layers))

    if context.direction_result is None:
        msg = "direction_result is required for guard mode but was not computed"
        raise ValueError(msg)
    direction = context.direction_result.direction

    # --- Load optional condition direction (AdaSteer dual-direction) ---
    condition_direction: Array | None = None
    if config.guard.condition_direction_path is not None:
        cond_path = Path(config.guard.condition_direction_path)
        if not cond_path.is_absolute():
            cond_path = config.output_dir.parent / cond_path
        log(
            f"Loading condition direction from {cond_path}",
            verbose=v,
            elapsed=time.monotonic() - context.t0,
        )
        cond_np = np.load(str(cond_path))
        condition_direction = ops.array(cond_np)
        expected_d = _get_transformer(model).embed_tokens.weight.shape[1]
        if condition_direction.shape[-1] != expected_d:
            msg = (
                "condition_direction d_model mismatch:"
                f" {condition_direction.shape[-1]} != {expected_d}"
            )
            raise ValueError(msg)

    # --- Load or build defensive embeddings ---
    defensive_embeds: Array | None = None
    if config.guard.defensive_embeddings_path is not None:
        def_path = Path(config.guard.defensive_embeddings_path)
        if not def_path.is_absolute():
            def_path = config.output_dir.parent / def_path
        log(
            f"Loading defensive embeddings from {def_path}",
            verbose=v,
            elapsed=time.monotonic() - context.t0,
        )
        def_np = np.load(str(def_path))
        defensive_embeds = ops.array(def_np)
        if defensive_embeds.ndim == 2:
            defensive_embeds = defensive_embeds[None, :, :]
    elif config.guard.defensive_prompt is not None:
        log(
            "Encoding defensive prompt to embeddings",
            verbose=v,
            elapsed=time.monotonic() - context.t0,
        )
        transformer = _get_transformer(model)
        def_ids = ops.array(
            tokenizer.encode(config.guard.defensive_prompt),
        )[None, :]
        defensive_embeds = transformer.embed_tokens(def_ids)

    # --- Calibrate tiers if requested ---
    tiers = list(config.guard.tiers)
    if config.guard.calibrate:
        cal_prompts = (
            context.harmless
            if config.guard.calibrate_prompts == "harmless"
            else context.harmful
        )
        if cal_prompts:
            log(
                f"Calibrating guard tiers from {len(cal_prompts)} prompts",
                verbose=v,
                elapsed=time.monotonic() - context.t0,
            )
            tiers = calibrate_guard_thresholds(
                model, tokenizer, cal_prompts, direction, guard_layers,
            )

    # If calibration produced new tiers, build a new config with them
    if tiers is not config.guard.tiers:
        from dataclasses import replace
        guard_cfg = replace(config.guard, tiers=tiers)
    else:
        guard_cfg = config.guard

    # --- Run guard generation for each prompt ---
    guard_results = [
        guard_generate(
            model,
            tokenizer,
            prompt,
            direction,
            guard_layers,
            guard_cfg,
            condition_direction=condition_direction,
            defensive_embeds=defensive_embeds,
        )
        for prompt in config.guard.prompts
    ]

    report_path = write_mode_report(
        config.output_dir,
        ModeReport(
            "guard_report.json",
            [_guard_to_dict(result) for result in guard_results],
        ),
    )

    # --- Export direction + tiers for GuardSession integration ---
    import json

    direction_path = config.output_dir / "guard_direction.npy"
    np.save(str(direction_path), np.array(direction))
    log(
        f"Exported guard direction to {direction_path}",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )

    tiers_data = [
        {"threshold": t.threshold, "zone": t.zone, "alpha": t.alpha}
        for t in tiers
    ]
    tiers_path = config.output_dir / "guard_tiers.json"
    tiers_path.write_text(json.dumps(tiers_data, indent=2))
    log(
        f"Exported guard tiers to {tiers_path}",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )

    log(
        f"Done — guard report written to {report_path}",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    total_rewinds = sum(r.total_rewinds for r in guard_results)
    total_broken = sum(1 for r in guard_results if r.circuit_broken)
    finish_mode_run(
        context,
        "guard",
        ["guard_report.json", "guard_direction.npy", "guard_tiers.json"],
        {
            "n_prompts": len(guard_results),
            "total_rewinds": total_rewinds,
            "circuit_broken": total_broken,
        },
    )
