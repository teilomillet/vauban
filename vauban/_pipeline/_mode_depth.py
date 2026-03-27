# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Depth early-mode runner."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from typing import cast as _cast

from vauban._pipeline._context import EarlyModeContext, log
from vauban._pipeline._helpers import is_default_data
from vauban._pipeline._mode_common import ModeReport, finish_mode_run, write_mode_report
from vauban._serializers import _depth_direction_to_dict, _depth_to_dict

if TYPE_CHECKING:
    from vauban.types import CausalLM, DepthResult, DirectionResult, Tokenizer


def _run_depth_mode(context: EarlyModeContext) -> None:
    """Run [depth] early-return mode and write its report."""
    config = context.config
    if config.depth is None:
        msg = "depth config is required for depth mode"
        raise ValueError(msg)
    v = config.verbose
    model = _cast("CausalLM", context.model)
    tokenizer = _cast("Tokenizer", context.tokenizer)

    from vauban.dataset import resolve_prompts
    from vauban.depth import depth_direction, depth_generate, depth_profile
    from vauban.measure import measure

    log(
        "Running depth analysis",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    depth_results: list[DepthResult] = []
    for prompt in config.depth.prompts:
        if config.depth.max_tokens > 0:
            result = depth_generate(model, tokenizer, prompt, config.depth)
        else:
            result = depth_profile(model, tokenizer, prompt, config.depth)
        depth_results.append(result)

    report: dict[str, object] = {
        "dtr_results": [_depth_to_dict(result) for result in depth_results],
    }

    if config.depth.extract_direction and len(depth_results) >= 2:
        log(
            "Extracting depth direction",
            verbose=v,
            elapsed=time.monotonic() - context.t0,
        )
        dir_prompts = config.depth.direction_prompts
        if dir_prompts is not None:
            dir_depth_results: list[DepthResult] = []
            for prompt in dir_prompts:
                if config.depth.max_tokens > 0:
                    result = depth_generate(model, tokenizer, prompt, config.depth)
                else:
                    result = depth_profile(model, tokenizer, prompt, config.depth)
                dir_depth_results.append(result)
        else:
            dir_depth_results = depth_results

        refusal_dir: DirectionResult | None = None
        if not is_default_data(config):
            log(
                "Computing refusal direction for cosine comparison",
                verbose=v,
                elapsed=time.monotonic() - context.t0,
            )
            harmful = resolve_prompts(config.harmful_path)
            harmless = resolve_prompts(config.harmless_path)
            refusal_dir = measure(
                model,
                tokenizer,
                harmful,
                harmless,
                config.depth.clip_quantile,
            )

        depth_dir_result = depth_direction(
            model,
            tokenizer,
            dir_depth_results,
            refusal_direction=refusal_dir,
            clip_quantile=config.depth.clip_quantile,
        )

        import numpy as np

        dir_path = config.output_dir / "depth_direction.npy"
        dir_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(dir_path), np.array(depth_dir_result.direction))
        report["direction"] = _depth_direction_to_dict(depth_dir_result)

    report_path = write_mode_report(
        config.output_dir,
        ModeReport("depth_report.json", report),
    )
    log(
        f"Done — depth report written to {report_path}",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    finish_mode_run(
        context,
        "depth",
        ["depth_report.json"],
        {"n_prompts": len(depth_results)},
    )
