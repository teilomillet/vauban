"""CAST early-mode runner."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING
from typing import cast as _cast

from vauban._pipeline._context import EarlyModeContext, log
from vauban._pipeline._mode_common import ModeReport, finish_mode_run, write_mode_report
from vauban._serializers import _cast_to_dict

if TYPE_CHECKING:
    from vauban._array import Array
    from vauban.types import CausalLM, Tokenizer


def _run_cast_mode(context: EarlyModeContext) -> None:
    """Run [cast] early-return mode and write its report."""
    import numpy as np

    from vauban import _ops as ops
    from vauban._forward import get_transformer as _get_transformer
    from vauban.cast import cast_generate
    from vauban.svf import load_svf_boundary

    config = context.config
    if config.cast is None:
        msg = "cast config is required for cast mode"
        raise ValueError(msg)
    v = config.verbose
    model = _cast("CausalLM", context.model)
    tokenizer = _cast("Tokenizer", context.tokenizer)

    log(
        "Running CAST conditional steering",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )

    n_layers = len(_get_transformer(model).layers)
    cast_layers = config.cast.layers or list(range(n_layers))

    baseline_activations: dict[int, Array] | None = None
    if (
        config.cast.externality_monitor
        and config.cast.baseline_activations_path is not None
    ):
        baseline_path = Path(config.cast.baseline_activations_path)
        if not baseline_path.is_absolute():
            baseline_path = config.output_dir.parent / baseline_path
        loaded = ops.load(str(baseline_path))
        if not isinstance(loaded, dict):
            msg = (
                "Expected dict from baseline activations file,"
                f" got {type(loaded).__name__}"
            )
            raise TypeError(msg)
        loaded_dict = _cast("dict[str, Array]", loaded)
        baseline_activations = {int(key): value for key, value in loaded_dict.items()}

    condition_direction: Array | None = None
    if config.cast.condition_direction_path is not None:
        cond_path = Path(config.cast.condition_direction_path)
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

    if config.cast.bank_path and config.cast.composition:
        from vauban._compose import compose_direction, load_bank

        log(
            f"Composing direction from bank: {config.cast.bank_path}",
            verbose=v,
            elapsed=time.monotonic() - context.t0,
        )
        bank = load_bank(config.cast.bank_path)
        composed = compose_direction(bank, config.cast.composition)
        cast_results = [
            cast_generate(
                model,
                tokenizer,
                prompt,
                composed,
                cast_layers,
                config.cast.alpha,
                config.cast.threshold,
                config.cast.max_tokens,
                condition_direction=condition_direction,
                alpha_tiers=config.cast.alpha_tiers,
                baseline_activations=baseline_activations,
                displacement_threshold=config.cast.displacement_threshold,
            )
            for prompt in config.cast.prompts
        ]
    elif config.cast.direction_source == "svf":
        if config.cast.svf_boundary_path is None:
            msg = "svf_boundary_path is required when direction_source='svf'"
            raise ValueError(msg)
        from vauban.cast import cast_generate_svf as _cast_generate_svf

        boundary = load_svf_boundary(Path(config.cast.svf_boundary_path))
        cast_results = [
            _cast_generate_svf(
                model,
                tokenizer,
                prompt,
                boundary,
                cast_layers,
                config.cast.alpha,
                config.cast.max_tokens,
            )
            for prompt in config.cast.prompts
        ]
    else:
        if context.direction_result is None:
            msg = "direction_result is required for cast mode but was not computed"
            raise ValueError(msg)
        cast_results = [
            cast_generate(
                model,
                tokenizer,
                prompt,
                context.direction_result.direction,
                cast_layers,
                config.cast.alpha,
                config.cast.threshold,
                config.cast.max_tokens,
                condition_direction=condition_direction,
                alpha_tiers=config.cast.alpha_tiers,
                baseline_activations=baseline_activations,
                displacement_threshold=config.cast.displacement_threshold,
            )
            for prompt in config.cast.prompts
        ]

    report_path = write_mode_report(
        config.output_dir,
        ModeReport(
            "cast_report.json",
            [_cast_to_dict(result) for result in cast_results],
        ),
    )
    log(
        f"Done — CAST report written to {report_path}",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    total_interventions = sum(r.interventions for r in cast_results)
    finish_mode_run(
        context,
        "cast",
        ["cast_report.json"],
        {"n_prompts": len(cast_results), "interventions": total_interventions},
    )
