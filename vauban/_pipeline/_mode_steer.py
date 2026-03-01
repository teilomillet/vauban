"""Steer early-mode runner."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING
from typing import cast as _cast

from vauban._pipeline._context import EarlyModeContext, log
from vauban._pipeline._mode_common import ModeReport, finish_mode_run, write_mode_report
from vauban._serializers import _steer_to_dict

if TYPE_CHECKING:
    from vauban.types import CausalLM, Tokenizer


def _run_steer_mode(context: EarlyModeContext) -> None:
    """Run [steer] early-return mode and write its report."""
    config = context.config
    if config.steer is None:
        msg = "steer config is required for steer mode"
        raise ValueError(msg)
    v = config.verbose
    model = _cast("CausalLM", context.model)
    tokenizer = _cast("Tokenizer", context.tokenizer)

    from vauban._forward import get_transformer as _get_transformer
    from vauban.probe import steer
    from vauban.svf import load_svf_boundary

    log(
        "Running steer generation",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )

    n_layers = len(_get_transformer(model).layers)
    steer_layers = config.steer.layers or list(range(n_layers))

    if config.steer.bank_path and config.steer.composition:
        from vauban._compose import compose_direction, load_bank

        log(
            f"Composing direction from bank: {config.steer.bank_path}",
            verbose=v,
            elapsed=time.monotonic() - context.t0,
        )
        bank = load_bank(config.steer.bank_path)
        composed = compose_direction(bank, config.steer.composition)
        steer_results = [
            steer(
                model,
                tokenizer,
                prompt,
                composed,
                steer_layers,
                config.steer.alpha,
                config.steer.max_tokens,
            )
            for prompt in config.steer.prompts
        ]
    elif config.steer.direction_source == "svf":
        if config.steer.svf_boundary_path is None:
            msg = "svf_boundary_path is required when direction_source='svf'"
            raise ValueError(msg)
        from vauban.probe import steer_svf as _steer_svf

        boundary = load_svf_boundary(Path(config.steer.svf_boundary_path))
        steer_results = [
            _steer_svf(
                model,
                tokenizer,
                prompt,
                boundary,
                steer_layers,
                config.steer.alpha,
                config.steer.max_tokens,
            )
            for prompt in config.steer.prompts
        ]
    else:
        if context.direction_result is None:
            msg = "direction_result is required for steer mode but was not computed"
            raise ValueError(msg)
        steer_results = [
            steer(
                model,
                tokenizer,
                prompt,
                context.direction_result.direction,
                steer_layers,
                config.steer.alpha,
                config.steer.max_tokens,
            )
            for prompt in config.steer.prompts
        ]

    report_path = write_mode_report(
        config.output_dir,
        ModeReport(
            "steer_report.json",
            [_steer_to_dict(result) for result in steer_results],
        ),
    )
    log(
        f"Done — steer report written to {report_path}",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    finish_mode_run(context, "steer", ["steer_report.json"], {})
