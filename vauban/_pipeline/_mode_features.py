"""Features early-mode runner."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from typing import cast as _cast

from vauban._pipeline._context import EarlyModeContext, log
from vauban._pipeline._mode_common import ModeReport, finish_mode_run, write_mode_report

if TYPE_CHECKING:
    from vauban._array import Array
    from vauban.types import CausalLM, Tokenizer


def _run_features_mode(context: EarlyModeContext) -> None:
    """Run [features] early-return mode: SAE training and report."""
    config = context.config
    if config.features is None:
        msg = "features config is required for features mode"
        raise ValueError(msg)
    v = config.verbose
    model = _cast("CausalLM", context.model)
    tokenizer = _cast("Tokenizer", context.tokenizer)

    log(
        "Training sparse autoencoders",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )

    from vauban.features import save_sae, train_sae_multi_layer
    from vauban.measure import load_prompts

    prompts = load_prompts(config.features.prompts_path)

    direction: Array | None = None
    if context.direction_result is not None:
        direction = context.direction_result.direction

    saes, result = train_sae_multi_layer(
        model,
        tokenizer,
        prompts,
        config.features.layers,
        d_sae=config.features.d_sae,
        l1_coeff=config.features.l1_coeff,
        n_epochs=config.features.n_epochs,
        learning_rate=config.features.learning_rate,
        batch_size=config.features.batch_size,
        token_position=config.features.token_position,
        dead_feature_threshold=config.features.dead_feature_threshold,
        direction=direction,
        model_path=config.model_path,
    )

    for layer_idx, sae in saes.items():
        sae_path = config.output_dir / f"sae_layer_{layer_idx}.safetensors"
        save_sae(sae, sae_path)

    report_path = write_mode_report(
        config.output_dir,
        ModeReport("features_report.json", result.to_dict()),
    )
    log(
        f"Done — features report written to {report_path}",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    finish_mode_run(
        context,
        "features",
        ["features_report.json"] + [f"sae_layer_{idx}.safetensors" for idx in saes],
        {"n_layers": len(result.layers)},
    )
