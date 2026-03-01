"""SVF early-mode runner."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from typing import cast as _cast

from vauban._pipeline._context import EarlyModeContext, log
from vauban._pipeline._mode_common import ModeReport, finish_mode_run, write_mode_report

if TYPE_CHECKING:
    from vauban.types import CausalLM, Tokenizer


def _run_svf_mode(context: EarlyModeContext) -> None:
    """Run [svf] early-return mode: train SVF boundary and write its report."""
    config = context.config
    if config.svf is None:
        msg = "svf config is required for svf mode"
        raise ValueError(msg)
    v = config.verbose
    model = _cast("CausalLM", context.model)
    tokenizer = _cast("Tokenizer", context.tokenizer)

    log(
        "Training SVF boundary",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )

    from vauban._forward import get_transformer as _get_transformer
    from vauban.measure import load_prompts
    from vauban.svf import save_svf_boundary, train_svf_boundary

    transformer = _get_transformer(model)
    d_model = transformer.embed_tokens.weight.shape[1]
    n_layers = len(transformer.layers)
    target_prompts = load_prompts(config.svf.prompts_target)
    opposite_prompts = load_prompts(config.svf.prompts_opposite)

    boundary, svf_result = train_svf_boundary(
        model,
        tokenizer,
        target_prompts,
        opposite_prompts,
        d_model,
        n_layers,
        projection_dim=config.svf.projection_dim,
        hidden_dim=config.svf.hidden_dim,
        n_epochs=config.svf.n_epochs,
        learning_rate=config.svf.learning_rate,
        layers=config.svf.layers,
    )

    boundary_path = config.output_dir / "svf_boundary.safetensors"
    save_svf_boundary(boundary, boundary_path)

    report = {
        "train_loss_history": svf_result.train_loss_history,
        "final_accuracy": svf_result.final_accuracy,
        "per_layer_separation": svf_result.per_layer_separation,
        "projection_dim": svf_result.projection_dim,
        "hidden_dim": svf_result.hidden_dim,
        "n_layers_trained": svf_result.n_layers_trained,
        "boundary_path": str(boundary_path),
    }
    report_path = write_mode_report(
        config.output_dir,
        ModeReport("svf_report.json", report),
    )
    log(
        f"Done — SVF report written to {report_path}",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    finish_mode_run(
        context,
        "svf",
        ["svf_report.json", "svf_boundary.safetensors"],
        {"final_accuracy": svf_result.final_accuracy},
    )
