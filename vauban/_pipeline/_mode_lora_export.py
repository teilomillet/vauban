"""LoRA export early-mode runner."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, cast

from vauban._pipeline._context import EarlyModeContext, log
from vauban._pipeline._mode_common import ModeReport, finish_mode_run, write_mode_report

if TYPE_CHECKING:
    from vauban._array import Array


def _run_lora_export_mode(context: EarlyModeContext) -> None:
    """Run [lora_export] mode: export measured direction as LoRA adapter."""
    config = context.config
    if config.lora_export is None:
        msg = "lora_export config is required for lora_export mode"
        raise ValueError(msg)
    if context.direction_result is None:
        msg = "lora_export requires a measured direction (direction_result is None)"
        raise ValueError(msg)

    v = config.verbose
    log(
        "Exporting measured direction as LoRA adapter",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )

    from vauban._forward import get_transformer
    from vauban._ops import tree_flatten
    from vauban.cut import sparsify_direction
    from vauban.lora import build_lora_weights, save_adapter_mlx, save_adapter_peft
    from vauban.measure import select_target_layers

    direction = context.direction_result.direction
    model = context.model

    # Apply sparsity if configured
    if config.cut.sparsity > 0.0:
        direction = sparsify_direction(direction, config.cut.sparsity)

    # Determine rank from direction shape
    rank = 1 if direction.ndim == 1 else direction.shape[0]

    # Resolve target layers (same logic as cut phase)
    layer_types = context.direction_result.layer_types

    if config.cut.layers is not None:
        target_layers = config.cut.layers
    elif config.cut.layer_strategy != "all":
        cosine_scores = context.direction_result.cosine_scores
        if not cosine_scores:
            msg = "Probe-guided layer selection requires cosine scores"
            raise ValueError(msg)
        target_layers = select_target_layers(
            cosine_scores,
            config.cut.layer_strategy,
            config.cut.layer_top_k,
            layer_types=layer_types,
            type_filter=config.cut.layer_type_filter,
        )
    else:
        target_layers = list(
            range(len(get_transformer(model).layers)),  # type: ignore[arg-type]
        )

    # Get flat weights (read-only)
    flat_weights = cast(
        "dict[str, Array]",
        dict(tree_flatten(model.parameters())),  # type: ignore[attr-defined]
    )

    # Build LoRA matrices
    matrices = build_lora_weights(
        direction,
        flat_weights,
        target_layers,
        alpha=config.cut.alpha,
        polarity=config.lora_export.polarity,
        layer_weights=config.cut.layer_weights,
    )

    # Save adapter
    output_dir = config.output_dir / "lora_adapter"
    fmt = config.lora_export.format

    if fmt == "mlx":
        adapter_path = save_adapter_mlx(
            matrices, output_dir, rank, config.model_path,
        )
    else:
        adapter_path = save_adapter_peft(
            matrices, output_dir, rank, config.model_path,
        )

    from vauban.types import LoraExportResult

    result = LoraExportResult(
        output_path=str(adapter_path),
        format=fmt,
        polarity=config.lora_export.polarity,
        rank=rank,
        n_weights=len(matrices),
        target_layers=target_layers,
    )
    result_dict = result.to_dict()

    report_path = write_mode_report(
        config.output_dir,
        ModeReport("lora_export_report.json", result_dict),
    )
    log(
        f"Done — LoRA adapter ({fmt}) written to {adapter_path}",
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
        "lora_export",
        ["lora_export_report.json"],
        {"n_weights": len(matrices), "rank": rank},
    )
