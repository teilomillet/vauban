# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Layer selection, cut application, export, and modified-model loading."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from vauban._pipeline._context import log

if TYPE_CHECKING:
    from vauban._array import Array
    from vauban._pipeline._run_state import RunState


def run_cut_phase(state: RunState) -> None:
    """Select layers, cut weights, export, and hydrate the modified model."""
    from vauban._forward import get_transformer
    from vauban._model_io import load_model
    from vauban._ops import tree_flatten
    from vauban.cut import (
        _biprojected_direction,
        cut,
        cut_biprojected,
        cut_subspace,
        sparsify_direction,
    )
    from vauban.dequantize import dequantize_model, is_quantized
    from vauban.export import export_model
    from vauban.measure import measure, select_target_layers
    from vauban.types import DirectionResult

    config = state.config
    layer_types: list[str] | None = None
    if state.direction_result is not None:
        layer_types = state.direction_result.layer_types
    elif state.subspace_result is not None:
        layer_types = state.subspace_result.layer_types

    if config.cut.layers is not None:
        state.target_layers = config.cut.layers
    elif config.cut.layer_strategy != "all":
        if not state.cosine_scores:
            msg = "Probe-guided layer selection requires 'direction' mode"
            raise ValueError(msg)
        state.target_layers = select_target_layers(
            state.cosine_scores,
            config.cut.layer_strategy,
            config.cut.layer_top_k,
            layer_types=layer_types,
            type_filter=config.cut.layer_type_filter,
        )
    else:
        state.target_layers = list(range(len(get_transformer(state.model).layers)))

    state.flat_weights = cast(
        "dict[str, Array]",
        dict(tree_flatten(state.model.parameters())),  # type: ignore[attr-defined]
    )

    if state.direction_result is not None and config.cut.sparsity > 0.0:
        state.direction_result = DirectionResult(
            direction=sparsify_direction(
                state.direction_result.direction,
                config.cut.sparsity,
            ),
            layer_index=state.direction_result.layer_index,
            cosine_scores=state.direction_result.cosine_scores,
            d_model=state.direction_result.d_model,
            model_path=state.direction_result.model_path,
            layer_types=state.direction_result.layer_types,
        )

    if (
        config.cut.false_refusal_ortho
        and config.borderline_path is not None
        and state.direction_result is not None
        and state.harmless is not None
    ):
        from vauban.dataset import resolve_prompts

        borderline = resolve_prompts(config.borderline_path)
        false_refusal_result = measure(
            state.model,
            state.tokenizer,
            borderline,
            state.harmless,
            config.measure.clip_quantile,
        )
        state.direction_result = DirectionResult(
            direction=_biprojected_direction(
                state.direction_result.direction,
                false_refusal_result.direction,
            ),
            layer_index=state.direction_result.layer_index,
            cosine_scores=state.direction_result.cosine_scores,
            d_model=state.direction_result.d_model,
            model_path=state.direction_result.model_path,
            layer_types=state.direction_result.layer_types,
        )

    log(
        f"Cutting {len(state.target_layers)} layers (alpha={config.cut.alpha})",
        verbose=state.verbose,
        elapsed=state.elapsed(),
    )
    layer_weights = config.cut.layer_weights
    if config.measure.mode == "subspace":
        if state.subspace_result is None:
            msg = "subspace_result is required for subspace cut but was None"
            raise ValueError(msg)
        state.modified_weights = cut_subspace(
            state.flat_weights,
            state.subspace_result.basis,
            state.target_layers,
            config.cut.alpha,
            config.cut.norm_preserve,
            layer_weights,
        )
    elif config.cut.biprojected:
        if (
            state.direction_result is None
            or state.harmful is None
            or state.harmless is None
        ):
            msg = (
                "direction_result, harmful, and harmless"
                " are required for biprojected cut"
            )
            raise ValueError(msg)
        harmless_acts = measure(
            state.model,
            state.tokenizer,
            state.harmless,
            state.harmful,
            config.measure.clip_quantile,
        )
        state.modified_weights = cut_biprojected(
            state.flat_weights,
            state.direction_result.direction,
            harmless_acts.direction,
            state.target_layers,
            config.cut.alpha,
            config.cut.norm_preserve,
            layer_weights,
        )
    else:
        if state.direction_result is None:
            msg = "direction_result is required for cut but was None"
            raise ValueError(msg)
        state.modified_weights = cut(
            state.flat_weights,
            state.direction_result.direction,
            state.target_layers,
            config.cut.alpha,
            config.cut.norm_preserve,
            layer_weights,
        )

    if config.measure.mode == "dbdi" and config.cut.dbdi_target == "both":
        if state.dbdi_result is None:
            msg = "dbdi_result is required for DBDI both-mode cut"
            raise ValueError(msg)
        state.modified_weights = cut(
            state.modified_weights,
            state.dbdi_result.hdd,
            state.target_layers,
            config.cut.alpha,
            config.cut.norm_preserve,
            layer_weights,
        )

    log(
        f"Exporting modified model to {config.output_dir}",
        verbose=state.verbose,
        elapsed=state.elapsed(),
    )
    export_model(config.model_path, state.modified_weights, config.output_dir)

    needs_modified = (
        config.eval.prompts_path is not None or state.surface_before is not None
    )
    if not needs_modified:
        return
    state.modified_model, _ = load_model(config.model_path)
    if is_quantized(state.modified_model):
        dequantize_model(state.modified_model)
    state.modified_model.load_weights(  # type: ignore[attr-defined]
        list(state.modified_weights.items()),
    )
