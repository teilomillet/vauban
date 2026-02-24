"""Vauban - MLX-native abliteration toolkit for Apple Silicon."""

from pathlib import Path

from vauban.config import load_config
from vauban.cut import (
    cut,
    cut_biprojected,
    cut_false_refusal_ortho,
    cut_subspace,
    save_weights,
    sparsify_direction,
    target_weight_keys,
)
from vauban.dataset import load_hf_prompts, resolve_prompts
from vauban.dequantize import dequantize_model, is_quantized
from vauban.evaluate import evaluate
from vauban.export import export_model
from vauban.measure import (
    default_eval_path,
    default_prompt_paths,
    find_instruction_boundary,
    load_prompts,
    measure,
    measure_dbdi,
    measure_subspace,
    select_target_layers,
    silhouette_scores,
)
from vauban.probe import multi_probe, probe, steer
from vauban.subspace import (
    effective_rank,
    explained_variance_ratio,
    grassmann_distance,
    orthonormalize,
    principal_angles,
    project_subspace,
    remove_subspace,
    subspace_overlap,
)
from vauban.surface import (
    aggregate,
    compare_surfaces,
    default_surface_path,
    find_threshold,
    load_surface_prompts,
    map_surface,
    scan,
)
from vauban.types import (
    CutConfig,
    DatasetRef,
    DBDIResult,
    DirectionResult,
    EvalResult,
    MeasureConfig,
    PipelineConfig,
    ProbeResult,
    SteerResult,
    SubspaceResult,
    SurfaceComparison,
    SurfaceConfig,
    SurfaceGroup,
    SurfaceGroupDelta,
    SurfacePoint,
    SurfacePrompt,
    SurfaceResult,
)

__version__ = "0.2.0"

__all__ = [
    "CutConfig",
    "DBDIResult",
    "DatasetRef",
    "DirectionResult",
    "EvalResult",
    "MeasureConfig",
    "PipelineConfig",
    "ProbeResult",
    "SteerResult",
    "SubspaceResult",
    "SurfaceComparison",
    "SurfaceConfig",
    "SurfaceGroup",
    "SurfaceGroupDelta",
    "SurfacePoint",
    "SurfacePrompt",
    "SurfaceResult",
    "aggregate",
    "compare_surfaces",
    "cut",
    "cut_biprojected",
    "cut_false_refusal_ortho",
    "cut_subspace",
    "default_eval_path",
    "default_prompt_paths",
    "default_surface_path",
    "dequantize_model",
    "effective_rank",
    "evaluate",
    "explained_variance_ratio",
    "export_model",
    "find_instruction_boundary",
    "find_threshold",
    "grassmann_distance",
    "is_quantized",
    "load_config",
    "load_hf_prompts",
    "load_prompts",
    "load_surface_prompts",
    "map_surface",
    "measure",
    "measure_dbdi",
    "measure_subspace",
    "multi_probe",
    "orthonormalize",
    "principal_angles",
    "probe",
    "project_subspace",
    "remove_subspace",
    "resolve_prompts",
    "run",
    "save_weights",
    "scan",
    "select_target_layers",
    "silhouette_scores",
    "sparsify_direction",
    "steer",
    "subspace_overlap",
    "target_weight_keys",
]


def run(config_path: str | Path) -> None:
    """Run the full measure -> cut -> evaluate pipeline from a TOML config."""
    import json

    import mlx_lm
    from mlx.utils import tree_flatten

    config = load_config(config_path)

    model, tokenizer = mlx_lm.load(config.model_path)  # type: ignore[invalid-assignment]

    # Auto-dequantize if model has quantized weights
    if is_quantized(model):
        dequantize_model(model)

    harmful = resolve_prompts(config.harmful_path)
    harmless = resolve_prompts(config.harmless_path)

    # Branch on measure mode
    direction_result = None
    subspace_result = None
    cosine_scores: list[float] = []

    clip_q = config.measure.clip_quantile

    if config.measure.mode == "subspace":
        subspace_result = measure_subspace(
            model, tokenizer, harmful, harmless,  # type: ignore[arg-type]
            config.measure.top_k, clip_q,
        )
    elif config.measure.mode == "dbdi":
        dbdi_result = measure_dbdi(
            model, tokenizer, harmful, harmless, clip_q,  # type: ignore[arg-type]
        )
        # Map DBDI result to direction_result based on dbdi_target
        if config.cut.dbdi_target in ("red", "both"):
            direction_result = DirectionResult(
                direction=dbdi_result.red,
                layer_index=dbdi_result.red_layer_index,
                cosine_scores=dbdi_result.red_cosine_scores,
                d_model=dbdi_result.d_model,
                model_path=dbdi_result.model_path,
            )
            cosine_scores = dbdi_result.red_cosine_scores
        else:  # "hdd"
            direction_result = DirectionResult(
                direction=dbdi_result.hdd,
                layer_index=dbdi_result.hdd_layer_index,
                cosine_scores=dbdi_result.hdd_cosine_scores,
                d_model=dbdi_result.d_model,
                model_path=dbdi_result.model_path,
            )
            cosine_scores = dbdi_result.hdd_cosine_scores
    else:
        direction_result = measure(
            model, tokenizer, harmful, harmless, clip_q,  # type: ignore[arg-type]
        )
        cosine_scores = direction_result.cosine_scores

    # Determine target layers
    if config.cut.layers is not None:
        # Explicit list always wins
        target_layers = config.cut.layers
    elif config.cut.layer_strategy != "all":
        # Probe-guided selection
        if not cosine_scores:
            msg = "Probe-guided layer selection requires 'direction' mode"
            raise ValueError(msg)
        target_layers = select_target_layers(
            cosine_scores, config.cut.layer_strategy, config.cut.layer_top_k,
        )
    else:
        # Default: all layers
        target_layers = list(range(len(model.model.layers)))

    # Flatten weights for cut
    flat_weights: dict[str, object] = dict(tree_flatten(model.parameters()))

    # Apply optional sparsification to direction
    if direction_result is not None and config.cut.sparsity > 0.0:
        direction_result = DirectionResult(
            direction=sparsify_direction(
                direction_result.direction, config.cut.sparsity,
            ),
            layer_index=direction_result.layer_index,
            cosine_scores=direction_result.cosine_scores,
            d_model=direction_result.d_model,
            model_path=direction_result.model_path,
        )

    # False refusal orthogonalization: measure borderline direction and
    # orthogonalize the refusal direction against it before cutting.
    if (
        config.cut.false_refusal_ortho
        and config.borderline_path is not None
        and direction_result is not None
    ):
        from vauban.cut import _biprojected_direction

        borderline = resolve_prompts(config.borderline_path)
        false_refusal_result = measure(
            model, tokenizer, borderline, harmless, clip_q,  # type: ignore[arg-type]
        )
        ortho_dir = _biprojected_direction(
            direction_result.direction, false_refusal_result.direction,
        )
        direction_result = DirectionResult(
            direction=ortho_dir,
            layer_index=direction_result.layer_index,
            cosine_scores=direction_result.cosine_scores,
            d_model=direction_result.d_model,
            model_path=direction_result.model_path,
        )

    # Extract direction for surface use
    surface_direction = None
    surface_layer = 0
    if direction_result is not None:
        surface_direction = direction_result.direction
        surface_layer = direction_result.layer_index
    elif subspace_result is not None:
        best = subspace_result.best_direction()
        surface_direction = best.direction
        surface_layer = best.layer_index

    # "Before" surface map (on original model, before cut)
    surface_before: SurfaceResult | None = None
    surface_prompts: list[SurfacePrompt] | None = None
    if config.surface is not None and surface_direction is not None:
        surface_prompts = load_surface_prompts(
            default_surface_path()
            if config.surface.prompts_path == "default"
            else config.surface.prompts_path,
        )
        surface_before = map_surface(
            model,
            tokenizer,  # type: ignore[arg-type]
            surface_prompts,
            surface_direction,
            surface_layer,
            generate=config.surface.generate,
            max_tokens=config.surface.max_tokens,
        )

    # Apply the appropriate cut
    lw = config.cut.layer_weights
    if config.measure.mode == "subspace":
        assert subspace_result is not None
        modified_weights = cut_subspace(
            flat_weights,  # type: ignore[arg-type]
            subspace_result.basis,
            target_layers,
            config.cut.alpha,
            config.cut.norm_preserve,
            lw,
        )
    elif config.cut.biprojected:
        assert direction_result is not None
        harmless_acts = measure(model, tokenizer, harmless, harmful, clip_q)  # type: ignore[arg-type]
        modified_weights = cut_biprojected(
            flat_weights,  # type: ignore[arg-type]
            direction_result.direction,
            harmless_acts.direction,
            target_layers,
            config.cut.alpha,
            config.cut.norm_preserve,
            lw,
        )
    else:
        assert direction_result is not None
        modified_weights = cut(
            flat_weights,  # type: ignore[arg-type]
            direction_result.direction,
            target_layers,
            config.cut.alpha,
            config.cut.norm_preserve,
            lw,
        )

    # For DBDI "both" mode: apply second cut with HDD direction
    if config.measure.mode == "dbdi" and config.cut.dbdi_target == "both":
        hdd_direction = DirectionResult(
            direction=dbdi_result.hdd,
            layer_index=dbdi_result.hdd_layer_index,
            cosine_scores=dbdi_result.hdd_cosine_scores,
            d_model=dbdi_result.d_model,
            model_path=dbdi_result.model_path,
        )
        modified_weights = cut(
            modified_weights,
            hdd_direction.direction,
            target_layers,
            config.cut.alpha,
            config.cut.norm_preserve,
            lw,
        )

    # Export as a complete loadable model directory
    export_model(config.model_path, modified_weights, config.output_dir)

    # Load modified model if needed for surface-after or eval
    needs_modified = (
        config.eval_prompts_path is not None or surface_before is not None
    )
    modified_model = None
    if needs_modified:
        modified_model, _ = mlx_lm.load(config.model_path)  # type: ignore[invalid-assignment]
        if is_quantized(modified_model):
            dequantize_model(modified_model)
        modified_model.load_weights(list(modified_weights.items()))

    # "After" surface map + comparison
    if surface_before is not None:
        assert modified_model is not None
        assert surface_prompts is not None
        assert surface_direction is not None
        assert config.surface is not None
        surface_after = map_surface(
            modified_model,
            tokenizer,  # type: ignore[arg-type]
            surface_prompts,
            surface_direction,
            surface_layer,
            generate=config.surface.generate,
            max_tokens=config.surface.max_tokens,
        )
        comparison = compare_surfaces(surface_before, surface_after)
        report_path = config.output_dir / "surface_report.json"
        report_path.write_text(
            json.dumps(_surface_comparison_to_dict(comparison), indent=2),
        )

    # Evaluate if eval prompts are provided
    if config.eval_prompts_path is not None and modified_model is not None:
        eval_prompts = load_prompts(config.eval_prompts_path)

        result = evaluate(
            model, modified_model, tokenizer, eval_prompts,  # type: ignore[arg-type]
        )

        report_path = config.output_dir / "eval_report.json"
        report = {
            "refusal_rate_original": result.refusal_rate_original,
            "refusal_rate_modified": result.refusal_rate_modified,
            "perplexity_original": result.perplexity_original,
            "perplexity_modified": result.perplexity_modified,
            "kl_divergence": result.kl_divergence,
            "num_prompts": result.num_prompts,
        }
        report_path.write_text(json.dumps(report, indent=2))


def _surface_comparison_to_dict(
    comparison: SurfaceComparison,
) -> dict[str, object]:
    """Serialize a SurfaceComparison to a JSON-compatible dict."""
    def _group_delta_to_dict(d: SurfaceGroupDelta) -> dict[str, object]:
        return {
            "name": d.name,
            "count": d.count,
            "refusal_rate_before": d.refusal_rate_before,
            "refusal_rate_after": d.refusal_rate_after,
            "refusal_rate_delta": d.refusal_rate_delta,
            "mean_projection_before": d.mean_projection_before,
            "mean_projection_after": d.mean_projection_after,
            "mean_projection_delta": d.mean_projection_delta,
        }

    return {
        "summary": {
            "refusal_rate_before": comparison.refusal_rate_before,
            "refusal_rate_after": comparison.refusal_rate_after,
            "refusal_rate_delta": comparison.refusal_rate_delta,
            "threshold_before": comparison.threshold_before,
            "threshold_after": comparison.threshold_after,
            "threshold_delta": comparison.threshold_delta,
            "total_scanned": comparison.before.total_scanned,
        },
        "category_deltas": [
            _group_delta_to_dict(d) for d in comparison.category_deltas
        ],
        "label_deltas": [
            _group_delta_to_dict(d) for d in comparison.label_deltas
        ],
    }
