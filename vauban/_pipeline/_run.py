"""Main pipeline entry point: run()."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

from vauban._pipeline._context import EarlyModeContext, log, write_experiment_log
from vauban._pipeline._helpers import (
    load_refusal_phrases,
    surface_gate_failures,
    write_measure_reports,
)
from vauban._pipeline._modes import dispatch_early_mode

if TYPE_CHECKING:
    from vauban._array import Array


def run(config_path: str | Path) -> None:
    """Run the full measure -> cut -> evaluate pipeline from a TOML config."""
    from vauban._model_io import load_model
    from vauban._serializers import (
        _detect_to_dict,
        _direction_transfer_to_dict,
        _surface_comparison_to_dict,
    )
    from vauban.config import load_config
    from vauban.cut import (
        cut,
        cut_biprojected,
        cut_subspace,
        sparsify_direction,
    )
    from vauban.dataset import resolve_prompts
    from vauban.dequantize import dequantize_model, is_quantized
    from vauban.detect import detect
    from vauban.evaluate import evaluate
    from vauban.export import export_model
    from vauban.measure import (
        load_prompts,
        measure,
        measure_dbdi,
        measure_diff,
        measure_subspace,
        measure_subspace_bank,
        select_target_layers,
    )
    from vauban.surface import (
        compare_surfaces,
        default_multilingual_surface_path,
        default_surface_path,
        load_surface_prompts,
        map_surface,
    )
    from vauban.types import (
        DirectionResult,
        SurfacePrompt,
        SurfaceResult,
    )

    config = load_config(config_path)
    v = config.verbose
    t0 = time.monotonic()

    log(
        f"Loading model {config.model_path}",
        verbose=v, elapsed=time.monotonic() - t0,
    )
    model, tokenizer = load_model(config.model_path)

    # Auto-dequantize if model has quantized weights
    if is_quantized(model):
        log(
            "Dequantizing model weights",
            verbose=v, elapsed=time.monotonic() - t0,
        )
        dequantize_model(model)

    early_mode_context = EarlyModeContext(
        config_path=config_path,
        config=config,
        model=model,
        tokenizer=tokenizer,
        t0=t0,
    )
    if dispatch_early_mode("before_prompts", early_mode_context):
        return

    log(
        "Loading prompts",
        verbose=v, elapsed=time.monotonic() - t0,
    )
    harmful = resolve_prompts(config.harmful_path)
    harmless = resolve_prompts(config.harmless_path)
    early_mode_context.harmful = harmful
    early_mode_context.harmless = harmless

    # Load custom refusal phrases if configured
    refusal_phrases: list[str] | None = None
    if config.eval.refusal_phrases_path is not None:
        refusal_phrases = load_refusal_phrases(config.eval.refusal_phrases_path)

    # Defense detection (runs before measure/cut)
    if config.detect is not None:
        log(
            "Running defense detection",
            verbose=v, elapsed=time.monotonic() - t0,
        )
        detect_result = detect(model, tokenizer, harmful, harmless, config.detect)
        report_path = config.output_dir / "detect_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(_detect_to_dict(detect_result), indent=2))

    # Branch on measure mode
    direction_result = None
    subspace_result = None
    dbdi_result = None
    diff_result = None
    cosine_scores: list[float] = []
    measure_reports: list[str] = []

    clip_q = config.measure.clip_quantile

    log(
        f"Measuring refusal direction (mode={config.measure.mode})",
        verbose=v, elapsed=time.monotonic() - t0,
    )
    if config.measure.mode == "subspace":
        subspace_result = measure_subspace(
            model, tokenizer, harmful, harmless,
            config.measure.top_k, clip_q,
        )
        # Subspace bank: measure additional named subspaces (Steer2Adapt)
        if config.measure.bank:
            base_dir = Path(config_path).resolve().parent
            bank_entries: list[tuple[str, list[str], list[str]]] = []
            for entry in config.measure.bank:
                bank_harmful = (
                    harmful if entry.harmful == "default"
                    else load_prompts(base_dir / entry.harmful)
                )
                bank_harmless = (
                    harmless if entry.harmless == "default"
                    else load_prompts(base_dir / entry.harmless)
                )
                bank_entries.append((entry.name, bank_harmful, bank_harmless))
            log(
                f"Measuring subspace bank ({len(bank_entries)} entries)",
                verbose=v, elapsed=time.monotonic() - t0,
            )
            bank_results = measure_subspace_bank(
                model,
                tokenizer,
                bank_entries, config.measure.top_k, clip_q,
            )
            # Save bank as subspace_bank.safetensors
            import mlx.core as mx

            bank_path = config.output_dir / "subspace_bank.safetensors"
            bank_path.parent.mkdir(parents=True, exist_ok=True)
            bank_arrays: dict[str, mx.array] = {}
            for name, bank_result in bank_results.items():
                bank_arrays[name] = bank_result.basis
            mx.save_safetensors(str(bank_path), bank_arrays)
            log(
                f"Subspace bank written to {bank_path}"
                f" ({len(bank_results)} subspaces)",
                verbose=v, elapsed=time.monotonic() - t0,
            )
    elif config.measure.mode == "diff":
        if config.measure.diff_model is None:
            msg = "measure.diff_model is required when mode='diff'"
            raise ValueError(msg)
        log(
            f"Loading base model for diff: {config.measure.diff_model}",
            verbose=v, elapsed=time.monotonic() - t0,
        )
        base_model, _ = load_model(config.measure.diff_model)
        if is_quantized(base_model):
            log(
                "Dequantizing base model weights",
                verbose=v, elapsed=time.monotonic() - t0,
            )
            dequantize_model(base_model)
        diff_result = measure_diff(
            base_model,
            model,
            top_k=config.measure.top_k,
            source_model_id=config.measure.diff_model,
            target_model_id=config.model_path,
        )
        measure_reports = write_measure_reports(
            config,
            direction_result=None,
            subspace_result=None,
            dbdi_result=None,
            diff_result=diff_result,
        )
        log(
            f"Diff report written to {config.output_dir / 'diff_report.json'}",
            verbose=v, elapsed=time.monotonic() - t0,
        )
        direction_result = diff_result.best_direction()
        cosine_scores = []
    elif config.measure.mode == "dbdi":
        dbdi_result = measure_dbdi(
            model, tokenizer, harmful, harmless, clip_q,
        )
        # Map DBDI result to direction_result based on dbdi_target
        if config.cut.dbdi_target in ("red", "both"):
            direction_result = DirectionResult(
                direction=dbdi_result.red,
                layer_index=dbdi_result.red_layer_index,
                cosine_scores=dbdi_result.red_cosine_scores,
                d_model=dbdi_result.d_model,
                model_path=dbdi_result.model_path,
                layer_types=dbdi_result.layer_types,
            )
            cosine_scores = dbdi_result.red_cosine_scores
        else:  # "hdd"
            direction_result = DirectionResult(
                direction=dbdi_result.hdd,
                layer_index=dbdi_result.hdd_layer_index,
                cosine_scores=dbdi_result.hdd_cosine_scores,
                d_model=dbdi_result.d_model,
                model_path=dbdi_result.model_path,
                layer_types=dbdi_result.layer_types,
            )
            cosine_scores = dbdi_result.hdd_cosine_scores
    else:
        direction_result = measure(
            model, tokenizer, harmful, harmless, clip_q,
        )
        cosine_scores = direction_result.cosine_scores

    # Direction transfer testing
    if (
        config.measure.transfer_models
        and direction_result is not None
    ):
        from vauban.transfer import check_direction_transfer
        from vauban.types import DirectionTransferResult

        transfer_results_list: list[DirectionTransferResult] = []
        for transfer_model_id in config.measure.transfer_models:
            log(
                f"Testing direction transfer on {transfer_model_id}",
                verbose=v, elapsed=time.monotonic() - t0,
            )
            t_model, _ = load_model(transfer_model_id)
            if is_quantized(t_model):
                dequantize_model(t_model)
            transfer_result = check_direction_transfer(
                t_model,
                tokenizer,
                direction_result.direction,
                harmful,
                harmless,
                transfer_model_id,
                clip_q,
            )
            transfer_results_list.append(transfer_result)

        # Write transfer report
        transfer_report_path = config.output_dir / "transfer_report.json"
        transfer_report_path.parent.mkdir(parents=True, exist_ok=True)
        transfer_report_path.write_text(
            json.dumps(
                [_direction_transfer_to_dict(r) for r in transfer_results_list],
                indent=2,
            ),
        )
        log(
            f"Transfer report written to {transfer_report_path}",
            verbose=v, elapsed=time.monotonic() - t0,
        )
        transfer_reports = ["transfer_report.json"]
    else:
        transfer_reports = []

    if config.measure.measure_only:
        if not measure_reports:
            measure_reports = write_measure_reports(
                config,
                direction_result=direction_result,
                subspace_result=subspace_result,
                dbdi_result=dbdi_result,
                diff_result=diff_result,
            )
        log(
            f"Measure-only mode complete — output written to {config.output_dir}",
            verbose=v, elapsed=time.monotonic() - t0,
        )
        write_experiment_log(
            config_path,
            config,
            "measure",
            measure_reports + transfer_reports,
            {},
            time.monotonic() - t0,
        )
        return

    early_mode_context.direction_result = direction_result
    if dispatch_early_mode("after_measure", early_mode_context):
        return

    # Resolve layer types from whichever result is available
    layer_types: list[str] | None = None
    if direction_result is not None:
        layer_types = direction_result.layer_types
    elif subspace_result is not None:
        layer_types = subspace_result.layer_types

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
            cosine_scores,
            config.cut.layer_strategy,
            config.cut.layer_top_k,
            layer_types=layer_types,
            type_filter=config.cut.layer_type_filter,
        )
    else:
        # Default: all layers
        from vauban._forward import get_transformer as _get_transformer

        target_layers = list(range(len(_get_transformer(model).layers)))

    # Flatten weights for cut
    from vauban._ops import tree_flatten

    flat_weights: dict[str, object] = dict(tree_flatten(model.parameters()))  # type: ignore[attr-defined]  # CausalLM is nn.Module at runtime

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
            layer_types=direction_result.layer_types,
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
            model, tokenizer, borderline, harmless, clip_q,
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
            layer_types=direction_result.layer_types,
        )

    # Extract direction for surface use
    surface_direction: Array | None = None
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
        log(
            "Mapping refusal surface (before cut)",
            verbose=v, elapsed=time.monotonic() - t0,
        )
        if config.surface.prompts_path == "default":
            surface_prompts_path = default_surface_path()
        elif config.surface.prompts_path == "default_multilingual":
            surface_prompts_path = default_multilingual_surface_path()
        else:
            surface_prompts_path = config.surface.prompts_path
        surface_prompts = load_surface_prompts(surface_prompts_path)
        surface_before = map_surface(
            model,
            tokenizer,
            surface_prompts,
            surface_direction,
            surface_layer,
            generate=config.surface.generate,
            max_tokens=config.surface.max_tokens,
            refusal_phrases=refusal_phrases,
            progress=config.surface.progress,
            refusal_mode=config.eval.refusal_mode,
        )

    # Apply the appropriate cut
    log(
        f"Cutting {len(target_layers)} layers (alpha={config.cut.alpha})",
        verbose=v, elapsed=time.monotonic() - t0,
    )
    lw = config.cut.layer_weights
    if config.measure.mode == "subspace":
        if subspace_result is None:
            msg = "subspace_result is required for subspace cut but was None"
            raise ValueError(msg)
        modified_weights = cut_subspace(
            flat_weights,  # type: ignore[arg-type]
            subspace_result.basis,
            target_layers,
            config.cut.alpha,
            config.cut.norm_preserve,
            lw,
        )
    elif config.cut.biprojected:
        if direction_result is None:
            msg = "direction_result is required for biprojected cut"
            raise ValueError(msg)
        harmless_acts = measure(model, tokenizer, harmless, harmful, clip_q)
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
        if direction_result is None:
            msg = "direction_result is required for cut but was None"
            raise ValueError(msg)
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
        if dbdi_result is None:
            msg = "dbdi_result is required for DBDI both-mode cut"
            raise ValueError(msg)
        modified_weights = cut(
            modified_weights,
            dbdi_result.hdd,
            target_layers,
            config.cut.alpha,
            config.cut.norm_preserve,
            lw,
        )

    # Export as a complete loadable model directory
    log(
        f"Exporting modified model to {config.output_dir}",
        verbose=v, elapsed=time.monotonic() - t0,
    )
    export_model(config.model_path, modified_weights, config.output_dir)

    # Load modified model if needed for surface-after or eval
    needs_modified = (
        config.eval.prompts_path is not None or surface_before is not None
    )
    modified_model = None
    if needs_modified:
        modified_model, _ = load_model(config.model_path)
        if is_quantized(modified_model):
            dequantize_model(modified_model)
        modified_model.load_weights(list(modified_weights.items()))  # type: ignore[attr-defined]  # CausalLM is nn.Module at runtime

    # "After" surface map + comparison
    if surface_before is not None:
        log(
            "Mapping refusal surface (after cut)",
            verbose=v, elapsed=time.monotonic() - t0,
        )
        if modified_model is None:
            msg = "modified_model is required for surface-after mapping"
            raise ValueError(msg)
        if surface_prompts is None:
            msg = "surface_prompts is required for surface-after mapping"
            raise ValueError(msg)
        if surface_direction is None:
            msg = "surface_direction is required for surface-after mapping"
            raise ValueError(msg)
        if config.surface is None:
            msg = "surface config is required for surface-after mapping"
            raise ValueError(msg)
        surface_after = map_surface(
            modified_model,
            tokenizer,
            surface_prompts,
            surface_direction,
            surface_layer,
            generate=config.surface.generate,
            max_tokens=config.surface.max_tokens,
            refusal_phrases=refusal_phrases,
            progress=config.surface.progress,
            refusal_mode=config.eval.refusal_mode,
        )
        comparison = compare_surfaces(surface_before, surface_after)
        report_path = config.output_dir / "surface_report.json"
        report_path.write_text(
            json.dumps(_surface_comparison_to_dict(comparison), indent=2),
        )
        gate_failures = surface_gate_failures(config.surface, comparison)
        if gate_failures:
            joined = "\n".join(f"- {failure}" for failure in gate_failures)
            msg = (
                "Surface quality gates failed:\n"
                f"{joined}\n"
                "Adjust [surface] gate thresholds or improve model behavior."
            )
            raise RuntimeError(msg)

    # Evaluate if eval prompts are provided
    eval_refusal_rate: float | None = None
    if config.eval.prompts_path is not None and modified_model is not None:
        log(
            "Evaluating modified model",
            verbose=v, elapsed=time.monotonic() - t0,
        )
        eval_prompts = load_prompts(config.eval.prompts_path)

        result = evaluate(
            model, modified_model, tokenizer, eval_prompts,
            refusal_phrases=refusal_phrases,
            max_tokens=config.eval.max_tokens,
            refusal_mode=config.eval.refusal_mode,
        )
        eval_refusal_rate = result.refusal_rate_modified

        report_path = config.output_dir / "eval_report.json"
        eval_report: dict[str, object] = {
            "refusal_rate_original": result.refusal_rate_original,
            "refusal_rate_modified": result.refusal_rate_modified,
            "perplexity_original": result.perplexity_original,
            "perplexity_modified": result.perplexity_modified,
            "kl_divergence": result.kl_divergence,
            "num_prompts": result.num_prompts,
        }
        report_path.write_text(json.dumps(eval_report, indent=2))

    # Collect reports and metrics for experiment log
    normal_reports: list[str] = []
    normal_metrics: dict[str, float] = {}
    if surface_before is not None:
        normal_reports.append("surface_report.json")
    if eval_refusal_rate is not None:
        normal_reports.append("eval_report.json")
        normal_metrics["refusal_rate_modified"] = eval_refusal_rate

    log(
        f"Done — output written to {config.output_dir}",
        verbose=v, elapsed=time.monotonic() - t0,
    )
    write_experiment_log(
        config_path, config, "default",
        normal_reports, normal_metrics,
        time.monotonic() - t0,
    )
