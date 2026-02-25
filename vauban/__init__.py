"""Vauban - MLX-native abliteration toolkit for Apple Silicon."""

from pathlib import Path

from vauban._serializers import (
    _detect_to_dict,
    _optimize_to_dict,
    _sic_to_dict,
    _softprompt_to_dict,
    _surface_comparison_to_dict,
)
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
from vauban.detect import detect
from vauban.evaluate import evaluate
from vauban.export import export_model
from vauban.measure import (
    default_eval_path,
    default_prompt_paths,
    detect_layer_types,
    find_instruction_boundary,
    load_prompts,
    measure,
    measure_dbdi,
    measure_subspace,
    select_target_layers,
    silhouette_scores,
)
from vauban.optimize import optimize
from vauban.probe import multi_probe, probe, steer
from vauban.sic import calibrate_threshold, sic_single
from vauban.sic import sic as sic_sanitize
from vauban.softprompt import _project_to_tokens, softprompt_attack
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
    DetectConfig,
    DetectResult,
    DirectionResult,
    EvalConfig,
    EvalResult,
    MeasureConfig,
    OptimizeConfig,
    OptimizeResult,
    PipelineConfig,
    ProbeResult,
    SICConfig,
    SICPromptResult,
    SICResult,
    SoftPromptConfig,
    SoftPromptResult,
    SteerResult,
    SubspaceResult,
    SurfaceComparison,
    SurfaceConfig,
    SurfaceGroup,
    SurfaceGroupDelta,
    SurfacePoint,
    SurfacePrompt,
    SurfaceResult,
    TransferEvalResult,
    TrialResult,
)

__version__ = "0.2.1"

__all__ = [
    "CutConfig",
    "DBDIResult",
    "DatasetRef",
    "DetectConfig",
    "DetectResult",
    "DirectionResult",
    "EvalConfig",
    "EvalResult",
    "MeasureConfig",
    "OptimizeConfig",
    "OptimizeResult",
    "PipelineConfig",
    "ProbeResult",
    "SICConfig",
    "SICPromptResult",
    "SICResult",
    "SoftPromptConfig",
    "SoftPromptResult",
    "SteerResult",
    "SubspaceResult",
    "SurfaceComparison",
    "SurfaceConfig",
    "SurfaceGroup",
    "SurfaceGroupDelta",
    "SurfacePoint",
    "SurfacePrompt",
    "SurfaceResult",
    "TransferEvalResult",
    "TrialResult",
    "aggregate",
    "calibrate_threshold",
    "compare_surfaces",
    "cut",
    "cut_biprojected",
    "cut_false_refusal_ortho",
    "cut_subspace",
    "default_eval_path",
    "default_prompt_paths",
    "default_surface_path",
    "dequantize_model",
    "detect",
    "detect_layer_types",
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
    "optimize",
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
    "sic_sanitize",
    "sic_single",
    "silhouette_scores",
    "softprompt_attack",
    "sparsify_direction",
    "steer",
    "subspace_overlap",
    "target_weight_keys",
    "validate",
]


def validate(config_path: str | Path) -> list[str]:
    """Validate a TOML config without loading any model.

    Checks:
    - TOML parses and all fields are well-typed
    - Referenced file paths exist on disk
    - Refusal phrases file is non-empty
    - Pipeline mode is unambiguous

    Returns a list of warnings (empty = clean).  Raises on hard errors.
    """
    import sys

    config = load_config(config_path)
    warnings: list[str] = []

    # Check data paths exist (skip HF dataset refs and "default")
    for name, p in [
        ("harmful", config.harmful_path),
        ("harmless", config.harmless_path),
    ]:
        if isinstance(p, Path) and not p.exists():
            warnings.append(f"[data].{name} file not found: {p}")

    if (
        isinstance(config.borderline_path, Path)
        and not config.borderline_path.exists()
    ):
        warnings.append(
            f"[data].borderline file not found: {config.borderline_path}"
        )

    # Eval prompts path
    if (
        config.eval.prompts_path is not None
        and not config.eval.prompts_path.exists()
    ):
        warnings.append(
            f"[eval].prompts file not found: {config.eval.prompts_path}"
        )

    # Refusal phrases file
    if config.eval.refusal_phrases_path is not None:
        rp = config.eval.refusal_phrases_path
        if not rp.exists():
            warnings.append(f"[eval].refusal_phrases file not found: {rp}")
        else:
            phrases = _load_refusal_phrases(rp)
            if len(phrases) < 2:
                warnings.append(
                    f"[eval].refusal_phrases has only {len(phrases)}"
                    " phrase(s) — consider adding more"
                )

    # Surface prompts path
    if config.surface is not None:
        sp = config.surface.prompts_path
        if isinstance(sp, Path) and not sp.exists():
            warnings.append(f"[surface].prompts file not found: {sp}")

    # Early-return mode conflicts
    early_modes = []
    if config.sic is not None:
        early_modes.append("[sic]")
    if config.optimize is not None:
        early_modes.append("[optimize]")
    if config.softprompt is not None:
        early_modes.append("[softprompt]")
    if len(early_modes) > 1:
        warnings.append(
            f"Multiple early-return modes active: {', '.join(early_modes)}"
            " — only the first will run (precedence: sic > optimize"
            " > softprompt)"
        )

    # Surface + eval without eval prompts is fine but worth noting
    if config.surface is not None and not early_modes:
        pass  # surface runs in normal pipeline

    # Print summary
    mode = "measure → cut → export"
    if config.sic is not None:
        mode = "SIC sanitization"
    elif config.optimize is not None:
        mode = "Optuna optimization"
    elif config.softprompt is not None:
        mode = "soft prompt attack"
    extras = []
    if config.detect is not None:
        extras.append("detect")
    if config.surface is not None and not early_modes:
        extras.append("surface")
    if config.eval.prompts_path is not None and not early_modes:
        extras.append("eval")
    mode_str = mode
    if extras:
        mode_str += f" + {', '.join(extras)}"

    print(f"Config:   {config_path}", file=sys.stderr)
    print(f"Model:    {config.model_path}", file=sys.stderr)
    print(f"Pipeline: {mode_str}", file=sys.stderr)
    print(f"Output:   {config.output_dir}", file=sys.stderr)

    if warnings:
        print(f"\nWarnings ({len(warnings)}):", file=sys.stderr)
        for w in warnings:
            print(f"  - {w}", file=sys.stderr)
    else:
        print("\nNo issues found.", file=sys.stderr)

    return warnings


def _load_refusal_phrases(path: Path) -> list[str]:
    """Load refusal phrases from a text file (one per line)."""
    phrases: list[str] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            phrases.append(stripped)
    if not phrases:
        msg = f"Refusal phrases file is empty: {path}"
        raise ValueError(msg)
    return phrases


def _log(msg: str, *, verbose: bool = True, elapsed: float | None = None) -> None:
    """Print a one-line status message to stderr."""
    if not verbose:
        return
    import sys
    prefix = f"[vauban {elapsed:+.1f}s]" if elapsed is not None else "[vauban]"
    print(f"{prefix} {msg}", file=sys.stderr, flush=True)


def run(config_path: str | Path) -> None:
    """Run the full measure -> cut -> evaluate pipeline from a TOML config."""
    import json
    import time

    import mlx.core as mx
    import mlx_lm
    from mlx.utils import tree_flatten

    config = load_config(config_path)
    v = config.verbose
    t0 = time.monotonic()

    _log(
        f"Loading model {config.model_path}",
        verbose=v, elapsed=time.monotonic() - t0,
    )
    model, tokenizer = mlx_lm.load(config.model_path)  # type: ignore[invalid-assignment]

    # Auto-dequantize if model has quantized weights
    if is_quantized(model):
        _log(
            "Dequantizing model weights",
            verbose=v, elapsed=time.monotonic() - t0,
        )
        dequantize_model(model)

    _log(
        "Loading prompts",
        verbose=v, elapsed=time.monotonic() - t0,
    )
    harmful = resolve_prompts(config.harmful_path)
    harmless = resolve_prompts(config.harmless_path)

    # Load custom refusal phrases if configured
    refusal_phrases: list[str] | None = None
    if config.eval.refusal_phrases_path is not None:
        refusal_phrases = _load_refusal_phrases(config.eval.refusal_phrases_path)

    # Defense detection (runs before measure/cut)
    if config.detect is not None:
        _log(
            "Running defense detection",
            verbose=v, elapsed=time.monotonic() - t0,
        )
        detect_result = detect(model, tokenizer, harmful, harmless, config.detect)  # type: ignore[arg-type]
        report_path = config.output_dir / "detect_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(_detect_to_dict(detect_result), indent=2))

    # Branch on measure mode
    direction_result = None
    subspace_result = None
    cosine_scores: list[float] = []

    clip_q = config.measure.clip_quantile

    _log(
        f"Measuring refusal direction (mode={config.measure.mode})",
        verbose=v, elapsed=time.monotonic() - t0,
    )
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
            model, tokenizer, harmful, harmless, clip_q,  # type: ignore[arg-type]
        )
        cosine_scores = direction_result.cosine_scores

    # SIC sanitization: standalone early-return mode
    if config.sic is not None:
        _log(
            f"Running SIC sanitization (mode={config.sic.mode})",
            verbose=v, elapsed=time.monotonic() - t0,
        )
        direction_vec = (
            direction_result.direction if direction_result is not None
            else None
        )
        layer_idx = (
            direction_result.layer_index if direction_result is not None
            else 0
        )
        if config.eval.prompts_path is not None:
            sic_prompts: list[str] = load_prompts(config.eval.prompts_path)
        else:
            sic_prompts = harmful[:config.eval.num_prompts]

        cal_prompts: list[str] | None = None
        if config.sic.calibrate:
            cal_prompts = (
                harmless if config.sic.calibrate_prompts == "harmless"
                else harmful
            )

        sic_result = sic_sanitize(
            model, tokenizer, sic_prompts, config.sic,  # type: ignore[arg-type]
            direction_vec, layer_idx, cal_prompts,
        )
        report_path = config.output_dir / "sic_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(_sic_to_dict(sic_result), indent=2),
        )
        _log(
            f"Done — SIC report written to {report_path}",
            verbose=v, elapsed=time.monotonic() - t0,
        )
        return

    # Optimization mode: search over cut parameters, write report, return early
    if config.optimize is not None and direction_result is not None:
        _log(
            f"Running optimization ({config.optimize.n_trials} trials)",
            verbose=v, elapsed=time.monotonic() - t0,
        )
        eval_prompts_opt: list[str] = []
        if config.eval.prompts_path is not None:
            eval_prompts_opt = load_prompts(config.eval.prompts_path)
        else:
            eval_prompts_opt = harmful[:config.eval.num_prompts]

        opt_result = optimize(
            model, tokenizer, direction_result,  # type: ignore[arg-type]
            eval_prompts_opt, config.optimize,
        )

        report_path = config.output_dir / "optimize_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(_optimize_to_dict(opt_result), indent=2),
        )
        _log(
            f"Done — optimize report written to {report_path}",
            verbose=v, elapsed=time.monotonic() - t0,
        )
        return

    # Soft prompt attack: optimize a learnable prefix, write report, return
    if config.softprompt is not None:
        _log(
            f"Running soft prompt attack (mode={config.softprompt.mode})",
            verbose=v, elapsed=time.monotonic() - t0,
        )
        direction_vec = (
            direction_result.direction if direction_result is not None else None
        )
        if config.eval.prompts_path is not None:
            sp_prompts: list[str] = load_prompts(config.eval.prompts_path)
        else:
            sp_prompts = harmful[:config.eval.num_prompts]

        # Load reference model for KL collision loss if configured
        ref_model = None
        if config.softprompt.ref_model is not None:
            ref_model, _ = mlx_lm.load(config.softprompt.ref_model)  # type: ignore[assignment]
            if is_quantized(ref_model):
                dequantize_model(ref_model)

        sp_result = softprompt_attack(
            model, tokenizer, sp_prompts, config.softprompt, direction_vec,  # type: ignore[arg-type]
            ref_model,
        )

        # Transfer evaluation: test optimized prefix on other models
        if config.softprompt.transfer_models:
            from vauban.softprompt import _evaluate_attack

            # Get discrete token IDs (project if continuous mode)
            if sp_result.token_ids is not None:
                transfer_token_ids = sp_result.token_ids
            elif sp_result.embeddings is not None:
                transfer_token_ids = _project_to_tokens(
                    sp_result.embeddings,
                    model.model.embed_tokens.weight,
                )
            else:
                transfer_token_ids = []

            transfer_results: list[TransferEvalResult] = []
            for transfer_model_id in config.softprompt.transfer_models:
                t_model, _ = mlx_lm.load(transfer_model_id)  # type: ignore[assignment]
                if is_quantized(t_model):
                    dequantize_model(t_model)
                # Re-embed the tokens in the transfer model's space
                t_token_array = mx.array(transfer_token_ids)[None, :]
                t_embeds = t_model.model.embed_tokens(t_token_array)
                mx.eval(t_embeds)
                t_success, t_responses = _evaluate_attack(
                    t_model, tokenizer, sp_prompts,  # type: ignore[arg-type]
                    t_embeds, config.softprompt,
                )
                transfer_results.append(TransferEvalResult(
                    model_id=transfer_model_id,
                    success_rate=t_success,
                    eval_responses=t_responses,
                ))

            # Reconstruct result with transfer results
            sp_result = SoftPromptResult(
                mode=sp_result.mode,
                success_rate=sp_result.success_rate,
                final_loss=sp_result.final_loss,
                loss_history=sp_result.loss_history,
                n_steps=sp_result.n_steps,
                n_tokens=sp_result.n_tokens,
                embeddings=sp_result.embeddings,
                token_ids=sp_result.token_ids,
                token_text=sp_result.token_text,
                eval_responses=sp_result.eval_responses,
                accessibility_score=sp_result.accessibility_score,
                per_prompt_losses=sp_result.per_prompt_losses,
                early_stopped=sp_result.early_stopped,
                transfer_results=transfer_results,
            )

        report_path = config.output_dir / "softprompt_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(_softprompt_to_dict(sp_result), indent=2),
        )
        _log(
            f"Done — softprompt report written to {report_path}",
            verbose=v, elapsed=time.monotonic() - t0,
        )
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
            layer_types=direction_result.layer_types,
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
        _log(
            "Mapping refusal surface (before cut)",
            verbose=v, elapsed=time.monotonic() - t0,
        )
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
            refusal_phrases=refusal_phrases,
            progress=config.surface.progress,
        )

    # Apply the appropriate cut
    _log(
        f"Cutting {len(target_layers)} layers (alpha={config.cut.alpha})",
        verbose=v, elapsed=time.monotonic() - t0,
    )
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
            layer_types=dbdi_result.layer_types,
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
    _log(
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
        modified_model, _ = mlx_lm.load(config.model_path)  # type: ignore[invalid-assignment]
        if is_quantized(modified_model):
            dequantize_model(modified_model)
        modified_model.load_weights(list(modified_weights.items()))

    # "After" surface map + comparison
    if surface_before is not None:
        _log(
            "Mapping refusal surface (after cut)",
            verbose=v, elapsed=time.monotonic() - t0,
        )
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
            refusal_phrases=refusal_phrases,
            progress=config.surface.progress,
        )
        comparison = compare_surfaces(surface_before, surface_after)
        report_path = config.output_dir / "surface_report.json"
        report_path.write_text(
            json.dumps(_surface_comparison_to_dict(comparison), indent=2),
        )

    # Evaluate if eval prompts are provided
    if config.eval.prompts_path is not None and modified_model is not None:
        _log(
            "Evaluating modified model",
            verbose=v, elapsed=time.monotonic() - t0,
        )
        eval_prompts = load_prompts(config.eval.prompts_path)

        result = evaluate(
            model, modified_model, tokenizer, eval_prompts,  # type: ignore[arg-type]
            refusal_phrases=refusal_phrases,
            max_tokens=config.eval.max_tokens,
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

    _log(
        f"Done — output written to {config.output_dir}",
        verbose=v, elapsed=time.monotonic() - t0,
    )
