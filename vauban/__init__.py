"""Vauban - MLX-native abliteration toolkit for Apple Silicon."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from typing import cast as _cast

if TYPE_CHECKING:
    from vauban._array import Array

from vauban._serializers import (
    _cast_to_dict,
    _depth_direction_to_dict,
    _depth_to_dict,
    _detect_to_dict,
    _direction_transfer_to_dict,
    _optimize_to_dict,
    _probe_to_dict,
    _sic_to_dict,
    _softprompt_to_dict,
    _steer_to_dict,
    _surface_comparison_to_dict,
)
from vauban.cast import cast_generate
from vauban.config import load_config
from vauban.config._mode_registry import (
    EarlyModePhase,
    active_early_mode_for_phase,
)
from vauban.config._validation import validate_config
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
from vauban.depth import depth_direction, depth_generate, depth_profile
from vauban.dequantize import dequantize_model, is_quantized
from vauban.detect import detect
from vauban.evaluate import evaluate
from vauban.export import export_model
from vauban.geometry import (
    DirectionGeometryResult,
    DirectionPair,
    analyze_directions,
)
from vauban.measure import (
    default_eval_path,
    default_prompt_paths,
    detect_layer_types,
    find_instruction_boundary,
    load_prompts,
    measure,
    measure_dbdi,
    measure_diff,
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
    default_multilingual_surface_path,
    default_surface_path,
    find_threshold,
    load_surface_prompts,
    map_surface,
    scan,
)
from vauban.types import (
    AlphaTier,
    CastConfig,
    CastResult,
    CausalLM,
    CutConfig,
    DatasetRef,
    DBDIResult,
    DepthConfig,
    DepthDirectionResult,
    DepthResult,
    DetectConfig,
    DetectResult,
    DiffResult,
    DirectionResult,
    DirectionTransferResult,
    EvalConfig,
    EvalResult,
    MeasureConfig,
    OptimizeConfig,
    OptimizeResult,
    PipelineConfig,
    ProbeConfig,
    ProbeResult,
    SICConfig,
    SICPromptResult,
    SICResult,
    SoftPromptConfig,
    SoftPromptResult,
    SteerConfig,
    SteerResult,
    SubspaceResult,
    SurfaceComparison,
    SurfaceConfig,
    SurfaceGroup,
    SurfaceGroupDelta,
    SurfacePoint,
    SurfacePrompt,
    SurfaceResult,
    TokenDepth,
    Tokenizer,
    TransferEvalResult,
    TrialResult,
)

__version__ = "0.2.5"

__all__ = [
    "AlphaTier",
    "CastConfig",
    "CastResult",
    "CutConfig",
    "DBDIResult",
    "DatasetRef",
    "DepthConfig",
    "DepthDirectionResult",
    "DepthResult",
    "DetectConfig",
    "DetectResult",
    "DiffResult",
    "DirectionGeometryResult",
    "DirectionPair",
    "DirectionResult",
    "DirectionTransferResult",
    "EvalConfig",
    "EvalResult",
    "MeasureConfig",
    "OptimizeConfig",
    "OptimizeResult",
    "PipelineConfig",
    "ProbeConfig",
    "ProbeResult",
    "SICConfig",
    "SICPromptResult",
    "SICResult",
    "SoftPromptConfig",
    "SoftPromptResult",
    "SteerConfig",
    "SteerResult",
    "SubspaceResult",
    "SurfaceComparison",
    "SurfaceConfig",
    "SurfaceGroup",
    "SurfaceGroupDelta",
    "SurfacePoint",
    "SurfacePrompt",
    "SurfaceResult",
    "TokenDepth",
    "TransferEvalResult",
    "TrialResult",
    "aggregate",
    "analyze_directions",
    "calibrate_threshold",
    "cast_generate",
    "compare_surfaces",
    "cut",
    "cut_biprojected",
    "cut_false_refusal_ortho",
    "cut_subspace",
    "default_eval_path",
    "default_multilingual_surface_path",
    "default_prompt_paths",
    "default_surface_path",
    "depth_direction",
    "depth_generate",
    "depth_profile",
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
    "measure_diff",
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
    """Validate a TOML config without loading any model."""
    return validate_config(config_path)


def _surface_gate_failures(
    surface: SurfaceConfig,
    comparison: SurfaceComparison,
) -> list[str]:
    """Return human-readable failures for configured surface gate thresholds."""
    failures: list[str] = []

    max_worst_after = surface.max_worst_cell_refusal_after
    if (
        max_worst_after is not None
        and comparison.worst_cell_refusal_rate_after > max_worst_after
    ):
        failures.append(
            (
                "worst-cell refusal rate after cut"
                f" {comparison.worst_cell_refusal_rate_after:.4f}"
                f" exceeds max_worst_cell_refusal_after={max_worst_after:.4f}"
            ),
        )

    max_worst_delta = surface.max_worst_cell_refusal_delta
    if (
        max_worst_delta is not None
        and comparison.worst_cell_refusal_rate_delta > max_worst_delta
    ):
        failures.append(
            (
                "worst-cell refusal-rate delta"
                f" {comparison.worst_cell_refusal_rate_delta:.4f}"
                f" exceeds max_worst_cell_refusal_delta={max_worst_delta:.4f}"
            ),
        )

    min_coverage = surface.min_coverage_score
    if (
        min_coverage is not None
        and comparison.coverage_score_after < min_coverage
    ):
        failures.append(
            (
                "coverage score after cut"
                f" {comparison.coverage_score_after:.4f}"
                f" is below min_coverage_score={min_coverage:.4f}"
            ),
        )

    return failures


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


def _is_default_data(config: PipelineConfig) -> bool:
    """Check whether data paths are just bundled defaults (not user-provided)."""
    h_default, hl_default = default_prompt_paths()
    return config.harmful_path == h_default and config.harmless_path == hl_default


def _log(msg: str, *, verbose: bool = True, elapsed: float | None = None) -> None:
    """Print a one-line status message to stderr."""
    if not verbose:
        return
    import sys
    prefix = f"[vauban {elapsed:+.1f}s]" if elapsed is not None else "[vauban]"
    print(f"{prefix} {msg}", file=sys.stderr, flush=True)


def _write_experiment_log(
    config_path: str | Path,
    config: PipelineConfig,
    mode: str,
    reports: list[str],
    metrics: dict[str, float],
    elapsed: float,
) -> None:
    """Append an experiment entry to output_dir/experiment_log.jsonl.

    Best-effort: never crashes the pipeline on I/O errors.
    """
    import datetime
    import json as _json

    try:
        entry = {
            "timestamp": datetime.datetime.now(
                tz=datetime.UTC,
            ).isoformat(timespec="seconds"),
            "config_path": str(Path(config_path).resolve()),
            "model_path": config.model_path,
            "pipeline_mode": mode,
            "output_dir": str(config.output_dir),
            "reports": reports,
            "metrics": metrics,
            "elapsed_seconds": round(elapsed, 2),
        }
        log_path = config.output_dir / "experiment_log.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as f:
            f.write(_json.dumps(entry) + "\n")
    except Exception:
        pass


def _write_measure_reports(
    config: PipelineConfig,
    direction_result: DirectionResult | None,
    subspace_result: SubspaceResult | None,
    dbdi_result: DBDIResult | None,
    diff_result: DiffResult | None,
) -> list[str]:
    """Write the JSON reports produced directly by the measure stage."""
    reports: list[str] = []

    if diff_result is not None:
        report_path = config.output_dir / "diff_report.json"
        report = {
            "source_model": diff_result.source_model,
            "target_model": diff_result.target_model,
            "best_layer": diff_result.best_layer,
            "d_model": diff_result.d_model,
            "best_layer_singular_values": diff_result.singular_values,
            "explained_variance_per_layer": diff_result.explained_variance,
            "per_layer_singular_values": diff_result.per_layer_singular_values,
        }
    elif subspace_result is not None:
        report_path = config.output_dir / "subspace_report.json"
        report = {
            "layer_index": subspace_result.layer_index,
            "d_model": subspace_result.d_model,
            "model_path": subspace_result.model_path,
            "singular_values": subspace_result.singular_values,
            "explained_variance": subspace_result.explained_variance,
            "layer_types": subspace_result.layer_types,
        }
    elif dbdi_result is not None:
        report_path = config.output_dir / "dbdi_report.json"
        report = {
            "hdd_layer_index": dbdi_result.hdd_layer_index,
            "red_layer_index": dbdi_result.red_layer_index,
            "hdd_cosine_scores": dbdi_result.hdd_cosine_scores,
            "red_cosine_scores": dbdi_result.red_cosine_scores,
            "d_model": dbdi_result.d_model,
            "model_path": dbdi_result.model_path,
            "layer_types": dbdi_result.layer_types,
        }
    elif direction_result is not None:
        report_path = config.output_dir / "direction_report.json"
        report = direction_result.to_dict()
    else:
        return reports

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    reports.append(report_path.name)
    return reports


@dataclass(slots=True)
class _EarlyModeContext:
    """Shared runtime context passed to early-mode handlers."""

    config_path: str | Path
    config: PipelineConfig
    model: object
    tokenizer: object
    t0: float
    harmful: list[str] | None = None
    harmless: list[str] | None = None
    direction_result: DirectionResult | None = None


def _run_depth_mode(context: _EarlyModeContext) -> None:
    """Run [depth] early-return mode and write its report."""
    import time

    config = context.config
    assert config.depth is not None
    v = config.verbose
    model = _cast("CausalLM", context.model)
    tokenizer = _cast("Tokenizer", context.tokenizer)

    _log(
        "Running depth analysis",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    depth_results: list[DepthResult] = []
    for prompt in config.depth.prompts:
        if config.depth.max_tokens > 0:
            dtr = depth_generate(
                model,
                tokenizer,
                prompt,
                config.depth,
            )
        else:
            dtr = depth_profile(
                model,
                tokenizer,
                prompt,
                config.depth,
            )
        depth_results.append(dtr)

    report: dict[str, object] = {
        "dtr_results": [_depth_to_dict(r) for r in depth_results],
    }

    if config.depth.extract_direction and len(depth_results) >= 2:
        _log(
            "Extracting depth direction",
            verbose=v,
            elapsed=time.monotonic() - context.t0,
        )
        dir_prompts = config.depth.direction_prompts
        if dir_prompts is not None:
            dir_depth_results: list[DepthResult] = []
            for prompt in dir_prompts:
                if config.depth.max_tokens > 0:
                    dtr = depth_generate(
                        model,
                        tokenizer,
                        prompt,
                        config.depth,
                    )
                else:
                    dtr = depth_profile(
                        model,
                        tokenizer,
                        prompt,
                        config.depth,
                    )
                dir_depth_results.append(dtr)
        else:
            dir_depth_results = depth_results

        refusal_dir: DirectionResult | None = None
        if not _is_default_data(config):
            _log(
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

    report_path = config.output_dir / "depth_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    _log(
        f"Done — depth report written to {report_path}",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    _write_experiment_log(
        context.config_path,
        config,
        "depth",
        ["depth_report.json"],
        {},
        time.monotonic() - context.t0,
    )


def _run_probe_mode(context: _EarlyModeContext) -> None:
    """Run [probe] early-return mode and write its report."""
    import time

    config = context.config
    assert config.probe is not None
    assert context.direction_result is not None
    v = config.verbose
    model = _cast("CausalLM", context.model)
    tokenizer = _cast("Tokenizer", context.tokenizer)

    _log(
        "Running probe inspection",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    probe_results = [
        probe(
            model,
            tokenizer,
            prompt,
            context.direction_result.direction,
        )
        for prompt in config.probe.prompts
    ]
    report_path = config.output_dir / "probe_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps([_probe_to_dict(r) for r in probe_results], indent=2),
    )
    _log(
        f"Done — probe report written to {report_path}",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    _write_experiment_log(
        context.config_path,
        config,
        "probe",
        ["probe_report.json"],
        {},
        time.monotonic() - context.t0,
    )


def _run_steer_mode(context: _EarlyModeContext) -> None:
    """Run [steer] early-return mode and write its report."""
    import time

    config = context.config
    assert config.steer is not None
    assert context.direction_result is not None
    v = config.verbose
    model = _cast("CausalLM", context.model)
    tokenizer = _cast("Tokenizer", context.tokenizer)

    _log(
        "Running steer generation",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    from vauban._forward import get_transformer as _get_transformer

    n_layers = len(_get_transformer(model).layers)
    steer_layers = config.steer.layers or list(range(n_layers))
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
    report_path = config.output_dir / "steer_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps([_steer_to_dict(r) for r in steer_results], indent=2),
    )
    _log(
        f"Done — steer report written to {report_path}",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    _write_experiment_log(
        context.config_path,
        config,
        "steer",
        ["steer_report.json"],
        {},
        time.monotonic() - context.t0,
    )


def _run_cast_mode(context: _EarlyModeContext) -> None:
    """Run [cast] early-return mode and write its report."""
    import time

    import numpy as np

    from vauban import _ops as ops

    config = context.config
    assert config.cast is not None
    assert context.direction_result is not None
    v = config.verbose
    model = _cast("CausalLM", context.model)
    tokenizer = _cast("Tokenizer", context.tokenizer)

    _log(
        "Running CAST conditional steering",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )

    # Load condition direction if configured
    condition_direction: Array | None = None
    if config.cast.condition_direction_path is not None:
        cond_path = Path(config.cast.condition_direction_path)
        if not cond_path.is_absolute():
            cond_path = config.output_dir.parent / cond_path
        _log(
            f"Loading condition direction from {cond_path}",
            verbose=v,
            elapsed=time.monotonic() - context.t0,
        )
        cond_np = np.load(str(cond_path))
        condition_direction = ops.array(cond_np)
        if condition_direction.shape[-1] != context.direction_result.d_model:
            msg = (
                f"condition_direction d_model mismatch:"
                f" {condition_direction.shape[-1]}"
                f" != {context.direction_result.d_model}"
            )
            raise ValueError(msg)

    from vauban._forward import get_transformer as _get_transformer

    cast_layers = config.cast.layers or list(range(len(_get_transformer(model).layers)))
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
        )
        for prompt in config.cast.prompts
    ]
    report_path = config.output_dir / "cast_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps([_cast_to_dict(r) for r in cast_results], indent=2),
    )
    _log(
        f"Done — CAST report written to {report_path}",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    _write_experiment_log(
        context.config_path,
        config,
        "cast",
        ["cast_report.json"],
        {},
        time.monotonic() - context.t0,
    )


def _run_sic_mode(context: _EarlyModeContext) -> None:
    """Run [sic] early-return mode and write its report."""
    import time

    config = context.config
    assert config.sic is not None
    assert context.harmful is not None
    assert context.harmless is not None
    v = config.verbose
    model = _cast("CausalLM", context.model)
    tokenizer = _cast("Tokenizer", context.tokenizer)

    _log(
        f"Running SIC sanitization (mode={config.sic.mode})",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    direction_vec = (
        context.direction_result.direction
        if context.direction_result is not None
        else None
    )
    layer_idx = (
        context.direction_result.layer_index
        if context.direction_result is not None
        else 0
    )
    if config.eval.prompts_path is not None:
        sic_prompts: list[str] = load_prompts(config.eval.prompts_path)
    else:
        sic_prompts = context.harmful[:config.eval.num_prompts]

    cal_prompts: list[str] | None = None
    if config.sic.calibrate:
        cal_prompts = (
            context.harmless
            if config.sic.calibrate_prompts == "harmless"
            else context.harmful
        )

    sic_result = sic_sanitize(
        model,
        tokenizer,
        sic_prompts,
        config.sic,
        direction_vec,
        layer_idx,
        cal_prompts,
    )
    report_path = config.output_dir / "sic_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(_sic_to_dict(sic_result), indent=2))
    _log(
        f"Done — SIC report written to {report_path}",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    _write_experiment_log(
        context.config_path,
        config,
        "sic",
        ["sic_report.json"],
        {},
        time.monotonic() - context.t0,
    )


def _run_optimize_mode(context: _EarlyModeContext) -> None:
    """Run [optimize] early-return mode and write its report."""
    import time

    config = context.config
    assert config.optimize is not None
    assert context.direction_result is not None
    assert context.harmful is not None
    v = config.verbose
    model = _cast("CausalLM", context.model)
    tokenizer = _cast("Tokenizer", context.tokenizer)

    _log(
        f"Running optimization ({config.optimize.n_trials} trials)",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    if config.eval.prompts_path is not None:
        eval_prompts_opt = load_prompts(config.eval.prompts_path)
    else:
        eval_prompts_opt = context.harmful[:config.eval.num_prompts]

    opt_result = optimize(
        model,
        tokenizer,
        context.direction_result,
        eval_prompts_opt,
        config.optimize,
    )

    report_path = config.output_dir / "optimize_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(_optimize_to_dict(opt_result), indent=2))
    _log(
        f"Done — optimize report written to {report_path}",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    _write_experiment_log(
        context.config_path,
        config,
        "optimize",
        ["optimize_report.json"],
        {"n_trials": float(config.optimize.n_trials)},
        time.monotonic() - context.t0,
    )


def _run_softprompt_mode(context: _EarlyModeContext) -> None:
    """Run [softprompt] early-return mode and write its report."""
    import time

    from vauban import _ops as ops
    from vauban._forward import force_eval
    from vauban._forward import get_transformer as _get_transformer
    from vauban._model_io import load_model

    config = context.config
    assert config.softprompt is not None
    assert context.harmful is not None
    v = config.verbose
    model = _cast("CausalLM", context.model)
    tokenizer = _cast("Tokenizer", context.tokenizer)

    _log(
        f"Running soft prompt attack (mode={config.softprompt.mode})",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    direction_vec = (
        context.direction_result.direction
        if context.direction_result is not None
        else None
    )
    if config.eval.prompts_path is not None:
        sp_prompts: list[str] = load_prompts(config.eval.prompts_path)
    else:
        pool_size = (
            config.softprompt.prompt_pool_size
            if config.softprompt.prompt_pool_size is not None
            else config.eval.num_prompts
        )
        sp_prompts = context.harmful[:pool_size]

    ref_model: object | None = None
    if config.softprompt.ref_model is not None:
        ref_model, _ = load_model(config.softprompt.ref_model)
        if is_quantized(ref_model):
            dequantize_model(ref_model)

    # Pre-load transfer models once (reused in GAN loop and/or post-hoc eval)
    transfer_models_loaded: (
        list[tuple[str, object, object]] | None
    ) = None
    if config.softprompt.transfer_models:
        transfer_models_loaded = []
        for transfer_model_id in config.softprompt.transfer_models:
            t_model, t_tok = load_model(transfer_model_id)
            if is_quantized(t_model):
                dequantize_model(t_model)
            transfer_models_loaded.append(
                (transfer_model_id, t_model, t_tok),
            )

    sp_result = softprompt_attack(
        model,
        tokenizer,
        sp_prompts,
        config.softprompt,
        direction_vec,
        ref_model,
        transfer_models=transfer_models_loaded,  # type: ignore[arg-type]
    )

    # Post-hoc transfer eval (skip if GAN loop already did per-round eval)
    if (
        transfer_models_loaded
        and not config.softprompt.gan_rounds
    ):
        from vauban.softprompt import _evaluate_attack

        if sp_result.token_ids is not None:
            transfer_token_ids = sp_result.token_ids
        elif sp_result.embeddings is not None:
            transfer_token_ids = _project_to_tokens(
                sp_result.embeddings,
                _get_transformer(model).embed_tokens.weight,
            )
        else:
            transfer_token_ids = []

        transfer_results: list[TransferEvalResult] = []
        for t_name, t_model, t_tok in transfer_models_loaded:
            t_token_array = ops.array(transfer_token_ids)[None, :]
            t_embeds = _get_transformer(t_model).embed_tokens(  # type: ignore[union-attr]
                t_token_array,
            )
            force_eval(t_embeds)
            t_success, t_responses = _evaluate_attack(
                t_model,
                t_tok,
                sp_prompts,
                t_embeds,
                config.softprompt,
            )
            transfer_results.append(
                TransferEvalResult(
                    model_id=t_name,
                    success_rate=t_success,
                    eval_responses=t_responses,
                ),
            )

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
            defense_eval=sp_result.defense_eval,
            gan_history=sp_result.gan_history,
        )

    report_path = config.output_dir / "softprompt_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(_softprompt_to_dict(sp_result), indent=2))
    _log(
        f"Done — softprompt report written to {report_path}",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    _write_experiment_log(
        context.config_path,
        config,
        "softprompt",
        ["softprompt_report.json"],
        {"success_rate": sp_result.success_rate},
        time.monotonic() - context.t0,
    )


type _EarlyModeRunner = Callable[[_EarlyModeContext], None]

_EARLY_MODE_RUNNERS: dict[str, _EarlyModeRunner] = {
    "depth": _run_depth_mode,
    "probe": _run_probe_mode,
    "steer": _run_steer_mode,
    "cast": _run_cast_mode,
    "sic": _run_sic_mode,
    "optimize": _run_optimize_mode,
    "softprompt": _run_softprompt_mode,
}


def _dispatch_early_mode(
    phase: EarlyModePhase,
    context: _EarlyModeContext,
) -> bool:
    """Run the active early-return mode for *phase* if one is enabled."""
    spec = active_early_mode_for_phase(context.config, phase)
    if spec is None:
        return False
    if spec.requires_direction and context.direction_result is None:
        return False
    runner = _EARLY_MODE_RUNNERS[spec.mode]
    runner(context)
    return True


def run(config_path: str | Path) -> None:
    """Run the full measure -> cut -> evaluate pipeline from a TOML config."""
    import json
    import time

    from vauban._model_io import load_model

    config = load_config(config_path)
    v = config.verbose
    t0 = time.monotonic()

    _log(
        f"Loading model {config.model_path}",
        verbose=v, elapsed=time.monotonic() - t0,
    )
    model, tokenizer = load_model(config.model_path)

    # Auto-dequantize if model has quantized weights
    if is_quantized(model):
        _log(
            "Dequantizing model weights",
            verbose=v, elapsed=time.monotonic() - t0,
        )
        dequantize_model(model)

    early_mode_context = _EarlyModeContext(
        config_path=config_path,
        config=config,
        model=model,
        tokenizer=tokenizer,
        t0=t0,
    )
    if _dispatch_early_mode("before_prompts", early_mode_context):
        return

    _log(
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
        refusal_phrases = _load_refusal_phrases(config.eval.refusal_phrases_path)

    # Defense detection (runs before measure/cut)
    if config.detect is not None:
        _log(
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

    _log(
        f"Measuring refusal direction (mode={config.measure.mode})",
        verbose=v, elapsed=time.monotonic() - t0,
    )
    if config.measure.mode == "subspace":
        subspace_result = measure_subspace(
            model, tokenizer, harmful, harmless,
            config.measure.top_k, clip_q,
        )
    elif config.measure.mode == "diff":
        assert config.measure.diff_model is not None
        _log(
            f"Loading base model for diff: {config.measure.diff_model}",
            verbose=v, elapsed=time.monotonic() - t0,
        )
        base_model, _ = load_model(config.measure.diff_model)
        if is_quantized(base_model):
            _log(
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
        measure_reports = _write_measure_reports(
            config,
            direction_result=None,
            subspace_result=None,
            dbdi_result=None,
            diff_result=diff_result,
        )
        _log(
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

        transfer_results_list: list[DirectionTransferResult] = []
        for transfer_model_id in config.measure.transfer_models:
            _log(
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
        _log(
            f"Transfer report written to {transfer_report_path}",
            verbose=v, elapsed=time.monotonic() - t0,
        )
        transfer_reports = ["transfer_report.json"]
    else:
        transfer_reports = []

    if config.measure.measure_only:
        if not measure_reports:
            measure_reports = _write_measure_reports(
                config,
                direction_result=direction_result,
                subspace_result=subspace_result,
                dbdi_result=dbdi_result,
                diff_result=diff_result,
            )
        _log(
            f"Measure-only mode complete — output written to {config.output_dir}",
            verbose=v, elapsed=time.monotonic() - t0,
        )
        _write_experiment_log(
            config_path,
            config,
            "measure",
            measure_reports + transfer_reports,
            {},
            time.monotonic() - t0,
        )
        return

    early_mode_context.direction_result = direction_result
    if _dispatch_early_mode("after_measure", early_mode_context):
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
        assert dbdi_result is not None
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
        modified_model, _ = load_model(config.model_path)
        if is_quantized(modified_model):
            dequantize_model(modified_model)
        modified_model.load_weights(list(modified_weights.items()))  # type: ignore[attr-defined]  # CausalLM is nn.Module at runtime

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
        gate_failures = _surface_gate_failures(config.surface, comparison)
        if gate_failures:
            joined = "\n".join(f"- {failure}" for failure in gate_failures)
            msg = (
                "Surface quality gates failed:\n"
                f"{joined}\n"
                "Adjust [surface] gate thresholds or improve model behavior."
            )
            raise RuntimeError(msg)

    # Evaluate if eval prompts are provided
    if config.eval.prompts_path is not None and modified_model is not None:
        _log(
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

    # Collect reports and metrics for experiment log
    normal_reports: list[str] = []
    normal_metrics: dict[str, float] = {}
    if surface_before is not None:
        normal_reports.append("surface_report.json")
    if config.eval.prompts_path is not None and modified_model is not None:
        normal_reports.append("eval_report.json")
        normal_metrics["refusal_rate_modified"] = result.refusal_rate_modified

    _log(
        f"Done — output written to {config.output_dir}",
        verbose=v, elapsed=time.monotonic() - t0,
    )
    _write_experiment_log(
        config_path, config, "default",
        normal_reports, normal_metrics,
        time.monotonic() - t0,
    )
