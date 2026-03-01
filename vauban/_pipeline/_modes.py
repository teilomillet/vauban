"""Early-return mode runners for each pipeline section."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING
from typing import cast as _cast

from vauban._pipeline._context import EarlyModeContext, log, write_experiment_log
from vauban._pipeline._helpers import is_default_data, write_arena_card
from vauban._serializers import (
    _cast_to_dict,
    _defend_to_dict,
    _depth_direction_to_dict,
    _depth_to_dict,
    _optimize_to_dict,
    _probe_to_dict,
    _sic_to_dict,
    _softprompt_to_dict,
    _steer_to_dict,
)
from vauban.config._mode_registry import (
    EarlyModePhase,
    active_early_mode_for_phase,
)
from vauban.types import (
    SoftPromptResult,
    TransferEvalResult,
)

if TYPE_CHECKING:
    from vauban._array import Array
    from vauban.types import (
        CausalLM,
        DepthResult,
        DirectionResult,
        Tokenizer,
    )


def _run_depth_mode(context: EarlyModeContext) -> None:
    """Run [depth] early-return mode and write its report."""
    config = context.config
    if config.depth is None:
        msg = "depth config is required for depth mode"
        raise ValueError(msg)
    v = config.verbose
    model = _cast("CausalLM", context.model)
    tokenizer = _cast("Tokenizer", context.tokenizer)

    from vauban.dataset import resolve_prompts
    from vauban.depth import depth_direction, depth_generate, depth_profile
    from vauban.measure import measure

    log(
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
        log(
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
        if not is_default_data(config):
            log(
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
    log(
        f"Done — depth report written to {report_path}",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    write_experiment_log(
        context.config_path,
        config,
        "depth",
        ["depth_report.json"],
        {},
        time.monotonic() - context.t0,
    )


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

    # Load prompt files
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

    # Save boundary
    boundary_path = config.output_dir / "svf_boundary.safetensors"
    save_svf_boundary(boundary, boundary_path)

    # Write report
    report: dict[str, object] = {
        "train_loss_history": svf_result.train_loss_history,
        "final_accuracy": svf_result.final_accuracy,
        "per_layer_separation": svf_result.per_layer_separation,
        "projection_dim": svf_result.projection_dim,
        "hidden_dim": svf_result.hidden_dim,
        "n_layers_trained": svf_result.n_layers_trained,
        "boundary_path": str(boundary_path),
    }
    report_path = config.output_dir / "svf_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    log(
        f"Done — SVF report written to {report_path}",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    write_experiment_log(
        context.config_path,
        config,
        "svf",
        ["svf_report.json", "svf_boundary.safetensors"],
        {"final_accuracy": svf_result.final_accuracy},
        time.monotonic() - context.t0,
    )


def _run_probe_mode(context: EarlyModeContext) -> None:
    """Run [probe] early-return mode and write its report."""
    config = context.config
    if config.probe is None:
        msg = "probe config is required for probe mode"
        raise ValueError(msg)
    if context.direction_result is None:
        msg = "direction_result is required for probe mode but was not computed"
        raise ValueError(msg)
    v = config.verbose
    model = _cast("CausalLM", context.model)
    tokenizer = _cast("Tokenizer", context.tokenizer)

    from vauban.probe import probe

    log(
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
    log(
        f"Done — probe report written to {report_path}",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    write_experiment_log(
        context.config_path,
        config,
        "probe",
        ["probe_report.json"],
        {},
        time.monotonic() - context.t0,
    )


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
            verbose=v, elapsed=time.monotonic() - context.t0,
        )
        bank = load_bank(config.steer.bank_path)
        composed = compose_direction(bank, config.steer.composition)
        steer_results = [
            steer(
                model, tokenizer, prompt, composed,
                steer_layers, config.steer.alpha, config.steer.max_tokens,
            )
            for prompt in config.steer.prompts
        ]
    elif config.steer.direction_source == "svf":
        if config.steer.svf_boundary_path is None:
            msg = "svf_boundary_path is required when direction_source='svf'"
            raise ValueError(msg)
        from vauban.probe import steer_svf as _steer_svf

        boundary = load_svf_boundary(
            Path(config.steer.svf_boundary_path),
        )
        steer_results = [
            _steer_svf(
                model, tokenizer, prompt, boundary,
                steer_layers, config.steer.alpha, config.steer.max_tokens,
            )
            for prompt in config.steer.prompts
        ]
    else:
        if context.direction_result is None:
            msg = (
                "direction_result is required for steer mode"
                " but was not computed"
            )
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
    report_path = config.output_dir / "steer_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps([_steer_to_dict(r) for r in steer_results], indent=2),
    )
    log(
        f"Done — steer report written to {report_path}",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    write_experiment_log(
        context.config_path,
        config,
        "steer",
        ["steer_report.json"],
        {},
        time.monotonic() - context.t0,
    )


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
        baseline_activations = {int(k): v for k, v in loaded_dict.items()}

    # Load condition direction (AdaSteer dual-direction) if configured.
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
                f"condition_direction d_model mismatch:"
                f" {condition_direction.shape[-1]} != {expected_d}"
            )
            raise ValueError(msg)

    if config.cast.bank_path and config.cast.composition:
        from vauban._compose import compose_direction, load_bank

        log(
            f"Composing direction from bank: {config.cast.bank_path}",
            verbose=v, elapsed=time.monotonic() - context.t0,
        )
        bank = load_bank(config.cast.bank_path)
        composed = compose_direction(bank, config.cast.composition)
        cast_results = [
            cast_generate(
                model, tokenizer, prompt, composed,
                cast_layers, config.cast.alpha, config.cast.threshold,
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
        from vauban.cast import cast_generate_svf as _cast_gen_svf

        boundary = load_svf_boundary(
            Path(config.cast.svf_boundary_path),
        )
        cast_results = [
            _cast_gen_svf(
                model, tokenizer, prompt, boundary,
                cast_layers, config.cast.alpha, config.cast.max_tokens,
            )
            for prompt in config.cast.prompts
        ]
    else:
        if context.direction_result is None:
            msg = (
                "direction_result is required for cast mode"
                " but was not computed"
            )
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
    report_path = config.output_dir / "cast_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps([_cast_to_dict(r) for r in cast_results], indent=2),
    )
    log(
        f"Done — CAST report written to {report_path}",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    write_experiment_log(
        context.config_path,
        config,
        "cast",
        ["cast_report.json"],
        {},
        time.monotonic() - context.t0,
    )


def _run_sic_mode(context: EarlyModeContext) -> None:
    """Run [sic] early-return mode and write its report."""
    config = context.config
    if config.sic is None:
        msg = "sic config is required for sic mode"
        raise ValueError(msg)
    if context.harmful is None:
        msg = "harmful prompts are required for sic mode but were not loaded"
        raise ValueError(msg)
    if context.harmless is None:
        msg = "harmless prompts are required for sic mode but were not loaded"
        raise ValueError(msg)
    v = config.verbose
    model = _cast("CausalLM", context.model)
    tokenizer = _cast("Tokenizer", context.tokenizer)

    from vauban.measure import load_prompts
    from vauban.sic import sic as sic_sanitize

    log(
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
    log(
        f"Done — SIC report written to {report_path}",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    write_experiment_log(
        context.config_path,
        config,
        "sic",
        ["sic_report.json"],
        {},
        time.monotonic() - context.t0,
    )


def _run_defend_mode(context: EarlyModeContext) -> None:
    """Run [defend] early-return mode and write its report."""
    from vauban.defend import defend_content
    from vauban.measure import load_prompts

    config = context.config
    if config.defend is None:
        msg = "defend config is required for defend mode"
        raise ValueError(msg)
    if context.harmful is None:
        msg = "harmful prompts are required for defend mode but were not loaded"
        raise ValueError(msg)
    v = config.verbose
    model = _cast("CausalLM", context.model)
    tokenizer = _cast("Tokenizer", context.tokenizer)

    log(
        "Running defense stack evaluation",
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
        defend_prompts: list[str] = load_prompts(config.eval.prompts_path)
    else:
        defend_prompts = context.harmful[:config.eval.num_prompts]

    defend_results = [
        defend_content(
            model, tokenizer, prompt,
            direction_vec, config.defend,
            layer_idx,
        )
        for prompt in defend_prompts
    ]

    total_blocked = sum(1 for r in defend_results if r.blocked)
    block_rate = total_blocked / len(defend_results) if defend_results else 0.0

    report: dict[str, object] = {
        "total_prompts": len(defend_results),
        "total_blocked": total_blocked,
        "block_rate": block_rate,
        "results": [_defend_to_dict(r) for r in defend_results],
    }
    report_path = config.output_dir / "defend_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    log(
        f"Done — defend report written to {report_path}",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    write_experiment_log(
        context.config_path,
        config,
        "defend",
        ["defend_report.json"],
        {"block_rate": block_rate},
        time.monotonic() - context.t0,
    )


def _run_optimize_mode(context: EarlyModeContext) -> None:
    """Run [optimize] early-return mode and write its report."""
    config = context.config
    if config.optimize is None:
        msg = "optimize config is required for optimize mode"
        raise ValueError(msg)
    if context.direction_result is None:
        msg = (
            "direction_result is required for optimize mode"
            " but was not computed"
        )
        raise ValueError(msg)
    if context.harmful is None:
        msg = (
            "harmful prompts are required for optimize mode"
            " but were not loaded"
        )
        raise ValueError(msg)
    v = config.verbose
    model = _cast("CausalLM", context.model)
    tokenizer = _cast("Tokenizer", context.tokenizer)

    from vauban.measure import load_prompts
    from vauban.optimize import optimize

    log(
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
    log(
        f"Done — optimize report written to {report_path}",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    write_experiment_log(
        context.config_path,
        config,
        "optimize",
        ["optimize_report.json"],
        {"n_trials": float(config.optimize.n_trials)},
        time.monotonic() - context.t0,
    )


def _run_compose_optimize_mode(context: EarlyModeContext) -> None:
    """Run [compose_optimize] early-return mode and write its report."""
    from vauban.measure import load_prompts
    from vauban.optimize import optimize_composition

    config = context.config
    if config.compose_optimize is None:
        msg = "compose_optimize config is required for compose_optimize mode"
        raise ValueError(msg)
    if context.harmful is None:
        msg = (
            "harmful prompts are required for compose_optimize mode"
            " but were not loaded"
        )
        raise ValueError(msg)
    v = config.verbose
    model = _cast("CausalLM", context.model)
    tokenizer = _cast("Tokenizer", context.tokenizer)

    log(
        f"Running composition optimization"
        f" ({config.compose_optimize.n_trials} trials)",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    if config.eval.prompts_path is not None:
        eval_prompts_co = load_prompts(config.eval.prompts_path)
    else:
        eval_prompts_co = context.harmful[:config.eval.num_prompts]

    co_result = optimize_composition(
        model, tokenizer, eval_prompts_co, config.compose_optimize,
    )

    report: dict[str, object] = {
        "n_trials": co_result.n_trials,
        "bank_entries": co_result.bank_entries,
        "best_refusal": (
            {
                "weights": co_result.best_refusal.weights,
                "refusal_rate": co_result.best_refusal.refusal_rate,
                "perplexity": co_result.best_refusal.perplexity,
            }
            if co_result.best_refusal is not None else None
        ),
        "best_balanced": (
            {
                "weights": co_result.best_balanced.weights,
                "refusal_rate": co_result.best_balanced.refusal_rate,
                "perplexity": co_result.best_balanced.perplexity,
            }
            if co_result.best_balanced is not None else None
        ),
    }
    report_path = config.output_dir / "compose_optimize_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    log(
        f"Done — compose_optimize report written to {report_path}",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    write_experiment_log(
        context.config_path,
        config,
        "compose_optimize",
        ["compose_optimize_report.json"],
        {"n_trials": float(config.compose_optimize.n_trials)},
        time.monotonic() - context.t0,
    )


def _run_softprompt_mode(context: EarlyModeContext) -> None:
    """Run [softprompt] early-return mode and write its report."""
    from vauban import _ops as ops
    from vauban._forward import force_eval
    from vauban._forward import get_transformer as _get_transformer
    from vauban._model_io import load_model
    from vauban.dequantize import dequantize_model, is_quantized
    from vauban.measure import load_prompts
    from vauban.softprompt import _project_to_tokens, softprompt_attack

    config = context.config
    if config.softprompt is None:
        msg = "softprompt config is required for softprompt mode"
        raise ValueError(msg)
    if context.harmful is None:
        msg = (
            "harmful prompts are required for softprompt mode"
            " but were not loaded"
        )
        raise ValueError(msg)
    v = config.verbose
    model = _cast("CausalLM", context.model)
    tokenizer = _cast("Tokenizer", context.tokenizer)

    log(
        f"Running soft prompt attack (mode={config.softprompt.mode})",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    direction_vec: Array | None = (
        context.direction_result.direction
        if context.direction_result is not None
        else None
    )

    # Externality mode: load target direction from explicit path
    if (
        config.softprompt.loss_mode == "externality"
        and config.softprompt.externality_target is not None
    ):
        loaded = ops.load(config.softprompt.externality_target)
        if isinstance(loaded, dict):
            loaded_dict = _cast("dict[str, Array]", loaded)
            direction_vec = next(iter(loaded_dict.values()))
        elif isinstance(loaded, tuple):
            msg = "Expected array or dict from externality target, got tuple"
            raise TypeError(msg)
        else:
            direction_vec = loaded

    if config.eval.prompts_path is not None:
        sp_prompts: list[str] = load_prompts(config.eval.prompts_path)
    else:
        pool_size = (
            config.softprompt.prompt_pool_size
            if config.softprompt.prompt_pool_size is not None
            else config.eval.num_prompts
        )
        sp_prompts = context.harmful[:pool_size]

    ref_model: CausalLM | None = None
    if config.softprompt.ref_model is not None:
        ref_model, _ = load_model(config.softprompt.ref_model)
        if is_quantized(ref_model):
            dequantize_model(ref_model)

    # Pre-load transfer models once (reused in GAN loop and/or post-hoc eval)
    transfer_models_loaded: (
        list[tuple[str, CausalLM, Tokenizer]] | None
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
        transfer_models=transfer_models_loaded,
        api_eval_config=config.api_eval,
        environment_config=config.environment,
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
            t_embeds = _get_transformer(t_model).embed_tokens(
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

    # API-based transfer evaluation (hit remote endpoints with the suffix)
    if config.api_eval and sp_result.token_text:
        from vauban.api_eval import evaluate_suffix_via_api

        api_results = evaluate_suffix_via_api(
            sp_result.token_text,
            sp_prompts,
            config.api_eval,
            config.softprompt.system_prompt if config.softprompt else None,
        )
        all_transfer = [*sp_result.transfer_results, *api_results]
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
            transfer_results=all_transfer,
            defense_eval=sp_result.defense_eval,
            gan_history=sp_result.gan_history,
        )
        for api_r in api_results:
            log(
                f"API eval {api_r.model_id}: {api_r.success_rate:.2%}",
                verbose=v,
                elapsed=time.monotonic() - context.t0,
            )

    report_path = config.output_dir / "softprompt_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(_softprompt_to_dict(sp_result), indent=2))

    # Write arena card alongside JSON report
    if sp_result.token_text:
        arena_path = config.output_dir / "arena_card.txt"
        write_arena_card(arena_path, sp_result, sp_prompts)
        log(
            f"Arena card written to {arena_path}",
            verbose=v,
            elapsed=time.monotonic() - context.t0,
        )

    log(
        f"Done — softprompt report written to {report_path}",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    write_experiment_log(
        context.config_path,
        config,
        "softprompt",
        ["softprompt_report.json"],
        {"success_rate": sp_result.success_rate},
        time.monotonic() - context.t0,
    )


def _run_circuit_mode(context: EarlyModeContext) -> None:
    """Run [circuit] early-return mode: activation patching and report."""
    config = context.config
    if config.circuit is None:
        msg = "circuit config is required for circuit mode"
        raise ValueError(msg)
    v = config.verbose
    model = _cast("CausalLM", context.model)
    tokenizer = _cast("Tokenizer", context.tokenizer)

    log(
        "Tracing circuit via activation patching",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )

    from vauban.circuit import trace_circuit

    direction: Array | None = None
    if context.direction_result is not None and config.circuit.attribute_direction:
        direction = context.direction_result.direction

    result = trace_circuit(
        model,
        tokenizer,
        config.circuit.clean_prompts,
        config.circuit.corrupt_prompts,
        metric=config.circuit.metric,
        granularity=config.circuit.granularity,
        layers=config.circuit.layers,
        token_position=config.circuit.token_position,
        direction=direction,
        attribute_direction=config.circuit.attribute_direction,
        logit_diff_tokens=config.circuit.logit_diff_tokens,
    )

    report: dict[str, object] = result.to_dict()
    report_path = config.output_dir / "circuit_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    log(
        f"Done — circuit report written to {report_path}",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    write_experiment_log(
        context.config_path,
        config,
        "circuit",
        ["circuit_report.json"],
        {"n_effects": len(result.effects)},
        time.monotonic() - context.t0,
    )


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

    saes, features_result = train_sae_multi_layer(
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

    # Save SAEs
    for layer_idx, sae in saes.items():
        sae_path = config.output_dir / f"sae_layer_{layer_idx}.safetensors"
        save_sae(sae, sae_path)

    report: dict[str, object] = features_result.to_dict()
    report_path = config.output_dir / "features_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    log(
        f"Done — features report written to {report_path}",
        verbose=v,
        elapsed=time.monotonic() - context.t0,
    )
    write_experiment_log(
        context.config_path,
        config,
        "features",
        ["features_report.json"]
        + [f"sae_layer_{idx}.safetensors" for idx in saes],
        {"n_layers": len(features_result.layers)},
        time.monotonic() - context.t0,
    )


type EarlyModeRunner = Callable[[EarlyModeContext], None]

EARLY_MODE_RUNNERS: dict[str, EarlyModeRunner] = {
    "depth": _run_depth_mode,
    "svf": _run_svf_mode,
    "features": _run_features_mode,
    "probe": _run_probe_mode,
    "steer": _run_steer_mode,
    "cast": _run_cast_mode,
    "sic": _run_sic_mode,
    "optimize": _run_optimize_mode,
    "compose_optimize": _run_compose_optimize_mode,
    "softprompt": _run_softprompt_mode,
    "defend": _run_defend_mode,
    "circuit": _run_circuit_mode,
}


def dispatch_early_mode(
    phase: EarlyModePhase,
    context: EarlyModeContext,
) -> bool:
    """Run the active early-return mode for *phase* if one is enabled."""
    spec = active_early_mode_for_phase(context.config, phase)
    if spec is None:
        return False
    if spec.requires_direction and context.direction_result is None:
        return False
    runner = EARLY_MODE_RUNNERS[spec.mode]
    runner(context)
    return True
