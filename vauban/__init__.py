"""Vauban - MLX-native abliteration toolkit for Apple Silicon."""

import json
import tomllib
from pathlib import Path

from vauban._serializers import (
    _depth_direction_to_dict,
    _depth_to_dict,
    _detect_to_dict,
    _optimize_to_dict,
    _probe_to_dict,
    _sic_to_dict,
    _softprompt_to_dict,
    _steer_to_dict,
    _surface_comparison_to_dict,
)
from vauban.config import load_config
from vauban.config._types import TomlDict
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
    DepthConfig,
    DepthDirectionResult,
    DepthResult,
    DetectConfig,
    DetectResult,
    DirectionResult,
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
    TransferEvalResult,
    TrialResult,
)

__version__ = "0.2.3"

__all__ = [
    "CutConfig",
    "DBDIResult",
    "DatasetRef",
    "DepthConfig",
    "DepthDirectionResult",
    "DepthResult",
    "DetectConfig",
    "DetectResult",
    "DirectionResult",
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
    "calibrate_threshold",
    "compare_surfaces",
    "cut",
    "cut_biprojected",
    "cut_false_refusal_ortho",
    "cut_subspace",
    "default_eval_path",
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

_EARLY_RETURN_PRECEDENCE: tuple[str, ...] = (
    "[depth]",
    "[probe]",
    "[steer]",
    "[sic]",
    "[optimize]",
    "[softprompt]",
)


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

    config_path = Path(config_path)
    raw = _load_raw_toml(config_path)

    # Check for unknown/typo'd top-level sections before parsing
    from vauban._suggestions import (
        check_unknown_keys,
        check_unknown_sections,
        check_value_constraints,
    )

    unknown_warnings = check_unknown_sections(raw)
    unknown_warnings.extend(check_unknown_keys(raw))
    unknown_warnings.extend(check_value_constraints(raw))

    config = load_config(config_path)
    warnings: list[str] = []
    for uw in unknown_warnings:
        _add_warning(warnings, "MEDIUM", uw)

    harmful_count = _validate_prompt_source(
        config.harmful_path,
        "[data].harmful",
        warnings,
        min_recommended=16,
        missing_fix=(
            "set [data].harmful to an existing JSONL path"
            ' or use [data].harmful = "default"'
        ),
    )
    harmless_count = _validate_prompt_source(
        config.harmless_path,
        "[data].harmless",
        warnings,
        min_recommended=16,
        missing_fix=(
            "set [data].harmless to an existing JSONL path"
            ' or use [data].harmless = "default"'
        ),
    )
    if config.borderline_path is not None:
        _validate_prompt_source(
            config.borderline_path,
            "[data].borderline",
            warnings,
            min_recommended=8,
            missing_fix=(
                "set [data].borderline to an existing JSONL path"
                " or disable [cut].false_refusal_ortho"
            ),
        )

    if (
        harmful_count is not None
        and harmless_count is not None
        and harmful_count > 0
        and harmless_count > 0
    ):
        ratio = (
            harmful_count / harmless_count
            if harmful_count >= harmless_count
            else harmless_count / harmful_count
        )
        if ratio > 4.0:
            _add_warning(
                warnings,
                "LOW",
                (
                    "[data] prompt set sizes are highly imbalanced"
                    f" (harmful={harmful_count}, harmless={harmless_count})"
                ),
                fix=(
                    "use similarly sized harmful/harmless datasets"
                    " for more stable direction estimates"
                ),
            )

    # Eval prompts path + schema
    if config.eval.prompts_path is not None:
        eval_count = _validate_prompt_jsonl_file(
            config.eval.prompts_path,
            "[eval].prompts",
            warnings,
            min_recommended=8,
            missing_fix=(
                "set [eval].prompts to an existing JSONL path"
                " or remove [eval] if you do not want eval reports"
            ),
        )
        if eval_count is not None and eval_count < 3:
            _add_warning(
                warnings,
                "MEDIUM",
                (
                    f"[eval].prompts has only {eval_count} prompt(s);"
                    " evaluation metrics may be noisy"
                ),
                fix="use at least 8-20 prompts for reliable evaluation",
            )

    # Refusal phrases file
    if config.eval.refusal_phrases_path is not None:
        rp = config.eval.refusal_phrases_path
        if not rp.exists():
            _add_warning(
                warnings,
                "HIGH",
                f"[eval].refusal_phrases file not found: {rp}",
                fix=(
                    "set [eval].refusal_phrases to an existing text file,"
                    " or remove it to use built-in refusal phrases"
                ),
            )
        else:
            try:
                phrases = _load_refusal_phrases(rp)
            except ValueError as exc:
                msg = (
                    f"{exc} — fix: add one refusal phrase per line in"
                    f" {rp}, or remove [eval].refusal_phrases"
                )
                raise ValueError(msg) from exc
            if len(phrases) < 2:
                _add_warning(
                    warnings,
                    "MEDIUM",
                    (
                        f"[eval].refusal_phrases has only {len(phrases)}"
                        " phrase(s)"
                    ),
                    fix=(
                        "add multiple refusal phrases to reduce false negatives"
                    ),
                )

    # Surface prompts path + schema
    if config.surface is not None:
        sp_raw = config.surface.prompts_path
        if sp_raw == "default":
            sp = default_surface_path()
        elif isinstance(sp_raw, Path):
            sp = sp_raw
        else:
            sp = Path(sp_raw)
        surface_count = _validate_surface_jsonl_file(
            sp,
            "[surface].prompts",
            warnings,
            missing_fix=(
                "set [surface].prompts to an existing JSONL path"
                ' or use [surface].prompts = "default"'
            ),
        )
        if surface_count is not None and surface_count < 8:
            _add_warning(
                warnings,
                "LOW",
                (
                    f"[surface].prompts has only {surface_count} record(s);"
                    " category/label aggregates may be unstable"
                ),
                fix="use a broader surface prompt set (16+ recommended)",
            )

    # Output dir sanity
    if config.output_dir.exists() and not config.output_dir.is_dir():
        _add_warning(
            warnings,
            "HIGH",
            (
                f"[output].dir points to a file, not a directory:"
                f" {config.output_dir}"
            ),
            fix="set [output].dir to a directory path",
        )

    # Early-return mode conflicts
    early_modes = _active_early_modes(config)
    if len(early_modes) > 1:
        _add_warning(
            warnings,
                "HIGH",
                f"Multiple early-return modes active: {', '.join(early_modes)}"
                " — only the first will run (precedence: depth > probe > steer"
                " > sic > optimize > softprompt)",
                fix=(
                    "keep one early-return mode per config,"
                    " and split other modes into separate TOML files"
            ),
        )

    # Depth direction: warn if extract_direction=True but not enough prompts
    if config.depth is not None and config.depth.extract_direction:
        effective = (
            config.depth.direction_prompts
            if config.depth.direction_prompts is not None
            else config.depth.prompts
        )
        if len(effective) < 2:
            src = (
                "direction_prompts"
                if config.depth.direction_prompts is not None
                else "prompts"
            )
            _add_warning(
                warnings,
                "HIGH",
                f"[depth].extract_direction = true but {src}"
                f" has only {len(effective)} entry — need >= 2",
                fix=(
                    "add at least 2 prompts to the selected source,"
                    " or set [depth].extract_direction = false"
                ),
            )

    # [eval] present without prompts in normal pipeline does not run eval.
    eval_raw = raw.get("eval")
    if (
        isinstance(eval_raw, dict)
        and "prompts" not in eval_raw
        and config.eval.prompts_path is None
        and not early_modes
    ):
        _add_warning(
            warnings,
            "LOW",
            (
                "[eval] section is present but [eval].prompts is not set;"
                " eval_report.json will not be produced in default pipeline"
            ),
            fix=(
                'set [eval].prompts = "eval.jsonl"'
                " or remove the [eval] section"
            ),
        )

    # Sections silently skipped by early-return modes
    if early_modes:
        # Determine which mode takes precedence
        active_mode = early_modes[0]
        skipped: list[str] = []

        # depth skips everything including [detect]; other modes still run detect
        if config.depth is not None and config.detect is not None:
            skipped.append("[detect]")

        # [surface] and [eval] are skipped by ALL early-return modes
        if config.surface is not None:
            skipped.append("[surface]")
        if config.eval.prompts_path is not None:
            skipped.append("[eval]")

        if skipped:
            _add_warning(
                warnings,
                "MEDIUM",
                f"{active_mode} early-return will skip:"
                f" {', '.join(skipped)}"
                f" — these sections have no effect in"
                f" {active_mode.strip('[]')} mode",
                fix=(
                    "remove skipped sections from this config,"
                    " or run them in a separate non-early-return config"
                ),
            )

    # Print summary
    mode = "measure → cut → export"
    if config.depth is not None:
        mode = "depth analysis"
    elif config.probe is not None:
        mode = "probe inspection"
    elif config.steer is not None:
        mode = "steer generation"
    elif config.sic is not None:
        mode = "SIC sanitization"
    elif config.optimize is not None:
        mode = "Optuna optimization"
    elif config.softprompt is not None:
        mode = "soft prompt attack"
    # Extras: only sections that actually run alongside the active mode
    extras = []
    if config.detect is not None and config.depth is None:
        # detect runs before measure, so it works with all modes except depth
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


def _load_raw_toml(path: Path) -> TomlDict:
    """Load raw TOML mapping for intent-level validation checks."""
    with path.open("rb") as f:
        raw: TomlDict = tomllib.load(f)
    return raw


def _active_early_modes(config: PipelineConfig) -> list[str]:
    """Return active early-return modes in runtime precedence order."""
    active: dict[str, bool] = {
        "[depth]": config.depth is not None,
        "[probe]": config.probe is not None,
        "[steer]": config.steer is not None,
        "[sic]": config.sic is not None,
        "[optimize]": config.optimize is not None,
        "[softprompt]": config.softprompt is not None,
    }
    return [name for name in _EARLY_RETURN_PRECEDENCE if active[name]]


def _add_warning(
    warnings: list[str],
    severity: str,
    message: str,
    *,
    fix: str | None = None,
) -> None:
    """Append a structured warning with optional fix guidance."""
    full = f"[{severity}] {message}"
    if fix is not None:
        full += f" — fix: {fix}"
    warnings.append(full)


def _validate_prompt_source(
    source: Path | DatasetRef,
    key: str,
    warnings: list[str],
    *,
    min_recommended: int,
    missing_fix: str,
) -> int | None:
    """Validate a prompt source (local JSONL or HF dataset reference)."""
    if isinstance(source, DatasetRef):
        if source.limit is not None and source.limit < 2:
            _add_warning(
                warnings,
                "MEDIUM",
                (
                    f"{key} uses HF dataset limit={source.limit};"
                    " this is likely too small for stable estimation"
                ),
                fix="increase [data.*].limit or remove it",
            )
        return None
    return _validate_prompt_jsonl_file(
        source,
        key,
        warnings,
        min_recommended=min_recommended,
        missing_fix=missing_fix,
    )


def _validate_prompt_jsonl_file(
    path: Path,
    key: str,
    warnings: list[str],
    *,
    min_recommended: int,
    missing_fix: str,
) -> int | None:
    """Validate JSONL prompt schema for files using {'prompt': str} lines."""
    if not path.exists():
        _add_warning(
            warnings,
            "HIGH",
            f"{key} file not found: {path}",
            fix=missing_fix,
        )
        return None

    valid_count = 0
    seen_non_empty = 0
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            seen_non_empty += 1
            try:
                obj_raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                _add_warning(
                    warnings,
                    "HIGH",
                    (
                        f"{key} line {line_no} is not valid JSON"
                        f" ({exc.msg}) in {path}"
                    ),
                    fix='use JSONL lines like {"prompt": "your text"}',
                )
                return None
            if not isinstance(obj_raw, dict):
                _add_warning(
                    warnings,
                    "HIGH",
                    f"{key} line {line_no} must be a JSON object in {path}",
                    fix='use JSONL lines like {"prompt": "your text"}',
                )
                return None
            prompt_raw = obj_raw.get("prompt")
            if not isinstance(prompt_raw, str) or not prompt_raw.strip():
                _add_warning(
                    warnings,
                    "HIGH",
                    (
                        f"{key} line {line_no} must contain a non-empty"
                        f" string key 'prompt' in {path}"
                    ),
                    fix='ensure each line has {"prompt": "..."}',
                )
                return None
            valid_count += 1

    if seen_non_empty == 0:
        _add_warning(
            warnings,
            "HIGH",
            f"{key} is empty: {path}",
            fix='add JSONL records like {"prompt": "..."}',
        )
        return 0

    if valid_count < min_recommended:
        _add_warning(
            warnings,
            "LOW",
            (
                f"{key} has only {valid_count} prompt(s);"
                " results may be noisy on small sets"
            ),
            fix=f"use at least {min_recommended} prompts",
        )
    return valid_count


def _validate_surface_jsonl_file(
    path: Path,
    key: str,
    warnings: list[str],
    *,
    missing_fix: str,
) -> int | None:
    """Validate JSONL surface schema for prompt/label/category records."""
    if not path.exists():
        _add_warning(
            warnings,
            "HIGH",
            f"{key} file not found: {path}",
            fix=missing_fix,
        )
        return None

    valid_count = 0
    seen_non_empty = 0
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            seen_non_empty += 1
            try:
                obj_raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                _add_warning(
                    warnings,
                    "HIGH",
                    (
                        f"{key} line {line_no} is not valid JSON"
                        f" ({exc.msg}) in {path}"
                    ),
                    fix=(
                        'use JSONL lines like {"prompt": "...",'
                        ' "label": "harmful", "category": "weapons"}'
                    ),
                )
                return None
            if not isinstance(obj_raw, dict):
                _add_warning(
                    warnings,
                    "HIGH",
                    f"{key} line {line_no} must be a JSON object in {path}",
                    fix=(
                        'use JSONL lines like {"prompt": "...",'
                        ' "label": "harmful", "category": "weapons"}'
                    ),
                )
                return None
            prompt_raw = obj_raw.get("prompt")
            label_raw = obj_raw.get("label")
            category_raw = obj_raw.get("category")
            if (
                not isinstance(prompt_raw, str) or not prompt_raw.strip()
                or not isinstance(label_raw, str) or not label_raw.strip()
                or not isinstance(category_raw, str) or not category_raw.strip()
            ):
                _add_warning(
                    warnings,
                    "HIGH",
                    (
                        f"{key} line {line_no} must include non-empty string"
                        " keys: prompt, label, category"
                    ),
                    fix=(
                        'use JSONL lines like {"prompt": "...",'
                        ' "label": "harmful", "category": "weapons"}'
                    ),
                )
                return None
            valid_count += 1

    if seen_non_empty == 0:
        _add_warning(
            warnings,
            "HIGH",
            f"{key} is empty: {path}",
            fix=(
                'add JSONL records with prompt/label/category keys'
            ),
        )
        return 0

    return valid_count


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

    # Depth analysis: earliest possible early-return (no direction needed)
    if config.depth is not None:
        _log(
            "Running depth analysis",
            verbose=v, elapsed=time.monotonic() - t0,
        )
        depth_results: list[DepthResult] = []
        for p in config.depth.prompts:
            if config.depth.max_tokens > 0:
                dr = depth_generate(model, tokenizer, p, config.depth)  # type: ignore[arg-type]
            else:
                dr = depth_profile(model, tokenizer, p, config.depth)  # type: ignore[arg-type]
            depth_results.append(dr)

        report: dict[str, object] = {
            "dtr_results": [_depth_to_dict(r) for r in depth_results],
        }

        # Optional depth direction extraction
        if config.depth.extract_direction and len(depth_results) >= 2:
            _log(
                "Extracting depth direction",
                verbose=v, elapsed=time.monotonic() - t0,
            )
            # Use direction_prompts if provided, else reuse depth_results
            dir_prompts = config.depth.direction_prompts
            if dir_prompts is not None:
                dir_depth_results: list[DepthResult] = []
                for p in dir_prompts:
                    if config.depth.max_tokens > 0:
                        ddr = depth_generate(model, tokenizer, p, config.depth)  # type: ignore[arg-type]
                    else:
                        ddr = depth_profile(model, tokenizer, p, config.depth)  # type: ignore[arg-type]
                    dir_depth_results.append(ddr)
            else:
                dir_depth_results = depth_results

            # Optionally compute refusal direction for cosine comparison
            refusal_dir: DirectionResult | None = None
            has_real_data = not _is_default_data(config)
            if has_real_data:
                _log(
                    "Computing refusal direction for cosine comparison",
                    verbose=v, elapsed=time.monotonic() - t0,
                )
                harmful = resolve_prompts(config.harmful_path)
                harmless = resolve_prompts(config.harmless_path)
                refusal_dir = measure(
                    model, tokenizer,  # type: ignore[arg-type]
                    harmful, harmless,
                    config.depth.clip_quantile,
                )

            depth_dir_result = depth_direction(
                model, tokenizer, dir_depth_results,  # type: ignore[arg-type]
                refusal_direction=refusal_dir,
                clip_quantile=config.depth.clip_quantile,
            )

            # Save direction as .npy
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
            verbose=v, elapsed=time.monotonic() - t0,
        )
        _write_experiment_log(
            config_path, config, "depth",
            ["depth_report.json"], {},
            time.monotonic() - t0,
        )
        return

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

    # Probe inspection: measure per-layer projections, write report, return
    if config.probe is not None and direction_result is not None:
        _log(
            "Running probe inspection",
            verbose=v, elapsed=time.monotonic() - t0,
        )
        probe_results = [
            probe(model, tokenizer, p, direction_result.direction)  # type: ignore[arg-type]
            for p in config.probe.prompts
        ]
        report_path = config.output_dir / "probe_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps([_probe_to_dict(r) for r in probe_results], indent=2),
        )
        _log(
            f"Done — probe report written to {report_path}",
            verbose=v, elapsed=time.monotonic() - t0,
        )
        _write_experiment_log(
            config_path, config, "probe",
            ["probe_report.json"], {},
            time.monotonic() - t0,
        )
        return

    # Steer generation: generate with direction removal, write report, return
    if config.steer is not None and direction_result is not None:
        _log(
            "Running steer generation",
            verbose=v, elapsed=time.monotonic() - t0,
        )
        steer_layers = config.steer.layers or list(
            range(len(model.model.layers)),
        )
        steer_results = [
            steer(
                model, tokenizer, p, direction_result.direction,  # type: ignore[arg-type]
                steer_layers,
                config.steer.alpha,
                config.steer.max_tokens,
            )
            for p in config.steer.prompts
        ]
        report_path = config.output_dir / "steer_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps([_steer_to_dict(r) for r in steer_results], indent=2),
        )
        _log(
            f"Done — steer report written to {report_path}",
            verbose=v, elapsed=time.monotonic() - t0,
        )
        _write_experiment_log(
            config_path, config, "steer",
            ["steer_report.json"], {},
            time.monotonic() - t0,
        )
        return

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
        _write_experiment_log(
            config_path, config, "sic",
            ["sic_report.json"], {},
            time.monotonic() - t0,
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
        _write_experiment_log(
            config_path, config, "optimize",
            ["optimize_report.json"],
            {"n_trials": float(config.optimize.n_trials)},
            time.monotonic() - t0,
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
        _write_experiment_log(
            config_path, config, "softprompt",
            ["softprompt_report.json"],
            {"success_rate": sp_result.success_rate},
            time.monotonic() - t0,
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
