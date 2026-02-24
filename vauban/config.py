"""TOML configuration loader for vauban pipelines."""

import tomllib
from pathlib import Path

from vauban.measure import default_prompt_paths
from vauban.types import (
    CutConfig,
    DatasetRef,
    DetectConfig,
    MeasureConfig,
    PipelineConfig,
    SurfaceConfig,
)

# tomllib.load() returns dict[str, Any]. We use this alias to avoid bare Any
# while acknowledging TOML values are dynamically typed.
type TomlDict = dict[str, object]


def load_config(path: str | Path) -> PipelineConfig:
    """Load a pipeline configuration from a TOML file.

    Resolves file paths relative to the config file's parent directory.
    Sections other than [model] are optional.

    Supports `harmful = "default"` / `harmless = "default"` in [data]
    to use the bundled prompt datasets.
    """
    path = Path(path)
    base_dir = path.parent

    with path.open("rb") as f:
        raw: TomlDict = tomllib.load(f)

    model_section = raw.get("model")
    if not isinstance(model_section, dict) or "path" not in model_section:
        msg = f"Config must have [model] section with 'path' key: {path}"
        raise ValueError(msg)

    model_path_raw = model_section["path"]
    if not isinstance(model_path_raw, str):
        msg = f"[model].path must be a string, got {type(model_path_raw).__name__}"
        raise TypeError(msg)
    model_path: str = model_path_raw

    harmful_path, harmless_path = _resolve_data_paths(base_dir, raw)

    cut_raw = raw.get("cut")
    cut_config = _parse_cut(cut_raw if isinstance(cut_raw, dict) else {})

    measure_raw = raw.get("measure")
    measure_config = _parse_measure(
        measure_raw if isinstance(measure_raw, dict) else {},
    )

    surface_config = _parse_surface(base_dir, raw)
    detect_config = _parse_detect(raw)

    eval_path: Path | None = None
    eval_section = raw.get("eval")
    if isinstance(eval_section, dict) and "prompts" in eval_section:
        prompts_raw = eval_section["prompts"]
        if isinstance(prompts_raw, str):
            eval_path = base_dir / prompts_raw

    output_section = raw.get("output")
    output_dir_str = "output"
    if isinstance(output_section, dict):
        dir_raw = output_section.get("dir")
        if isinstance(dir_raw, str):
            output_dir_str = dir_raw
    output_dir = base_dir / output_dir_str

    borderline_path = _resolve_borderline_path(base_dir, raw)

    # Validate: false_refusal_ortho requires borderline_path
    if cut_config.false_refusal_ortho and borderline_path is None:
        msg = (
            "[cut].false_refusal_ortho = true requires"
            " [data].borderline to be set"
        )
        raise ValueError(msg)

    return PipelineConfig(
        model_path=model_path,
        harmful_path=harmful_path,
        harmless_path=harmless_path,
        cut=cut_config,
        measure=measure_config,
        surface=surface_config,
        detect=detect_config,
        eval_prompts_path=eval_path,
        output_dir=output_dir,
        borderline_path=borderline_path,
    )


def _resolve_data_paths(
    base_dir: Path,
    raw: TomlDict,
) -> tuple[Path | DatasetRef, Path | DatasetRef]:
    """Resolve harmful/harmless paths.

    Supports 'default', local files, and HF datasets.
    """
    sec = raw.get("data")
    if not isinstance(sec, dict):
        msg = "Config must have [data] section with 'harmful' and 'harmless' keys"
        raise ValueError(msg)

    harmful_raw = sec.get("harmful")  # type: ignore[arg-type]
    harmless_raw = sec.get("harmless")  # type: ignore[arg-type]

    harmful = _resolve_single_data(base_dir, harmful_raw, "harmful")
    harmless = _resolve_single_data(base_dir, harmless_raw, "harmless")
    return harmful, harmless


def _resolve_borderline_path(
    base_dir: Path,
    raw: TomlDict,
) -> Path | DatasetRef | None:
    """Resolve optional [data].borderline path for false refusal orthogonalization.

    Returns None if borderline is not specified.
    """
    sec = raw.get("data")
    if not isinstance(sec, dict):
        return None
    borderline_raw = sec.get("borderline")  # type: ignore[arg-type]
    if borderline_raw is None:
        return None
    return _resolve_single_data(base_dir, borderline_raw, "borderline")


def _resolve_single_data(
    base_dir: Path,
    value: object,
    name: str,
) -> Path | DatasetRef:
    """Resolve a single data source (harmful or harmless).

    Supports:
    - ``"default"`` — bundled prompts
    - ``"hf:repo/name"`` — HF dataset short form
    - ``{hf = "repo/name", ...}`` — HF dataset table form
    - ``"path/to/file.jsonl"`` — local file
    """
    if isinstance(value, str):
        if value == "default":
            harmful, harmless = default_prompt_paths()
            return harmful if name == "harmful" else harmless
        if value.startswith("hf:"):
            repo_id = value[3:]
            if not repo_id:
                msg = f"[data].{name}: empty repo_id after 'hf:' prefix"
                raise ValueError(msg)
            return DatasetRef(repo_id=repo_id)
        return base_dir / value

    if isinstance(value, dict):
        hf_repo = value.get("hf")  # type: ignore[arg-type]
        if not isinstance(hf_repo, str):
            msg = f"[data.{name}] table must have 'hf' key with a string value"
            raise TypeError(msg)
        split_raw = value.get("split", "train")  # type: ignore[arg-type]
        column_raw = value.get("column", "prompt")  # type: ignore[arg-type]
        config_raw = value.get("config")  # type: ignore[arg-type]
        limit_raw = value.get("limit")  # type: ignore[arg-type]
        return DatasetRef(
            repo_id=hf_repo,
            split=str(split_raw),
            column=str(column_raw),
            config=str(config_raw) if config_raw is not None else None,
            limit=int(limit_raw) if isinstance(limit_raw, int | float) else None,
        )

    msg = f"[data].{name} must be a string or table, got {type(value).__name__}"
    raise TypeError(msg)


def _resolve_path(
    base_dir: Path,
    raw: TomlDict,
    section: str,
    key: str,
) -> Path:
    """Resolve a path from a TOML section relative to the config directory."""
    sec = raw.get(section)
    if not isinstance(sec, dict):
        msg = f"Config must have [{section}] section with '{key}' key"
        raise ValueError(msg)
    if key not in sec:
        msg = f"Config must have [{section}] section with '{key}' key"
        raise ValueError(msg)
    value = sec[key]  # type: ignore[index]
    if not isinstance(value, str):
        msg = (
            f"[{section}].{key} must be a string,"
            f" got {type(value).__name__}"
        )
        raise TypeError(msg)
    return base_dir / value


def _parse_surface(base_dir: Path, raw: TomlDict) -> SurfaceConfig | None:
    """Parse the optional [surface] section into a SurfaceConfig.

    Returns None if the section is absent.
    """
    sec = raw.get("surface")
    if sec is None:
        return None
    if not isinstance(sec, dict):
        msg = f"[surface] must be a table, got {type(sec).__name__}"
        raise TypeError(msg)

    prompts_raw = sec.get("prompts", "default")  # type: ignore[arg-type]
    if not isinstance(prompts_raw, str):
        msg = f"[surface].prompts must be a string, got {type(prompts_raw).__name__}"
        raise TypeError(msg)
    prompts_path: Path | str = (
        "default" if prompts_raw == "default" else base_dir / prompts_raw
    )

    generate_raw = sec.get("generate", True)  # type: ignore[arg-type]
    if not isinstance(generate_raw, bool):
        msg = f"[surface].generate must be a boolean, got {type(generate_raw).__name__}"
        raise TypeError(msg)

    max_tokens_raw = sec.get("max_tokens", 20)  # type: ignore[arg-type]
    if not isinstance(max_tokens_raw, int):
        msg = (
            f"[surface].max_tokens must be an integer,"
            f" got {type(max_tokens_raw).__name__}"
        )
        raise TypeError(msg)

    return SurfaceConfig(
        prompts_path=prompts_path,
        generate=generate_raw,
        max_tokens=max_tokens_raw,
    )


def _parse_detect(raw: TomlDict) -> DetectConfig | None:
    """Parse the optional [detect] section into a DetectConfig.

    Returns None if the section is absent.
    """
    sec = raw.get("detect")
    if sec is None:
        return None
    if not isinstance(sec, dict):
        msg = f"[detect] must be a table, got {type(sec).__name__}"
        raise TypeError(msg)

    mode_raw = sec.get("mode", "full")  # type: ignore[arg-type]
    if not isinstance(mode_raw, str):
        msg = f"[detect].mode must be a string, got {type(mode_raw).__name__}"
        raise TypeError(msg)
    valid_modes = ("fast", "probe", "full")
    if mode_raw not in valid_modes:
        msg = (
            f"[detect].mode must be one of {valid_modes!r},"
            f" got {mode_raw!r}"
        )
        raise ValueError(msg)

    top_k_raw = sec.get("top_k", 5)  # type: ignore[arg-type]
    if not isinstance(top_k_raw, int):
        msg = f"[detect].top_k must be an integer, got {type(top_k_raw).__name__}"
        raise TypeError(msg)

    clip_quantile_raw = sec.get("clip_quantile", 0.0)  # type: ignore[arg-type]
    if not isinstance(clip_quantile_raw, int | float):
        msg = (
            f"[detect].clip_quantile must be a number,"
            f" got {type(clip_quantile_raw).__name__}"
        )
        raise TypeError(msg)

    alpha_raw = sec.get("alpha", 1.0)  # type: ignore[arg-type]
    if not isinstance(alpha_raw, int | float):
        msg = f"[detect].alpha must be a number, got {type(alpha_raw).__name__}"
        raise TypeError(msg)

    max_tokens_raw = sec.get("max_tokens", 100)  # type: ignore[arg-type]
    if not isinstance(max_tokens_raw, int):
        msg = (
            f"[detect].max_tokens must be an integer,"
            f" got {type(max_tokens_raw).__name__}"
        )
        raise TypeError(msg)

    return DetectConfig(
        mode=mode_raw,
        top_k=int(top_k_raw),
        clip_quantile=float(clip_quantile_raw),
        alpha=float(alpha_raw),
        max_tokens=max_tokens_raw,
    )


def _parse_cut(raw: TomlDict) -> CutConfig:
    """Parse the [cut] section into a CutConfig."""
    layers_raw = raw.get("layers")
    layers: list[int] | None
    if layers_raw is None or layers_raw == "auto":
        layers = None
    elif isinstance(layers_raw, list):
        layers = [int(x) for x in layers_raw if isinstance(x, int | float)]
    else:
        msg = f"[cut].layers must be 'auto' or a list of ints, got {layers_raw!r}"
        raise TypeError(msg)

    alpha_raw = raw.get("alpha", 1.0)
    if not isinstance(alpha_raw, int | float):
        msg = f"[cut].alpha must be a number, got {type(alpha_raw).__name__}"
        raise TypeError(msg)

    layer_strategy_raw = raw.get("layer_strategy", "all")
    if not isinstance(layer_strategy_raw, str):
        msg = (
            f"[cut].layer_strategy must be a string,"
            f" got {type(layer_strategy_raw).__name__}"
        )
        raise TypeError(msg)
    valid_strategies = ("all", "above_median", "top_k")
    if layer_strategy_raw not in valid_strategies:
        msg = (
            f"[cut].layer_strategy must be one of {valid_strategies!r},"
            f" got {layer_strategy_raw!r}"
        )
        raise ValueError(msg)

    layer_top_k_raw = raw.get("layer_top_k", 10)
    if not isinstance(layer_top_k_raw, int | float):
        msg = (
            f"[cut].layer_top_k must be an integer,"
            f" got {type(layer_top_k_raw).__name__}"
        )
        raise TypeError(msg)

    layer_weights_raw = raw.get("layer_weights")
    layer_weights: list[float] | None = None
    if layer_weights_raw is not None:
        if not isinstance(layer_weights_raw, list):
            msg = (
                f"[cut].layer_weights must be a list of numbers,"
                f" got {type(layer_weights_raw).__name__}"
            )
            raise TypeError(msg)
        layer_weights = [
            float(x)  # type: ignore[arg-type]
            for x in layer_weights_raw
        ]

    sparsity_raw = raw.get("sparsity", 0.0)
    if not isinstance(sparsity_raw, int | float):
        msg = (
            f"[cut].sparsity must be a number,"
            f" got {type(sparsity_raw).__name__}"
        )
        raise TypeError(msg)
    sparsity = float(sparsity_raw)
    if not 0.0 <= sparsity < 1.0:
        msg = f"[cut].sparsity must be in [0.0, 1.0), got {sparsity}"
        raise ValueError(msg)

    dbdi_target_raw = raw.get("dbdi_target", "red")
    if not isinstance(dbdi_target_raw, str):
        msg = (
            f"[cut].dbdi_target must be a string,"
            f" got {type(dbdi_target_raw).__name__}"
        )
        raise TypeError(msg)
    valid_dbdi_targets = ("red", "hdd", "both")
    if dbdi_target_raw not in valid_dbdi_targets:
        msg = (
            f"[cut].dbdi_target must be one of {valid_dbdi_targets!r},"
            f" got {dbdi_target_raw!r}"
        )
        raise ValueError(msg)

    false_refusal_ortho_raw = raw.get("false_refusal_ortho", False)
    if not isinstance(false_refusal_ortho_raw, bool):
        msg = (
            f"[cut].false_refusal_ortho must be a boolean,"
            f" got {type(false_refusal_ortho_raw).__name__}"
        )
        raise TypeError(msg)

    return CutConfig(
        alpha=float(alpha_raw),
        layers=layers,
        norm_preserve=bool(raw.get("norm_preserve", False)),
        biprojected=bool(raw.get("biprojected", False)),
        layer_strategy=layer_strategy_raw,
        layer_top_k=int(layer_top_k_raw),
        layer_weights=layer_weights,
        sparsity=sparsity,
        dbdi_target=dbdi_target_raw,
        false_refusal_ortho=false_refusal_ortho_raw,
    )


def _parse_measure(raw: TomlDict) -> MeasureConfig:
    """Parse the optional [measure] section into a MeasureConfig."""
    mode_raw = raw.get("mode", "direction")
    if not isinstance(mode_raw, str):
        msg = f"[measure].mode must be a string, got {type(mode_raw).__name__}"
        raise TypeError(msg)
    if mode_raw not in ("direction", "subspace", "dbdi"):
        msg = (
            f"[measure].mode must be 'direction', 'subspace', or 'dbdi',"
            f" got {mode_raw!r}"
        )
        raise ValueError(msg)

    top_k_raw = raw.get("top_k", 5)
    if not isinstance(top_k_raw, int):
        msg = f"[measure].top_k must be an integer, got {type(top_k_raw).__name__}"
        raise TypeError(msg)

    clip_quantile_raw = raw.get("clip_quantile", 0.0)
    if not isinstance(clip_quantile_raw, int | float):
        msg = (
            f"[measure].clip_quantile must be a number,"
            f" got {type(clip_quantile_raw).__name__}"
        )
        raise TypeError(msg)
    clip_quantile = float(clip_quantile_raw)
    if not 0.0 <= clip_quantile < 0.5:
        msg = f"[measure].clip_quantile must be in [0.0, 0.5), got {clip_quantile}"
        raise ValueError(msg)

    return MeasureConfig(
        mode=mode_raw, top_k=int(top_k_raw), clip_quantile=clip_quantile,
    )
