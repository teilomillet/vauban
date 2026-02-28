"""TOML configuration loader and path resolution."""

import tomllib
from pathlib import Path
from typing import cast

from vauban.config._registry import (
    ConfigParseContext,
    parse_registered_section,
    parse_registered_sections,
)
from vauban.config._types import TomlDict
from vauban.measure import default_prompt_paths
from vauban.types import DatasetRef, DepthConfig, PipelineConfig


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
        msg = (
            f"[model].path must be a string,"
            f" got {type(model_path_raw).__name__}"
        )
        raise TypeError(msg)
    model_path: str = model_path_raw

    parse_context = ConfigParseContext(base_dir=base_dir, raw=raw)
    depth_config = cast(
        "DepthConfig | None",
        parse_registered_section(parse_context, "depth"),
    )
    harmful_path, harmless_path = _resolve_data_paths(
        base_dir, raw, depth_only=depth_config is not None,
    )

    parsed_sections = parse_registered_sections(
        parse_context,
        depth_override=depth_config,
    )
    cast_config = parsed_sections.cast
    cut_config = parsed_sections.cut
    measure_config = parsed_sections.measure
    surface_config = parsed_sections.surface
    detect_config = parsed_sections.detect
    optimize_config = parsed_sections.optimize
    softprompt_config = parsed_sections.softprompt
    sic_config = parsed_sections.sic
    probe_config = parsed_sections.probe
    steer_config = parsed_sections.steer
    eval_config = parsed_sections.eval
    api_eval_config = parsed_sections.api_eval

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

    # -- backend --
    from vauban._backend import SUPPORTED_BACKENDS

    backend = "mlx"
    backend_raw = raw.get("backend")
    if backend_raw is not None:
        if not isinstance(backend_raw, str):
            msg = (
                f"backend must be a string,"
                f" got {type(backend_raw).__name__}"
            )
            raise TypeError(msg)
        if backend_raw not in SUPPORTED_BACKENDS:
            msg = (
                f"backend must be one of {sorted(SUPPORTED_BACKENDS)},"
                f" got {backend_raw!r}"
            )
            raise ValueError(msg)
        backend = backend_raw

    # -- verbose --
    verbose = True
    verbose_raw = raw.get("verbose")
    if verbose_raw is not None:
        if not isinstance(verbose_raw, bool):
            msg = (
                f"verbose must be a boolean,"
                f" got {type(verbose_raw).__name__}"
            )
            raise TypeError(msg)
        verbose = verbose_raw

    return PipelineConfig(
        model_path=model_path,
        harmful_path=harmful_path,
        harmless_path=harmless_path,
        backend=backend,
        cut=cut_config,
        measure=measure_config,
        surface=surface_config,
        detect=detect_config,
        optimize=optimize_config,
        softprompt=softprompt_config,
        sic=sic_config,
        depth=depth_config,
        probe=probe_config,
        steer=steer_config,
        cast=cast_config,
        eval=eval_config,
        api_eval=api_eval_config,
        output_dir=output_dir,
        borderline_path=borderline_path,
        verbose=verbose,
    )


def _resolve_data_paths(
    base_dir: Path,
    raw: TomlDict,
    *,
    depth_only: bool = False,
) -> tuple[Path | DatasetRef, Path | DatasetRef]:
    """Resolve harmful/harmless paths.

    Supports 'default', local files, and HF datasets.
    When *depth_only* is True and [data] is absent, falls back to bundled
    defaults since depth mode never uses the data paths.
    """
    sec = raw.get("data")
    if not isinstance(sec, dict):
        if depth_only:
            harmful, harmless = default_prompt_paths()
            return harmful, harmless
        msg = (
            "Config must have [data] section"
            " with 'harmful' and 'harmless' keys"
        )
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
    """Resolve optional [data].borderline path.

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
            msg = (
                f"[data.{name}] table must have"
                " 'hf' key with a string value"
            )
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
            limit=(
                int(limit_raw)
                if isinstance(limit_raw, int | float)
                else None
            ),
        )

    msg = (
        f"[data].{name} must be a string or table,"
        f" got {type(value).__name__}"
    )
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
