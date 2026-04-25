# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""TOML configuration loader and path resolution."""

import tomllib
from pathlib import Path
from typing import cast

from vauban.config._parse_meta import parse_meta
from vauban.config._registry import (
    SECTION_PARSE_SPECS,
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

    # Detect standalone modes that need no local model.
    # [model] and [data] are optional for these.
    _standalone = (
        _is_standalone_api_eval(raw)
        or _is_standalone_remote(raw)
        or _is_standalone_ai_act(raw)
        or _is_standalone_benchmark(raw)
        or _is_standalone_behavior_report(raw)
        or _is_standalone_token_audit(raw)
    )

    model_section = raw.get("model")
    if _standalone:
        model_path = ""
        if isinstance(model_section, dict):
            _mp = model_section.get("path")
            if isinstance(_mp, str):
                model_path = _mp
    else:
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
        model_path = model_path_raw

    parse_context = ConfigParseContext(base_dir=base_dir, raw=raw)
    depth_config = cast(
        "DepthConfig | None",
        parse_registered_section(parse_context, "depth"),
    )
    harmful_path, harmless_path = _resolve_data_paths(
        base_dir, raw, depth_only=depth_config is not None or _standalone,
    )

    parsed_sections = parse_registered_sections(
        parse_context,
        depth_override=depth_config,
    )

    output_section = raw.get("output")
    output_dir_str = "output"
    if isinstance(output_section, dict):
        dir_raw = output_section.get("dir")
        if isinstance(dir_raw, str):
            output_dir_str = dir_raw
    output_dir = base_dir / output_dir_str

    meta_config = parse_meta(raw, config_path=path)

    borderline_path = _resolve_borderline_path(base_dir, raw)

    # Validate: false_refusal_ortho requires borderline_path
    if parsed_sections.cut.false_refusal_ortho and borderline_path is None:
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

    # Build section kwargs from registry specs — each spec target_field
    # maps 1:1 to a PipelineConfig field.  Adding a new spec is the only
    # step needed; no per-field lines here.
    section_kwargs: dict[str, object] = {
        spec.target_field: getattr(parsed_sections, spec.target_field)
        for spec in SECTION_PARSE_SPECS
    }

    return PipelineConfig(
        model_path=model_path,
        harmful_path=harmful_path,
        harmless_path=harmless_path,
        backend=backend,
        **section_kwargs,  # type: ignore[arg-type]
        meta=meta_config,
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

    sec_typed = cast("TomlDict", sec)
    harmful_raw = sec_typed.get("harmful")
    harmless_raw = sec_typed.get("harmless")

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
    sec_typed = cast("TomlDict", sec)
    borderline_raw = sec_typed.get("borderline")
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

        from vauban.benchmarks import (
            KNOWN_BENCHMARKS,
            generate_infix_prompts,
            resolve_benchmark,
        )

        if value in KNOWN_BENCHMARKS:
            return resolve_benchmark(value)
        # Only treat as infix sentinel for bare names (no path
        # separators or file extensions) to avoid hijacking
        # filenames like "harmbench_infix.jsonl".
        if (
            value.endswith("_infix")
            and "/" not in value
            and "." not in value
        ):
            base_name = value[:-6]
            if base_name in KNOWN_BENCHMARKS:
                return generate_infix_prompts(base_name)

        if value.startswith("hf:"):
            repo_id = value[3:]
            if not repo_id:
                msg = f"[data].{name}: empty repo_id after 'hf:' prefix"
                raise ValueError(msg)
            return DatasetRef(repo_id=repo_id)
        return base_dir / value

    if isinstance(value, dict):
        from vauban.config._parse_helpers import SectionReader

        reader = SectionReader(f"[data.{name}]", cast("TomlDict", value))
        hf_repo = reader.string("hf")
        split = reader.string("split", default="train")
        column = reader.string("column", default="prompt")
        config_val = reader.optional_string("config")
        limit = reader.optional_integer("limit")
        return DatasetRef(
            repo_id=hf_repo,
            split=split,
            column=column,
            config=config_val,
            limit=limit,
        )

    msg = (
        f"[data].{name} must be a string or table,"
        f" got {type(value).__name__}"
    )
    raise TypeError(msg)


def _is_standalone_remote(raw: TomlDict) -> bool:
    """Check whether the raw TOML represents a standalone remote config.

    Returns True when the ``remote`` top-level key is present, even if the
    section is malformed, so the remote parser can surface the specific
    validation error before model/data checks run.
    """
    return "remote" in raw


def _is_standalone_api_eval(raw: TomlDict) -> bool:
    """Check whether the raw TOML represents a standalone api_eval config.

    Returns True when ``[api_eval]`` is present and ``token_text`` is set,
    meaning the pipeline can run without a local model.
    """
    sec = raw.get("api_eval")
    if not isinstance(sec, dict):
        return False
    api_eval = cast("TomlDict", sec)
    token_text = api_eval.get("token_text")
    return isinstance(token_text, str) and bool(token_text)


def _is_standalone_ai_act(raw: TomlDict) -> bool:
    """Check whether the raw TOML represents a standalone AI Act report."""
    return "ai_act" in raw


def _is_standalone_benchmark(raw: TomlDict) -> bool:
    """Check whether the raw TOML represents a standalone benchmark."""
    return "benchmark" in raw


def _is_standalone_behavior_report(raw: TomlDict) -> bool:
    """Check whether the raw TOML represents a standalone behavior report."""
    return "behavior_report" in raw


def _is_standalone_token_audit(raw: TomlDict) -> bool:
    """Check whether the raw TOML represents a standalone token audit."""
    return "token_audit" in raw
