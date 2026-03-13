"""TOML configuration loader and path resolution."""

import tomllib
from pathlib import Path
from typing import cast

from vauban.config._parse_meta import parse_meta
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

    # Detect standalone api_eval: [api_eval] present with token_text set.
    # Standalone mode needs no local model, so [model] and [data] are optional.
    _standalone = _is_standalone_api_eval(raw)

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
    sss_config = parsed_sections.sss
    awareness_config = parsed_sections.awareness
    eval_config = parsed_sections.eval
    api_eval_config = parsed_sections.api_eval
    svf_config = parsed_sections.svf
    compose_optimize_config = parsed_sections.compose_optimize

    environment_config = parsed_sections.environment
    scan_config = parsed_sections.scan
    policy_config = parsed_sections.policy
    intent_config = parsed_sections.intent
    defend_config = parsed_sections.defend
    circuit_config = parsed_sections.circuit
    features_config = parsed_sections.features
    linear_probe_config = parsed_sections.linear_probe
    fusion_config = parsed_sections.fusion
    repbend_config = parsed_sections.repbend
    lora_export_config = parsed_sections.lora_export
    lora_load_config = parsed_sections.lora_load
    lora_analysis_config = parsed_sections.lora_analysis
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
        sss=sss_config,
        awareness=awareness_config,
        cast=cast_config,
        svf=svf_config,
        compose_optimize=compose_optimize_config,
        environment=environment_config,
        scan=scan_config,
        policy=policy_config,
        intent=intent_config,
        defend=defend_config,
        circuit=circuit_config,
        features=features_config,
        linear_probe=linear_probe_config,
        fusion=fusion_config,
        repbend=repbend_config,
        lora_export=lora_export_config,
        lora_load=lora_load_config,
        lora_analysis=lora_analysis_config,
        eval=eval_config,
        api_eval=api_eval_config,
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


def _is_standalone_api_eval(raw: TomlDict) -> bool:
    """Check whether the raw TOML represents a standalone api_eval config.

    Returns True when ``[api_eval]`` is present and ``token_text`` is set,
    meaning the pipeline can run without a local model.
    """
    sec = raw.get("api_eval")
    if not isinstance(sec, dict):
        return False
    return isinstance(sec.get("token_text"), str) and bool(sec["token_text"])
