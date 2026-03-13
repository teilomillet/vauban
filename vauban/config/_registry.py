"""Typed parser registry for TOML config sections."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from vauban.config._parse_api_eval import _parse_api_eval
from vauban.config._parse_cast import _parse_cast
from vauban.config._parse_circuit import _parse_circuit
from vauban.config._parse_compose_optimize import _parse_compose_optimize
from vauban.config._parse_cut import _parse_cut
from vauban.config._parse_defend import _parse_defend
from vauban.config._parse_depth import _parse_depth
from vauban.config._parse_detect import _parse_detect
from vauban.config._parse_environment import _parse_environment
from vauban.config._parse_eval import _parse_eval
from vauban.config._parse_features import _parse_features
from vauban.config._parse_fusion import _parse_fusion
from vauban.config._parse_intent import _parse_intent
from vauban.config._parse_linear_probe import _parse_linear_probe
from vauban.config._parse_lora_analysis import _parse_lora_analysis
from vauban.config._parse_lora_export import _parse_lora_export
from vauban.config._parse_lora_load import _parse_lora_load
from vauban.config._parse_measure import _parse_measure
from vauban.config._parse_optimize import _parse_optimize
from vauban.config._parse_policy import _parse_policy
from vauban.config._parse_probe import _parse_probe
from vauban.config._parse_repbend import _parse_repbend
from vauban.config._parse_scan import _parse_scan
from vauban.config._parse_sic import _parse_sic
from vauban.config._parse_softprompt import _parse_softprompt
from vauban.config._parse_sss import _parse_sss
from vauban.config._parse_steer import _parse_steer
from vauban.config._parse_surface import _parse_surface
from vauban.config._parse_svf import _parse_svf
from vauban.config._types import TomlDict
from vauban.types import (
    ApiEvalConfig,
    CastConfig,
    CircuitConfig,
    ComposeOptimizeConfig,
    CutConfig,
    DefenseStackConfig,
    DepthConfig,
    DetectConfig,
    EnvironmentConfig,
    EvalConfig,
    FeaturesConfig,
    FusionConfig,
    IntentConfig,
    LinearProbeConfig,
    LoraAnalysisConfig,
    LoraExportConfig,
    LoraLoadConfig,
    MeasureConfig,
    OptimizeConfig,
    PolicyConfig,
    ProbeConfig,
    RepBendConfig,
    ScanConfig,
    SICConfig,
    SoftPromptConfig,
    SSSConfig,
    SteerConfig,
    SurfaceConfig,
    SVFConfig,
)


@dataclass(frozen=True, slots=True)
class ConfigParseContext:
    """Shared parse context for section adapters."""

    base_dir: Path
    raw: TomlDict


type _SectionParserResult = (
    DepthConfig
    | None
    | ApiEvalConfig
    | CastConfig
    | CircuitConfig
    | ComposeOptimizeConfig
    | CutConfig
    | DefenseStackConfig
    | EnvironmentConfig
    | FeaturesConfig
    | FusionConfig
    | IntentConfig
    | LinearProbeConfig
    | LoraAnalysisConfig
    | LoraExportConfig
    | LoraLoadConfig
    | MeasureConfig
    | OptimizeConfig
    | PolicyConfig
    | RepBendConfig
    | ScanConfig
    | SurfaceConfig
    | DetectConfig
    | OptimizeConfig
    | SoftPromptConfig
    | SICConfig
    | ProbeConfig
    | SSSConfig
    | SteerConfig
    | EvalConfig
    | SVFConfig
)

type SectionParser[T] = Callable[[ConfigParseContext], T]


@dataclass(frozen=True, slots=True)
class SectionParseSpec[T]:
    """Typed parser registration for a TOML section."""

    section: str
    target_field: str
    parser: SectionParser[T]
    order: int


@dataclass(frozen=True, slots=True)
class ParsedSectionValues:
    """All parsed config sections assembled from the parser registry."""

    depth: DepthConfig | None
    cast: CastConfig | None
    cut: CutConfig
    measure: MeasureConfig
    surface: SurfaceConfig | None
    detect: DetectConfig | None
    optimize: OptimizeConfig | None
    compose_optimize: ComposeOptimizeConfig | None
    softprompt: SoftPromptConfig | None
    sic: SICConfig | None
    probe: ProbeConfig | None
    steer: SteerConfig | None
    sss: SSSConfig | None
    eval: EvalConfig
    api_eval: ApiEvalConfig | None
    svf: SVFConfig | None
    environment: EnvironmentConfig | None
    scan: ScanConfig | None
    policy: PolicyConfig | None
    intent: IntentConfig | None
    defend: DefenseStackConfig | None
    circuit: CircuitConfig | None
    features: FeaturesConfig | None
    linear_probe: LinearProbeConfig | None
    fusion: FusionConfig | None
    repbend: RepBendConfig | None
    lora_export: LoraExportConfig | None
    lora_load: LoraLoadConfig | None
    lora_analysis: LoraAnalysisConfig | None


@dataclass(frozen=True, slots=True)
class _DepthOverrideSentinel:
    """Internal sentinel for optional depth override."""


_DEPTH_OVERRIDE_UNSET = _DepthOverrideSentinel()


def _parse_depth_adapter(context: ConfigParseContext) -> DepthConfig | None:
    """Adapter for parsers that accept the full raw config mapping."""
    return _parse_depth(context.raw)


def _parse_cast_adapter(context: ConfigParseContext) -> CastConfig | None:
    """Adapter for parsers that accept the full raw config mapping."""
    return _parse_cast(context.raw)


def _parse_cut_adapter(context: ConfigParseContext) -> CutConfig:
    """Adapter for parsers that accept a section table mapping."""
    cut_raw = context.raw.get("cut")
    cut_section = (
        cast("TomlDict", cut_raw)
        if isinstance(cut_raw, dict)
        else cast("TomlDict", {})
    )
    return _parse_cut(cut_section)


def _parse_measure_adapter(context: ConfigParseContext) -> MeasureConfig:
    """Adapter for parsers that accept a section table mapping."""
    measure_raw = context.raw.get("measure")
    measure_section = (
        cast("TomlDict", measure_raw)
        if isinstance(measure_raw, dict)
        else cast("TomlDict", {})
    )
    return _parse_measure(measure_section)


def _parse_surface_adapter(context: ConfigParseContext) -> SurfaceConfig | None:
    """Adapter for parsers that require base_dir + raw config mapping."""
    return _parse_surface(context.base_dir, context.raw)


def _parse_detect_adapter(context: ConfigParseContext) -> DetectConfig | None:
    """Adapter for parsers that accept the full raw config mapping."""
    return _parse_detect(context.raw)


def _parse_optimize_adapter(context: ConfigParseContext) -> OptimizeConfig | None:
    """Adapter for parsers that accept the full raw config mapping."""
    return _parse_optimize(context.raw)


def _parse_compose_optimize_adapter(
    context: ConfigParseContext,
) -> ComposeOptimizeConfig | None:
    """Adapter for parsers that require base_dir + raw config mapping."""
    return _parse_compose_optimize(context.base_dir, context.raw)


def _parse_softprompt_adapter(
    context: ConfigParseContext,
) -> SoftPromptConfig | None:
    """Adapter for parsers that accept the full raw config mapping."""
    return _parse_softprompt(context.raw, context.base_dir)


def _parse_sic_adapter(context: ConfigParseContext) -> SICConfig | None:
    """Adapter for parsers that accept the full raw config mapping."""
    return _parse_sic(context.raw)


def _parse_probe_adapter(context: ConfigParseContext) -> ProbeConfig | None:
    """Adapter for parsers that accept the full raw config mapping."""
    return _parse_probe(context.raw)


def _parse_steer_adapter(context: ConfigParseContext) -> SteerConfig | None:
    """Adapter for parsers that accept the full raw config mapping."""
    return _parse_steer(context.raw)


def _parse_sss_adapter(context: ConfigParseContext) -> SSSConfig | None:
    """Adapter for parsers that accept the full raw config mapping."""
    return _parse_sss(context.raw)


def _parse_eval_adapter(context: ConfigParseContext) -> EvalConfig:
    """Adapter for parsers that require base_dir + raw config mapping."""
    return _parse_eval(context.base_dir, context.raw)


def _parse_api_eval_adapter(
    context: ConfigParseContext,
) -> ApiEvalConfig | None:
    """Adapter for parsers that accept the full raw config mapping."""
    return _parse_api_eval(context.raw)


def _parse_svf_adapter(
    context: ConfigParseContext,
) -> SVFConfig | None:
    """Adapter for parsers that require base_dir + raw config mapping."""
    return _parse_svf(context.base_dir, context.raw)


def _parse_environment_adapter(
    context: ConfigParseContext,
) -> EnvironmentConfig | None:
    """Adapter for parsers that accept the full raw config mapping."""
    return _parse_environment(context.raw)


def _parse_scan_adapter(
    context: ConfigParseContext,
) -> ScanConfig | None:
    """Adapter for parsers that accept the full raw config mapping."""
    return _parse_scan(context.raw)


def _parse_policy_adapter(
    context: ConfigParseContext,
) -> PolicyConfig | None:
    """Adapter for parsers that accept the full raw config mapping."""
    return _parse_policy(context.raw)


def _parse_intent_adapter(
    context: ConfigParseContext,
) -> IntentConfig | None:
    """Adapter for parsers that accept the full raw config mapping."""
    return _parse_intent(context.raw)


def _parse_defend_adapter(
    context: ConfigParseContext,
) -> DefenseStackConfig | None:
    """Adapter for parsers that accept the full raw config mapping."""
    return _parse_defend(context.raw)


def _parse_circuit_adapter(
    context: ConfigParseContext,
) -> CircuitConfig | None:
    """Adapter for parsers that accept the full raw config mapping."""
    return _parse_circuit(context.raw)


def _parse_features_adapter(
    context: ConfigParseContext,
) -> FeaturesConfig | None:
    """Adapter for parsers that require base_dir + raw config mapping."""
    return _parse_features(context.base_dir, context.raw)


def _parse_linear_probe_adapter(
    context: ConfigParseContext,
) -> LinearProbeConfig | None:
    """Adapter for parsers that accept the full raw config mapping."""
    return _parse_linear_probe(context.raw)


def _parse_fusion_adapter(
    context: ConfigParseContext,
) -> FusionConfig | None:
    """Adapter for parsers that require base_dir + raw config mapping."""
    return _parse_fusion(context.base_dir, context.raw)


def _parse_repbend_adapter(
    context: ConfigParseContext,
) -> RepBendConfig | None:
    """Adapter for parsers that accept the full raw config mapping."""
    return _parse_repbend(context.raw)


def _parse_lora_export_adapter(
    context: ConfigParseContext,
) -> LoraExportConfig | None:
    """Adapter for parsers that accept the full raw config mapping."""
    return _parse_lora_export(context.raw)


def _parse_lora_load_adapter(
    context: ConfigParseContext,
) -> LoraLoadConfig | None:
    """Adapter for parsers that accept the full raw config mapping."""
    return _parse_lora_load(context.raw)


def _parse_lora_analysis_adapter(
    context: ConfigParseContext,
) -> LoraAnalysisConfig | None:
    """Adapter for parsers that accept the full raw config mapping."""
    return _parse_lora_analysis(context.raw)


SECTION_PARSE_SPECS: tuple[SectionParseSpec[_SectionParserResult], ...] = (
    SectionParseSpec("lora", "lora_load", _parse_lora_load_adapter, 5),
    SectionParseSpec("depth", "depth", _parse_depth_adapter, 10),
    SectionParseSpec("cast", "cast", _parse_cast_adapter, 20),
    SectionParseSpec("cut", "cut", _parse_cut_adapter, 30),
    SectionParseSpec("measure", "measure", _parse_measure_adapter, 40),
    SectionParseSpec("surface", "surface", _parse_surface_adapter, 50),
    SectionParseSpec("detect", "detect", _parse_detect_adapter, 60),
    SectionParseSpec("optimize", "optimize", _parse_optimize_adapter, 70),
    SectionParseSpec(
        "compose_optimize", "compose_optimize",
        _parse_compose_optimize_adapter, 75,
    ),
    SectionParseSpec("softprompt", "softprompt", _parse_softprompt_adapter, 80),
    SectionParseSpec("sic", "sic", _parse_sic_adapter, 90),
    SectionParseSpec("svf", "svf", _parse_svf_adapter, 95),
    SectionParseSpec("probe", "probe", _parse_probe_adapter, 100),
    SectionParseSpec("steer", "steer", _parse_steer_adapter, 110),
    SectionParseSpec("sss", "sss", _parse_sss_adapter, 112),
    SectionParseSpec("eval", "eval", _parse_eval_adapter, 120),
    SectionParseSpec("api_eval", "api_eval", _parse_api_eval_adapter, 130),
    SectionParseSpec(
        "environment", "environment",
        _parse_environment_adapter, 135,
    ),
    SectionParseSpec("scan", "scan", _parse_scan_adapter, 140),
    SectionParseSpec("policy", "policy", _parse_policy_adapter, 145),
    SectionParseSpec("intent", "intent", _parse_intent_adapter, 150),
    SectionParseSpec("defend", "defend", _parse_defend_adapter, 155),
    SectionParseSpec("circuit", "circuit", _parse_circuit_adapter, 160),
    SectionParseSpec("features", "features", _parse_features_adapter, 165),
    SectionParseSpec(
        "linear_probe", "linear_probe",
        _parse_linear_probe_adapter, 170,
    ),
    SectionParseSpec("fusion", "fusion", _parse_fusion_adapter, 175),
    SectionParseSpec("repbend", "repbend", _parse_repbend_adapter, 180),
    SectionParseSpec("lora_export", "lora_export", _parse_lora_export_adapter, 185),
    SectionParseSpec(
        "lora_analysis", "lora_analysis", _parse_lora_analysis_adapter, 190,
    ),
)


_SECTION_PARSE_SPEC_BY_NAME: dict[str, SectionParseSpec[_SectionParserResult]] = {
    spec.section: spec for spec in SECTION_PARSE_SPECS
}


def parse_registered_section(
    context: ConfigParseContext,
    section: str,
) -> _SectionParserResult:
    """Parse one section using the registry."""
    spec = _SECTION_PARSE_SPEC_BY_NAME.get(section)
    if spec is None:
        msg = f"Unknown parser section: {section!r}"
        raise KeyError(msg)
    return spec.parser(context)


def parse_registered_sections(
    context: ConfigParseContext,
    *,
    depth_override: DepthConfig | None | _DepthOverrideSentinel = _DEPTH_OVERRIDE_UNSET,
) -> ParsedSectionValues:
    """Parse all sections in registry order and return typed assembled values."""
    parsed: dict[str, _SectionParserResult] = {}
    for spec in SECTION_PARSE_SPECS:
        if (
            spec.section == "depth"
            and not isinstance(depth_override, _DepthOverrideSentinel)
        ):
            parsed[spec.target_field] = depth_override
            continue
        parsed[spec.target_field] = spec.parser(context)

    return ParsedSectionValues(
        depth=cast("DepthConfig | None", parsed["depth"]),
        cast=cast("CastConfig | None", parsed["cast"]),
        cut=cast("CutConfig", parsed["cut"]),
        measure=cast("MeasureConfig", parsed["measure"]),
        surface=cast("SurfaceConfig | None", parsed["surface"]),
        detect=cast("DetectConfig | None", parsed["detect"]),
        optimize=cast("OptimizeConfig | None", parsed["optimize"]),
        compose_optimize=cast(
            "ComposeOptimizeConfig | None",
            parsed["compose_optimize"],
        ),
        softprompt=cast("SoftPromptConfig | None", parsed["softprompt"]),
        sic=cast("SICConfig | None", parsed["sic"]),
        probe=cast("ProbeConfig | None", parsed["probe"]),
        steer=cast("SteerConfig | None", parsed["steer"]),
        sss=cast("SSSConfig | None", parsed["sss"]),
        eval=cast("EvalConfig", parsed["eval"]),
        api_eval=cast("ApiEvalConfig | None", parsed["api_eval"]),
        svf=cast("SVFConfig | None", parsed["svf"]),
        environment=cast(
            "EnvironmentConfig | None", parsed["environment"],
        ),
        scan=cast("ScanConfig | None", parsed["scan"]),
        policy=cast("PolicyConfig | None", parsed["policy"]),
        intent=cast("IntentConfig | None", parsed["intent"]),
        defend=cast("DefenseStackConfig | None", parsed["defend"]),
        circuit=cast("CircuitConfig | None", parsed["circuit"]),
        features=cast("FeaturesConfig | None", parsed["features"]),
        linear_probe=cast(
            "LinearProbeConfig | None", parsed["linear_probe"],
        ),
        fusion=cast("FusionConfig | None", parsed["fusion"]),
        repbend=cast("RepBendConfig | None", parsed["repbend"]),
        lora_export=cast("LoraExportConfig | None", parsed["lora_export"]),
        lora_load=cast("LoraLoadConfig | None", parsed["lora_load"]),
        lora_analysis=cast("LoraAnalysisConfig | None", parsed["lora_analysis"]),
    )
