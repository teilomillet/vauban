"""Typed parser registry for TOML config sections."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from vauban.config._parse_api_eval import _parse_api_eval
from vauban.config._parse_awareness import _parse_awareness
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
from vauban.config._parse_flywheel import _parse_flywheel
from vauban.config._parse_fusion import _parse_fusion
from vauban.config._parse_intent import _parse_intent
from vauban.config._parse_jailbreak import _parse_jailbreak
from vauban.config._parse_linear_probe import _parse_linear_probe
from vauban.config._parse_lora_analysis import _parse_lora_analysis
from vauban.config._parse_lora_export import _parse_lora_export
from vauban.config._parse_lora_load import _parse_lora_load
from vauban.config._parse_measure import _parse_measure
from vauban.config._parse_optimize import _parse_optimize
from vauban.config._parse_policy import _parse_policy
from vauban.config._parse_probe import _parse_probe
from vauban.config._parse_remote import _parse_remote
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
    AwarenessConfig,
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
    FlywheelConfig,
    FusionConfig,
    IntentConfig,
    JailbreakConfig,
    LinearProbeConfig,
    LoraAnalysisConfig,
    LoraExportConfig,
    LoraLoadConfig,
    MeasureConfig,
    OptimizeConfig,
    PolicyConfig,
    ProbeConfig,
    RemoteConfig,
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
    | FlywheelConfig
    | FusionConfig
    | IntentConfig
    | JailbreakConfig
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
    | SoftPromptConfig
    | SICConfig
    | ProbeConfig
    | RemoteConfig
    | SSSConfig
    | AwarenessConfig
    | SteerConfig
    | EvalConfig
    | SVFConfig
)

type SectionParser[T] = Callable[..., T]

# How the parser receives its arguments from ConfigParseContext.
type CallConvention = Literal[
    "raw",             # parser(raw)
    "base_dir_raw",    # parser(base_dir, raw)
    "raw_base_dir",    # parser(raw, base_dir)
    "section_table",   # extract raw[section], parser(section_dict)
]


@dataclass(frozen=True, slots=True)
class SectionParseSpec[T]:
    """Typed parser registration for a TOML section."""

    section: str
    target_field: str
    parser: SectionParser[T]
    order: int
    call: CallConvention = "raw"


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
    awareness: AwarenessConfig | None
    eval: EvalConfig
    api_eval: ApiEvalConfig | None
    svf: SVFConfig | None
    environment: EnvironmentConfig | None
    scan: ScanConfig | None
    policy: PolicyConfig | None
    intent: IntentConfig | None
    jailbreak: JailbreakConfig | None
    defend: DefenseStackConfig | None
    circuit: CircuitConfig | None
    features: FeaturesConfig | None
    linear_probe: LinearProbeConfig | None
    flywheel: FlywheelConfig | None
    remote: RemoteConfig | None
    fusion: FusionConfig | None
    repbend: RepBendConfig | None
    lora_export: LoraExportConfig | None
    lora_load: LoraLoadConfig | None
    lora_analysis: LoraAnalysisConfig | None


@dataclass(frozen=True, slots=True)
class _DepthOverrideSentinel:
    """Internal sentinel for optional depth override."""


_DEPTH_OVERRIDE_UNSET = _DepthOverrideSentinel()


def _call_parser(
    spec: SectionParseSpec[_SectionParserResult],
    context: ConfigParseContext,
) -> _SectionParserResult:
    """Invoke a section parser using its declared calling convention.

    Resolves the parser via module globals so that ``monkeypatch.setattr``
    on the registry module works in tests.
    """
    # Resolve through module globals to support monkeypatching.
    parser = globals().get(spec.parser.__name__, spec.parser)
    if spec.call == "raw":
        return parser(context.raw)
    if spec.call == "base_dir_raw":
        return parser(context.base_dir, context.raw)
    if spec.call == "raw_base_dir":
        return parser(context.raw, context.base_dir)
    # section_table: extract the section dict, default to empty
    sec_raw = context.raw.get(spec.section)
    sec = (
        cast("TomlDict", sec_raw)
        if isinstance(sec_raw, dict)
        else cast("TomlDict", {})
    )
    return parser(sec)


SECTION_PARSE_SPECS: tuple[SectionParseSpec[_SectionParserResult], ...] = (
    SectionParseSpec("lora", "lora_load", _parse_lora_load, 5),
    SectionParseSpec("depth", "depth", _parse_depth, 10),
    SectionParseSpec("cast", "cast", _parse_cast, 20),
    SectionParseSpec("cut", "cut", _parse_cut, 30, call="section_table"),
    SectionParseSpec("measure", "measure", _parse_measure, 40, call="section_table"),
    SectionParseSpec("surface", "surface", _parse_surface, 50, call="base_dir_raw"),
    SectionParseSpec("detect", "detect", _parse_detect, 60),
    SectionParseSpec("optimize", "optimize", _parse_optimize, 70),
    SectionParseSpec(
        "compose_optimize", "compose_optimize",
        _parse_compose_optimize, 75, call="base_dir_raw",
    ),
    SectionParseSpec(
        "softprompt", "softprompt",
        _parse_softprompt, 80, call="raw_base_dir",
    ),
    SectionParseSpec("sic", "sic", _parse_sic, 90),
    SectionParseSpec("svf", "svf", _parse_svf, 95, call="base_dir_raw"),
    SectionParseSpec("probe", "probe", _parse_probe, 100),
    SectionParseSpec("steer", "steer", _parse_steer, 110),
    SectionParseSpec("sss", "sss", _parse_sss, 112),
    SectionParseSpec("awareness", "awareness", _parse_awareness, 113),
    SectionParseSpec("eval", "eval", _parse_eval, 120, call="base_dir_raw"),
    SectionParseSpec("api_eval", "api_eval", _parse_api_eval, 130),
    SectionParseSpec(
        "environment", "environment",
        _parse_environment, 135,
    ),
    SectionParseSpec("scan", "scan", _parse_scan, 140),
    SectionParseSpec("policy", "policy", _parse_policy, 145),
    SectionParseSpec("intent", "intent", _parse_intent, 150),
    SectionParseSpec("jailbreak", "jailbreak", _parse_jailbreak, 152),
    SectionParseSpec("defend", "defend", _parse_defend, 155),
    SectionParseSpec("circuit", "circuit", _parse_circuit, 160),
    SectionParseSpec(
        "features", "features",
        _parse_features, 165, call="base_dir_raw",
    ),
    SectionParseSpec(
        "linear_probe", "linear_probe",
        _parse_linear_probe, 170,
    ),
    SectionParseSpec(
        "fusion", "fusion",
        _parse_fusion, 175, call="base_dir_raw",
    ),
    SectionParseSpec("repbend", "repbend", _parse_repbend, 180),
    SectionParseSpec(
        "lora_export", "lora_export", _parse_lora_export, 185,
    ),
    SectionParseSpec(
        "lora_analysis", "lora_analysis", _parse_lora_analysis, 190,
    ),
    SectionParseSpec("flywheel", "flywheel", _parse_flywheel, 195),
    SectionParseSpec("remote", "remote", _parse_remote, 200),
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
    return _call_parser(spec, context)


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
        parsed[spec.target_field] = _call_parser(spec, context)

    return ParsedSectionValues(**parsed)  # type: ignore[arg-type]
