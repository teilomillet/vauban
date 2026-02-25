"""Typed parser registry for TOML config sections."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from vauban.config._parse_cut import _parse_cut
from vauban.config._parse_depth import _parse_depth
from vauban.config._parse_detect import _parse_detect
from vauban.config._parse_eval import _parse_eval
from vauban.config._parse_measure import _parse_measure
from vauban.config._parse_optimize import _parse_optimize
from vauban.config._parse_probe import _parse_probe
from vauban.config._parse_sic import _parse_sic
from vauban.config._parse_softprompt import _parse_softprompt
from vauban.config._parse_steer import _parse_steer
from vauban.config._parse_surface import _parse_surface
from vauban.config._types import TomlDict
from vauban.types import (
    CutConfig,
    DepthConfig,
    DetectConfig,
    EvalConfig,
    MeasureConfig,
    OptimizeConfig,
    ProbeConfig,
    SICConfig,
    SoftPromptConfig,
    SteerConfig,
    SurfaceConfig,
)


@dataclass(frozen=True, slots=True)
class ConfigParseContext:
    """Shared parse context for section adapters."""

    base_dir: Path
    raw: TomlDict


type _SectionParserResult = (
    DepthConfig
    | None
    | CutConfig
    | MeasureConfig
    | SurfaceConfig
    | DetectConfig
    | OptimizeConfig
    | SoftPromptConfig
    | SICConfig
    | ProbeConfig
    | SteerConfig
    | EvalConfig
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
    cut: CutConfig
    measure: MeasureConfig
    surface: SurfaceConfig | None
    detect: DetectConfig | None
    optimize: OptimizeConfig | None
    softprompt: SoftPromptConfig | None
    sic: SICConfig | None
    probe: ProbeConfig | None
    steer: SteerConfig | None
    eval: EvalConfig


@dataclass(frozen=True, slots=True)
class _DepthOverrideSentinel:
    """Internal sentinel for optional depth override."""


_DEPTH_OVERRIDE_UNSET = _DepthOverrideSentinel()


def _parse_depth_adapter(context: ConfigParseContext) -> DepthConfig | None:
    """Adapter for parsers that accept the full raw config mapping."""
    return _parse_depth(context.raw)


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


def _parse_softprompt_adapter(
    context: ConfigParseContext,
) -> SoftPromptConfig | None:
    """Adapter for parsers that accept the full raw config mapping."""
    return _parse_softprompt(context.raw)


def _parse_sic_adapter(context: ConfigParseContext) -> SICConfig | None:
    """Adapter for parsers that accept the full raw config mapping."""
    return _parse_sic(context.raw)


def _parse_probe_adapter(context: ConfigParseContext) -> ProbeConfig | None:
    """Adapter for parsers that accept the full raw config mapping."""
    return _parse_probe(context.raw)


def _parse_steer_adapter(context: ConfigParseContext) -> SteerConfig | None:
    """Adapter for parsers that accept the full raw config mapping."""
    return _parse_steer(context.raw)


def _parse_eval_adapter(context: ConfigParseContext) -> EvalConfig:
    """Adapter for parsers that require base_dir + raw config mapping."""
    return _parse_eval(context.base_dir, context.raw)


SECTION_PARSE_SPECS: tuple[SectionParseSpec[_SectionParserResult], ...] = (
    SectionParseSpec("depth", "depth", _parse_depth_adapter, 10),
    SectionParseSpec("cut", "cut", _parse_cut_adapter, 20),
    SectionParseSpec("measure", "measure", _parse_measure_adapter, 30),
    SectionParseSpec("surface", "surface", _parse_surface_adapter, 40),
    SectionParseSpec("detect", "detect", _parse_detect_adapter, 50),
    SectionParseSpec("optimize", "optimize", _parse_optimize_adapter, 60),
    SectionParseSpec("softprompt", "softprompt", _parse_softprompt_adapter, 70),
    SectionParseSpec("sic", "sic", _parse_sic_adapter, 80),
    SectionParseSpec("probe", "probe", _parse_probe_adapter, 90),
    SectionParseSpec("steer", "steer", _parse_steer_adapter, 100),
    SectionParseSpec("eval", "eval", _parse_eval_adapter, 110),
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
        cut=cast("CutConfig", parsed["cut"]),
        measure=cast("MeasureConfig", parsed["measure"]),
        surface=cast("SurfaceConfig | None", parsed["surface"]),
        detect=cast("DetectConfig | None", parsed["detect"]),
        optimize=cast("OptimizeConfig | None", parsed["optimize"]),
        softprompt=cast("SoftPromptConfig | None", parsed["softprompt"]),
        sic=cast("SICConfig | None", parsed["sic"]),
        probe=cast("ProbeConfig | None", parsed["probe"]),
        steer=cast("SteerConfig | None", parsed["steer"]),
        eval=cast("EvalConfig", parsed["eval"]),
    )
