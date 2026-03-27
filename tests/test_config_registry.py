"""Tests for parser registry structure and loader parity."""

import tomllib
from pathlib import Path
from typing import cast

import pytest

from vauban.config import load_config
from vauban.config._loader import _resolve_borderline_path, _resolve_data_paths
from vauban.config._parse_cast import _parse_cast
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
from vauban.config._registry import (
    SECTION_PARSE_SPECS,
    ConfigParseContext,
    SectionParseSpec,
    _call_parser,
    parse_registered_sections,
)
from vauban.config._types import TomlDict
from vauban.types import (
    CutConfig,
    DepthConfig,
    EvalConfig,
    MeasureConfig,
    PipelineConfig,
)

_EXPECTED_SECTION_ORDER: list[str] = [
    "lora",
    "depth",
    "ai_act",
    "cast",
    "cut",
    "measure",
    "surface",
    "detect",
    "optimize",
    "compose_optimize",
    "softprompt",
    "sic",
    "svf",
    "probe",
    "steer",
    "sss",
    "awareness",
    "eval",
    "api_eval",
    "environment",
    "scan",
    "policy",
    "intent",
    "jailbreak",
    "defend",
    "circuit",
    "features",
    "linear_probe",
    "fusion",
    "repbend",
    "lora_export",
    "lora_analysis",
    "flywheel",
    "remote",
]

_MODEL = '[model]\npath = "test-model"\n'
_DATA = '[data]\nharmful = "harmful.jsonl"\nharmless = "harmless.jsonl"\n'


def _manual_load_config_pre_registry(path: Path) -> PipelineConfig:
    """Legacy loader assembly used for strict parity checks."""
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

    depth_config = _parse_depth(raw)
    cast_config = _parse_cast(raw)
    harmful_path, harmless_path = _resolve_data_paths(
        base_dir,
        raw,
        depth_only=depth_config is not None,
    )

    cut_raw = raw.get("cut")
    cut_section = (
        cast("TomlDict", cut_raw)
        if isinstance(cut_raw, dict)
        else cast("TomlDict", {})
    )
    cut_config = _parse_cut(
        cut_section,
    )

    measure_raw = raw.get("measure")
    measure_config = _parse_measure(
        cast("TomlDict", measure_raw)
        if isinstance(measure_raw, dict)
        else cast("TomlDict", {}),
    )

    surface_config = _parse_surface(base_dir, raw)
    detect_config = _parse_detect(raw)
    optimize_config = _parse_optimize(raw)
    softprompt_config = _parse_softprompt(raw)
    sic_config = _parse_sic(raw)
    probe_config = _parse_probe(raw)
    steer_config = _parse_steer(raw)
    eval_config = _parse_eval(base_dir, raw)

    output_section = raw.get("output")
    output_dir_str = "output"
    if isinstance(output_section, dict):
        dir_raw = output_section.get("dir")
        if isinstance(dir_raw, str):
            output_dir_str = dir_raw
    output_dir = base_dir / output_dir_str

    borderline_path = _resolve_borderline_path(base_dir, raw)
    if cut_config.false_refusal_ortho and borderline_path is None:
        msg = (
            "[cut].false_refusal_ortho = true requires"
            " [data].borderline to be set"
        )
        raise ValueError(msg)

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
        model_path=model_path_raw,
        harmful_path=harmful_path,
        harmless_path=harmless_path,
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
        output_dir=output_dir,
        borderline_path=borderline_path,
        verbose=verbose,
    )


def _assert_loader_exception_parity(tmp_path: Path, toml_text: str) -> None:
    """Assert exception type/message parity with legacy loader flow."""
    path = tmp_path / "bad.toml"
    path.write_text(toml_text)

    with pytest.raises(Exception) as expected_exc_info:
        _manual_load_config_pre_registry(path)
    expected_exc = expected_exc_info.value

    with pytest.raises(type(expected_exc)) as actual_exc_info:
        load_config(path)
    assert str(actual_exc_info.value) == str(expected_exc)


def test_section_parse_specs_order_is_strict() -> None:
    sections = [spec.section for spec in SECTION_PARSE_SPECS]
    assert sections == _EXPECTED_SECTION_ORDER


def test_section_parse_specs_have_unique_names_and_targets() -> None:
    sections = [spec.section for spec in SECTION_PARSE_SPECS]
    targets = [spec.target_field for spec in SECTION_PARSE_SPECS]
    orders = [spec.order for spec in SECTION_PARSE_SPECS]

    assert len(sections) == len(set(sections))
    assert len(targets) == len(set(targets))
    assert len(orders) == len(set(orders))
    assert orders == sorted(orders)


def test_call_parser_base_dir_raw(tmp_path: Path) -> None:
    """base_dir_raw convention passes (base_dir, raw) to the parser."""
    calls: list[tuple[Path, TomlDict]] = []

    def fake(base_dir: Path, raw: TomlDict) -> str:
        calls.append((base_dir, raw))
        return "ok"

    spec = SectionParseSpec("test", "test", fake, 999, call="base_dir_raw")
    raw: TomlDict = {"key": "val"}
    ctx = ConfigParseContext(base_dir=tmp_path, raw=raw)
    result = _call_parser(spec, ctx)

    assert result == "ok"
    assert calls == [(tmp_path, raw)]


def test_call_parser_raw_base_dir(tmp_path: Path) -> None:
    """raw_base_dir convention passes (raw, base_dir) to the parser."""
    calls: list[tuple[TomlDict, Path]] = []

    def fake(raw: TomlDict, base_dir: Path) -> str:
        calls.append((raw, base_dir))
        return "ok"

    spec = SectionParseSpec("test", "test", fake, 999, call="raw_base_dir")
    raw: TomlDict = {"key": "val"}
    ctx = ConfigParseContext(base_dir=tmp_path, raw=raw)
    result = _call_parser(spec, ctx)

    assert result == "ok"
    assert calls == [(raw, tmp_path)]


def test_call_parser_section_table(tmp_path: Path) -> None:
    """section_table convention extracts the section dict."""
    calls: list[TomlDict] = []

    def fake(sec: TomlDict) -> str:
        calls.append(sec)
        return "ok"

    spec = SectionParseSpec("mysec", "mysec", fake, 999, call="section_table")
    raw: TomlDict = {"mysec": {"alpha": 0.25}}
    ctx = ConfigParseContext(base_dir=tmp_path, raw=raw)
    _call_parser(spec, ctx)

    assert calls == [cast("TomlDict", {"alpha": 0.25})]


def test_call_parser_section_table_missing_defaults_empty(
    tmp_path: Path,
) -> None:
    """section_table convention defaults to empty dict when absent."""
    calls: list[TomlDict] = []

    def fake(sec: TomlDict) -> str:
        calls.append(sec)
        return "ok"

    spec = SectionParseSpec("mysec", "mysec", fake, 999, call="section_table")
    ctx = ConfigParseContext(base_dir=tmp_path, raw={})
    _call_parser(spec, ctx)

    assert calls == [cast("TomlDict", {})]


def test_parse_registered_sections_with_empty_raw(
    tmp_path: Path,
) -> None:
    """parse_registered_sections with empty raw returns sensible defaults."""
    context = ConfigParseContext(base_dir=tmp_path, raw={})
    parsed = parse_registered_sections(context)

    # Required fields get defaults
    assert isinstance(parsed.cut, CutConfig)
    assert isinstance(parsed.measure, MeasureConfig)
    assert isinstance(parsed.eval, EvalConfig)

    # Optional fields are None when absent
    assert parsed.depth is None
    assert parsed.cast is None
    assert parsed.softprompt is None
    assert parsed.flywheel is None


def test_parse_registered_sections_depth_override(
    tmp_path: Path,
) -> None:
    """depth_override bypasses the depth parser."""
    override = DepthConfig(prompts=["override"])
    parsed = parse_registered_sections(
        ConfigParseContext(base_dir=tmp_path, raw={}),
        depth_override=override,
    )

    assert parsed.depth == override


def test_loader_parity_minimal_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(_MODEL + _DATA)

    expected = _manual_load_config_pre_registry(config_path)
    actual = load_config(config_path)

    assert actual == expected


def test_loader_parity_depth_only_without_data(tmp_path: Path) -> None:
    config_path = tmp_path / "depth.toml"
    config_path.write_text(
        _MODEL
        + "[depth]\n"
        'prompts = ["What is 2+2?", "Explain gravity"]\n',
    )

    expected = _manual_load_config_pre_registry(config_path)
    actual = load_config(config_path)

    assert actual == expected


def test_loader_parity_full_optional_config(tmp_path: Path) -> None:
    config_path = tmp_path / "full.toml"
    config_path.write_text(
        _MODEL
        + "[data]\n"
        + 'harmful = "harmful.jsonl"\n'
        + 'harmless = "harmless.jsonl"\n'
        + 'borderline = "borderline.jsonl"\n'
        + "[cut]\nalpha = 0.7\nlayers = [1, 2]\n"
        + "[measure]\nmode = \"direction\"\ntop_k = 7\n"
        + '[surface]\nprompts = "surface.jsonl"\ngenerate = false\nmax_tokens = 17\n'
        + '[detect]\nmode = "probe"\ntop_k = 3\n'
        + "[optimize]\nn_trials = 3\nalpha_min = 0.2\nalpha_max = 1.1\n"
        + '[softprompt]\nmode = "continuous"\nn_tokens = 4\n'
        + '[sic]\nmode = "generation"\nmax_iterations = 4\n'
        + '[probe]\nprompts = ["probe prompt"]\n'
        + '[steer]\nprompts = ["steer prompt"]\nalpha = 1.1\n'
        + '[cast]\nprompts = ["cast prompt"]\nthreshold = 0.2\n'
        + '[eval]\nprompts = "eval.jsonl"\nmax_tokens = 45\nnum_prompts = 9\n'
        + '[output]\ndir = "out"\n'
        + "verbose = false\n"
    )

    expected = _manual_load_config_pre_registry(config_path)
    actual = load_config(config_path)

    assert actual == expected


@pytest.mark.parametrize(
    ("_section_name", "body"),
    [
        ("depth", _MODEL + "[depth]\nprompts = 123\n"),
        ("cut", _MODEL + _DATA + '[cut]\nalpha = "bad"\n'),
        ("measure", _MODEL + _DATA + "[measure]\nmode = 12\n"),
        ("surface", _MODEL + _DATA + "[surface]\ngenerate = 12\n"),
        ("detect", _MODEL + _DATA + "[detect]\nmode = 12\n"),
        (
            "optimize",
            _MODEL + _DATA + "[optimize]\nalpha_min = 1.0\nalpha_max = 0.5\n",
        ),
        ("softprompt", _MODEL + _DATA + '[softprompt]\nmode = "invalid"\n'),
        ("sic", _MODEL + _DATA + "[sic]\nmax_iterations = 0\n"),
        ("probe", _MODEL + _DATA + "[probe]\nprompts = 5\n"),
        ("steer", _MODEL + _DATA + "[steer]\nprompts = 5\n"),
        ("cast", _MODEL + _DATA + "[cast]\nprompts = 5\n"),
        ("eval", _MODEL + _DATA + "[eval]\nmax_tokens = 0\n"),
    ],
)
def test_loader_exception_parity_for_section_errors(
    tmp_path: Path,
    _section_name: str,
    body: str,
) -> None:
    _assert_loader_exception_parity(tmp_path, body)
