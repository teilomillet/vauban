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
    parse_registered_section,
    parse_registered_sections,
)
from vauban.config._types import TomlDict
from vauban.types import (
    ApiEvalConfig,
    ApiEvalEndpoint,
    CastConfig,
    CircuitConfig,
    ComposeOptimizeConfig,
    CutConfig,
    DefenseStackConfig,
    DepthConfig,
    DetectConfig,
    EnvironmentConfig,
    EnvironmentTarget,
    EnvironmentTask,
    EvalConfig,
    FeaturesConfig,
    IntentConfig,
    MeasureConfig,
    OptimizeConfig,
    PipelineConfig,
    PolicyConfig,
    ProbeConfig,
    ScanConfig,
    SICConfig,
    SoftPromptConfig,
    SSSConfig,
    SteerConfig,
    SurfaceConfig,
    SVFConfig,
    ToolSchema,
)

_EXPECTED_SECTION_ORDER: list[str] = [
    "lora",
    "depth",
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
    "eval",
    "api_eval",
    "environment",
    "scan",
    "policy",
    "intent",
    "defend",
    "circuit",
    "features",
    "linear_probe",
    "fusion",
    "repbend",
    "lora_export",
    "lora_analysis",
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


def test_eval_adapter_receives_base_dir_and_full_raw(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[Path, TomlDict]] = []

    def fake_parse_eval(base_dir: Path, raw: TomlDict) -> EvalConfig:
        calls.append((base_dir, raw))
        return EvalConfig(max_tokens=7)

    monkeypatch.setattr("vauban.config._registry._parse_eval", fake_parse_eval)

    raw: TomlDict = {"model": {"path": "test"}}
    context = ConfigParseContext(base_dir=tmp_path, raw=raw)
    parsed = parse_registered_section(context, "eval")

    assert parsed == EvalConfig(max_tokens=7)
    assert calls == [(tmp_path, raw)]


def test_surface_adapter_receives_base_dir_and_full_raw(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[Path, TomlDict]] = []

    def fake_parse_surface(base_dir: Path, raw: TomlDict) -> SurfaceConfig | None:
        calls.append((base_dir, raw))
        return SurfaceConfig(prompts_path="default", max_tokens=21)

    monkeypatch.setattr("vauban.config._registry._parse_surface", fake_parse_surface)

    raw: TomlDict = {"model": {"path": "test"}, "surface": {"prompts": "default"}}
    context = ConfigParseContext(base_dir=tmp_path, raw=raw)
    parsed = parse_registered_section(context, "surface")

    assert parsed == SurfaceConfig(prompts_path="default", max_tokens=21)
    assert calls == [(tmp_path, raw)]


def test_section_table_adapters_pass_only_section_tables(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cut_calls: list[TomlDict] = []
    measure_calls: list[TomlDict] = []

    def fake_parse_cut(raw: TomlDict) -> CutConfig:
        cut_calls.append(raw)
        return CutConfig(alpha=0.25)

    def fake_parse_measure(raw: TomlDict) -> MeasureConfig:
        measure_calls.append(raw)
        return MeasureConfig(mode="dbdi")

    monkeypatch.setattr("vauban.config._registry._parse_cut", fake_parse_cut)
    monkeypatch.setattr("vauban.config._registry._parse_measure", fake_parse_measure)

    raw: TomlDict = {
        "model": {"path": "test"},
        "cut": {"alpha": 0.25},
        "measure": {"mode": "dbdi"},
    }
    context = ConfigParseContext(base_dir=tmp_path, raw=raw)

    parse_registered_section(context, "cut")
    parse_registered_section(context, "measure")

    assert cut_calls == [cast("TomlDict", {"alpha": 0.25})]
    assert measure_calls == [cast("TomlDict", {"mode": "dbdi"})]


def test_section_table_adapters_use_empty_table_when_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cut_calls: list[TomlDict] = []

    def fake_parse_cut(raw: TomlDict) -> CutConfig:
        cut_calls.append(raw)
        return CutConfig()

    monkeypatch.setattr("vauban.config._registry._parse_cut", fake_parse_cut)

    context = ConfigParseContext(base_dir=tmp_path, raw={})
    parse_registered_section(context, "cut")

    assert cut_calls == [cast("TomlDict", {})]


def test_parse_registered_sections_respects_registry_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    call_order: list[str] = []

    def fake_depth(raw: TomlDict) -> DepthConfig | None:
        call_order.append("depth")
        return DepthConfig(prompts=["depth"])

    def fake_cut(raw: TomlDict) -> CutConfig:
        call_order.append("cut")
        return CutConfig(alpha=2.0)

    def fake_cast(raw: TomlDict) -> CastConfig | None:
        call_order.append("cast")
        return CastConfig(prompts=["cast"], threshold=0.3)

    def fake_measure(raw: TomlDict) -> MeasureConfig:
        call_order.append("measure")
        return MeasureConfig(mode="direction", top_k=9)

    def fake_surface(base_dir: Path, raw: TomlDict) -> SurfaceConfig | None:
        call_order.append("surface")
        return SurfaceConfig(prompts_path=base_dir / "surface.jsonl")

    def fake_detect(raw: TomlDict) -> DetectConfig | None:
        call_order.append("detect")
        return DetectConfig(mode="probe")

    def fake_optimize(raw: TomlDict) -> OptimizeConfig | None:
        call_order.append("optimize")
        return OptimizeConfig(n_trials=2)

    def fake_compose_optimize(
        base_dir: Path, raw: TomlDict,
    ) -> ComposeOptimizeConfig | None:
        call_order.append("compose_optimize")
        return ComposeOptimizeConfig(bank_path="bank.safetensors")

    def fake_softprompt(
        raw: TomlDict, base_dir: Path | None = None,
    ) -> SoftPromptConfig | None:
        call_order.append("softprompt")
        return SoftPromptConfig(n_tokens=8)

    def fake_sic(raw: TomlDict) -> SICConfig | None:
        call_order.append("sic")
        return SICConfig(max_iterations=4)

    def fake_svf(base_dir: Path, raw: TomlDict) -> SVFConfig | None:
        call_order.append("svf")
        return None

    def fake_probe(raw: TomlDict) -> ProbeConfig | None:
        call_order.append("probe")
        return ProbeConfig(prompts=["probe"])

    def fake_steer(raw: TomlDict) -> SteerConfig | None:
        call_order.append("steer")
        return SteerConfig(prompts=["steer"], alpha=1.5)

    def fake_sss(raw: TomlDict) -> SSSConfig | None:
        call_order.append("sss")
        return SSSConfig(prompts=["sss"])

    def fake_eval(base_dir: Path, raw: TomlDict) -> EvalConfig:
        call_order.append("eval")
        return EvalConfig(max_tokens=222)

    def fake_api_eval(raw: TomlDict) -> ApiEvalConfig | None:
        call_order.append("api_eval")
        return ApiEvalConfig(
            endpoints=[
                ApiEvalEndpoint(
                    name="fake",
                    base_url="https://api.fake.com/v1",
                    model="m",
                    api_key_env="K",
                ),
            ],
        )

    def fake_environment(raw: TomlDict) -> EnvironmentConfig | None:
        call_order.append("environment")
        return EnvironmentConfig(
            system_prompt="test",
            tools=[ToolSchema(name="t", description="d", parameters={})],
            target=EnvironmentTarget(function="t"),
            task=EnvironmentTask(content="do it"),
            injection_surface="t",
        )

    def fake_scan(raw: TomlDict) -> ScanConfig | None:
        call_order.append("scan")
        return ScanConfig(threshold=0.5)

    def fake_policy(raw: TomlDict) -> PolicyConfig | None:
        call_order.append("policy")
        return PolicyConfig()

    def fake_intent(raw: TomlDict) -> IntentConfig | None:
        call_order.append("intent")
        return IntentConfig(mode="judge")

    def fake_defend(raw: TomlDict) -> DefenseStackConfig | None:
        call_order.append("defend")
        return DefenseStackConfig(fail_fast=False)

    def fake_circuit(raw: TomlDict) -> CircuitConfig | None:
        call_order.append("circuit")
        return CircuitConfig(
            clean_prompts=["a"], corrupt_prompts=["b"],
        )

    def fake_features(
        base_dir: Path, raw: TomlDict,
    ) -> FeaturesConfig | None:
        call_order.append("features")
        return None

    monkeypatch.setattr("vauban.config._registry._parse_depth", fake_depth)
    monkeypatch.setattr("vauban.config._registry._parse_cast", fake_cast)
    monkeypatch.setattr("vauban.config._registry._parse_cut", fake_cut)
    monkeypatch.setattr("vauban.config._registry._parse_measure", fake_measure)
    monkeypatch.setattr("vauban.config._registry._parse_surface", fake_surface)
    monkeypatch.setattr("vauban.config._registry._parse_detect", fake_detect)
    monkeypatch.setattr("vauban.config._registry._parse_optimize", fake_optimize)
    monkeypatch.setattr(
        "vauban.config._registry._parse_compose_optimize",
        fake_compose_optimize,
    )
    monkeypatch.setattr("vauban.config._registry._parse_softprompt", fake_softprompt)
    monkeypatch.setattr("vauban.config._registry._parse_sic", fake_sic)
    monkeypatch.setattr("vauban.config._registry._parse_svf", fake_svf)
    monkeypatch.setattr("vauban.config._registry._parse_probe", fake_probe)
    monkeypatch.setattr("vauban.config._registry._parse_steer", fake_steer)
    monkeypatch.setattr("vauban.config._registry._parse_sss", fake_sss)
    monkeypatch.setattr("vauban.config._registry._parse_eval", fake_eval)
    monkeypatch.setattr("vauban.config._registry._parse_api_eval", fake_api_eval)
    monkeypatch.setattr(
        "vauban.config._registry._parse_environment", fake_environment,
    )
    monkeypatch.setattr("vauban.config._registry._parse_scan", fake_scan)
    monkeypatch.setattr("vauban.config._registry._parse_policy", fake_policy)
    monkeypatch.setattr("vauban.config._registry._parse_intent", fake_intent)
    monkeypatch.setattr("vauban.config._registry._parse_defend", fake_defend)
    monkeypatch.setattr("vauban.config._registry._parse_circuit", fake_circuit)
    monkeypatch.setattr("vauban.config._registry._parse_features", fake_features)

    def fake_linear_probe(raw: TomlDict) -> None:
        call_order.append("linear_probe")
        return None

    def fake_fusion(base_dir: Path, raw: TomlDict) -> None:
        call_order.append("fusion")
        return None

    def fake_repbend(raw: TomlDict) -> None:
        call_order.append("repbend")
        return None

    def fake_lora_export(raw: TomlDict) -> None:
        call_order.append("lora_export")
        return None

    def fake_lora_load(raw: TomlDict) -> None:
        call_order.append("lora")
        return None

    def fake_lora_analysis(raw: TomlDict) -> None:
        call_order.append("lora_analysis")
        return None

    monkeypatch.setattr(
        "vauban.config._registry._parse_linear_probe", fake_linear_probe,
    )
    monkeypatch.setattr("vauban.config._registry._parse_fusion", fake_fusion)
    monkeypatch.setattr("vauban.config._registry._parse_repbend", fake_repbend)
    monkeypatch.setattr(
        "vauban.config._registry._parse_lora_export", fake_lora_export,
    )
    monkeypatch.setattr(
        "vauban.config._registry._parse_lora_load", fake_lora_load,
    )
    monkeypatch.setattr(
        "vauban.config._registry._parse_lora_analysis", fake_lora_analysis,
    )

    context = ConfigParseContext(base_dir=tmp_path, raw={})
    parsed = parse_registered_sections(context)

    assert call_order == _EXPECTED_SECTION_ORDER
    assert parsed.depth == DepthConfig(prompts=["depth"])
    assert parsed.cast == CastConfig(prompts=["cast"], threshold=0.3)
    assert parsed.cut.alpha == 2.0
    assert parsed.measure.top_k == 9
    assert parsed.surface == SurfaceConfig(prompts_path=tmp_path / "surface.jsonl")
    assert parsed.detect == DetectConfig(mode="probe")
    assert parsed.optimize == OptimizeConfig(n_trials=2)
    assert parsed.compose_optimize == ComposeOptimizeConfig(
        bank_path="bank.safetensors",
    )
    assert parsed.softprompt == SoftPromptConfig(n_tokens=8)
    assert parsed.sic == SICConfig(max_iterations=4)
    assert parsed.probe == ProbeConfig(prompts=["probe"])
    assert parsed.steer == SteerConfig(prompts=["steer"], alpha=1.5)
    assert parsed.eval.max_tokens == 222
    assert parsed.api_eval is not None
    assert parsed.api_eval.endpoints[0].name == "fake"
    assert parsed.environment is not None
    assert parsed.environment.injection_surface == "t"
    assert parsed.scan is not None
    assert parsed.scan.threshold == 0.5
    assert parsed.policy is not None
    assert parsed.intent is not None
    assert parsed.intent.mode == "judge"
    assert parsed.defend is not None
    assert parsed.defend.fail_fast is False
    assert parsed.circuit is not None
    assert parsed.circuit.clean_prompts == ["a"]
    assert parsed.features is None


def test_parse_registered_sections_depth_override_bypasses_depth_parser(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_depth(_: TomlDict) -> DepthConfig | None:
        msg = "depth parser should not run when depth_override is provided"
        raise AssertionError(msg)

    monkeypatch.setattr("vauban.config._registry._parse_depth", fail_depth)

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
