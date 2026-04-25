# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for validation rule registry and early-mode metadata parity."""

from pathlib import Path

import pytest

import vauban
from vauban import validate
from vauban.behavior import BehaviorReport, BehaviorSuiteRef, ReportModelRef
from vauban.config._mode_registry import (
    EARLY_MODE_SPECS,
    active_early_mode_for_phase,
    active_early_modes,
)
from vauban.config._validation import VALIDATION_RULE_SPECS, validate_config
from vauban.types import (
    AIActConfig,
    ApiEvalConfig,
    ApiEvalEndpoint,
    AuditConfig,
    AwarenessConfig,
    BehaviorDiffConfig,
    BehaviorReportConfig,
    BehaviorTraceConfig,
    BehaviorTracePromptConfig,
    CastConfig,
    CircuitConfig,
    ComposeOptimizeConfig,
    DefenseStackConfig,
    DepthConfig,
    FeaturesConfig,
    FlywheelConfig,
    FusionConfig,
    GuardConfig,
    InterventionEvalConfig,
    InterventionEvalPrompt,
    JailbreakConfig,
    LinearProbeConfig,
    LoraAnalysisConfig,
    LoraExportConfig,
    OptimizeConfig,
    PipelineConfig,
    ProbeConfig,
    RemoteConfig,
    RepBendConfig,
    SICConfig,
    SoftPromptConfig,
    SSSConfig,
    SteerConfig,
    SVFConfig,
)

_EXPECTED_RULE_ORDER: list[str] = [
    "unknown_schema",
    "prompt_sources",
    "eval_prompts",
    "refusal_phrases",
    "surface_prompts",
    "output_dir",
    "ai_act_readiness",
    "early_mode_conflicts",
    "depth_extract_direction",
    "eval_without_prompts",
    "skipped_sections",
]

_EXPECTED_EARLY_MODE_ORDER: list[str] = [
    "[remote]",
    "[api_eval]",
    "[ai_act]",
    "[behavior_diff]",
    "[behavior_trace]",
    "[behavior_report]",
    "[depth]",
    "[svf]",
    "[features]",
    "[probe]",
    "[steer]",
    "[intervention_eval]",
    "[sss]",
    "[awareness]",
    "[audit]",
    "[guard]",
    "[cast]",
    "[sic]",
    "[optimize]",
    "[compose_optimize]",
    "[softprompt]",
    "[jailbreak]",
    "[defend]",
    "[circuit]",
    "[linear_probe]",
    "[fusion]",
    "[repbend]",
    "[lora_export]",
    "[lora_analysis]",
    "[flywheel]",
]


def _write_prompt_jsonl(path: Path, count: int) -> None:
    lines = [f'{{"prompt": "prompt {i}"}}' for i in range(count)]
    path.write_text("\n".join(lines) + "\n")


def _write_surface_jsonl(path: Path, count: int) -> None:
    lines = [
        (
            f'{{"prompt": "surface {i}", "label": "harmful",'
            ' "category": "weapons"}'
        )
        for i in range(count)
    ]
    path.write_text("\n".join(lines) + "\n")


def _write_conflict_fixture(tmp_path: Path) -> Path:
    harmful = tmp_path / "harmful.jsonl"
    harmless = tmp_path / "harmless.jsonl"
    surface = tmp_path / "surface.jsonl"
    eval_prompts = tmp_path / "eval.jsonl"

    _write_prompt_jsonl(harmful, 20)
    _write_prompt_jsonl(harmless, 20)
    _write_surface_jsonl(surface, 10)
    _write_prompt_jsonl(eval_prompts, 10)

    config_path = tmp_path / "conflict.toml"
    config_path.write_text(
        '[model]\npath = "test-model"\n'
        '[data]\nharmful = "harmful.jsonl"\nharmless = "harmless.jsonl"\n'
        '[depth]\nprompts = ["What is 2+2?", "Explain gravity"]\n'
        '[probe]\nprompts = ["How do I pick a lock?"]\n'
        "[detect]\n"
        '[surface]\nprompts = "surface.jsonl"\n'
        '[eval]\nprompts = "eval.jsonl"\n'
    )
    return config_path


def test_validation_rule_registry_order_is_stable() -> None:
    names = [spec.name for spec in VALIDATION_RULE_SPECS]
    assert names == _EXPECTED_RULE_ORDER


def test_validation_rule_registry_names_and_order_values_are_unique() -> None:
    names = [spec.name for spec in VALIDATION_RULE_SPECS]
    orders = [spec.order for spec in VALIDATION_RULE_SPECS]

    assert len(names) == len(set(names))
    assert len(orders) == len(set(orders))
    assert orders == sorted(orders)


def test_validate_wrapper_delegates_to_validation_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str | Path] = []

    def fake_validate_config(config_path: str | Path) -> list[str]:
        calls.append(config_path)
        return ["sentinel"]

    monkeypatch.setattr(
        "vauban.config._validation.validate_config",
        fake_validate_config,
    )

    result = vauban.validate("run.toml")

    assert result == ["sentinel"]
    assert calls == ["run.toml"]


def test_validation_warning_content_and_order_for_conflict_fixture(
    tmp_path: Path,
) -> None:
    config_path = _write_conflict_fixture(tmp_path)

    warnings = validate_config(config_path)

    assert warnings == [
            (
                "[HIGH] Multiple early-return modes active: [depth], [probe]"
                " — only the first will run (precedence: remote"
                " > api_eval > ai_act > behavior_diff > behavior_trace"
                " > behavior_report"
                " > depth > svf > features"
            " > probe > steer > intervention_eval > sss > awareness"
            " > audit > guard > cast > sic"
            " > optimize > compose_optimize"
            " > softprompt > jailbreak > defend > circuit > linear_probe > fusion"
            " > repbend > lora_export > lora_analysis > flywheel)"
            " — fix: keep one early-return mode per config,"
            " and split other modes into separate TOML files"
        ),
        (
            "[MEDIUM] [depth] early-return will skip: [detect], [surface], [eval]"
            " — these sections have no effect in depth mode"
            " — fix: remove skipped sections from this config,"
            " or run them in a separate non-early-return config"
        ),
    ]


def test_validation_summary_pipeline_line_is_unchanged(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_conflict_fixture(tmp_path)

    validate(config_path)
    captured = capsys.readouterr()

    assert f"Config:   {config_path}" in captured.err
    assert "Model:    test-model" in captured.err
    assert "Pipeline: depth analysis" in captured.err
    assert f"Output:   {tmp_path / 'output'}" in captured.err
    assert "+ detect" not in captured.err
    assert "+ surface" not in captured.err
    assert "+ eval" not in captured.err


def test_early_mode_registry_order_is_stable() -> None:
    sections = [spec.section for spec in EARLY_MODE_SPECS]
    assert sections == _EXPECTED_EARLY_MODE_ORDER


def test_active_early_modes_precedence_matches_legacy_behavior() -> None:
    config = PipelineConfig(
        model_path="test-model",
        harmful_path=Path("harmful.jsonl"),
        harmless_path=Path("harmless.jsonl"),
        api_eval=ApiEvalConfig(
            endpoints=[ApiEvalEndpoint(
                name="ep", base_url="https://api.example.com/v1",
                model="m", api_key_env="K",
            )],
            token_text="tokens",
            prompts=["prompt"],
        ),
        ai_act=AIActConfig(
            company_name="Example Energy",
            system_name="Customer Assistant",
            intended_purpose="Answers customer questions.",
        ),
        behavior_diff=BehaviorDiffConfig(
            baseline_trace=Path("base.jsonl"),
            candidate_trace=Path("candidate.jsonl"),
        ),
        behavior_trace=BehaviorTraceConfig(
            prompts=[
                BehaviorTracePromptConfig(
                    prompt_id="p1",
                    text="Explain rainbows.",
                ),
            ],
        ),
        behavior_report=BehaviorReportConfig(
            report=BehaviorReport(
                title="Behavior Report",
                baseline=ReportModelRef(
                    label="base",
                    model_path="base-model",
                    role="baseline",
                ),
                candidate=ReportModelRef(
                    label="candidate",
                    model_path="candidate-model",
                ),
                suite=BehaviorSuiteRef(
                    name="smoke",
                    description="Smoke behavior suite.",
                    categories=("benign_request",),
                    metrics=("compliance_rate",),
                ),
            ),
        ),
        audit=AuditConfig(company_name="Test", system_name="Test"),
        depth=DepthConfig(prompts=["a", "b"]),
        svf=SVFConfig(
            prompts_target=Path("target.jsonl"),
            prompts_opposite=Path("opposite.jsonl"),
        ),
        probe=ProbeConfig(prompts=["probe"]),
        steer=SteerConfig(prompts=["steer"]),
        intervention_eval=InterventionEvalConfig(
            prompts=[
                InterventionEvalPrompt(
                    prompt_id="p1",
                    prompt="Explain rainbows.",
                ),
            ],
        ),
        sss=SSSConfig(prompts=["sss"]),
        awareness=AwarenessConfig(prompts=["awareness"]),
        guard=GuardConfig(prompts=["guard"]),
        cast=CastConfig(prompts=["cast"]),
        sic=SICConfig(),
        optimize=OptimizeConfig(),
        compose_optimize=ComposeOptimizeConfig(bank_path="bank.safetensors"),
        softprompt=SoftPromptConfig(),
        jailbreak=JailbreakConfig(),
        defend=DefenseStackConfig(),
        features=FeaturesConfig(
            prompts_path=Path("prompts.jsonl"),
            layers=[0, 1],
        ),
        circuit=CircuitConfig(
            clean_prompts=["clean"],
            corrupt_prompts=["corrupt"],
        ),
        linear_probe=LinearProbeConfig(layers=[0, 1]),
        fusion=FusionConfig(
            harmful_prompts=["harmful"],
            benign_prompts=["benign"],
        ),
        repbend=RepBendConfig(layers=[0, 1]),
        lora_export=LoraExportConfig(),
        lora_analysis=LoraAnalysisConfig(adapter_path="adapter/"),
        flywheel=FlywheelConfig(),
        remote=RemoteConfig(
            backend="jsinfer",
            api_key_env="K",
            models=["m"],
            prompts=["p"],
        ),
    )

    assert active_early_modes(config) == _EXPECTED_EARLY_MODE_ORDER

    standalone = active_early_mode_for_phase(config, "standalone")
    before_prompts = active_early_mode_for_phase(config, "before_prompts")
    after_measure = active_early_mode_for_phase(config, "after_measure")

    assert standalone is not None
    assert standalone.mode == "remote"
    assert before_prompts is not None
    assert before_prompts.mode == "behavior_trace"
    assert after_measure is not None
    assert after_measure.mode == "probe"
