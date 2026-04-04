# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tail coverage for config loader, registry, schema, and validation rules."""

from __future__ import annotations

import typing as typing_mod
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from vauban.config import load_config
from vauban.config._loader import _resolve_single_data
from vauban.config._parse_softprompt_loss import _parse_softprompt_loss
from vauban.config._registry import ConfigParseContext, parse_registered_section
from vauban.config._schema import (
    _dataclass_description,
    _json_compatible_default,
    _schema_for_tuple,
    _schema_for_type,
    _SchemaState,
)
from vauban.config._validation_models import ValidationCollector, ValidationContext
from vauban.config._validation_rules import (
    _rule_ai_act_readiness,
    _rule_prompt_sources,
    _rule_refusal_phrases,
    _rule_surface_prompts,
    _rule_unknown_schema,
)
from vauban.types import (
    AIActConfig,
    EvalConfig,
    PipelineConfig,
    SurfaceConfig,
)


def _write_jsonl(path: Path, count: int) -> None:
    lines = [f'{{"prompt": "prompt {i}"}}' for i in range(count)]
    path.write_text("\n".join(lines))


def _make_context(
    *,
    harmful_path: Path,
    harmless_path: Path,
    eval_config: EvalConfig | None = None,
    surface_config: SurfaceConfig | None = None,
    ai_act_config: AIActConfig | None = None,
    borderline_path: Path | None = None,
) -> ValidationContext:
    config = PipelineConfig(
        model_path="test-model",
        harmful_path=harmful_path,
        harmless_path=harmless_path,
        eval=eval_config or EvalConfig(),
        surface=surface_config,
        ai_act=ai_act_config,
        borderline_path=borderline_path,
        output_dir=Path("/tmp/output"),
    )
    return ValidationContext(
        config_path=Path("/tmp/config.toml"),
        raw={},
        config=config,
    )


def _make_ai_act() -> AIActConfig:
    return AIActConfig(
        company_name="Acme",
        system_name="Assistant",
        intended_purpose="testing",
    )


class _NoDocSchemaHelper:
    value: int


class TestLoaderTail:
    def test_supported_backend_is_preserved(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "cfg.toml"
        toml_file.write_text(
            'backend = "torch"\n'
            '[model]\npath = "mlx-community/tiny"\n'
            "[data]\n"
            'harmful = "harmful.jsonl"\n'
            'harmless = "harmless.jsonl"\n'
        )

        config = load_config(toml_file)

        assert config.backend == "torch"

    def test_resolve_single_data_benchmark_and_infix(self, tmp_path: Path) -> None:
        base_dir = tmp_path

        with (
            patch("vauban.benchmarks.KNOWN_BENCHMARKS", frozenset({"harmbench"})),
            patch(
                "vauban.benchmarks.resolve_benchmark",
                side_effect=lambda name: base_dir / f"{name}.jsonl",
            ),
            patch(
                "vauban.benchmarks.generate_infix_prompts",
                side_effect=lambda name: base_dir / f"{name}_infix.jsonl",
            ),
        ):
            benchmark = _resolve_single_data(base_dir, "harmbench", "harmful")
            infix = _resolve_single_data(base_dir, "harmbench_infix", "harmful")

        assert benchmark == base_dir / "harmbench.jsonl"
        assert infix == base_dir / "harmbench_infix.jsonl"


class TestSoftPromptLossTail:
    def test_invalid_constraint_list_element_raises(self) -> None:
        with pytest.raises(ValueError, match="not one of"):
            _parse_softprompt_loss(
                {"token_constraint": ["ascii", "bogus"]},
                None,
            )


class TestRegistryTail:
    def test_unknown_section_raises_key_error(self, tmp_path: Path) -> None:
        context = ConfigParseContext(base_dir=tmp_path, raw={})

        with pytest.raises(KeyError, match="Unknown parser section"):
            parse_registered_section(context, "missing")


class TestSchemaTail:
    def test_remaining_helper_branches(self) -> None:
        state = _SchemaState(defs={})

        assert _schema_for_type(typing_mod.List, state) == {  # noqa: UP006
            "type": "array",
            "items": {"type": "object"},
        }
        assert _schema_for_type(typing_mod.Set, state) == {  # noqa: UP006
            "type": "array",
            "items": {"type": "object"},
        }
        assert _schema_for_tuple((int, Ellipsis), state) == {
            "type": "array",
            "items": {"type": "integer"},
        }
        assert _json_compatible_default(Path("nested/file.json")) == "nested/file.json"
        assert _dataclass_description(_NoDocSchemaHelper) == "_NoDocSchemaHelper"


class TestValidationRulesTail:
    def test_unknown_schema_collects_warnings(self, tmp_path: Path) -> None:
        harmful = tmp_path / "harmful.jsonl"
        harmless = tmp_path / "harmless.jsonl"
        _write_jsonl(harmful, 4)
        _write_jsonl(harmless, 4)
        ctx = _make_context(harmful_path=harmful, harmless_path=harmless)
        collector = ValidationCollector()

        with (
            patch(
                "vauban.config._validation_rules.check_unknown_sections",
                return_value=["unknown section"],
            ),
            patch(
                "vauban.config._validation_rules.check_unknown_keys",
                return_value=["unknown key"],
            ),
            patch(
                "vauban.config._validation_rules.check_value_constraints",
                return_value=["bad value"],
            ),
        ):
            _rule_unknown_schema(ctx, collector)

        rendered = collector.render()
        assert any("unknown section" in item for item in rendered)
        assert any("unknown key" in item for item in rendered)
        assert any("bad value" in item for item in rendered)

    def test_prompt_source_ratio_uses_reciprocal_branch(self, tmp_path: Path) -> None:
        harmful = tmp_path / "harmful.jsonl"
        harmless = tmp_path / "harmless.jsonl"
        _write_jsonl(harmful, 2)
        _write_jsonl(harmless, 20)
        ctx = _make_context(harmful_path=harmful, harmless_path=harmless)
        collector = ValidationCollector()

        _rule_prompt_sources(ctx, collector)

        rendered = collector.render()
        assert any("highly imbalanced" in item for item in rendered)

    def test_refusal_phrase_loader_value_error(self, tmp_path: Path) -> None:
        harmful = tmp_path / "harmful.jsonl"
        harmless = tmp_path / "harmless.jsonl"
        refusal = tmp_path / "refusal.txt"
        _write_jsonl(harmful, 4)
        _write_jsonl(harmless, 4)
        refusal.write_text("bad\n")
        ctx = _make_context(
            harmful_path=harmful,
            harmless_path=harmless,
            eval_config=EvalConfig(refusal_phrases_path=refusal),
        )
        collector = ValidationCollector()

        with patch(
            "vauban.config._validation_rules._load_refusal_phrases",
            side_effect=ValueError("malformed refusal file"),
        ):
            _rule_refusal_phrases(ctx, collector)

        rendered = collector.render()
        assert any("malformed refusal file" in item for item in rendered)

    def test_surface_path_object_and_delta_gate(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        harmful = tmp_path / "harmful.jsonl"
        harmless = tmp_path / "harmless.jsonl"
        surface_path = tmp_path / "surface.jsonl"
        _write_jsonl(harmful, 4)
        _write_jsonl(harmless, 4)
        seen: list[Path] = []

        def fake_validate(
            candidate: Path,
            _label: str,
            collector: ValidationCollector,
            *,
            missing_fix: str,
        ) -> int:
            seen.append(candidate)
            collector.add("LOW", "validated", fix=missing_fix)
            return 4

        monkeypatch.setattr(
            "vauban.config._validation_rules._validate_surface_jsonl_file",
            fake_validate,
        )
        ctx = _make_context(
            harmful_path=harmful,
            harmless_path=harmless,
            surface_config=SurfaceConfig(
                prompts_path=surface_path,
                generate=False,
                max_worst_cell_refusal_after=None,
                max_worst_cell_refusal_delta=0.25,
            ),
        )
        collector = ValidationCollector()

        _rule_surface_prompts(ctx, collector)

        assert seen == [surface_path]
        rendered = collector.render()
        assert any("projection-only mode" in item for item in rendered)

    def test_ai_act_low_level_warnings_and_transparency_paths(
        self,
        tmp_path: Path,
    ) -> None:
        harmful = tmp_path / "harmful.jsonl"
        harmless = tmp_path / "harmless.jsonl"
        report = tmp_path / "report.txt"
        report.write_text("ok")
        _write_jsonl(harmful, 4)
        _write_jsonl(harmless, 4)

        ai_act = replace(
            _make_ai_act(),
            eu_market=True,
            role="deployer",
            ai_literacy_record=report,
            transparency_notice=report,
            interacts_with_natural_persons=True,
            interaction_obvious_to_persons=False,
        )
        ctx = _make_context(
            harmful_path=harmful,
            harmless_path=harmless,
            ai_act_config=ai_act,
        )
        collector = ValidationCollector()

        _rule_ai_act_readiness(ctx, collector)

        rendered = collector.render()
        assert any("non-obvious human interaction" in item for item in rendered)

    def test_ai_act_obviousness_and_article50_branches(self, tmp_path: Path) -> None:
        harmful = tmp_path / "harmful.jsonl"
        harmless = tmp_path / "harmless.jsonl"
        report = tmp_path / "report.txt"
        report.write_text("ok")
        _write_jsonl(harmful, 4)
        _write_jsonl(harmless, 4)

        ai_act = replace(
            _make_ai_act(),
            eu_market=True,
            role="research",
            ai_literacy_record=report,
            interacts_with_natural_persons=False,
            interaction_obvious_to_persons=True,
            exposes_emotion_recognition_or_biometric_categorization=False,
            uses_emotion_recognition=True,
            uses_biometric_categorization=True,
            public_interest_text_editorial_responsibility=True,
            public_interest_text_human_review_or_editorial_control=False,
        )
        ctx = _make_context(
            harmful_path=harmful,
            harmless_path=harmless,
            ai_act_config=ai_act,
        )
        collector = ValidationCollector()

        _rule_ai_act_readiness(ctx, collector)

        rendered = collector.render()
        assert any("interaction_obvious_to_persons" in item for item in rendered)
        assert any("emotion_recognition" in item for item in rendered)
        assert any("biometric_categorization" in item for item in rendered)
        assert any(
            "public_interest_text_editorial_responsibility" in item
            for item in rendered
        )

    def test_ai_act_exception_and_sensitive_trait_branches(
        self,
        tmp_path: Path,
    ) -> None:
        harmful = tmp_path / "harmful.jsonl"
        harmless = tmp_path / "harmless.jsonl"
        _write_jsonl(harmful, 4)
        _write_jsonl(harmless, 4)

        ai_act = replace(
            _make_ai_act(),
            eu_market=True,
            role="research",
            interacts_with_natural_persons=False,
            interaction_obvious_to_persons=False,
            uses_emotion_recognition=False,
            emotion_recognition_medical_or_safety_exception=True,
            uses_biometric_categorization=False,
            biometric_categorization_infers_sensitive_traits=True,
            public_interest_text_editorial_responsibility=True,
            public_interest_text_human_review_or_editorial_control=False,
        )
        ctx = _make_context(
            harmful_path=harmful,
            harmless_path=harmless,
            ai_act_config=ai_act,
        )
        collector = ValidationCollector()

        _rule_ai_act_readiness(ctx, collector)

        rendered = collector.render()
        assert any(
            "emotion_recognition_medical_or_safety_exception" in item
            for item in rendered
        )
        assert any(
            "biometric_categorization_infers_sensitive_traits" in item
            for item in rendered
        )
        assert any(
            "public_interest_text_editorial_responsibility" in item
            for item in rendered
        )

    def test_ai_act_article6_3_and_high_risk_deployer_paths(
        self,
        tmp_path: Path,
    ) -> None:
        harmful = tmp_path / "harmful.jsonl"
        harmless = tmp_path / "harmless.jsonl"
        evidence = tmp_path / "evidence.txt"
        evidence.write_text("ok")
        _write_jsonl(harmful, 4)
        _write_jsonl(harmless, 4)

        carve_out = replace(
            _make_ai_act(),
            eu_market=True,
            role="deployer",
            ai_literacy_record=evidence,
            annex_iii_narrow_procedural_task=True,
            annex_iii_does_not_materially_influence_decision_outcome=True,
            annex_iii_use_cases=[],
            uses_profiling_or_similarly_significant_decision_support=False,
            annex_i_product_or_safety_component=False,
            annex_i_third_party_conformity_assessment=False,
        )
        deployer = replace(
            _make_ai_act(),
            eu_market=True,
            role="deployer",
            ai_literacy_record=evidence,
            transparency_notice=evidence,
            annex_iii_use_cases=["annex_iii_1_biometric"],
            operation_monitoring_procedure=None,
            human_oversight_procedure=None,
            risk_owner=None,
            log_retention_procedure=None,
            provides_input_data_for_high_risk_system=True,
            input_data_governance_procedure=None,
            workplace_deployment=True,
            employee_or_worker_representative_notice=None,
            makes_or_assists_decisions_about_natural_persons=True,
            affected_person_notice=None,
            decision_with_legal_or_similarly_significant_effects=True,
            explanation_request_procedure=None,
            public_sector_use=True,
            eu_database_registration_record=None,
        )

        collector = ValidationCollector()
        _rule_ai_act_readiness(
            _make_context(
                harmful_path=harmful,
                harmless_path=harmless,
                ai_act_config=carve_out,
            ),
            collector,
        )
        _rule_ai_act_readiness(
            _make_context(
                harmful_path=harmful,
                harmless_path=harmless,
                ai_act_config=deployer,
            ),
            collector,
        )

        rendered = collector.render()
        assert any("Article 6(3) carve-out" in item for item in rendered)
        assert any("operation monitoring procedure" in item for item in rendered)
        assert any("human oversight procedure" in item for item in rendered)
        assert any("risk_owner" in item for item in rendered)
        assert any("log_retention_procedure" in item for item in rendered)
        assert any("input data for a high-risk system" in item for item in rendered)
        assert any("workplace deployment" in item for item in rendered)
        assert any("affected_person_notice" in item for item in rendered)
        assert any("explanation_request_procedure" in item for item in rendered)
        assert any("eu_database_registration_record" in item for item in rendered)
