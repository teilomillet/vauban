# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for mode runners F: audit, jailbreak, lora_analysis."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from tests.conftest import make_direction_result, make_early_mode_context
from vauban._pipeline._mode_audit import _run_audit_mode
from vauban._pipeline._mode_jailbreak import _run_jailbreak_mode
from vauban._pipeline._mode_lora_analysis import _run_lora_analysis_mode
from vauban.types import (
    AuditConfig,
    AuditFinding,
    AuditResult,
    DefenseStackConfig,
    DefenseStackResult,
    JailbreakConfig,
    JailbreakTemplate,
    LoraAnalysisConfig,
    LoraAnalysisResult,
    LoraLayerAnalysis,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _make_audit_result() -> AuditResult:
    """Build a minimal audit result."""
    return AuditResult(
        company_name="Acme",
        system_name="Shield",
        model_path="test-model",
        thoroughness="standard",
        overall_risk="low",
        findings=[
            AuditFinding(
                category="guard",
                severity="low",
                title="Guard finding",
                description="guard works",
                evidence="none",
                remediation="none",
            ),
        ],
        detect_hardened=False,
        detect_confidence=0.5,
        jailbreak_success_rate=0.25,
        jailbreak_total=4,
        softprompt_success_rate=None,
        bijection_success_rate=None,
        surface_refusal_rate=None,
        surface_coverage=None,
        guard_circuit_break_rate=None,
    )


def _make_lora_analysis_result(
    adapter_path: str,
    total_params: int,
) -> LoraAnalysisResult:
    """Build a minimal LoRA analysis result."""
    return LoraAnalysisResult(
        adapter_path=adapter_path,
        layers=[
            LoraLayerAnalysis(
                key="layer.0",
                frobenius_norm=1.0,
                singular_values=[1.0],
                effective_rank=1.0,
                variance_cutoff=1,
                direction_alignment=0.5,
            ),
        ],
        total_params=total_params,
        mean_effective_rank=1.0,
        norm_profile=[1.0],
    )


class TestAuditMode:
    """Tests for ``_run_audit_mode``."""

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        ctx = make_early_mode_context(tmp_path)
        with pytest.raises(ValueError, match="audit config is required"):
            _run_audit_mode(ctx)

    def test_happy_path_resolves_prompts_and_writes_pdf(
        self,
        tmp_path: Path,
    ) -> None:
        audit_cfg = AuditConfig(company_name="Acme", system_name="Shield")
        ctx = make_early_mode_context(
            tmp_path,
            harmful=None,
            harmless=None,
            audit=audit_cfg,
        )
        result = _make_audit_result()

        def _fake_run_audit(
            model: object,
            tokenizer: object,
            harmful: list[str],
            harmless: list[str],
            config: AuditConfig,
            model_path: str,
            *,
            direction_result: object | None = None,
            log_fn: Callable[[str], None] | None = None,
        ) -> AuditResult:
            del model, tokenizer, harmful, harmless
            del config, model_path, direction_result
            assert callable(log_fn)
            log_fn("audit-progress")
            return result

        with (
            patch(
                "vauban.dataset.resolve_prompts",
                side_effect=[["harm"], ["safe"]],
            ),
            patch("vauban.audit.run_audit", side_effect=_fake_run_audit) as mock_run,
            patch("vauban.audit.audit_result_to_dict", return_value={"risk": "low"}),
            patch("vauban.audit.audit_result_to_markdown", return_value="# Audit"),
            patch(
                "vauban.audit_pdf.render_audit_report_pdf",
                return_value=b"pdf-bytes",
            ),
            patch("vauban._pipeline._mode_audit.finish_mode_run") as mock_finish,
        ):
            _run_audit_mode(ctx)

        assert (tmp_path / "audit_report.json").exists()
        assert (tmp_path / "audit_report.md").read_text() == "# Audit"
        assert (tmp_path / audit_cfg.pdf_report_filename).read_bytes() == b"pdf-bytes"
        assert mock_run.call_args[0][2] == ["harm"]
        assert mock_run.call_args[0][3] == ["safe"]
        metadata = mock_finish.call_args[0][3]
        assert metadata["findings"] == 1
        assert metadata["jailbreak_success_rate"] == 0.25

    def test_pdf_import_error_uses_existing_prompts(
        self,
        tmp_path: Path,
    ) -> None:
        audit_cfg = AuditConfig(company_name="Acme", system_name="Shield")
        ctx = make_early_mode_context(
            tmp_path,
            harmful=["harm"],
            harmless=["safe"],
            audit=audit_cfg,
        )
        result = _make_audit_result()

        with (
            patch("vauban.audit.run_audit", return_value=result) as mock_run,
            patch("vauban.audit.audit_result_to_dict", return_value={}),
            patch("vauban.audit.audit_result_to_markdown", return_value="md"),
            patch.dict(sys.modules, {"vauban.audit_pdf": None}),
            patch("vauban._pipeline._mode_audit.log") as mock_log,
            patch("vauban._pipeline._mode_audit.finish_mode_run") as mock_finish,
        ):
            _run_audit_mode(ctx)

        assert mock_run.call_args[0][2] == ["harm"]
        assert mock_run.call_args[0][3] == ["safe"]
        log_messages = [call.args[0] for call in mock_log.call_args_list]
        assert any("skipping PDF generation" in msg for msg in log_messages)
        report_files = mock_finish.call_args[0][2]
        assert report_files == ["audit_report.json", "audit_report.md"]


class TestJailbreakMode:
    """Tests for ``_run_jailbreak_mode``."""

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        ctx = make_early_mode_context(tmp_path)
        with pytest.raises(ValueError, match="jailbreak config is required"):
            _run_jailbreak_mode(ctx)

    def test_harmful_payload_mode_requires_harmful_prompts(
        self,
        tmp_path: Path,
    ) -> None:
        ctx = make_early_mode_context(
            tmp_path,
            harmful=None,
            jailbreak=JailbreakConfig(),
        )
        template = JailbreakTemplate(
            strategy="persona",
            name="p",
            template="{payload}",
        )

        with (
            patch("vauban.jailbreak.load_templates", return_value=[template]),
            pytest.raises(ValueError, match="harmful prompts required"),
        ):
            _run_jailbreak_mode(ctx)

    def test_happy_path_filters_templates_and_uses_default_defense_stack(
        self,
        tmp_path: Path,
    ) -> None:
        cfg = JailbreakConfig(strategies=["persona"])
        ctx = make_early_mode_context(
            tmp_path,
            harmful=["payload1", "payload2"],
            jailbreak=cfg,
        )
        template = JailbreakTemplate(
            strategy="persona",
            name="persona-template",
            template="{payload}",
        )
        expanded = [
            (template, "jailbreak-1"),
            (template, "jailbreak-2"),
        ]
        results = [
            DefenseStackResult(blocked=True, layer_that_blocked="policy"),
            DefenseStackResult(blocked=False, layer_that_blocked=None),
        ]

        with (
            patch(
                "vauban.jailbreak.load_templates",
                return_value=[template],
            ),
            patch(
                "vauban.jailbreak.filter_by_strategy",
                return_value=[template],
            ) as mock_filter,
            patch(
                "vauban.jailbreak.apply_templates",
                return_value=expanded,
            ),
            patch(
                "vauban.defend.defend_content",
                side_effect=results,
            ),
            patch("vauban._pipeline._mode_jailbreak.finish_mode_run") as mock_finish,
        ):
            _run_jailbreak_mode(ctx)

        assert mock_filter.called
        assert (tmp_path / "jailbreak_report.json").exists()
        metadata = mock_finish.call_args[0][3]
        assert metadata["block_rate"] == 0.5

    def test_custom_payload_path_reuses_defend_config_and_direction(
        self,
        tmp_path: Path,
    ) -> None:
        defend_cfg = DefenseStackConfig()
        cfg = JailbreakConfig(payloads_from="payloads.jsonl")
        direction_result = make_direction_result()
        ctx = make_early_mode_context(
            tmp_path,
            direction_result=direction_result,
            jailbreak=cfg,
            defend=defend_cfg,
        )
        template = JailbreakTemplate(
            strategy="roleplay",
            name="roleplay-template",
            template="{payload}",
        )

        def _fake_defend_content(
            model: object,
            tokenizer: object,
            prompt: str,
            direction_vec: object | None,
            defend_config: DefenseStackConfig,
            layer_idx: int,
        ) -> DefenseStackResult:
            del model, tokenizer
            assert prompt == "expanded-prompt"
            assert direction_vec is direction_result.direction
            assert defend_config is defend_cfg
            assert layer_idx == direction_result.layer_index
            return DefenseStackResult(blocked=True, layer_that_blocked="scan")

        with (
            patch(
                "vauban.jailbreak.load_templates",
                return_value=[template],
            ),
            patch("vauban.measure.load_prompts", return_value=["payload"]),
            patch(
                "vauban.jailbreak.apply_templates",
                return_value=[(template, "expanded-prompt")],
            ),
            patch(
                "vauban.defend.defend_content",
                side_effect=_fake_defend_content,
            ),
            patch("vauban._pipeline._mode_jailbreak.finish_mode_run") as mock_finish,
        ):
            _run_jailbreak_mode(ctx)

        metadata = mock_finish.call_args[0][3]
        assert metadata["block_rate"] == 1.0


class TestLoraAnalysisMode:
    """Tests for ``_run_lora_analysis_mode``."""

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        ctx = make_early_mode_context(tmp_path)
        with pytest.raises(ValueError, match="lora_analysis config is required"):
            _run_lora_analysis_mode(ctx)

    def test_single_adapter_path_aligns_with_direction(
        self,
        tmp_path: Path,
    ) -> None:
        direction_result = make_direction_result()
        cfg = LoraAnalysisConfig(adapter_path="one.safetensors")
        ctx = make_early_mode_context(
            tmp_path,
            direction_result=direction_result,
            lora_analysis=cfg,
        )
        result = _make_lora_analysis_result("one.safetensors", total_params=10)

        with (
            patch("vauban.lora.analyze_adapter", return_value=result) as mock_analyze,
            patch(
                "vauban._pipeline._mode_lora_analysis.finish_mode_run",
            ) as mock_finish,
        ):
            _run_lora_analysis_mode(ctx)

        assert (tmp_path / "lora_analysis_report.json").exists()
        assert mock_analyze.call_args.kwargs["direction"] is direction_result.direction
        metadata = mock_finish.call_args[0][3]
        assert metadata["n_adapters"] == 1
        assert metadata["total_layers"] == 1

    def test_multiple_adapter_paths_without_direction(
        self,
        tmp_path: Path,
    ) -> None:
        cfg = LoraAnalysisConfig(
            adapter_paths=["one.safetensors", "two.safetensors"],
            align_with_direction=False,
        )
        ctx = make_early_mode_context(tmp_path, lora_analysis=cfg)
        results = [
            _make_lora_analysis_result("one.safetensors", total_params=10),
            _make_lora_analysis_result("two.safetensors", total_params=20),
        ]

        with (
            patch(
                "vauban.lora.analyze_adapter",
                side_effect=results,
            ) as mock_analyze,
            patch(
                "vauban._pipeline._mode_lora_analysis.finish_mode_run",
            ) as mock_finish,
        ):
            _run_lora_analysis_mode(ctx)

        assert mock_analyze.call_count == 2
        assert mock_analyze.call_args.kwargs["direction"] is None
        metadata = mock_finish.call_args[0][3]
        assert metadata["n_adapters"] == 2
        assert metadata["total_layers"] == 2
