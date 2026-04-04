# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Extra coverage for AI Act config parsing and mode runner branches."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

import vauban.config._parse_ai_act as parse_ai_act_module
from tests.conftest import make_early_mode_context
from vauban._pipeline._mode_ai_act import _run_ai_act_mode
from vauban.types import AIActConfig

if TYPE_CHECKING:
    from pathlib import Path


def _base_ai_act_section(tmp_path: Path) -> dict[str, object]:
    """Build a minimal valid `[ai_act]` table for branch-focused tests."""
    return {
        "company_name": "Example Energy",
        "system_name": "Customer Assistant",
        "intended_purpose": "Answers customer questions.",
        "annex_iii_use_cases": [],
        "technical_report_paths": [
            str(tmp_path / "reports" / "abs.json"),
            "reports/rel.json",
        ],
    }


def _parse_ai_act(base_dir: Path, section: dict[str, object]) -> AIActConfig:
    """Parse a wrapped `[ai_act]` section and assert it exists."""
    cfg = parse_ai_act_module._parse_ai_act(base_dir, {"ai_act": section})
    assert cfg is not None
    return cfg


class TestParseAIActExtra:
    """Targeted branch coverage for the AI Act parser."""

    def test_rejects_non_table_section(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="\\[ai_act\\] must be a table"):
            parse_ai_act_module._parse_ai_act(tmp_path, {"ai_act": []})

    @pytest.mark.parametrize(
        ("field", "message"),
        [
            ("system_name", "must not be empty"),
            ("intended_purpose", "must not be empty"),
            ("sector", "must not be empty"),
        ],
    )
    def test_rejects_blank_required_strings(
        self,
        tmp_path: Path,
        field: str,
        message: str,
    ) -> None:
        section = _base_ai_act_section(tmp_path)
        section[field] = ""

        with pytest.raises(ValueError, match=message):
            _parse_ai_act(tmp_path, section)

    def test_rejects_unknown_annex_iii_use_cases(self, tmp_path: Path) -> None:
        section = _base_ai_act_section(tmp_path)
        section["annex_iii_use_cases"] = ["unknown_use_case"]

        with pytest.raises(ValueError, match="unknown values"):
            _parse_ai_act(tmp_path, section)

    def test_rejects_blank_risk_owner_and_contact(
        self,
        tmp_path: Path,
    ) -> None:
        section = _base_ai_act_section(tmp_path)
        section["risk_owner"] = ""

        with pytest.raises(ValueError, match="risk_owner must not be empty"):
            _parse_ai_act(tmp_path, section)

        section = _base_ai_act_section(tmp_path)
        section["compliance_contact"] = ""

        with pytest.raises(
            ValueError,
            match="compliance_contact must not be empty",
        ):
            _parse_ai_act(tmp_path, section)

    def test_resolves_absolute_paths_and_all_role_branches(
        self,
        tmp_path: Path,
    ) -> None:
        absolute_notice = tmp_path / "notices" / "literacy.md"
        section = _base_ai_act_section(tmp_path)
        section["ai_literacy_record"] = str(absolute_notice)
        section["pdf_report_filename"] = "ai_act_custom.pdf"

        for role in ("provider", "modifier", "research"):
            with patch.object(
                parse_ai_act_module.SectionReader,
                "literal",
                side_effect=("deployer_readiness", role),
            ):
                cfg = _parse_ai_act(tmp_path, section)
                assert cfg.role == role
                assert cfg.ai_literacy_record == absolute_notice
                assert cfg.technical_report_paths == [
                    tmp_path / "reports" / "abs.json",
                    tmp_path / "reports" / "rel.json",
                ]
                assert cfg.pdf_report_filename == "ai_act_custom.pdf"

    def test_rejects_invalid_report_kind(self, tmp_path: Path) -> None:
        section = _base_ai_act_section(tmp_path)

        with (
            patch.object(
                parse_ai_act_module.SectionReader,
                "literal",
                side_effect=("invalid",),
            ),
            pytest.raises(
                ValueError,
                match="report_kind must be 'deployer_readiness'",
            ),
        ):
            _parse_ai_act(tmp_path, section)


@dataclass(frozen=True, slots=True)
class _DummyAIActArtifacts:
    """Minimal artifact bundle for AI Act mode error-path tests."""

    report: dict[str, object]
    coverage_ledger: dict[str, object] = field(default_factory=dict)
    control_library: dict[str, object] = field(default_factory=dict)
    controls_matrix: dict[str, object] = field(default_factory=dict)
    annex_iii_classification: dict[str, object] = field(default_factory=dict)
    risk_register: dict[str, object] = field(default_factory=dict)
    fria_prep: dict[str, object] = field(default_factory=dict)
    evidence_manifest: dict[str, object] = field(default_factory=dict)
    integrity: dict[str, object] = field(default_factory=dict)
    executive_summary_markdown: str = "summary"
    auditor_appendix_markdown: str = "appendix"
    remediation_markdown: str = "remediation"
    fria_prep_markdown: str = "fria"
    pdf_report_bytes: bytes | None = None
    pdf_report_filename: str | None = None


class TestAIActModeExtra:
    """Targeted branch coverage for the standalone AI Act mode runner."""

    def test_controls_overview_must_be_a_dict(self, tmp_path: Path) -> None:
        ctx = make_early_mode_context(tmp_path, ai_act=AIActConfig(
            company_name="Example Energy",
            system_name="Customer Assistant",
            intended_purpose="Answers customer questions.",
        ))
        artifacts = _DummyAIActArtifacts(report={"controls_overview": []})

        with (
            patch(
                "vauban.ai_act.generate_deployer_readiness_artifacts",
                return_value=artifacts,
            ),
            pytest.raises(TypeError, match="missing controls_overview"),
        ):
            _run_ai_act_mode(ctx)

    def test_controls_overview_metrics_must_be_ints(
        self,
        tmp_path: Path,
    ) -> None:
        ctx = make_early_mode_context(tmp_path, ai_act=AIActConfig(
            company_name="Example Energy",
            system_name="Customer Assistant",
            intended_purpose="Answers customer questions.",
        ))
        artifacts = _DummyAIActArtifacts(
            report={
                "controls_overview": {
                    "pass": "bad",
                    "fail": 0,
                    "unknown": 0,
                    "not_applicable": 0,
                },
            },
        )

        with (
            patch(
                "vauban.ai_act.generate_deployer_readiness_artifacts",
                return_value=artifacts,
            ),
            pytest.raises(
                TypeError,
                match="controls_overview\\['pass'\\] must be an int",
            ),
        ):
            _run_ai_act_mode(ctx)
