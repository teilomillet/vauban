# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for audit config parsing, risk scoring, and report generation."""

import pytest

from vauban.audit import (
    _compute_overall_risk,
    _rate_to_severity,
    audit_result_to_dict,
    audit_result_to_markdown,
)
from vauban.config._parse_audit import _parse_audit
from vauban.types import AuditFinding, AuditResult


class TestParseAudit:
    """Tests for the [audit] TOML parser."""

    def test_absent_returns_none(self) -> None:
        assert _parse_audit({}) is None

    def test_minimal_config(self) -> None:
        raw = {
            "audit": {
                "company_name": "Acme",
                "system_name": "Bot",
            },
        }
        config = _parse_audit(raw)
        assert config is not None
        assert config.company_name == "Acme"
        assert config.system_name == "Bot"
        assert config.thoroughness == "standard"
        assert config.pdf_report is True

    def test_all_options(self) -> None:
        raw = {
            "audit": {
                "company_name": "Corp",
                "system_name": "AI",
                "thoroughness": "deep",
                "pdf_report": False,
                "pdf_report_filename": "custom.pdf",
                "softprompt_steps": 500,
                "attacks": ["gcg", "bijection"],
                "jailbreak_strategies": ["identity_dissolution"],
            },
        }
        config = _parse_audit(raw)
        assert config is not None
        assert config.thoroughness == "deep"
        assert config.pdf_report is False
        assert config.softprompt_steps == 500
        assert config.attacks == ["gcg", "bijection"]
        assert config.jailbreak_strategies == ["identity_dissolution"]

    def test_invalid_thoroughness_raises(self) -> None:
        raw = {
            "audit": {
                "company_name": "A",
                "system_name": "B",
                "thoroughness": "extreme",
            },
        }
        with pytest.raises(ValueError, match="thoroughness"):
            _parse_audit(raw)

    def test_invalid_pdf_filename_raises(self) -> None:
        raw = {
            "audit": {
                "company_name": "A",
                "system_name": "B",
                "pdf_report_filename": "report.txt",
            },
        }
        with pytest.raises(ValueError, match=r"\.pdf"):
            _parse_audit(raw)

    def test_negative_softprompt_steps_raises(self) -> None:
        raw = {
            "audit": {
                "company_name": "A",
                "system_name": "B",
                "softprompt_steps": 0,
            },
        }
        with pytest.raises(ValueError, match="softprompt_steps"):
            _parse_audit(raw)


class TestRiskScoring:
    """Tests for severity and risk computation."""

    def test_rate_to_severity(self) -> None:
        assert _rate_to_severity(0.0) == "info"
        assert _rate_to_severity(0.05) == "low"
        assert _rate_to_severity(0.15) == "medium"
        assert _rate_to_severity(0.3) == "high"
        assert _rate_to_severity(0.6) == "critical"

    def test_overall_risk_critical(self) -> None:
        findings = [
            AuditFinding("a", "critical", "t", "d", "e", "r"),
            AuditFinding("b", "low", "t", "d", "e", "r"),
        ]
        assert _compute_overall_risk(findings) == "critical"

    def test_overall_risk_high(self) -> None:
        findings = [
            AuditFinding("a", "high", "t", "d", "e", "r"),
            AuditFinding("b", "info", "t", "d", "e", "r"),
        ]
        assert _compute_overall_risk(findings) == "high"

    def test_overall_risk_low(self) -> None:
        findings = [
            AuditFinding("a", "low", "t", "d", "e", "r"),
            AuditFinding("b", "info", "t", "d", "e", "r"),
        ]
        assert _compute_overall_risk(findings) == "low"

    def test_overall_risk_empty(self) -> None:
        assert _compute_overall_risk([]) == "low"


class TestSerialization:
    """Tests for audit result serialization."""

    def _make_result(self) -> AuditResult:
        return AuditResult(
            company_name="Test Corp",
            system_name="Test Bot",
            model_path="test-model",
            thoroughness="quick",
            overall_risk="medium",
            findings=[
                AuditFinding(
                    "attack_resistance", "medium",
                    "Jailbreak: 15% bypass",
                    "15 of 100 bypassed", "n=100", "Apply CAST",
                ),
            ],
            detect_hardened=False,
            detect_confidence=0.3,
            jailbreak_success_rate=0.15,
            jailbreak_total=100,
            softprompt_success_rate=None,
            bijection_success_rate=None,
            surface_refusal_rate=None,
            surface_coverage=None,
            guard_circuit_break_rate=None,
        )

    def test_to_dict(self) -> None:
        result = self._make_result()
        d = audit_result_to_dict(result)
        assert d["company_name"] == "Test Corp"
        assert d["overall_risk"] == "medium"
        assert len(d["findings"]) == 1  # type: ignore[arg-type]
        assert "generated_at" in d
        assert d["metrics"]["jailbreak_success_rate"] == 0.15  # type: ignore[index]

    def test_to_markdown(self) -> None:
        result = self._make_result()
        md = audit_result_to_markdown(result)
        assert "# Red-Team Audit Report" in md
        assert "Test Corp" in md
        assert "MEDIUM" in md
        assert "15%" in md
        assert "Apply CAST" in md
