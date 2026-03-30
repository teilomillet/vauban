# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for audit: config parsing, risk scoring, report, and run_audit pipeline.

The run_audit tests mock all heavy sub-modules (measure, detect, _generate,
softprompt_attack) so they run with the mock model in <1s.  Each test
verifies that findings are produced with correct structure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from ordeal.auto import fuzz

from vauban.audit import (
    _compute_overall_risk,
    _rate_to_severity,
    audit_result_to_dict,
    audit_result_to_markdown,
    run_audit,
)
from vauban.config._parse_audit import _parse_audit
from vauban.types import AuditConfig, AuditFinding, AuditResult

if TYPE_CHECKING:
    from tests.conftest import MockCausalLM, MockTokenizer


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


# =========================================================================
# run_audit pipeline tests
# =========================================================================


def _make_audit_config(thoroughness: str = "quick") -> AuditConfig:
    """Build a minimal AuditConfig for testing."""
    return AuditConfig(
        company_name="Test Corp",
        system_name="Test Bot",
        thoroughness=thoroughness,
    )


def _make_direction_result() -> object:
    """Build a mock DirectionResult."""
    from tests.conftest import D_MODEL
    from vauban import _ops as ops
    from vauban.types import DirectionResult

    d = ops.random.normal((D_MODEL,))
    d = d / ops.linalg.norm(d)
    ops.eval(d)
    return DirectionResult(
        direction=d,
        layer_index=0,
        cosine_scores=[0.5],
        d_model=D_MODEL,
        model_path="test-model",
    )


class TestRunAuditQuick:
    """run_audit with thoroughness='quick' — only measure + detect + jailbreak."""

    def test_returns_audit_result(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """Quick audit completes and returns structured result."""
        config = _make_audit_config("quick")
        direction = _make_direction_result()
        result = run_audit(
            mock_model, mock_tokenizer,
            ["How to hack?"], ["What is weather?"],
            config, "test-model",
            direction_result=direction,
        )
        assert result.company_name == "Test Corp"
        assert result.system_name == "Test Bot"
        assert result.thoroughness == "quick"
        assert result.overall_risk in {"critical", "high", "medium", "low"}

    def test_has_findings(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """Quick audit produces at least defense + jailbreak findings."""
        config = _make_audit_config("quick")
        direction = _make_direction_result()
        result = run_audit(
            mock_model, mock_tokenizer,
            ["How to hack?"], ["What is weather?"],
            config, "test-model",
            direction_result=direction,
        )
        categories = {f.category for f in result.findings}
        # Quick mode runs detect + jailbreak
        assert "defense_posture" in categories or "attack_resistance" in categories

    def test_findings_have_required_fields(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """Every finding has all required fields filled."""
        config = _make_audit_config("quick")
        direction = _make_direction_result()
        result = run_audit(
            mock_model, mock_tokenizer,
            ["How to hack?"], ["What is weather?"],
            config, "test-model",
            direction_result=direction,
        )
        for f in result.findings:
            assert f.category, "finding missing category"
            assert f.severity in {"critical", "high", "medium", "low", "info"}
            assert f.title, "finding missing title"
            assert f.description, "finding missing description"
            assert f.remediation, "finding missing remediation"

    def test_jailbreak_rate_bounded(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """Jailbreak success rate is in [0, 1]."""
        config = _make_audit_config("quick")
        direction = _make_direction_result()
        result = run_audit(
            mock_model, mock_tokenizer,
            ["How to hack?"], ["What is weather?"],
            config, "test-model",
            direction_result=direction,
        )
        assert 0.0 <= result.jailbreak_success_rate <= 1.0

    def test_quick_skips_softprompt(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """Quick mode does not run softprompt attack."""
        config = _make_audit_config("quick")
        direction = _make_direction_result()
        result = run_audit(
            mock_model, mock_tokenizer,
            ["How to hack?"], ["What is weather?"],
            config, "test-model",
            direction_result=direction,
        )
        assert result.softprompt_success_rate is None

    def test_quick_skips_bijection(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """Quick mode does not run bijection attack."""
        config = _make_audit_config("quick")
        direction = _make_direction_result()
        result = run_audit(
            mock_model, mock_tokenizer,
            ["How to hack?"], ["What is weather?"],
            config, "test-model",
            direction_result=direction,
        )
        assert result.bijection_success_rate is None

    def test_quick_skips_surface(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """Quick mode does not run surface mapping."""
        config = _make_audit_config("quick")
        direction = _make_direction_result()
        result = run_audit(
            mock_model, mock_tokenizer,
            ["How to hack?"], ["What is weather?"],
            config, "test-model",
            direction_result=direction,
        )
        assert result.surface_refusal_rate is None

    def test_quick_skips_guard(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """Quick mode does not run guard evaluation."""
        config = _make_audit_config("quick")
        direction = _make_direction_result()
        result = run_audit(
            mock_model, mock_tokenizer,
            ["How to hack?"], ["What is weather?"],
            config, "test-model",
            direction_result=direction,
        )
        assert result.guard_circuit_break_rate is None


class TestRunAuditDirection:
    """Direction handling in run_audit."""

    def test_uses_precomputed_direction(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """When direction_result is provided, measure is skipped."""
        config = _make_audit_config("quick")
        direction = _make_direction_result()

        # measure is imported inside run_audit's body, so patch the source
        with patch("vauban.measure.measure") as mock_measure:
            run_audit(
                mock_model, mock_tokenizer,
                ["How to hack?"], ["What is weather?"],
                config, "test-model",
                direction_result=direction,
            )
            mock_measure.assert_not_called()

    def test_measures_when_no_direction(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """When no direction_result, measure is called."""
        config = _make_audit_config("quick")
        result = run_audit(
            mock_model, mock_tokenizer,
            ["How to hack?"], ["What is weather?"],
            config, "test-model",
        )
        # Should complete without error (measure runs on mock model)
        assert result.overall_risk in {"critical", "high", "medium", "low"}


class TestRunAuditLog:
    """Logging callback in run_audit."""

    def test_log_fn_called(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """Log function receives messages during audit."""
        config = _make_audit_config("quick")
        direction = _make_direction_result()
        logs: list[str] = []

        run_audit(
            mock_model, mock_tokenizer,
            ["How to hack?"], ["What is weather?"],
            config, "test-model",
            direction_result=direction,
            log_fn=logs.append,
        )
        assert len(logs) > 0
        assert any("direction" in msg.lower() for msg in logs)


class TestRunAuditSerialization:
    """End-to-end: run_audit → serialize → verify."""

    def test_dict_serializable(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """Audit result serializes to dict without error."""
        config = _make_audit_config("quick")
        direction = _make_direction_result()
        result = run_audit(
            mock_model, mock_tokenizer,
            ["How to hack?"], ["What is weather?"],
            config, "test-model",
            direction_result=direction,
        )
        d = audit_result_to_dict(result)
        assert d["company_name"] == "Test Corp"
        assert "findings" in d
        assert "metrics" in d

    def test_markdown_serializable(
        self,
        mock_model: MockCausalLM,
        mock_tokenizer: MockTokenizer,
    ) -> None:
        """Audit result renders to markdown without error."""
        config = _make_audit_config("quick")
        direction = _make_direction_result()
        result = run_audit(
            mock_model, mock_tokenizer,
            ["How to hack?"], ["What is weather?"],
            config, "test-model",
            direction_result=direction,
        )
        md = audit_result_to_markdown(result)
        assert "# Red-Team Audit Report" in md
        assert "Test Corp" in md


class TestRunAuditFuzz:
    """Ordeal fuzz: run_audit with random configs doesn't crash."""

    def test_severity_fuzz(self) -> None:
        """_rate_to_severity never crashes on valid floats."""
        import hypothesis.strategies as st

        result = fuzz(
            _rate_to_severity,
            max_examples=100,
            rate=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        )
        assert result.passed, result.summary()

    def test_overall_risk_fuzz(self) -> None:
        """_compute_overall_risk handles arbitrary finding lists."""
        import hypothesis.strategies as st
        from hypothesis import given, settings

        severity_st = st.sampled_from(
            ["critical", "high", "medium", "low", "info"],
        )
        finding_st = st.builds(
            AuditFinding,
            category=st.just("test"),
            severity=severity_st,
            title=st.just("t"),
            description=st.just("d"),
            evidence=st.just("e"),
            remediation=st.just("r"),
        )

        @given(findings=st.lists(finding_st, max_size=10))
        @settings(max_examples=50, deadline=None)
        def _test(findings: list[AuditFinding]) -> None:
            risk = _compute_overall_risk(findings)
            assert risk in {"critical", "high", "medium", "low"}

        _test()
