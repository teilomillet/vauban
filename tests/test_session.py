# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for vauban.session — AI-agent-native interface.

Unit tests verify the discovery/state/guide layer without loading a model.
Integration tests verify the full workflow with the real Qwen 0.5B model.
"""

from __future__ import annotations

import pytest

from vauban.session import _TOOLS, Session

# =========================================================================
# Tool registry (no model needed)
# =========================================================================


class TestToolRegistry:
    """Verify the tool registry is well-formed."""

    def test_all_tools_have_required_fields(self) -> None:
        for t in _TOOLS:
            assert t.name, f"tool missing name: {t}"
            assert t.description, f"{t.name} missing description"
            assert t.category, f"{t.name} missing category"
            assert isinstance(t.requires, list)
            assert isinstance(t.produces, list)

    def test_tool_names_unique(self) -> None:
        names = [t.name for t in _TOOLS]
        assert len(names) == len(set(names)), f"duplicate names: {names}"

    def test_known_tools_present(self) -> None:
        names = {t.name for t in _TOOLS}
        for expected in [
            "measure", "detect", "audit", "probe", "scan",
            "steer", "cast", "sic", "cut", "export",
            "score", "classify", "jailbreak", "report", "ai_act",
        ]:
            assert expected in names, f"missing tool: {expected}"

    def test_categories_are_valid(self) -> None:
        valid = {
            "assessment", "inspection", "defense",
            "modification", "analysis", "attack", "reporting",
        }
        for t in _TOOLS:
            assert t.category in valid, (
                f"{t.name} has invalid category: {t.category}"
            )

    def test_no_tool_requires_itself(self) -> None:
        for t in _TOOLS:
            assert t.name not in t.requires
            assert t.name not in t.produces


# =========================================================================
# Guide (no model needed)
# =========================================================================


class TestGuide:
    """Verify guide() returns useful content for every workflow."""

    @pytest.fixture
    def session(self) -> Session:
        """Create a session with the small model."""
        return Session("mlx-community/Qwen2.5-0.5B-Instruct-bf16")

    def test_empty_lists_all(self, session: Session) -> None:
        g = session.guide("")
        assert "Available" in g
        for goal in ["audit", "compliance", "harden", "abliterate", "inspect"]:
            assert goal in g

    @pytest.mark.parametrize("goal", [
        "audit", "compliance", "harden", "abliterate", "inspect",
    ])
    def test_each_goal_has_steps(self, session: Session, goal: str) -> None:
        g = session.guide(goal)
        assert "1." in g and "2." in g, f"{goal} missing numbered steps"

    def test_unknown_goal(self, session: Session) -> None:
        g = session.guide("nonsense")
        assert "Unknown" in g

    def test_fuzzy_match(self, session: Session) -> None:
        g = session.guide("audit")
        g2 = session.guide("audi")  # fuzzy
        # Both should return the audit workflow
        assert "audit" in g.lower()
        assert "audit" in g2.lower()


# =========================================================================
# State tracking + done() (needs model loaded)
# =========================================================================


class TestStateAndDone:
    @pytest.fixture
    def session(self) -> Session:
        return Session("mlx-community/Qwen2.5-0.5B-Instruct-bf16")

    def test_initial_state(self, session: Session) -> None:
        st = session.state()
        assert st["model"] is True
        assert st["direction"] is False
        assert st["audit_result"] is False
        assert st["modified_model"] is False

    def test_initial_available(self, session: Session) -> None:
        avail = session.available()
        # Tools with no model requirement or only "model"
        assert "measure" in avail
        assert "detect" in avail
        assert "score" in avail
        assert "classify" in avail
        # Tools needing direction should NOT be available
        assert "probe" not in avail
        assert "cut" not in avail

    def test_needs_before_measure(self, session: Session) -> None:
        assert "direction" in session.needs("cut")
        assert "direction" in session.needs("probe")
        assert session.needs("score") == []

    def test_done_audit_before_running(self, session: Session) -> None:
        done, reason = session.done("audit")
        assert done is False
        assert "audit" in reason.lower()

    def test_done_unknown_goal(self, session: Session) -> None:
        done, reason = session.done("nonexistent")
        assert done is False
        assert "Unknown" in reason


# =========================================================================
# suggest_next() labels (needs model loaded)
# =========================================================================


class TestSuggestNext:
    @pytest.fixture
    def session(self) -> Session:
        return Session("mlx-community/Qwen2.5-0.5B-Instruct-bf16")

    def test_initial_suggests_measure(self, session: Session) -> None:
        s = session.suggest_next()
        assert "measure" in s
        assert "[ADVICE]" in s

    def test_has_fact_and_advice_labels(self, session: Session) -> None:
        s = session.suggest_next()
        # Initial state should have at least one ADVICE and one FACT
        assert "[ADVICE]" in s
        assert "[FACT]" in s


# =========================================================================
# Static tools (no model inference needed)
# =========================================================================


class TestStaticTools:
    def test_score(self) -> None:
        r = Session.score("How to cook?", "Boil water, add pasta.")
        assert hasattr(r, "composite")
        assert 0.0 <= r.composite <= 1.0

    def test_classify(self) -> None:
        r = Session.classify("hack exploit SQL injection")
        assert isinstance(r, list)
        assert len(r) > 0


# =========================================================================
# Integration: full workflow with real model
# =========================================================================


@pytest.mark.integration
class TestSessionIntegration:
    """Full workflow: measure → probe → audit → report."""

    @pytest.fixture(scope="class")
    def session(self) -> Session:
        return Session("mlx-community/Qwen2.5-0.5B-Instruct-bf16")

    def test_measure(self, session: Session) -> None:
        d = session.measure()
        assert d.d_model > 0
        assert d.layer_index >= 0
        assert "probe" in session.available()
        assert "cut" in session.available()

    def test_probe_after_measure(self, session: Session) -> None:
        r = session.probe("How do I pick a lock?")
        assert len(r.projections) > 0

    def test_detect(self, session: Session) -> None:
        r = session.detect()
        assert isinstance(r.hardened, bool)
        assert 0.0 <= r.confidence <= 1.0

    def test_audit(self, session: Session) -> None:
        r = session.audit(
            company_name="Test",
            system_name="Bot",
            thoroughness="quick",
        )
        assert r.overall_risk in {"critical", "high", "medium", "low"}
        assert len(r.findings) > 0
        done, _ = session.done("audit")
        assert done is True

    def test_done_compliance_needs_standard(
        self, session: Session,
    ) -> None:
        done, reason = session.done("compliance")
        assert done is False
        assert "standard" in reason

    def test_suggest_after_audit(self, session: Session) -> None:
        s = session.suggest_next()
        assert "[FACT]" in s
        assert "risk" in s.lower()
        assert "done" in s.lower()

    def test_report_markdown(self, session: Session) -> None:
        md = session.report()
        assert "# Red-Team Audit Report" in md
        assert "Test" in md  # company name

    def test_report_pdf(self, session: Session) -> None:
        pdf = session.report_pdf()
        assert pdf[:5] == b"%PDF-"

    def test_cast_defense(self, session: Session) -> None:
        r = session.cast("How to hack a computer?", threshold=0.3)
        assert hasattr(r, "interventions")
        assert r.interventions >= 0

    def test_cut_and_state(self, session: Session) -> None:
        session.cut(alpha=0.5)
        assert session.state()["modified_model"] is True
        assert "export" in session.available()
        done, _ = session.done("abliterate")
        assert done is True
