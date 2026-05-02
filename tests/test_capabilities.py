# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for the user-facing capability catalog."""

from vauban.capabilities import (
    capabilities_by_question,
    capability_by_name,
    capability_catalog,
    unassigned_early_modes,
)
from vauban.session import _TOOLS


def test_capability_names_are_unique() -> None:
    """Capability names should be stable identifiers."""
    names = [spec.name for spec in capability_catalog()]

    assert len(names) == len(set(names))


def test_capabilities_expose_at_least_one_surface() -> None:
    """Every catalog entry must point to a usable surface."""
    for spec in capability_catalog():
        assert spec.surfaces(), spec.name


def test_capabilities_explain_first_use_and_proof() -> None:
    """Every catalog entry should help a user start and verify the result."""
    for spec in capability_catalog():
        assert spec.use_when, spec.name
        assert spec.first_cli or spec.first_python, spec.name
        assert spec.proof, spec.name


def test_capability_catalog_covers_all_early_modes() -> None:
    """No TOML mode should be hidden outside the capability map."""
    assert unassigned_early_modes() == ()


def test_capability_catalog_covers_all_session_tools() -> None:
    """No Session tool should be hidden outside the capability map."""
    assigned = {
        tool
        for spec in capability_catalog()
        for tool in spec.session_tools
    }
    tool_names = {tool.name for tool in _TOOLS}

    assert sorted(tool_names - assigned) == []


def test_capability_lookup_by_name() -> None:
    """Lookup should return the stable catalog entry."""
    spec = capability_by_name("runtime_evidence_and_portability")

    assert spec is not None
    assert spec.question == "runtime_evidence"
    assert "behavior_trace" in spec.toml_sections
    assert spec.first_cli
    assert spec.proof


def test_capabilities_by_question_filters_catalog() -> None:
    """Question filters should return only matching entries."""
    specs = capabilities_by_question("model_transformation")

    assert specs
    assert all(spec.question == "model_transformation" for spec in specs)


def test_external_endpoint_capability_mentions_api_traces() -> None:
    """Endpoint access should make API behavior traces discoverable."""
    spec = capability_by_name("external_endpoint_audits")

    assert spec is not None
    assert "behavior_trace" in spec.toml_sections
    assert "call_api_behavior_trace" in spec.python_entrypoints
    assert "behavior_trace.jsonl" in spec.artifacts


def test_behavior_change_capability_mentions_release_gates() -> None:
    """Behavior change reports should expose release gate artifacts."""
    spec = capability_by_name("model_behavior_change_reports")

    assert spec is not None
    assert "release_gate" in spec.artifacts
    assert "docs/capabilities/change-release-gates.md" in spec.docs


def test_compliance_capability_mentions_public_sector_starter() -> None:
    """Public-sector readiness should be discoverable from capabilities."""
    spec = capability_by_name("compliance_and_release_readiness")

    assert spec is not None
    assert spec.first_cli
    assert "public_sector_readiness" in spec.first_cli[0]
    assert "docs/capabilities/public-sector-adoption.md" in spec.docs
    assert "public_sector_readiness_report.pdf" in spec.artifacts
