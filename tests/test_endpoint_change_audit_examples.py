# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for endpoint change audit examples."""

from pathlib import Path

from vauban.config import load_config

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLE_DIR = _REPO_ROOT / "examples" / "endpoint_change_audit"
_DOC_PATH = _REPO_ROOT / "docs" / "capabilities" / "change-release-gates.md"


def test_endpoint_change_audit_examples_roundtrip() -> None:
    """Endpoint trace and diff examples should load as valid configs."""
    baseline = load_config(_EXAMPLE_DIR / "baseline_trace.toml")
    candidate = load_config(_EXAMPLE_DIR / "candidate_trace.toml")
    diff = load_config(_EXAMPLE_DIR / "diff.toml")

    assert baseline.model_path == ""
    assert baseline.behavior_trace is not None
    assert baseline.behavior_trace.runtime_backend == "api"
    assert baseline.behavior_trace.api is not None
    assert baseline.behavior_trace.api.api_key_env == "BASELINE_API_KEY"
    assert candidate.behavior_trace is not None
    assert candidate.behavior_trace.api is not None
    assert candidate.behavior_trace.api.api_key_env == "CANDIDATE_API_KEY"
    assert diff.behavior_diff is not None
    assert diff.behavior_diff.transformation_kind == "endpoint_update"
    assert diff.behavior_diff.claim_strength == "black_box_behavioral_diff"
    assert len(diff.behavior_diff.thresholds) == 3


def test_public_endpoint_change_docs_do_not_name_specific_competitors() -> None:
    """Public positioning should describe Vauban's workflow directly."""
    public_text = "\n".join((
        (_EXAMPLE_DIR / "README.md").read_text(encoding="utf-8"),
        _DOC_PATH.read_text(encoding="utf-8"),
    )).lower()

    assert "giskard" not in public_text
