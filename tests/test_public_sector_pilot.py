# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for the public-sector pilot kit."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from vauban._pipeline import run
from vauban.config import load_config
from vauban.integrity import verify_ai_act_integrity

if TYPE_CHECKING:
    import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLE_DIR = _REPO_ROOT / "examples" / "public_sector_pilot"


def _copy_pilot(tmp_path: Path) -> Path:
    """Copy the pilot kit so generated outputs stay outside the repo."""
    target = tmp_path / "examples" / "public_sector_pilot"
    shutil.copytree(_EXAMPLE_DIR, target)
    return target


def test_public_sector_pilot_configs_roundtrip() -> None:
    """Pilot configs should load without model or endpoint access."""
    diff = load_config(_EXAMPLE_DIR / "behavior_diff.toml")
    readiness = load_config(_EXAMPLE_DIR / "readiness.toml")

    assert diff.behavior_diff is not None
    assert diff.behavior_diff.claim_strength == "black_box_behavioral_diff"
    assert len(diff.behavior_diff.thresholds) == 3
    assert readiness.ai_act is not None
    assert readiness.ai_act.role == "deployer"
    assert readiness.ai_act.public_sector_use is True
    assert readiness.ai_act.bundle_signature_secret_env == (
        "VAUBAN_PILOT_SIGNING_SECRET"
    )


def test_public_sector_pilot_runs_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pilot should produce a review gate and signed readiness bundle."""
    pilot_dir = _copy_pilot(tmp_path)
    monkeypatch.setenv(
        "VAUBAN_PILOT_SIGNING_SECRET",
        "public-sector-pilot-demo-secret",
    )

    run(pilot_dir / "behavior_diff.toml")
    report_dir = tmp_path / "output" / "examples" / "public_sector_pilot" / "report"
    report_path = report_dir / "behavior_diff_report.json"
    behavior_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert behavior_payload["release_gate"]["status"] == "review"
    assert behavior_payload["report"]["access"]["claim_strength"] == (
        "black_box_behavioral_diff"
    )

    run(pilot_dir / "readiness.toml")
    readiness_dir = (
        tmp_path / "output" / "examples" / "public_sector_pilot" / "readiness"
    )
    readiness_report = readiness_dir / "ai_act_readiness_report.json"
    integrity_path = readiness_dir / "ai_act_integrity.json"
    readiness_payload = json.loads(readiness_report.read_text(encoding="utf-8"))
    assert readiness_payload["technical_artifacts"]["n_attached"] == 1
    assert readiness_payload["integrity"]["status"] == "signed"
    assert (readiness_dir / "public_sector_pilot_readiness_report.pdf").exists()

    verification = verify_ai_act_integrity(
        integrity_path,
        require_signature=True,
    )
    assert verification.passed is True
    assert verification.signature_valid is True
