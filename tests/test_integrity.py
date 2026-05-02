# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for evidence-bundle integrity verification."""

from __future__ import annotations

import hashlib
import hmac
import json
import sys
from typing import TYPE_CHECKING

import pytest

from vauban.integrity import verify_ai_act_integrity

if TYPE_CHECKING:
    from pathlib import Path


def _sha256(data: bytes) -> str:
    """Return SHA-256 for test payloads."""
    return hashlib.sha256(data).hexdigest()


def _write_manifest(
    path: Path,
    artifact_hashes: dict[str, str],
    *,
    signature: str | None = None,
    signature_env_var: str | None = None,
) -> None:
    """Write a minimal AI Act integrity manifest."""
    payload: dict[str, object] = {
        "integrity_version": "ai_act_bundle_integrity_v1",
        "rulebook": {"sha256": "rulebook-sha"},
        "bundle_fingerprint": "bundle-fingerprint",
        "evidence_manifest_sha256": "evidence-sha",
        "artifact_hashes": artifact_hashes,
        "signature_status": "signed" if signature is not None else "unsigned",
        "signature_algorithm": "hmac-sha256" if signature is not None else "none",
        "signature_env_var": signature_env_var,
        "signature": signature,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _signature(secret: str, artifact_hashes: dict[str, str]) -> str:
    """Return the expected HMAC signature for the test manifest."""
    payload: dict[str, object] = {
        "bundle_fingerprint": "bundle-fingerprint",
        "rulebook_sha256": "rulebook-sha",
        "evidence_manifest_sha256": "evidence-sha",
        "artifact_hashes": artifact_hashes,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(secret.encode("utf-8"), encoded, hashlib.sha256).hexdigest()


def test_verify_ai_act_integrity_accepts_matching_artifact(
    tmp_path: Path,
) -> None:
    """A manifest with matching artifact hashes should pass."""
    artifact = tmp_path / "report.json"
    artifact.write_text('{"status":"ok"}', encoding="utf-8")
    manifest = tmp_path / "ai_act_integrity.json"
    _write_manifest(manifest, {"report.json": _sha256(artifact.read_bytes())})

    result = verify_ai_act_integrity(manifest)

    assert result.passed is True
    assert result.artifacts[0].status == "passed"
    assert result.signature_checked is False


def test_verify_ai_act_integrity_reports_mismatch(tmp_path: Path) -> None:
    """A changed artifact should fail verification."""
    artifact = tmp_path / "report.json"
    artifact.write_text('{"status":"changed"}', encoding="utf-8")
    manifest = tmp_path / "ai_act_integrity.json"
    _write_manifest(manifest, {"report.json": "0" * 64})

    result = verify_ai_act_integrity(manifest)

    assert result.passed is False
    assert result.artifacts[0].status == "mismatch"
    assert result.errors


def test_verify_ai_act_integrity_checks_signature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A signed manifest should verify when the signing secret is available."""
    artifact = tmp_path / "report.json"
    artifact.write_text('{"status":"ok"}', encoding="utf-8")
    artifact_hashes = {"report.json": _sha256(artifact.read_bytes())}
    signature = _signature("secret", artifact_hashes)
    manifest = tmp_path / "ai_act_integrity.json"
    _write_manifest(
        manifest,
        artifact_hashes,
        signature=signature,
        signature_env_var="VAUBAN_TEST_SIGNING_SECRET",
    )
    monkeypatch.setenv("VAUBAN_TEST_SIGNING_SECRET", "secret")

    result = verify_ai_act_integrity(
        manifest,
        require_signature=True,
    )

    assert result.passed is True
    assert result.signature_checked is True
    assert result.signature_valid is True


def test_cli_verify_bundle_reports_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI should expose integrity verification for reviewers."""
    artifact = tmp_path / "report.json"
    artifact.write_text('{"status":"ok"}', encoding="utf-8")
    manifest = tmp_path / "ai_act_integrity.json"
    _write_manifest(manifest, {"report.json": _sha256(artifact.read_bytes())})
    monkeypatch.setattr(
        sys,
        "argv",
        ["vauban", "verify-bundle", str(manifest)],
    )
    from vauban.__main__ import main

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "Status: passed" in captured.out
