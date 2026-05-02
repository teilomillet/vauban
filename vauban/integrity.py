# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Integrity verification for Vauban evidence bundles."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ArtifactStatus = Literal["passed", "missing", "mismatch"]


@dataclass(frozen=True, slots=True)
class ArtifactVerification:
    """Verification result for one bundle artifact."""

    filename: str
    expected_sha256: str
    actual_sha256: str | None
    status: ArtifactStatus

    def to_dict(self) -> dict[str, str | None]:
        """Serialize the artifact verification result."""
        return {
            "filename": self.filename,
            "expected_sha256": self.expected_sha256,
            "actual_sha256": self.actual_sha256,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class IntegrityVerificationResult:
    """Verification result for an AI Act readiness integrity manifest."""

    manifest_path: Path
    base_dir: Path
    artifacts: tuple[ArtifactVerification, ...]
    signature_status: str
    signature_checked: bool
    signature_valid: bool | None
    signature_env_var: str | None
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Return whether all artifact and requested signature checks passed."""
        return not self.errors and all(
            artifact.status == "passed" for artifact in self.artifacts
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the full verification result."""
        return {
            "manifest_path": str(self.manifest_path),
            "base_dir": str(self.base_dir),
            "passed": self.passed,
            "artifact_count": len(self.artifacts),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "signature_status": self.signature_status,
            "signature_checked": self.signature_checked,
            "signature_valid": self.signature_valid,
            "signature_env_var": self.signature_env_var,
            "errors": list(self.errors),
        }


def verify_ai_act_integrity(
    manifest_path: str | Path,
    *,
    base_dir: str | Path | None = None,
    secret_env: str | None = None,
    require_signature: bool = False,
) -> IntegrityVerificationResult:
    """Verify an AI Act readiness bundle integrity manifest.

    The verifier checks every artifact hash listed in `ai_act_integrity.json`.
    If a signing secret is available, it also verifies the HMAC signature over
    the same signature input used by bundle generation.
    """
    manifest_path_obj = Path(manifest_path)
    effective_base_dir = (
        Path(base_dir)
        if base_dir is not None
        else manifest_path_obj.parent
    )
    manifest = _load_manifest(manifest_path_obj)
    artifact_hashes = _string_mapping(
        manifest.get("artifact_hashes"),
        "artifact_hashes",
    )
    artifacts = tuple(
        _verify_artifact(effective_base_dir, filename, expected_hash)
        for filename, expected_hash in sorted(artifact_hashes.items())
    )
    errors = [
        (
            f"{artifact.filename}: {artifact.status}"
            if artifact.actual_sha256 is None
            else (
                f"{artifact.filename}: expected {artifact.expected_sha256},"
                f" got {artifact.actual_sha256}"
            )
        )
        for artifact in artifacts
        if artifact.status != "passed"
    ]

    signature_env_var = secret_env or _optional_string(
        manifest.get("signature_env_var"),
    )
    signature_status = _optional_string(
        manifest.get("signature_status"),
    ) or "unknown"
    signature_checked = False
    signature_valid: bool | None = None
    expected_signature = _optional_string(manifest.get("signature"))
    secret = (
        os.environ.get(signature_env_var)
        if signature_env_var is not None
        else None
    )

    if secret_env is not None and secret is None:
        errors.append(f"signature secret env var {secret_env!r} is not set")
    if require_signature and signature_status != "signed":
        errors.append("signature verification was required but manifest is unsigned")
    if require_signature and secret is None:
        errors.append("signature verification was required but no secret is available")

    if secret is not None:
        signature_checked = True
        if expected_signature is None:
            signature_valid = False
            errors.append("signature secret was provided but manifest has no signature")
        else:
            signature_input = _signature_input(manifest, artifact_hashes)
            actual_signature = _signature_hex(secret, signature_input)
            signature_valid = hmac.compare_digest(
                expected_signature,
                actual_signature,
            )
            if not signature_valid:
                errors.append("signature mismatch")

    return IntegrityVerificationResult(
        manifest_path=manifest_path_obj,
        base_dir=effective_base_dir,
        artifacts=artifacts,
        signature_status=signature_status,
        signature_checked=signature_checked,
        signature_valid=signature_valid,
        signature_env_var=signature_env_var,
        errors=tuple(errors),
    )


def format_integrity_verification(
    result: IntegrityVerificationResult,
) -> str:
    """Render an integrity verification result for CLI output."""
    missing = sum(1 for item in result.artifacts if item.status == "missing")
    mismatched = sum(1 for item in result.artifacts if item.status == "mismatch")
    if result.signature_checked:
        signature = "valid" if result.signature_valid else "invalid"
    elif result.signature_status == "signed":
        signature = "not checked"
    else:
        signature = result.signature_status
    lines = [
        f"Integrity manifest: {result.manifest_path}",
        f"Artifact base dir: {result.base_dir}",
        (
            f"Artifacts: {len(result.artifacts)} checked,"
            f" {missing} missing, {mismatched} mismatched"
        ),
        f"Signature: {signature}",
        f"Status: {'passed' if result.passed else 'failed'}",
    ]
    if result.errors:
        lines.append("Errors:")
        lines.extend(f"  - {error}" for error in result.errors)
    return "\n".join(lines) + "\n"


def _load_manifest(path: Path) -> dict[str, object]:
    """Load an integrity manifest as a JSON object."""
    raw_payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_payload, dict):
        msg = "integrity manifest must be a JSON object"
        raise ValueError(msg)
    return dict(raw_payload)


def _string_mapping(value: object, label: str) -> dict[str, str]:
    """Return a string-to-string mapping or raise a clear error."""
    if not isinstance(value, dict):
        msg = f"{label} must be a JSON object"
        raise ValueError(msg)
    result: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str) or not isinstance(raw_value, str):
            msg = f"{label} must map strings to strings"
            raise ValueError(msg)
        result[raw_key] = raw_value
    return result


def _optional_string(value: object) -> str | None:
    """Return a non-empty string value when present."""
    if isinstance(value, str) and value:
        return value
    return None


def _required_string(value: object, label: str) -> str:
    """Return a required string field from the manifest."""
    if not isinstance(value, str) or not value:
        msg = f"{label} must be a non-empty string"
        raise ValueError(msg)
    return value


def _verify_artifact(
    base_dir: Path,
    filename: str,
    expected_sha256: str,
) -> ArtifactVerification:
    """Verify one artifact hash from the manifest."""
    path = base_dir / filename
    if not path.exists():
        return ArtifactVerification(
            filename=filename,
            expected_sha256=expected_sha256,
            actual_sha256=None,
            status="missing",
        )
    actual_sha256 = _sha256_bytes(path.read_bytes())
    status: ArtifactStatus = (
        "passed" if actual_sha256 == expected_sha256 else "mismatch"
    )
    return ArtifactVerification(
        filename=filename,
        expected_sha256=expected_sha256,
        actual_sha256=actual_sha256,
        status=status,
    )


def _signature_input(
    manifest: dict[str, object],
    artifact_hashes: dict[str, str],
) -> dict[str, object]:
    """Reconstruct the signed payload from an integrity manifest."""
    raw_rulebook = manifest.get("rulebook")
    if not isinstance(raw_rulebook, dict):
        msg = "rulebook must be a JSON object"
        raise ValueError(msg)
    rulebook = dict(raw_rulebook)
    return {
        "bundle_fingerprint": _required_string(
            manifest.get("bundle_fingerprint"),
            "bundle_fingerprint",
        ),
        "rulebook_sha256": _required_string(
            rulebook.get("sha256"),
            "rulebook.sha256",
        ),
        "evidence_manifest_sha256": _required_string(
            manifest.get("evidence_manifest_sha256"),
            "evidence_manifest_sha256",
        ),
        "artifact_hashes": artifact_hashes,
    }


def _sha256_bytes(payload: bytes) -> str:
    """Return a SHA-256 hex digest for bytes."""
    return hashlib.sha256(payload).hexdigest()


def _signature_hex(secret: str, payload: object) -> str:
    """Return an HMAC-SHA256 signature for a canonical JSON payload."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(
        secret.encode("utf-8"),
        encoded,
        hashlib.sha256,
    ).hexdigest()
