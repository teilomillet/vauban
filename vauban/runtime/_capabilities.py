# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Runtime capability declarations and epistemic claim mapping."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vauban.behavior import (
    AccessClaimBoundary,
    AccessLevel,
    AccessPolicy,
    access_claim_boundary,
    access_policy_for_level,
)
from vauban.runtime._types import BackendCapabilities

if TYPE_CHECKING:
    from vauban.runtime._types import ForwardTrace, SupportLevel


_CAPABILITY_LABELS: tuple[tuple[str, str], ...] = (
    ("logits", "Runtime logits"),
    ("logprobs", "Runtime token logprobs"),
    ("activations", "Runtime activation traces"),
    ("interventions", "Runtime activation interventions"),
    ("kv_cache", "Runtime KV-cache access"),
    ("weight_access", "Runtime weight access"),
    ("mutable_weights", "Runtime mutable weights"),
)


def mlx_capabilities() -> BackendCapabilities:
    """Return the reference MLX runtime capability declaration."""
    return BackendCapabilities(
        name="mlx",
        device_kinds=("cpu", "gpu"),
        logits="full",
        logprobs="full",
        activations="full",
        interventions="full",
        kv_cache="full",
        weight_access="full",
        mutable_weights="full",
    )


def torch_capabilities() -> BackendCapabilities:
    """Return the current PyTorch runtime primitive capability declaration."""
    return BackendCapabilities(
        name="torch",
        device_kinds=("cpu", "cuda"),
        logits="partial",
        logprobs="partial",
        activations="partial",
        interventions="partial",
        kv_cache="unsupported",
        weight_access="partial",
        mutable_weights="unsupported",
    )


def max_capabilities() -> BackendCapabilities:
    """Return the placeholder MAX runtime capability declaration."""
    return BackendCapabilities(
        name="max",
        device_kinds=("cpu", "gpu"),
        logits="unsupported",
        logprobs="unsupported",
        activations="unsupported",
        interventions="unsupported",
        kv_cache="unsupported",
        weight_access="unsupported",
        mutable_weights="unsupported",
    )


def access_level_for_capabilities(
    capabilities: BackendCapabilities,
) -> AccessLevel:
    """Return the strongest report access level justified by runtime support."""
    if capabilities.activations != "unsupported":
        return "activations"
    if capabilities.weight_access != "unsupported":
        return "weights"
    if capabilities.logprobs != "unsupported":
        return "logprobs"
    return "black_box"


def access_boundary_for_capabilities(
    capabilities: BackendCapabilities,
) -> AccessClaimBoundary:
    """Return the behavior-report claim boundary implied by capabilities."""
    policy = access_policy_for_capabilities(capabilities)
    return access_claim_boundary(policy)


def access_policy_for_capabilities(
    capabilities: BackendCapabilities,
) -> AccessPolicy:
    """Return a behavior-report access policy from declared capabilities."""
    return access_policy_for_level(
        access_level_for_capabilities(capabilities),
        available_evidence=_available_capability_evidence(capabilities),
        missing_evidence=_missing_capability_evidence(capabilities),
        notes=_partial_capability_notes(capabilities),
    )


def access_policy_for_trace(
    capabilities: BackendCapabilities,
    trace: ForwardTrace,
) -> AccessPolicy:
    """Return a behavior-report access policy from collected runtime evidence."""
    _validate_trace_supported_by_capabilities(capabilities, trace)
    return access_policy_for_level(
        _access_level_for_trace(trace),
        available_evidence=_available_trace_evidence(trace),
        missing_evidence=_missing_trace_evidence(capabilities, trace),
        notes=_partial_capability_notes(capabilities),
    )


def declared_capabilities(name: str) -> BackendCapabilities:
    """Return declared capabilities for a known runtime backend name."""
    if name == "mlx":
        return mlx_capabilities()
    if name == "torch":
        return torch_capabilities()
    if name == "max":
        return max_capabilities()
    msg = f"Unknown runtime backend: {name!r}"
    raise ValueError(msg)


def _available_capability_evidence(
    capabilities: BackendCapabilities,
) -> tuple[str, ...]:
    """Return human-readable labels for supported runtime capabilities."""
    evidence: list[str] = []
    for capability, label in _CAPABILITY_LABELS:
        support = capabilities.support_level(capability)
        if support != "unsupported":
            evidence.append(_support_label(label, support))
    return tuple(evidence)


def _missing_capability_evidence(
    capabilities: BackendCapabilities,
) -> tuple[str, ...]:
    """Return human-readable labels for unsupported runtime capabilities."""
    missing: list[str] = []
    for capability, label in _CAPABILITY_LABELS:
        if capabilities.support_level(capability) == "unsupported":
            missing.append(label)
    return tuple(missing)


def _partial_capability_notes(
    capabilities: BackendCapabilities,
) -> tuple[str, ...]:
    """Return report notes for partially supported capabilities."""
    notes: list[str] = []
    for capability, label in _CAPABILITY_LABELS:
        if capabilities.support_level(capability) == "partial":
            notes.append(f"{label} support is partial for backend {capabilities.name}.")
    return tuple(notes)


def _support_label(label: str, support: SupportLevel) -> str:
    """Attach support level to a capability evidence label."""
    return f"{label} ({support})"


def _validate_trace_supported_by_capabilities(
    capabilities: BackendCapabilities,
    trace: ForwardTrace,
) -> None:
    """Reject traces that contain evidence the backend says it cannot produce."""
    if trace.logits is not None and capabilities.logits == "unsupported":
        msg = "runtime trace contains logits but backend declares logits unsupported"
        raise ValueError(msg)
    if trace.logprobs is not None and capabilities.logprobs == "unsupported":
        msg = (
            "runtime trace contains logprobs but backend declares logprobs"
            " unsupported"
        )
        raise ValueError(msg)
    if trace.activations and capabilities.activations == "unsupported":
        msg = (
            "runtime trace contains activations but backend declares activations"
            " unsupported"
        )
        raise ValueError(msg)
    if trace.interventions and capabilities.interventions == "unsupported":
        msg = (
            "runtime trace contains interventions but backend declares"
            " interventions unsupported"
        )
        raise ValueError(msg)


def _access_level_for_trace(trace: ForwardTrace) -> AccessLevel:
    """Return the strongest access level justified by actual trace evidence."""
    if trace.activations:
        return "activations"
    if trace.logprobs is not None or trace.logits is not None:
        return "logprobs"
    return "black_box"


def _available_trace_evidence(trace: ForwardTrace) -> tuple[str, ...]:
    """Return evidence labels for an observed runtime trace."""
    evidence = ["Runtime forward trace"]
    if trace.logits is not None:
        evidence.append("Runtime logits")
    if trace.logprobs is not None:
        evidence.append("Runtime token logprobs")
    if trace.activations:
        evidence.append("Runtime activation traces")
    if trace.interventions:
        evidence.append("Runtime activation interventions")
    return tuple(evidence)


def _missing_trace_evidence(
    capabilities: BackendCapabilities,
    trace: ForwardTrace,
) -> tuple[str, ...]:
    """Return missing evidence labels for one observed runtime trace."""
    missing: list[str] = []
    if trace.logits is None and capabilities.logits != "unsupported":
        missing.append("Runtime logits not collected")
    if trace.logprobs is None and capabilities.logprobs != "unsupported":
        missing.append("Runtime token logprobs not collected")
    if not trace.activations and capabilities.activations != "unsupported":
        missing.append("Runtime activation traces not collected")
    if not trace.interventions and capabilities.interventions != "unsupported":
        missing.append("Runtime activation interventions not collected")
    if capabilities.weight_access != "unsupported":
        missing.append("Runtime weights not attached to this trace")
    missing.extend(_missing_capability_evidence(capabilities))
    return tuple(missing)
