# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Runtime capability declarations and epistemic claim mapping."""

from __future__ import annotations

from vauban.behavior import (
    AccessClaimBoundary,
    AccessLevel,
    access_claim_boundary,
    access_policy_for_level,
)
from vauban.runtime._types import BackendCapabilities


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
    policy = access_policy_for_level(access_level_for_capabilities(capabilities))
    return access_claim_boundary(policy)


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
