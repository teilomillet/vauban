# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Runtime evidence serialization for regression-safe report plumbing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vauban.behavior import AccessPolicy, EvidenceRef, JsonValue
from vauban.runtime._capabilities import access_policy_for_trace

if TYPE_CHECKING:
    from vauban.runtime._types import (
        BackendCapabilities,
        ForwardTrace,
        RuntimeValue,
        TensorLike,
    )


@dataclass(frozen=True, slots=True)
class RuntimeReportEvidence:
    """Report-ready runtime evidence with an explicit access policy."""

    access: AccessPolicy
    evidence: tuple[EvidenceRef, ...]
    capabilities: dict[str, RuntimeValue]
    trace: dict[str, RuntimeValue]

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize runtime evidence for JSON report payloads."""
        return {
            "access": self.access.to_dict(),
            "evidence": [ref.to_dict() for ref in self.evidence],
            "capabilities": _runtime_dict_to_json(self.capabilities),
            "trace": _runtime_dict_to_json(self.trace),
        }


def runtime_report_evidence(
    capabilities: BackendCapabilities,
    trace: ForwardTrace,
    *,
    prefix: str = "runtime",
) -> RuntimeReportEvidence:
    """Build a report-ready evidence package from one runtime trace."""
    capability_ref = EvidenceRef(
        evidence_id=f"{prefix}.capabilities",
        kind="run_report",
        description="Runtime capability declaration used for access boundaries.",
    )
    return RuntimeReportEvidence(
        access=access_policy_for_trace(capabilities, trace),
        evidence=(capability_ref, *runtime_evidence_refs(trace, prefix=prefix)),
        capabilities=runtime_capability_snapshot(capabilities),
        trace=forward_trace_summary(trace),
    )


def runtime_capability_snapshot(
    capabilities: BackendCapabilities,
) -> dict[str, RuntimeValue]:
    """Return a stable serialized capability snapshot."""
    return capabilities.to_dict()


def forward_trace_summary(trace: ForwardTrace) -> dict[str, RuntimeValue]:
    """Return a stable, JSON-compatible summary of runtime evidence."""
    device: dict[str, RuntimeValue] = {
        "kind": trace.device.kind,
        "label": trace.device.label,
    }
    activation_shapes: dict[str, RuntimeValue] = {
        str(layer): _shape(tensor)
        for layer, tensor in sorted(trace.activations.items())
    }
    interventions: list[RuntimeValue] = [
        record.to_dict() for record in trace.interventions
    ]
    profile: list[RuntimeValue] = []
    for stage in trace.profile:
        stage_device: dict[str, RuntimeValue] | None = None
        if stage.device is not None:
            stage_device = {
                "kind": stage.device.kind,
                "label": stage.device.label,
            }
        profile.append({
            "name": stage.name,
            "device": stage_device,
            "memory_bytes": stage.memory_bytes,
            "metadata": dict(stage.metadata),
        })
    logits_shape: RuntimeValue = _shape_or_none(trace.logits)
    logprobs_shape: RuntimeValue = _shape_or_none(trace.logprobs)
    summary: dict[str, RuntimeValue] = {}
    summary["device"] = device
    summary["logits_shape"] = logits_shape
    summary["logprobs_shape"] = logprobs_shape
    summary["activation_shapes"] = activation_shapes
    summary["interventions"] = interventions
    summary["profile"] = profile
    return summary


def runtime_evidence_refs(
    trace: ForwardTrace,
    *,
    prefix: str = "runtime",
) -> tuple[EvidenceRef, ...]:
    """Return report evidence references justified by a forward trace."""
    refs = [
        EvidenceRef(
            evidence_id=f"{prefix}.trace",
            kind="trace",
            description="Runtime forward trace summary.",
        ),
    ]
    if trace.logprobs is not None:
        refs.append(
            EvidenceRef(
                evidence_id=f"{prefix}.logprobs",
                kind="logprobs",
                description="Runtime token-probability evidence.",
            ),
        )
    if trace.activations:
        refs.append(
            EvidenceRef(
                evidence_id=f"{prefix}.activations",
                kind="activation",
                description="Runtime activation evidence.",
            ),
        )
    return tuple(refs)


def _shape_or_none(tensor: TensorLike | None) -> list[RuntimeValue] | None:
    """Serialize tensor shape if present."""
    if tensor is None:
        return None
    return _shape(tensor)


def _shape(tensor: TensorLike) -> list[RuntimeValue]:
    """Serialize tensor shape as JSON-compatible runtime values."""
    return [int(dim) for dim in tensor.shape]


def _runtime_dict_to_json(data: dict[str, RuntimeValue]) -> dict[str, JsonValue]:
    """Convert runtime JSON-compatible values to behavior JSON-compatible values."""
    return {key: _runtime_value_to_json(value) for key, value in data.items()}


def _runtime_value_to_json(value: RuntimeValue) -> JsonValue:
    """Convert a runtime value into the behavior-report JSON alias."""
    if isinstance(value, list):
        return [_runtime_value_to_json(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _runtime_value_to_json(item)
            for key, item in value.items()
        }
    return value
