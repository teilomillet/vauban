# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Runtime primitive contracts for Vauban backend execution."""

from vauban.runtime._capabilities import (
    access_boundary_for_capabilities,
    access_level_for_capabilities,
    declared_capabilities,
    max_capabilities,
    mlx_capabilities,
    torch_capabilities,
)
from vauban.runtime._evidence import (
    forward_trace_summary,
    runtime_capability_snapshot,
    runtime_evidence_refs,
)
from vauban.runtime._profiling import StageTimer, profile_stage
from vauban.runtime._protocols import ModelRuntime, RuntimeStage
from vauban.runtime._registry import (
    available_runtime_backends,
    create_runtime,
    runtime_capabilities,
)
from vauban.runtime._types import (
    ActivationIntervention,
    BackendCapabilities,
    DeviceKind,
    DeviceRef,
    ForwardRequest,
    ForwardTrace,
    InterventionRecord,
    LoadedModel,
    ModelRef,
    RuntimeBackendName,
    RuntimeScalar,
    RuntimeValue,
    StageProfile,
    SupportLevel,
    TensorLike,
    TokenizedPrompt,
    TokenizeRequest,
)

__all__ = [
    "ActivationIntervention",
    "BackendCapabilities",
    "DeviceKind",
    "DeviceRef",
    "ForwardRequest",
    "ForwardTrace",
    "InterventionRecord",
    "LoadedModel",
    "ModelRef",
    "ModelRuntime",
    "RuntimeBackendName",
    "RuntimeScalar",
    "RuntimeStage",
    "RuntimeValue",
    "StageProfile",
    "StageTimer",
    "SupportLevel",
    "TensorLike",
    "TokenizeRequest",
    "TokenizedPrompt",
    "access_boundary_for_capabilities",
    "access_level_for_capabilities",
    "available_runtime_backends",
    "create_runtime",
    "declared_capabilities",
    "forward_trace_summary",
    "max_capabilities",
    "mlx_capabilities",
    "profile_stage",
    "runtime_capabilities",
    "runtime_capability_snapshot",
    "runtime_evidence_refs",
    "torch_capabilities",
]
