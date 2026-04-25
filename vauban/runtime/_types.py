# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Typed runtime primitives for Vauban backend execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

type RuntimeBackendName = Literal["mlx", "torch", "max"]
type SupportLevel = Literal["unsupported", "partial", "full"]
type DeviceKind = Literal["cpu", "gpu", "cuda", "mps"]
type RuntimeScalar = str | int | float | bool | None
type RuntimeValue = RuntimeScalar | list[RuntimeValue] | dict[str, RuntimeValue]


@runtime_checkable
class TensorLike(Protocol):
    """Minimal structural tensor surface exposed by runtime traces."""

    @property
    def shape(self) -> tuple[int, ...]: ...


@runtime_checkable
class ActivationIntervention(Protocol):
    """Reversible activation intervention applied at one runtime layer."""

    name: str
    layer_index: int

    def apply(self, activation: TensorLike) -> TensorLike:
        """Return the intervened activation."""
        ...


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """Declared evidence-producing support for a Vauban runtime backend."""

    name: RuntimeBackendName
    device_kinds: tuple[DeviceKind, ...]
    logits: SupportLevel
    logprobs: SupportLevel
    activations: SupportLevel
    interventions: SupportLevel
    kv_cache: SupportLevel
    weight_access: SupportLevel
    mutable_weights: SupportLevel

    def __post_init__(self) -> None:
        """Validate capability declarations."""
        if not self.device_kinds:
            msg = "device_kinds must not be empty"
            raise ValueError(msg)

    def supports(self, capability: str) -> bool:
        """Return whether a named capability is at least partially supported."""
        level = self.support_level(capability)
        return level != "unsupported"

    def support_level(self, capability: str) -> SupportLevel:
        """Return a named support level."""
        if capability == "logits":
            return self.logits
        if capability == "logprobs":
            return self.logprobs
        if capability == "activations":
            return self.activations
        if capability == "interventions":
            return self.interventions
        if capability == "kv_cache":
            return self.kv_cache
        if capability == "weight_access":
            return self.weight_access
        if capability == "mutable_weights":
            return self.mutable_weights
        msg = f"Unknown runtime capability: {capability!r}"
        raise ValueError(msg)

    def to_dict(self) -> dict[str, RuntimeValue]:
        """Serialize capability declarations for reports and diagnostics."""
        return {
            "name": self.name,
            "device_kinds": list(self.device_kinds),
            "logits": self.logits,
            "logprobs": self.logprobs,
            "activations": self.activations,
            "interventions": self.interventions,
            "kv_cache": self.kv_cache,
            "weight_access": self.weight_access,
            "mutable_weights": self.mutable_weights,
        }


@dataclass(frozen=True, slots=True)
class DeviceRef:
    """Runtime device metadata attached to traces."""

    kind: DeviceKind
    label: str

    def __post_init__(self) -> None:
        """Validate device metadata."""
        if not self.label.strip():
            msg = "device label must not be empty"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class StageProfile:
    """Timing and optional memory metadata for one runtime stage."""

    name: str
    duration_s: float
    device: DeviceRef | None = None
    memory_bytes: int | None = None
    metadata: dict[str, RuntimeValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate stage profile metadata."""
        if not self.name.strip():
            msg = "stage profile name must not be empty"
            raise ValueError(msg)
        if self.duration_s < 0.0:
            msg = "stage profile duration_s must be non-negative"
            raise ValueError(msg)
        if self.memory_bytes is not None and self.memory_bytes < 0:
            msg = "stage profile memory_bytes must be non-negative"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ModelRef:
    """Reference to a model runtime should load."""

    model_path: str
    revision: str | None = None
    metadata: dict[str, RuntimeValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate model reference fields."""
        if not self.model_path.strip():
            msg = "model_path must not be empty"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """Loaded backend-specific model handle with honest capabilities."""

    ref: ModelRef
    backend: RuntimeBackendName
    capabilities: BackendCapabilities
    model: object
    tokenizer: object | None = None
    metadata: dict[str, RuntimeValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate loaded model metadata."""
        if self.backend != self.capabilities.name:
            msg = (
                "loaded model backend must match capability backend: "
                f"{self.backend!r} != {self.capabilities.name!r}"
            )
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class TokenizeRequest:
    """Request to tokenize text through a loaded model tokenizer."""

    text: str
    apply_chat_template: bool = False

    def __post_init__(self) -> None:
        """Validate tokenize request fields."""
        if not self.text:
            msg = "text must not be empty"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class TokenizedPrompt:
    """Tokenized prompt emitted by a runtime tokenizer stage."""

    token_ids: tuple[int, ...]
    text: str
    profile: tuple[StageProfile, ...] = ()

    def __post_init__(self) -> None:
        """Validate tokenized prompt fields."""
        if not self.token_ids:
            msg = "token_ids must not be empty"
            raise ValueError(msg)
        if any(token_id < 0 for token_id in self.token_ids):
            msg = "token_ids must be non-negative"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ForwardRequest:
    """Runtime forward-pass request."""

    prompt_ids: tuple[int, ...]
    collect_layers: tuple[int, ...] = ()
    interventions: tuple[ActivationIntervention, ...] = ()
    return_logits: bool = True
    return_logprobs: bool = False

    def __post_init__(self) -> None:
        """Validate forward request fields."""
        if not self.prompt_ids:
            msg = "prompt_ids must not be empty"
            raise ValueError(msg)
        if any(token_id < 0 for token_id in self.prompt_ids):
            msg = "prompt_ids must be non-negative"
            raise ValueError(msg)
        if any(layer < 0 for layer in self.collect_layers):
            msg = "collect_layers must be non-negative"
            raise ValueError(msg)
        if any(intervention.layer_index < 0 for intervention in self.interventions):
            msg = "intervention layer indexes must be non-negative"
            raise ValueError(msg)
        if any(not intervention.name.strip() for intervention in self.interventions):
            msg = "intervention names must not be empty"
            raise ValueError(msg)
        if self.return_logprobs and not self.return_logits:
            msg = "return_logprobs requires return_logits"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class InterventionRecord:
    """Metadata for one activation intervention applied during forward."""

    name: str
    layer_index: int

    def __post_init__(self) -> None:
        """Validate intervention metadata."""
        if not self.name.strip():
            msg = "intervention record name must not be empty"
            raise ValueError(msg)
        if self.layer_index < 0:
            msg = "intervention record layer_index must be non-negative"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, RuntimeValue]:
        """Serialize intervention metadata."""
        return {
            "name": self.name,
            "layer_index": self.layer_index,
        }


@dataclass(frozen=True, slots=True)
class ForwardTrace:
    """Observed tensors and metadata from one forward pass."""

    logits: TensorLike | None
    logprobs: TensorLike | None
    activations: dict[int, TensorLike]
    device: DeviceRef
    interventions: tuple[InterventionRecord, ...] = ()
    profile: tuple[StageProfile, ...] = ()

    def __post_init__(self) -> None:
        """Validate forward trace consistency."""
        if any(layer < 0 for layer in self.activations):
            msg = "activation layer indexes must be non-negative"
            raise ValueError(msg)
        if any(record.layer_index < 0 for record in self.interventions):
            msg = "intervention layer indexes must be non-negative"
            raise ValueError(msg)
        if self.logprobs is not None and self.logits is None:
            msg = "logprobs require logits in the same trace"
            raise ValueError(msg)
