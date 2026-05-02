# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Torch reference primitive for activation projection and intervention."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Literal, Protocol, SupportsFloat, cast

from vauban.runtime._types import InterventionRecord, RuntimeValue, TensorLike

type ActivationPrimitiveMode = Literal[
    "project",
    "subtract",
    "add",
    "subspace_project",
    "subspace_remove",
    "subspace_add",
]
type DirectionInterventionMode = Literal[
    "subtract",
    "add",
    "subspace_remove",
    "subspace_add",
]


class TorchActivationTensor(TensorLike, Protocol):
    """Torch tensor surface needed by activation primitives."""

    @property
    def dtype(self) -> object:
        """Return tensor dtype."""
        ...

    @property
    def device(self) -> object:
        """Return tensor device."""
        ...

    def to(self, *, device: object | None = None, dtype: object | None = None) -> (
        TorchActivationTensor
    ):
        """Move or cast the tensor."""
        ...

    def detach(self) -> TorchActivationTensor:
        """Return a detached tensor view."""
        ...

    def float(self) -> TorchActivationTensor:
        """Return a float32-compatible tensor view."""
        ...

    def mean(self) -> TorchActivationTensor:
        """Return the tensor mean."""
        ...

    def min(self) -> TorchActivationTensor:
        """Return the tensor minimum."""
        ...

    def max(self) -> TorchActivationTensor:
        """Return the tensor maximum."""
        ...

    def item(self) -> SupportsFloat:
        """Return a scalar Python value."""
        ...

    def transpose(self, dim0: int, dim1: int) -> TorchActivationTensor:
        """Transpose two dimensions."""
        ...

    def __mul__(self, other: object) -> TorchActivationTensor:
        """Multiply by another tensor or scalar."""
        ...

    def __rmul__(self, other: object) -> TorchActivationTensor:
        """Multiply by another tensor or scalar."""
        ...

    def __add__(self, other: object) -> TorchActivationTensor:
        """Add another tensor or scalar."""
        ...

    def __sub__(self, other: object) -> TorchActivationTensor:
        """Subtract another tensor or scalar."""
        ...


class _TorchActivationOps(Protocol):
    """Subset of torch used by the reference activation primitive."""

    def sum(
        self,
        input_tensor: TorchActivationTensor,
        *,
        dim: int,
        keepdim: bool = False,
    ) -> TorchActivationTensor:
        """Sum a tensor along one dimension."""
        ...

    def matmul(
        self,
        input_tensor: TorchActivationTensor,
        other: TorchActivationTensor,
    ) -> TorchActivationTensor:
        """Matrix multiply two tensors."""
        ...


@dataclass(frozen=True, slots=True)
class TorchActivationPrimitiveRequest:
    """Request for the Torch activation projection/intervention primitive."""

    activation: TorchActivationTensor
    direction: TorchActivationTensor
    layer_index: int
    mode: ActivationPrimitiveMode = "project"
    alpha: float = 1.0
    name: str = "direction_projection"

    def __post_init__(self) -> None:
        """Validate the primitive request."""
        if self.layer_index < 0:
            msg = "layer_index must be non-negative"
            raise ValueError(msg)
        if not self.name.strip():
            msg = "name must not be empty"
            raise ValueError(msg)
        if not isfinite(self.alpha):
            msg = "alpha must be finite"
            raise ValueError(msg)
        _validate_activation_direction_shapes(self.activation, self.direction)


@dataclass(frozen=True, slots=True)
class TorchActivationPrimitiveResult:
    """Result from a Torch activation projection/intervention primitive."""

    projection: TorchActivationTensor
    activation: TorchActivationTensor
    direction: TorchActivationTensor
    layer_index: int
    mode: ActivationPrimitiveMode
    alpha: float
    name: str
    intervened_activation: TorchActivationTensor | None = None

    def intervention_record(self) -> InterventionRecord | None:
        """Return intervention metadata when the primitive changed activations."""
        if self.intervened_activation is None:
            return None
        return InterventionRecord(name=self.name, layer_index=self.layer_index)

    def artifact_metadata(self) -> dict[str, RuntimeValue]:
        """Return stable trace metadata for primitive evidence artifacts."""
        metadata: dict[str, RuntimeValue] = {}
        metadata["primitive"] = "activation_projection"
        metadata["name"] = self.name
        metadata["layer_index"] = self.layer_index
        metadata["mode"] = self.mode
        metadata["alpha"] = self.alpha
        metadata["activation_shape"] = _shape_as_runtime_value(self.activation)
        metadata["direction_shape"] = _shape_as_runtime_value(self.direction)
        metadata["projection_shape"] = _shape_as_runtime_value(self.projection)
        metadata["projection_summary"] = _tensor_summary(self.projection)
        metadata["intervened"] = self.intervened_activation is not None
        metadata["device"] = str(self.activation.device)
        metadata["dtype"] = str(self.activation.dtype)
        return metadata


@dataclass(slots=True)
class TorchDirectionIntervention:
    """Activation intervention backed by the Torch projection primitive."""

    name: str
    layer_index: int
    direction: TorchActivationTensor
    alpha: float = 1.0
    mode: DirectionInterventionMode = "subtract"
    _last_result: TorchActivationPrimitiveResult | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def apply(self, activation: TensorLike) -> TensorLike:
        """Apply direction intervention to one activation tensor."""
        result = run_torch_activation_primitive(
            TorchActivationPrimitiveRequest(
                activation=cast("TorchActivationTensor", activation),
                direction=self.direction,
                layer_index=self.layer_index,
                mode=self.mode,
                alpha=self.alpha,
                name=self.name,
            ),
        )
        if result.intervened_activation is None:
            msg = "direction intervention did not produce an activation"
            raise RuntimeError(msg)
        self._last_result = result
        return result.intervened_activation

    def primitive_metadata(self) -> dict[str, RuntimeValue] | None:
        """Return metadata from the most recent primitive application."""
        if self._last_result is None:
            return None
        return self._last_result.artifact_metadata()


class _PrimitiveMetadataProvider(Protocol):
    """Intervention object that can expose primitive metadata."""

    def primitive_metadata(self) -> dict[str, RuntimeValue] | None:
        """Return primitive metadata after an intervention runs."""
        ...


def primitive_metadata_for_intervention(
    intervention: object,
) -> dict[str, RuntimeValue] | None:
    """Return primitive metadata from an intervention when available."""
    metadata_fn = getattr(intervention, "primitive_metadata", None)
    if not callable(metadata_fn):
        return None
    return cast("_PrimitiveMetadataProvider", intervention).primitive_metadata()


def run_torch_activation_primitive(
    request: TorchActivationPrimitiveRequest,
) -> TorchActivationPrimitiveResult:
    """Run the Torch reference projection/intervention primitive."""
    import torch

    ops = cast("_TorchActivationOps", torch)
    direction = request.direction.to(
        device=request.activation.device,
        dtype=request.activation.dtype,
    )
    projection = _project(ops, request.activation, direction, request.mode)
    intervened: TorchActivationTensor | None = None
    if request.mode == "subtract":
        intervened = request.activation - (request.alpha * projection * direction)
    elif request.mode == "add":
        intervened = request.activation + (request.alpha * projection * direction)
    elif request.mode == "subspace_remove":
        component = _subspace_component(ops, projection, direction)
        intervened = request.activation - (request.alpha * component)
    elif request.mode == "subspace_add":
        component = _subspace_component(ops, projection, direction)
        intervened = request.activation + (request.alpha * component)

    return TorchActivationPrimitiveResult(
        projection=projection,
        activation=request.activation,
        direction=direction,
        layer_index=request.layer_index,
        mode=request.mode,
        alpha=request.alpha,
        name=request.name,
        intervened_activation=intervened,
    )


def _validate_activation_direction_shapes(
    activation: TorchActivationTensor,
    direction: TorchActivationTensor,
) -> None:
    """Reject shape pairs that cannot express last-dimension projection."""
    if len(activation.shape) < 1:
        msg = "activation must have at least one dimension"
        raise ValueError(msg)
    if len(direction.shape) not in (1, 2):
        msg = "direction must be a rank-1 direction or rank-2 subspace basis"
        raise ValueError(msg)
    direction_width = (
        direction.shape[0] if len(direction.shape) == 1 else direction.shape[1]
    )
    if activation.shape[-1] != direction_width:
        msg = (
            "direction dimension must match activation last dimension:"
            f" {direction_width} != {activation.shape[-1]}"
        )
        raise ValueError(msg)


def _shape_as_list(tensor: TensorLike) -> list[int]:
    """Return a tensor shape as JSON-serializable integers."""
    return [int(dim) for dim in tensor.shape]


def _shape_as_runtime_value(tensor: TensorLike) -> RuntimeValue:
    """Return tensor shape through the recursive runtime-value alias."""
    values: list[RuntimeValue] = []
    for dim in tensor.shape:
        values.append(int(dim))
    return values


def _project(
    ops: _TorchActivationOps,
    activation: TorchActivationTensor,
    direction: TorchActivationTensor,
    mode: ActivationPrimitiveMode,
) -> TorchActivationTensor:
    """Return rank-1 projection or subspace coefficients."""
    if mode in ("project", "subtract", "add"):
        if len(direction.shape) != 1:
            msg = f"{mode} mode requires a rank-1 direction"
            raise ValueError(msg)
        return ops.sum(activation * direction, dim=-1, keepdim=True)
    if len(direction.shape) != 2:
        msg = f"{mode} mode requires a rank-2 subspace basis"
        raise ValueError(msg)
    return ops.matmul(activation, direction.transpose(0, 1))


def _subspace_component(
    ops: _TorchActivationOps,
    coefficients: TorchActivationTensor,
    basis: TorchActivationTensor,
) -> TorchActivationTensor:
    """Reconstruct a projected component from subspace coefficients."""
    return ops.matmul(coefficients, basis)


def _tensor_summary(tensor: TorchActivationTensor) -> RuntimeValue:
    """Return compact numeric summary statistics for a tensor."""
    values: dict[str, RuntimeValue] = {}
    summary_tensor = tensor.detach().float()
    values["mean"] = float(summary_tensor.mean().item())
    values["min"] = float(summary_tensor.min().item())
    values["max"] = float(summary_tensor.max().item())
    return values
