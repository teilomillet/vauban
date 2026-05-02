# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Torch activation projection/intervention primitive."""

from __future__ import annotations

from typing import Protocol, cast

import pytest

from vauban.runtime import (
    ForwardRequest,
    LoadedModel,
    ModelRef,
    TorchActivationPrimitiveRequest,
    TorchActivationTensor,
    TorchDirectionIntervention,
    TorchRuntime,
    run_torch_activation_primitive,
    torch_capabilities,
)

torch = cast("_TorchModule", pytest.importorskip("torch"))


class _TorchModule(Protocol):
    """Subset of torch used by these primitive tests."""

    float32: object
    long: object

    def tensor(
        self,
        data: object,
        *,
        dtype: object | None = None,
        device: object | None = None,
    ) -> TorchActivationTensor:
        """Create a tensor."""

    def ones(
        self,
        size: tuple[int, ...],
        *,
        dtype: object | None = None,
    ) -> TorchActivationTensor:
        """Create a tensor of ones."""

    def allclose(
        self,
        input_tensor: TorchActivationTensor,
        other: TorchActivationTensor,
    ) -> bool:
        """Return whether two tensors are numerically close."""


class _FakeEmbedding:
    """Small Torch embedding surface for runtime integration tests."""

    def __init__(self, torch_module: _TorchModule) -> None:
        """Initialize fake embedding."""
        self._torch = torch_module
        self.weight = torch_module.ones((1,), dtype=torch_module.float32)

    def __call__(self, token_ids: TorchActivationTensor) -> TorchActivationTensor:
        """Embed tokens as deterministic activations."""
        batch = int(token_ids.shape[0])
        seq_len = int(token_ids.shape[1])
        return self._torch.ones((batch, seq_len, 4), dtype=self._torch.float32)


class _FakeLayer:
    """Small Torch layer that shifts hidden states."""

    def __init__(self, increment: float) -> None:
        """Initialize fake layer."""
        self.increment = increment

    def __call__(
        self,
        h: TorchActivationTensor,
        mask: object | None = None,
        cache: object | None = None,
    ) -> TorchActivationTensor:
        """Apply a deterministic activation shift."""
        _ = mask, cache
        return h + self.increment


class _FakeNorm:
    """Identity norm for runtime integration tests."""

    def __call__(self, h: TorchActivationTensor) -> TorchActivationTensor:
        """Return hidden states unchanged."""
        return h


class _FakeTransformer:
    """Minimal transformer surface consumed by TorchRuntime."""

    def __init__(self, torch_module: _TorchModule) -> None:
        """Initialize fake transformer."""
        self.embed_tokens = _FakeEmbedding(torch_module)
        self.layers = [_FakeLayer(1.0), _FakeLayer(2.0)]
        self.norm = _FakeNorm()


class _FakeTorchModel:
    """Minimal Torch model surface consumed by TorchRuntime."""

    def __init__(self, torch_module: _TorchModule) -> None:
        """Initialize fake model."""
        self.model = _FakeTransformer(torch_module)
        self.device = "cpu"


class _FakeTokenizer:
    """Minimal tokenizer surface for a loaded runtime model."""

    def encode(self, text: str) -> list[int]:
        """Encode text into deterministic token IDs."""
        return [ord(char) % 32 for char in text]


def test_torch_activation_primitive_projects_last_dimension() -> None:
    """Projection keeps one scalar per activation position."""
    activation = torch.tensor(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]],
        dtype=torch.float32,
    )
    direction = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

    result = run_torch_activation_primitive(
        TorchActivationPrimitiveRequest(
            activation=activation,
            direction=direction,
            layer_index=2,
        ),
    )

    expected = torch.tensor([[[2.0], [5.0]]], dtype=torch.float32)
    assert result.intervened_activation is None
    assert torch.allclose(result.projection, expected)
    assert result.artifact_metadata()["projection_shape"] == [1, 2, 1]


def test_torch_activation_primitive_subtracts_direction_component() -> None:
    """Subtract mode removes the projected component along the direction."""
    activation = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float32)
    direction = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

    result = run_torch_activation_primitive(
        TorchActivationPrimitiveRequest(
            activation=activation,
            direction=direction,
            layer_index=0,
            mode="subtract",
            name="remove_middle",
        ),
    )

    expected = torch.tensor([[[1.0, 0.0, 3.0]]], dtype=torch.float32)
    assert result.intervened_activation is not None
    assert torch.allclose(result.intervened_activation, expected)
    assert result.intervention_record() is not None
    assert result.artifact_metadata()["intervened"] is True


def test_torch_activation_primitive_adds_direction_component() -> None:
    """Add mode amplifies the projected component along the direction."""
    activation = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float32)
    direction = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

    result = run_torch_activation_primitive(
        TorchActivationPrimitiveRequest(
            activation=activation,
            direction=direction,
            layer_index=0,
            mode="add",
            alpha=0.5,
            name="add_middle",
        ),
    )

    expected = torch.tensor([[[1.0, 3.0, 3.0]]], dtype=torch.float32)
    assert result.intervened_activation is not None
    assert torch.allclose(result.intervened_activation, expected)


def test_torch_activation_primitive_projects_subspace_coefficients() -> None:
    """Subspace project mode returns one coefficient per basis vector."""
    activation = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float32)
    basis = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )

    result = run_torch_activation_primitive(
        TorchActivationPrimitiveRequest(
            activation=activation,
            direction=basis,
            layer_index=0,
            mode="subspace_project",
            name="edge_basis",
        ),
    )

    expected = torch.tensor([[[1.0, 3.0]]], dtype=torch.float32)
    assert result.intervened_activation is None
    assert torch.allclose(result.projection, expected)
    assert result.artifact_metadata()["projection_shape"] == [1, 1, 2]


def test_torch_activation_primitive_removes_subspace_component() -> None:
    """Subspace remove mode subtracts the reconstructed component."""
    activation = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float32)
    basis = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )

    result = run_torch_activation_primitive(
        TorchActivationPrimitiveRequest(
            activation=activation,
            direction=basis,
            layer_index=0,
            mode="subspace_remove",
            name="remove_edges",
        ),
    )

    expected = torch.tensor([[[0.0, 2.0, 0.0]]], dtype=torch.float32)
    assert result.intervened_activation is not None
    assert torch.allclose(result.intervened_activation, expected)


def test_torch_activation_primitive_rejects_shape_mismatch() -> None:
    """Direction dimension must match the activation hidden size."""
    activation = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float32)
    direction = torch.tensor([1.0, 0.0], dtype=torch.float32)

    with pytest.raises(ValueError, match="direction dimension"):
        TorchActivationPrimitiveRequest(
            activation=activation,
            direction=direction,
            layer_index=0,
        )


def test_torch_direction_intervention_integrates_with_runtime_forward() -> None:
    """The primitive-backed intervention works through TorchRuntime.forward."""
    loaded = LoadedModel(
        ref=ModelRef("fake-torch-model"),
        backend="torch",
        capabilities=torch_capabilities(),
        model=_FakeTorchModel(torch),
        tokenizer=_FakeTokenizer(),
    )
    direction = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)
    intervention = TorchDirectionIntervention(
        name="remove_middle",
        layer_index=0,
        direction=direction,
    )

    trace = TorchRuntime().forward(
        loaded,
        ForwardRequest(
            prompt_ids=(1, 2),
            collect_layers=(0,),
            interventions=(intervention,),
            return_logits=False,
        ),
    )

    expected = torch.tensor(
        [[[2.0, 0.0, 2.0, 2.0], [2.0, 0.0, 2.0, 2.0]]],
        dtype=torch.float32,
    )
    activation = cast("TorchActivationTensor", trace.activations[0])
    assert torch.allclose(activation, expected)
    assert trace.interventions[0].to_dict()["metadata"]["primitive"] == (
        "activation_projection"
    )
