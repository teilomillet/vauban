# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""CUDA regressions for CPU inputs combined with CUDA activations."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from tests.conftest import (
    D_MODEL,
    NUM_HEADS,
    NUM_LAYERS,
    VOCAB_SIZE,
    MockCausalLM,
    MockTokenizer,
)
from vauban import _ops as ops
from vauban.cast import cast_generate
from vauban.cut import cut
from vauban.features import SparseAutoencoder, feature_direction_alignment
from vauban.guard import GuardSession, guard_generate
from vauban.lora import direction_to_lora
from vauban.probe import probe, steer
from vauban.scan import scan
from vauban.sensitivity import LayerSensitivity, SensitivityProfile, directional_gain
from vauban.sss import sss_generate
from vauban.svf import SVFBoundary, svf_gradient
from vauban.types import GuardConfig, ScanConfig, SSSConfig

if TYPE_CHECKING:
    from types import ModuleType

    from vauban._array import Array


def _torch_module() -> ModuleType:
    """Return torch, skipping when CUDA is unavailable."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return torch


def _cuda_model() -> MockCausalLM:
    """Return the torch mock model on CUDA, skipping when unavailable."""
    _torch_module()
    return MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS).to("cuda")


def test_probe_and_steer_accept_cpu_direction_for_cuda_model() -> None:
    model = _cuda_model()
    tokenizer = MockTokenizer(VOCAB_SIZE)
    direction = ops.ones((D_MODEL,))

    probe_result = probe(model, tokenizer, "hello", direction)
    steer_result = steer(model, tokenizer, "hello", direction, [0], max_tokens=1)

    assert len(probe_result.projections) == NUM_LAYERS
    assert len(steer_result.text) == 1


def test_cast_guard_and_scan_accept_cpu_direction_for_cuda_model() -> None:
    model = _cuda_model()
    tokenizer = MockTokenizer(VOCAB_SIZE)
    direction = ops.ones((D_MODEL,))

    cast_result = cast_generate(
        model, tokenizer, "hello", direction, [0], max_tokens=1,
    )
    guard_result = guard_generate(
        model,
        tokenizer,
        "hello",
        direction,
        [0],
        GuardConfig(prompts=["hello"], layers=[0], max_tokens=1),
    )
    scan_result = scan(
        model,
        tokenizer,
        "hello",
        ScanConfig(target_layer=0),
        direction,
    )

    assert len(cast_result.text) == 1
    assert guard_result.tokens_generated >= 0
    assert len(scan_result.per_token_projections) > 0


def test_guard_session_accepts_cpu_direction_for_cuda_activation() -> None:
    torch = _torch_module()
    direction = ops.ones((D_MODEL,))
    activation = torch.ones((D_MODEL,), device="cuda")
    session = GuardSession(direction)

    verdict = session.check(activation)

    assert verdict.action in {"pass", "steer", "rewind", "break"}


def test_weight_transforms_accept_cpu_direction_for_cuda_weights() -> None:
    torch = _torch_module()
    direction = ops.ones((D_MODEL,))
    weight = torch.randn((D_MODEL, D_MODEL), device="cuda")
    weights = {"model.layers.0.self_attn.o_proj.weight": weight}

    cut_weights = cut(weights, direction, [0])
    lora_a, lora_b = direction_to_lora(direction, weight)

    assert str(cut_weights["model.layers.0.self_attn.o_proj.weight"].device).startswith(
        "cuda",
    )
    assert str(lora_a.device).startswith("cuda")
    assert str(lora_b.device).startswith("cuda")


def test_svf_and_sae_accept_cpu_parameters_or_direction_on_cuda() -> None:
    torch = _torch_module()
    h = torch.randn((D_MODEL,), device="cuda")
    boundary = SVFBoundary(D_MODEL, projection_dim=4, hidden_dim=8, n_layers=2)

    score, grad = svf_gradient(boundary, h, 0)

    sae = SparseAutoencoder(D_MODEL, d_sae=4)
    ref = torch.zeros((1,), device="cuda")
    sae.set_parameters([ops.to_device_like(p, ref) for p in sae.parameters()])
    alignment = feature_direction_alignment(sae, ops.ones((D_MODEL,)))

    assert math.isfinite(score)
    assert str(grad.device).startswith("cuda")
    assert len(alignment) == 4


def test_sensitivity_and_sss_accept_cpu_direction_for_cuda_state() -> None:
    torch = _torch_module()
    model = _cuda_model()
    tokenizer = MockTokenizer(VOCAB_SIZE)
    direction = ops.ones((D_MODEL,))
    h = torch.randn((1, 2, D_MODEL), device="cuda")

    def layer_fn(x: Array) -> Array:
        return x * 2.0

    gain = directional_gain(layer_fn, h, direction)
    profile = SensitivityProfile(
        layers=[
            LayerSensitivity(
                0,
                directional_gain=1.0,
                correlation=1.0,
                effective_rank=1.0,
            ),
            LayerSensitivity(
                1,
                directional_gain=1.0,
                correlation=1.0,
                effective_rank=1.0,
            ),
        ],
        valley_layers=[0],
    )
    result = sss_generate(
        model,
        tokenizer,
        "hello",
        direction,
        SSSConfig(prompts=["hello"], layers=[0], max_tokens=1),
        profile,
    )

    assert gain > 0.0
    assert len(result.text) == 1
