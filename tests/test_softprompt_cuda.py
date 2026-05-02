# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""CUDA regression tests for soft prompt optimization paths."""

from __future__ import annotations

import math

import pytest

from tests.conftest import (
    D_MODEL,
    NUM_HEADS,
    NUM_LAYERS,
    VOCAB_SIZE,
    MockCausalLM,
    MockTokenizer,
)
from vauban.softprompt._cold import _cold_attack
from vauban.softprompt._continuous import _continuous_attack
from vauban.softprompt._egd import _egd_attack
from vauban.softprompt._gcg import _gcg_attack
from vauban.types import SoftPromptConfig, SoftPromptResult


def _cuda_model() -> MockCausalLM:
    """Return the torch mock model on CUDA, skipping when unavailable."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS).to("cuda")


def _assert_valid_cuda_result(result: SoftPromptResult) -> None:
    """Assert a short CUDA softprompt run completed with finite loss."""
    assert math.isfinite(result.final_loss)
    assert len(result.loss_history) == 1


def test_continuous_attack_runs_on_cuda_model() -> None:
    model = _cuda_model()
    tokenizer = MockTokenizer(VOCAB_SIZE)
    config = SoftPromptConfig(
        mode="continuous",
        n_tokens=2,
        n_steps=1,
        learning_rate=0.01,
        max_gen_tokens=1,
        seed=1,
    )

    result = _continuous_attack(model, tokenizer, ["hello"], config, None)

    _assert_valid_cuda_result(result)
    assert result.embeddings is not None
    assert str(result.embeddings.device).startswith("cuda")


@pytest.mark.parametrize("mode", ["egd", "cold", "gcg"])
def test_discrete_attacks_run_on_cuda_model(mode: str) -> None:
    model = _cuda_model()
    tokenizer = MockTokenizer(VOCAB_SIZE)
    config = SoftPromptConfig(
        mode=mode,
        n_tokens=2,
        n_steps=1,
        batch_size=2,
        top_k=4,
        learning_rate=0.01,
        max_gen_tokens=1,
        seed=1,
    )

    if mode == "egd":
        result = _egd_attack(model, tokenizer, ["hello"], config, None)
    elif mode == "cold":
        result = _cold_attack(model, tokenizer, ["hello"], config, None)
    else:
        result = _gcg_attack(model, tokenizer, ["hello"], config, None)

    _assert_valid_cuda_result(result)
    assert result.token_ids is not None
    assert len(result.token_ids) == 2
