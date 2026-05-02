# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""CUDA regression tests for direct model-call generation paths."""

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
from vauban.depth import depth_generate, depth_profile
from vauban.environment._loop import _generate_response
from vauban.evaluate import _generate, _kl_divergence, _perplexity
from vauban.fusion import fuse_and_generate
from vauban.intent import _check_alignment_judge
from vauban.optimize import _kl_from_precomputed, _precompute_logits
from vauban.sic import _generate_with_messages as _sic_generate_with_messages
from vauban.surface._scan import (
    _generate_with_messages as _surface_generate_with_messages,
)
from vauban.types import DepthConfig, FusionConfig, IntentConfig, IntentState


def _cuda_model() -> MockCausalLM:
    """Return the torch mock model on CUDA, skipping when unavailable."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS).to("cuda")


def test_evaluate_helpers_run_on_cuda_model() -> None:
    model = _cuda_model()
    tokenizer = MockTokenizer(VOCAB_SIZE)

    text = _generate(model, tokenizer, "hello", max_tokens=2)
    perplexity = _perplexity(model, tokenizer, ["hello"])
    kl = _kl_divergence(model, model, tokenizer, ["hello"])

    assert len(text) == 2
    assert math.isfinite(perplexity)
    assert math.isfinite(kl)


def test_message_generation_helpers_run_on_cuda_model() -> None:
    model = _cuda_model()
    tokenizer = MockTokenizer(VOCAB_SIZE)
    messages = [{"role": "user", "content": "hello"}]

    surface_text = _surface_generate_with_messages(
        model, tokenizer, messages, max_tokens=2,
    )
    sic_text = _sic_generate_with_messages(
        model, tokenizer, messages, max_tokens=2,
    )
    environment_text = _generate_response(
        model, tokenizer, messages, max_tokens=2,
    )

    assert len(surface_text) == 2
    assert len(sic_text) == 2
    assert len(environment_text) == 2


def test_intent_judge_generation_runs_on_cuda_model() -> None:
    model = _cuda_model()
    tokenizer = MockTokenizer(VOCAB_SIZE)
    config = IntentConfig(mode="judge", max_tokens=2)
    intent_state = IntentState(user_request="open the file", activation=None)

    result = _check_alignment_judge(
        model,
        tokenizer,
        "read the file",
        intent_state,
        config,
    )

    assert result.mode == "judge"
    assert result.score in {0.0, 1.0}


def test_optimize_kl_helpers_run_on_cuda_model() -> None:
    model = _cuda_model()
    tokenizer = MockTokenizer(VOCAB_SIZE)
    prompts = ["hello"]

    logits = _precompute_logits(model, tokenizer, prompts)
    kl = _kl_from_precomputed(model, tokenizer, prompts, logits)

    assert len(logits) == 1
    assert str(logits[0].device).startswith("cuda")
    assert math.isfinite(kl)


def test_depth_paths_run_on_cuda_model_with_full_vocab_top_k() -> None:
    model = _cuda_model()
    tokenizer = MockTokenizer(VOCAB_SIZE)
    static_config = DepthConfig(prompts=["hello"], top_k_logits=VOCAB_SIZE)
    generate_config = DepthConfig(
        prompts=["hello"],
        top_k_logits=VOCAB_SIZE,
        max_tokens=1,
    )

    static_result = depth_profile(model, tokenizer, "hello", static_config)
    generate_result = depth_generate(model, tokenizer, "hello", generate_config)

    assert len(static_result.tokens) > 0
    assert len(generate_result.tokens) == 1
    assert math.isfinite(static_result.deep_thinking_ratio)


def test_fusion_generation_runs_on_cuda_model() -> None:
    model = _cuda_model()
    tokenizer = MockTokenizer(VOCAB_SIZE)
    config = FusionConfig(
        harmful_prompts=["harm"],
        benign_prompts=["safe"],
        layer=1,
        n_tokens=1,
        temperature=0.0,
    )

    result = fuse_and_generate(model, tokenizer, "harm", "safe", config)

    assert result.layer == 1
    assert len(result.output) == 1
