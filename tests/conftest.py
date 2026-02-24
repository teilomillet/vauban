"""Shared test fixtures: tiny mock model conforming to vauban protocols.

The mock matches real mlx-lm model interfaces:
- Layers return mx.array (not tuples) — cache is mutated in-place
- KV cache uses MockKVCache with update_and_fetch()
- make_cache() returns list[MockKVCache]
"""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
D_MODEL = 16
NUM_LAYERS = 2
VOCAB_SIZE = 32
NUM_HEADS = 2


class MockKVCache:
    """Minimal KV cache matching real mlx-lm KVCache interface."""

    def __init__(self) -> None:
        self.offset: int = 0
        self.keys: mx.array | None = None
        self.values: mx.array | None = None

    def update_and_fetch(
        self, keys: mx.array, values: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Append new keys/values and return the full cache."""
        if self.keys is not None and self.values is not None:
            self.keys = mx.concatenate([self.keys, keys], axis=1)
            self.values = mx.concatenate([self.values, values], axis=1)
        else:
            self.keys = keys
            self.values = values
        self.offset = self.keys.shape[1]
        return self.keys, self.values


class MockAttention(nn.Module):
    """Minimal multi-head attention with in-place cache mutation."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: MockKVCache | None = None,
    ) -> mx.array:
        """Forward pass. Mutates cache in-place, returns output tensor."""
        _b, s, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        # Simple dot-product attention (no head splitting for brevity)
        scale = self.head_dim**-0.5
        scores = (q @ k.transpose(0, 2, 1)) * scale
        if mask is not None and mask.ndim == 2:
            scores = scores + mask[-s:, : scores.shape[-1]]
        weights = mx.softmax(scores, axis=-1)
        out = weights @ v
        return self.o_proj(out)


class MockMLP(nn.Module):
    """Minimal MLP with gate/up/down projections."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.up_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.down_proj = nn.Linear(d_model * 2, d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class MockTransformerBlock(nn.Module):
    """Single transformer block — returns mx.array, cache mutated in-place."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.self_attn = MockAttention(d_model, num_heads)
        self.mlp = MockMLP(d_model)
        self.input_layernorm = nn.RMSNorm(d_model)
        self.post_attention_layernorm = nn.RMSNorm(d_model)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: MockKVCache | None = None,
    ) -> mx.array:
        """Forward pass. Returns hidden state only (cache mutated in-place)."""
        r = self.input_layernorm(x)
        attn_out = self.self_attn(r, mask, cache)
        h = x + attn_out
        r = self.post_attention_layernorm(h)
        h = h + self.mlp(r)
        return h


class MockTransformerModel(nn.Module):
    """Inner transformer conforming to TransformerModel protocol."""

    def __init__(
        self, d_model: int, num_layers: int, vocab_size: int, num_heads: int,
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.layers = [
            MockTransformerBlock(d_model, num_heads) for _ in range(num_layers)
        ]
        self.norm = nn.RMSNorm(d_model)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[MockKVCache] | None = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
        mask = mask.astype(h.dtype)

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            h = layer(h, mask, layer_cache)
        return self.norm(h)


class MockCausalLM(nn.Module):
    """Top-level model conforming to CausalLM protocol."""

    def __init__(
        self, d_model: int, num_layers: int, vocab_size: int, num_heads: int,
    ) -> None:
        super().__init__()
        self.model = MockTransformerModel(d_model, num_layers, vocab_size, num_heads)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[MockKVCache] | None = None,
    ) -> mx.array:
        h = self.model(inputs, cache)
        return self.lm_head(h)

    def make_cache(self) -> list[MockKVCache]:
        """Create a fresh KV cache for each layer."""
        return [MockKVCache() for _ in self.model.layers]


class MockTokenizer:
    """Minimal tokenizer conforming to Tokenizer protocol."""

    def __init__(self, vocab_size: int) -> None:
        self._vocab_size = vocab_size
        self.eos_token_id: int = vocab_size - 1

    def encode(self, text: str) -> list[int]:
        """Map each character to a token id (mod vocab_size)."""
        return [ord(c) % self._vocab_size for c in text]

    def decode(self, token_ids: list[int]) -> str:
        """Map token ids back to characters."""
        return "".join(chr(t + 65) for t in token_ids)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = True,
    ) -> str | list[int]:
        """Template with detectable boundary: [USER]{content}[/USER][ASST]."""
        parts: list[str] = []
        for m in messages:
            if m["role"] == "user":
                parts.append(f"[USER]{m['content']}[/USER]")
            else:
                parts.append(m["content"])
        parts.append("[ASST]")
        text = "".join(parts)
        if tokenize:
            return self.encode(text)
        return text


@pytest.fixture
def mock_model() -> MockCausalLM:
    """A tiny 2-layer, 16-dim mock model."""
    model = MockCausalLM(D_MODEL, NUM_LAYERS, VOCAB_SIZE, NUM_HEADS)
    mx.eval(model.parameters())
    return model


@pytest.fixture
def mock_tokenizer() -> MockTokenizer:
    """A mock tokenizer."""
    return MockTokenizer(VOCAB_SIZE)


@pytest.fixture
def mock_cache(mock_model: MockCausalLM) -> list[MockKVCache]:
    """A fresh KV cache for the mock model."""
    return mock_model.make_cache()


@pytest.fixture
def direction() -> mx.array:
    """A random unit direction vector."""
    d = mx.random.normal((D_MODEL,))
    d = d / mx.linalg.norm(d)
    mx.eval(d)
    return d


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to the test fixtures directory."""
    return FIXTURES_DIR
