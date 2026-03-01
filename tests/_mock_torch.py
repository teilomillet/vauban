"""PyTorch mock model classes for testing.

Parallel to _mock_mlx.py — same API surface, torch.nn.Module based.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as f

from vauban._nn_torch import create_additive_causal_mask


class TorchRMSNorm(nn.Module):
    """RMSNorm compatible with torch < 2.4 (no native torch.nn.RMSNorm)."""

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class TorchMockKVCache:
    """Minimal KV cache matching mlx-lm KVCache interface."""

    def __init__(self) -> None:
        self.offset: int = 0
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None

    def update_and_fetch(
        self, keys: torch.Tensor, values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new keys/values and return the full cache."""
        if self.keys is not None and self.values is not None:
            self.keys = torch.cat([self.keys, keys], dim=1)
            self.values = torch.cat([self.values, values], dim=1)
        else:
            self.keys = keys
            self.values = values
        self.offset = self.keys.shape[1]
        return self.keys, self.values


class TorchMockAttention(nn.Module):
    """Minimal multi-head attention with in-place cache mutation."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        cache: TorchMockKVCache | None = None,
    ) -> torch.Tensor:
        """Forward pass. Mutates cache in-place, returns output tensor."""
        _b, s, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        scale = self.head_dim**-0.5
        scores = (q @ k.permute(0, 2, 1)) * scale
        if mask is not None and mask.ndim == 2:
            scores = scores + mask[-s:, : scores.shape[-1]]
        weights = f.softmax(scores, dim=-1)
        out = weights @ v
        return self.o_proj(out)


class TorchMockMLP(nn.Module):
    """Minimal MLP with gate/up/down projections."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.up_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.down_proj = nn.Linear(d_model * 2, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(f.silu(self.gate_proj(x)) * self.up_proj(x))


class TorchMockTransformerBlock(nn.Module):
    """Single transformer block — returns tensor, cache mutated in-place."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.self_attn = TorchMockAttention(d_model, num_heads)
        self.mlp = TorchMockMLP(d_model)
        self.input_layernorm = TorchRMSNorm(d_model)
        self.post_attention_layernorm = TorchRMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        cache: TorchMockKVCache | None = None,
    ) -> torch.Tensor:
        """Forward pass. Returns hidden state only (cache mutated in-place)."""
        r = self.input_layernorm(x)
        attn_out = self.self_attn(r, mask, cache)
        h = x + attn_out
        r = self.post_attention_layernorm(h)
        h = h + self.mlp(r)
        return h


class TorchMockTransformerModel(nn.Module):
    """Inner transformer conforming to TransformerModel protocol."""

    def __init__(
        self, d_model: int, num_layers: int, vocab_size: int, num_heads: int,
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TorchMockTransformerBlock(d_model, num_heads) for _ in range(num_layers)
        ])
        self.norm = TorchRMSNorm(d_model)

    def forward(
        self,
        inputs: torch.Tensor,
        cache: list[TorchMockKVCache] | None = None,
    ) -> torch.Tensor:
        h = self.embed_tokens(inputs)
        mask = create_additive_causal_mask(h.shape[1])
        mask = mask.to(h.dtype)

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            h = layer(h, mask, layer_cache)
        return self.norm(h)


class TorchMockCausalLM(nn.Module):
    """Top-level model conforming to CausalLM protocol."""

    def __init__(
        self, d_model: int, num_layers: int, vocab_size: int, num_heads: int,
    ) -> None:
        super().__init__()
        self.model = TorchMockTransformerModel(
            d_model, num_layers, vocab_size, num_heads,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        inputs: torch.Tensor,
        cache: list[TorchMockKVCache] | None = None,
    ) -> torch.Tensor:
        h = self.model(inputs, cache)
        return self.lm_head(h)

    def make_cache(self) -> list[TorchMockKVCache]:
        """Create a fresh KV cache for each layer."""
        return [TorchMockKVCache() for _ in self.model.layers]

    def parameters(self) -> dict[str, torch.Tensor]:  # type: ignore[override]
        """Return nested parameter dict — matches MLX model.parameters() convention."""
        return dict(self.named_parameters())
