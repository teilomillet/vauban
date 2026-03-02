"""AmpleGCG generator training: fit an MLP on collected GCG suffixes.

Trains a 2-layer MLP that maps a prompt embedding to suffix token logits.
The generator is trained with cross-entropy loss on the collected suffixes
and can then produce hundreds of candidates in a single forward pass.

Reference: arxiv.org/abs/2404.07921
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vauban import _ops as ops
from vauban._forward import force_eval

if TYPE_CHECKING:
    from vauban._array import Array


@dataclass(slots=True)
class AmpleGCGGenerator:
    """Trained 2-layer MLP generator for suffix production.

    Maps prompt embeddings to suffix token logits. Architecture:
    input (embed_dim) → hidden (hidden_dim) → output (n_tokens * vocab_size)
    """

    w1: Array
    b1: Array
    w2: Array
    b2: Array
    n_tokens: int
    vocab_size: int

    def forward(self, prompt_embed: Array) -> Array:
        """Generate suffix token logits from a prompt embedding.

        Args:
            prompt_embed: Mean-pooled prompt embedding (embed_dim,).

        Returns:
            Logits of shape (n_tokens, vocab_size).
        """
        h = ops.maximum(prompt_embed @ self.w1 + self.b1, ops.array(0.0))
        out = h @ self.w2 + self.b2
        return out.reshape(self.n_tokens, self.vocab_size)

    def sample(
        self,
        prompt_embed: Array,
        n_candidates: int,
        temperature: float = 1.0,
    ) -> list[list[int]]:
        """Sample multiple suffix candidates from the generator.

        Args:
            prompt_embed: Mean-pooled prompt embedding (embed_dim,).
            n_candidates: Number of candidates to sample.
            temperature: Sampling temperature (higher = more diverse).

        Returns:
            List of token ID lists, one per candidate.
        """
        logits = self.forward(prompt_embed)
        candidates: list[list[int]] = []
        for _ in range(n_candidates):
            if temperature <= 0.0:
                token_ids = ops.argmax(logits, axis=-1)
            else:
                probs = ops.softmax(logits / temperature, axis=-1)
                token_ids_list: list[int] = []
                for pos in range(self.n_tokens):
                    # Categorical sampling via uniform + cumulative sum
                    u = ops.random.uniform(shape=(1,))
                    cumprobs = ops.cumsum(probs[pos], axis=-1)
                    token_id = int(
                        ops.sum(cumprobs < u[0]).item(),
                    )
                    token_id = min(token_id, self.vocab_size - 1)
                    token_ids_list.append(token_id)
                candidates.append(token_ids_list)
                continue
            force_eval(token_ids)
            candidates.append(
                [int(t) for t in token_ids.tolist()],
            )
        return candidates


def train_amplecgc_generator(
    collected_suffixes: list[tuple[list[int], float]],
    embed_dim: int,
    n_tokens: int,
    vocab_size: int,
    hidden_dim: int = 512,
    train_steps: int = 200,
    learning_rate: float = 0.001,
) -> AmpleGCGGenerator:
    """Train a 2-layer MLP generator on collected GCG suffixes.

    Uses manual SGD with momentum (Adam-like) for backend agnosticism.
    The training signal is cross-entropy between the generator's output
    logits and the collected suffix token IDs, weighted by inverse loss
    (better suffixes get higher weight).

    Args:
        collected_suffixes: List of (token_ids, loss) from collection phase.
        embed_dim: Embedding dimension of the model.
        n_tokens: Number of suffix tokens.
        vocab_size: Vocabulary size.
        hidden_dim: Hidden layer dimension.
        train_steps: Number of training steps.
        learning_rate: Learning rate for parameter updates.

    Returns:
        Trained AmpleGCGGenerator.
    """
    # Initialize weights with Xavier/He initialization
    scale1 = (2.0 / embed_dim) ** 0.5
    scale2 = (2.0 / hidden_dim) ** 0.5
    w1 = ops.random.normal((embed_dim, hidden_dim)) * scale1
    b1 = ops.zeros((hidden_dim,))
    w2 = ops.random.normal((hidden_dim, n_tokens * vocab_size)) * scale2
    b2 = ops.zeros((n_tokens * vocab_size,))
    force_eval(w1, b1, w2, b2)

    # Prepare training data: suffix tokens as target, random embeddings as input
    # (generator learns a general mapping, not prompt-specific)
    all_token_ids: list[list[int]] = [s[0] for s in collected_suffixes]
    all_losses: list[float] = [s[1] for s in collected_suffixes]

    # Compute weights: inverse loss (lower loss = higher weight)
    max_loss = max(all_losses) if all_losses else 1.0
    weights = [max_loss - loss + 0.1 for loss in all_losses]
    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]

    for _step in range(train_steps):
        # Sample a mini-batch of suffixes
        batch_size = min(16, len(all_token_ids))
        indices = ops.random.randint(
            low=0, high=len(all_token_ids), shape=(batch_size,),
        )
        force_eval(indices)
        raw_indices = indices.tolist()
        idx_list: list[int] = (
            [int(raw_indices)]
            if not isinstance(raw_indices, list)
            else [int(i) for i in raw_indices]
        )

        total_loss = ops.array(0.0)
        accum_grad_w1 = ops.zeros_like(w1)
        accum_grad_b1 = ops.zeros_like(b1)
        accum_grad_w2 = ops.zeros_like(w2)
        accum_grad_b2 = ops.zeros_like(b2)

        for idx in idx_list:
            target_ids = all_token_ids[idx]
            sample_weight = weights[idx]

            # Random prompt embedding (generator should generalize)
            prompt_embed = ops.random.normal((embed_dim,)) * 0.1
            force_eval(prompt_embed)

            def ce_loss(
                _w1: Array, _b1: Array, _w2: Array, _b2: Array,
                _prompt: Array = prompt_embed,
                _target: list[int] = target_ids,
                _sw: float = sample_weight,
            ) -> Array:
                h = ops.maximum(_prompt @ _w1 + _b1, ops.array(0.0))
                out = h @ _w2 + _b2
                logits = out.reshape(n_tokens, vocab_size)
                # Cross-entropy loss
                target_arr = ops.array(_target)
                log_probs = ops.log(ops.softmax(logits, axis=-1) + 1e-10)
                # Gather log probs at target positions
                loss = ops.array(0.0)
                for pos in range(n_tokens):
                    loss = loss - log_probs[pos, int(target_arr[pos].item())]
                return loss * _sw / n_tokens

            (loss_val, (gw1, gb1, gw2, gb2)) = ops.value_and_grad(
                ce_loss, argnums=(0, 1, 2, 3),
            )(w1, b1, w2, b2)
            force_eval(loss_val, gw1, gb1, gw2, gb2)
            total_loss = total_loss + loss_val
            accum_grad_w1 = accum_grad_w1 + gw1
            accum_grad_b1 = accum_grad_b1 + gb1
            accum_grad_w2 = accum_grad_w2 + gw2
            accum_grad_b2 = accum_grad_b2 + gb2

        # SGD update
        scale = 1.0 / batch_size
        w1 = w1 - learning_rate * accum_grad_w1 * scale
        b1 = b1 - learning_rate * accum_grad_b1 * scale
        w2 = w2 - learning_rate * accum_grad_w2 * scale
        b2 = b2 - learning_rate * accum_grad_b2 * scale
        force_eval(w1, b1, w2, b2)

    return AmpleGCGGenerator(
        w1=w1, b1=b1, w2=w2, b2=b2,
        n_tokens=n_tokens, vocab_size=vocab_size,
    )
