"""Scanning helpers for refusal surface mapping."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from vauban import _ops as ops
from vauban._forward import embed_and_mask, extract_logits, force_eval, make_cache
from vauban.evaluate import DEFAULT_REFUSAL_PHRASES, _judge_single
from vauban.types import CausalLM, SurfacePoint, SurfacePrompt, Tokenizer

if TYPE_CHECKING:
    from vauban._array import Array


def scan(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompts: list[SurfacePrompt],
    direction: Array,
    direction_layer: int,
    *,
    generate: bool = True,
    max_tokens: int = 60,
    refusal_phrases: list[str] | None = None,
    progress: bool = True,
    refusal_mode: str = "phrases",
) -> list[SurfacePoint]:
    """Scan prompts: probe projections and optionally generate + detect refusal."""
    if refusal_phrases is None:
        refusal_phrases = DEFAULT_REFUSAL_PHRASES

    points: list[SurfacePoint] = []
    total = len(prompts)

    for i, sp in enumerate(prompts):
        if progress:
            print(
                f"\r  Scanning {i + 1}/{total}: {sp.prompt[:50]}...",
                end="",
                file=sys.stderr,
                flush=True,
            )

        messages = _surface_messages(sp)
        projections = _probe_with_messages(
            model,
            tokenizer,
            messages,
            direction,
        )
        direction_proj = projections[direction_layer]

        refused: bool | None = None
        response: str | None = None
        if generate:
            response = _generate_with_messages(
                model,
                tokenizer,
                messages,
                max_tokens,
            )
            if refusal_mode == "judge":
                refused = _judge_single(model, tokenizer, sp.prompt, response)
            else:
                lower = response.lower()
                refused = any(
                    phrase.lower() in lower for phrase in refusal_phrases
                )

        points.append(
            SurfacePoint(
                prompt=sp.prompt,
                label=sp.label,
                category=sp.category,
                projections=projections,
                direction_projection=direction_proj,
                refused=refused,
                response=response,
                style=sp.style,
                language=sp.language,
                turn_depth=sp.turn_depth,
                framing=sp.framing,
                messages=sp.messages,
            ),
        )

    if progress:
        print("", file=sys.stderr)

    return points


def _surface_messages(surface_prompt: SurfacePrompt) -> list[dict[str, str]]:
    """Return full chat history for probing and generation."""
    if surface_prompt.messages is not None:
        return [
            {"role": message["role"], "content": message["content"]}
            for message in surface_prompt.messages
        ]
    return [{"role": "user", "content": surface_prompt.prompt}]


def _probe_with_messages(
    model: CausalLM,
    tokenizer: Tokenizer,
    messages: list[dict[str, str]],
    direction: Array,
) -> list[float]:
    """Compute per-layer projections for a full message list."""
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    if not isinstance(text, str):
        msg = "apply_chat_template must return str when tokenize=False"
        raise TypeError(msg)
    token_ids = ops.array(tokenizer.encode(text))[None, :]

    transformer = model.model
    h, mask = embed_and_mask(transformer, token_ids)

    projections: list[float] = []
    for layer in transformer.layers:
        h = layer(h, mask)
        last_token = h[0, -1, :]
        proj = ops.sum(last_token * direction)
        force_eval(proj)
        projections.append(float(proj.item()))
    return projections


def _generate_with_messages(
    model: CausalLM,
    tokenizer: Tokenizer,
    messages: list[dict[str, str]],
    max_tokens: int,
    eos_token_id: int | None = None,
) -> str:
    """Greedy generation over an arbitrary chat message list."""
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    if not isinstance(text, str):
        msg = "apply_chat_template must return str when tokenize=False"
        raise TypeError(msg)
    tokens = tokenizer.encode(text)
    generated: list[int] = []

    if eos_token_id is None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)

    cache = make_cache(model)
    token_ids = ops.array([tokens])
    for _ in range(max_tokens):
        result = model(token_ids, cache=cache)  # type: ignore[call-non-callable]
        logits = extract_logits(result)
        next_token = int(ops.argmax(logits[:, -1, :], axis=-1).item())
        generated.append(next_token)
        if eos_token_id is not None and next_token == eos_token_id:
            break
        token_ids = ops.array([[next_token]])

    return tokenizer.decode(generated)
