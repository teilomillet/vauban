---
title: "Composability — Unix Philosophy for LLM Safety Tools"
description: "Small tools that do one thing well. Measure, cut, probe, steer, cast, guard — each with typed inputs and outputs. TOML config wires them together. Any piece swappable."
keywords: "composable AI tools, modular safety pipeline, TOML configuration, Unix philosophy AI, typed pipeline"
---

<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Composability

Vauban follows the Unix philosophy: small tools that do one thing well, connected through simple interfaces. Nothing is glued together — parts are assembled. Any piece can be swapped without touching the rest.

## The building blocks

Each pipeline module is a focused operation with clear inputs and outputs:

| Module | Input | Output |
|--------|-------|--------|
| **Measure** | Model + prompt pairs | Direction (unit vector) |
| **Cut** | Model + direction | Modified model weights |
| **Evaluate** | Model | Refusal rate, perplexity, KL divergence |
| **Probe** | Model + direction + prompt | Per-layer projection strengths |
| **Surface** | Model + direction + prompt set | Projection map with refusal decisions |
| **Steer** | Model + direction + prompt | Generated text (steered) |
| **CAST** | Model + direction + prompt | Generated text (conditionally steered) |
| **SIC** | Model + prompt | Sanitized prompt |
| **Guard** | Model + direction | Circuit breaker with KV rewind |
| **Softprompt** | Model + target prompt | Optimized adversarial tokens |
| **Export** | Modified weights | Loadable model directory |

The output of Measure (a direction) is the input to Cut, Probe, Surface, Steer, CAST, and Guard. The output of Cut (modified weights) is the input to Evaluate and Export. These are not hardcoded pipelines — they are components you wire together.

> **Frozen dataclass** — a Python dataclass with `frozen=True` and `slots=True`. Once created, its fields cannot be modified. This guarantees that a result passed between pipeline stages cannot be silently mutated. Every config and result type in Vauban is frozen.

## TOML as the wiring diagram

The TOML config file declares which modules run and how they connect. There is no procedural pipeline code that users write or modify. The config is the pipeline.

A config that measures a direction and maps the refusal surface:

```toml
[model]
name = "mlx-community/Qwen2.5-1.5B-Instruct-bf16"

[measure]
mode = "direction"

[surface]
prompt_file = "prompts.jsonl"
```

A config that adds softprompt attack against CAST defense:

```toml
[model]
name = "mlx-community/Qwen2.5-1.5B-Instruct-bf16"

[measure]
mode = "direction"

[cast]
threshold = 1.5

[softprompt]
mode = "gcg"
n_tokens = 20
steps = 200
```

The pipeline runner reads the config, determines which sections are present, resolves dependencies (softprompt needs a direction, so measure runs first), and executes in order. Adding or removing a module is adding or removing a TOML section.

## Interfaces are dataclasses

Every boundary between modules is a typed, frozen dataclass. The direction that Measure produces is not a raw array — it is a `MeasureResult` with fields for the direction vector, the layers it was extracted from, and the metadata about how it was computed.

This matters for three reasons:

**Type safety.** The type checker (`ty`) verifies that a module receiving a `MeasureResult` actually uses the fields that exist on it. There is no stringly-typed dictionary passing.

> **Type checker** — a tool that reads your code *without running it* and verifies that values match their declared types. If a function expects a `MeasureResult` but you pass it a string, the type checker catches this before you ever run the code. Vauban uses `ty` for type checking — it catches bugs at development time rather than at runtime.

**Immutability.** A frozen dataclass cannot be mutated after creation. When Measure hands a direction to CAST, CAST cannot accidentally modify it. The direction that CAST uses is guaranteed to be the direction that Measure produced.

**Serialization.** Dataclasses serialize cleanly to JSON for reports and logging. The complete state of a pipeline run is captured in structured, typed data.

> **Serialization** — converting an in-memory object (like a direction result) into a format that can be saved to a file or sent over a network (like JSON). Deserialization is the reverse: reading the file back into an object. Clean serialization means pipeline results can be saved, shared, and inspected without loss of information.

> **Dataclass** — a Python class that primarily holds data fields, declared with the `@dataclass` decorator. With `frozen=True`, instances are immutable after creation. With `slots=True`, they use less memory and prevent accidental attribute creation.

## Swapping components

The composable design means any component can be replaced without affecting others:

**Measurement mode.** Switching from mean-difference (`mode = "direction"`) to SVD (`mode = "subspace"`) to weight-diff (`mode = "diff"`) changes how the direction is computed but not how downstream modules consume it. Cut, CAST, and Probe all accept a direction regardless of how it was measured.

**Backend.** The `_ops` and `_array` abstraction layers mean pipeline code never imports MLX or PyTorch directly. Switching backends changes the tensor operations underneath but not the pipeline logic above.

**Defense layers.** CAST, SIC, and Guard are independent modules. You can run any combination. Adding Guard to an existing CAST config is adding a `[guard]` section — no code changes, no rewiring.

**Attack mode.** GCG and EGD are different optimization algorithms for the same softprompt objective. Switching between them is `mode = "gcg"` vs `mode = "egd"` in the same `[softprompt]` section.

## The Session API

For programmatic use (particularly by AI assistants), the `Session` class composes these modules into a stateful interface:

```python
from vauban import Session

s = Session("mlx-community/Qwen2.5-1.5B-Instruct-bf16")
s.measure()       # extract direction
s.cast()          # enable runtime defense
s.audit(prompts)  # check behavior
s.state()         # inspect current pipeline state
```

Each method corresponds to a pipeline module. The Session tracks which operations have been performed and what results are available. It is the same composability as the TOML config, expressed as method calls instead of config sections.

The Session is self-describing: an agent can call `s.tools()` to discover available operations, their parameters, and their current applicability given the pipeline state.

!!! note "Composition order"
    Some compositions have implicit ordering. Measure must run before Cut (you need a direction to project out). CAST must have a direction before it can monitor. The pipeline runner resolves this automatically from the TOML config. The Session tracks it via state. But the modules themselves have no knowledge of each other — they depend on interfaces, not implementations.

## Related pages

- [Attack-Defense Duality](attack-defense-duality.md) — how attack and defense modules compose
- [Reproducibility](reproducibility.md) — how TOML configs capture complete experiment definitions
- [Last-Mile Reliability](last-mile-reliability.md) — how defense composition addresses production needs
