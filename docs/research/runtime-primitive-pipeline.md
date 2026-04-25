<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Runtime Primitive Pipeline Design Note

This is a handoff note for implementing Vauban's next backend layer.

The goal is not to make PyTorch imitate MLX, or to hide framework differences
behind a large abstraction. The goal is to define Vauban's own runtime
primitives: small, typed, composable stages that expose exactly what evidence a
backend can produce.

## Current Status

Verified in the current tree:

- Vauban now has access-aware behavior report primitives in
  `vauban/behavior/_primitives.py`.
- Those primitives describe report claims, evidence, access levels, and
  limitations.
- They do not yet define runtime execution primitives.
- Runtime code is still spread across `_ops`, `_forward`, `_model_io`, MLX
  helpers, and Torch helpers.

This means report claim strength is becoming epistemic, but backend execution is
not yet shaped the same way.

## Design Claim

Vauban should have a runtime primitive pipeline.

Each stage should have a narrow typed input, a narrow typed output, declared
capabilities, and optional profiling metadata. Pipeline code should combine
stages instead of depending directly on MLX, PyTorch, MAX, or framework-specific
model objects.

Reason: Vauban's product surface is the Model Behavior Change Report. Reports
depend on evidence. Evidence depends on what the runtime actually observed:
outputs, logits, logprobs, activations, weights, interventions, KV cache state,
or only black-box text. Backend support must therefore be explicit data, not an
implicit assumption.

## Core Principles

1. Vauban owns the semantics.

   A backend implements Vauban primitives. Vauban should not become a thin alias
   layer over MLX or PyTorch APIs.

2. Capabilities are explicit.

   If a backend cannot collect activations, apply an intervention, expose
   logprobs, or run on GPU, it must say so through a typed capability object.
   Missing support should narrow the report claim, not silently fall back to a
   weaker behavior.

3. MLX is the first reference implementation.

   MLX is already the primary runtime and gives direct access to eager arrays,
   layer activations, and intervention points. Implementing the primitive
   contract on MLX first gives us a regression baseline before GPU work.

4. PyTorch/GPU comes after the MLX contract passes.

   The RTX 4070 Ti path should implement the same Vauban contract and compare
   against MLX at the level of shapes, dtypes, device placement, metric
   equivalence, and behavior-report artifacts. It does not need bit-identical
   tensors.

5. Pipelines stay small and profileable.

   Loading, tokenization, forward pass, activation capture, intervention,
   scoring, and reporting should be isolated stages. That makes failures,
   performance regressions, memory pressure, and unsupported backend features
   easier to locate.

## Proposed Runtime Package

Start with a small package, not a rewrite:

```text
vauban/runtime/
  __init__.py
  _types.py          # frozen dataclasses for requests, traces, devices
  _capabilities.py   # declared backend support and claim mapping
  _protocols.py      # Protocol contracts implemented by runtimes
  _registry.py       # backend selection and construction
  _profiling.py      # optional per-stage timing and memory records
  _mlx.py            # first concrete implementation
  _torch.py          # later concrete implementation
```

Reason: this keeps backend semantics separate from report primitives, config
parsing, and high-level pipeline modes.

## Primitive Shape

The first contract should be deliberately small:

```python
from dataclasses import dataclass
from typing import Literal, Protocol

type BackendName = Literal["mlx", "torch", "max"]
type SupportLevel = Literal["unsupported", "partial", "full"]
type DeviceKind = Literal["cpu", "gpu", "cuda", "mps"]


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    name: BackendName
    device_kinds: tuple[DeviceKind, ...]
    logits: SupportLevel
    logprobs: SupportLevel
    activations: SupportLevel
    interventions: SupportLevel
    kv_cache: SupportLevel
    weight_access: SupportLevel
    mutable_weights: SupportLevel


@dataclass(frozen=True, slots=True)
class ForwardRequest:
    prompt_ids: tuple[int, ...]
    collect_layers: tuple[int, ...] = ()
    return_logits: bool = True
    return_logprobs: bool = False


@dataclass(frozen=True, slots=True)
class ForwardTrace:
    logits: object | None
    logprobs: object | None
    activations: dict[int, object]
    device: str
    profile: tuple["StageProfile", ...]


class ModelRuntime(Protocol):
    capabilities: BackendCapabilities

    def load(self, model_path: str) -> "LoadedModel": ...

    def forward(self, request: ForwardRequest) -> ForwardTrace: ...
```

This sketch is intentionally not final code. In implementation, tensor fields
should use a precise Vauban tensor alias or wrapper rather than `object` if the
type checker can express it cleanly.

## Pipeline Stages

The runtime should behave like a composable pipeline:

```text
ModelRef
  -> LoadModel
  -> Tokenize
  -> PrepareBatch
  -> Forward
  -> CollectActivations
  -> ApplyIntervention
  -> ScoreOutputs
  -> EmitEvidence
  -> RenderReport
```

Stages can be added, removed, or combined by config and mode logic. Each stage
should emit typed data and optional profile records.

Reason: this makes Vauban easier to reason about. It also lets us profile only
the expensive section, such as tokenization, prefill, decode, activation
collection, or projection scoring.

## Epistemic Contract

The backend contract and report contract must connect.

Examples:

- If the runtime returns only text outputs, Vauban can support a behavioral
  profile or black-box behavioral diff.
- If the runtime returns logits or logprobs, Vauban can support a
  distributional diff.
- If the runtime returns activations, Vauban can support activation diagnostics.
- If the runtime exposes base and transformed weights, Vauban can support the
  strongest model-change audit claims.

This is the main reason to make capabilities explicit. A backend capability gap
is not just an engineering detail. It changes what the report is allowed to
claim.

## Implementation Order

1. Define the runtime dataclasses, protocols, and capability model.

   Reason: this creates the target without changing behavior.

2. Implement the MLX runtime adapter first.

   Reason: MLX is the known-good reference path and already exposes the
   internals Vauban needs.

3. Add contract tests that MLX must pass.

   These tests should cover model loading, device declaration, forward traces,
   logits, optional logprobs, activation collection, intervention support, and
   generated evidence metadata.

4. Add narrow regression fixtures.

   The fixtures should verify stable shapes, stable report keys, stable
   capability declarations, and stable behavior metrics. They should avoid
   claiming bit-identical numerical reproducibility.

5. Implement the PyTorch runtime adapter against the same contract.

   Reason: GPU support should be a second implementation of Vauban semantics,
   not a separate semantic branch.

6. Validate CPU fallback and CUDA use separately.

   CPU fallback should be explicit. CUDA use should be verified through device
   metadata and a minimal tensor/model smoke test.

7. Evaluate MAX/Mojo only against this contract.

   MAX can become a runtime backend if it can satisfy enough of the same
   primitives. If it only supports black-box inference for a given model, then
   it should declare that and produce lower-strength evidence.

## Testing Strategy

Use three layers of tests:

- Contract tests: every backend implementation must return the same typed
  structures and honest capabilities.
- MLX regression tests: MLX establishes the reference behavior and report
  artifacts.
- Cross-backend equivalence tests: Torch/GPU is compared against MLX for shapes,
  supported evidence, metric direction, and report structure.

Reason: fixing PyTorch failures one by one before defining the contract risks
encoding accidental MLX behavior instead of Vauban behavior.

## Non-Goals

- Do not rewrite every pipeline mode at once.
- Do not add a large generic framework abstraction.
- Do not hide unsupported features with silent fallbacks.
- Do not claim GPU acceleration until measured on the target hardware.
- Do not claim MAX parity until it passes the same runtime contract.

## Open Questions

- What is the smallest MLX model fixture that can exercise logits,
  activations, and one intervention path deterministically?
- Should tensor values remain raw backend tensors, or should Vauban introduce a
  tiny tensor wrapper at the runtime boundary?
- Which existing mode should be migrated first: measure, probe, scan, or a
  minimal behavior trace path?
- How much profiling metadata should be always-on versus opt-in?

## Handoff Summary

Build Vauban runtime support as a pipeline of primitives.

Start with MLX so we have a trusted regression target. Then make PyTorch/CUDA
implement the same contract. Treat GPU acceleration as an implementation detail
under a capability-tested runtime, not as a separate product path.

The design goal is ease of mind: small modules, explicit claims, explicit
capabilities, profileable stages, and reports whose epistemic strength follows
from what the backend actually observed.
