<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Trace-First Inference Stack

This is a handoff note for Vauban's next backend layer.

The one-word primitive is **Trace**.

Vauban should not start by hardcoding a large set of framework-facing concepts
such as activation trace, intervention plan, weight view, model runtime, or
device policy. Those may become useful later, but they should emerge from the
execution record. The first practical primitive should be the evidence-bearing
trace of an inference run.

```text
model + input + requested observations
  -> trace
  -> evidence
  -> report claim
```

Reason: Vauban's product surface is the Model Behavior Change Report. Reports
need evidence. Evidence comes from what the inference stack actually observed,
not from what a backend abstraction promised in theory.

## Current Status

Verified in the current tree:

- Vauban has access-aware behavior report primitives in
  `vauban/behavior/_primitives.py`.
- Those primitives describe report claims, evidence, access levels, and
  limitations.
- Runtime execution is still spread across `_ops`, `_forward`, `_model_io`, MLX
  helpers, and Torch helpers.
- The current runtime layer does not yet emit a single evidence-bearing trace.

This means report claim strength is becoming epistemic, but inference execution
is not yet shaped in the same epistemic way.

## Design Claim

Vauban should build a trace-first inference stack.

A backend executes inference. A trace records what happened. The report consumes
evidence from the trace. Backend capabilities bound what a trace can honestly
contain.

```text
executor -> trace -> artifacts -> evidence -> claim
```

This keeps the design practical. The trace is the core. Other concepts are
levels inside or around the trace, not a large upfront framework.

## Levels

Use different levels of primitives, but keep the root small:

```text
Trace       full evidence-bearing record of a run
Span        one stage inside the run
Artifact    a produced or consumed value
Capability  what an executor can honestly produce
Profile     timing, memory, device, transfer, and sync data
```

Examples of artifact kinds:

```text
text
tokens
logits
logprobs
activation
intervention_result
weight_snapshot
metric
report_evidence
```

The important design move is that `activation`, `logits`, `logprobs`, and
`intervention_result` are artifact kinds. They are not the deepest primitives.

## USL Lens

The Universal Scaling Law does not discover code primitives automatically, but
it gives a useful pressure test.

See: [Universal Scaling Law](https://teilo.xyz/collections/universal-scaling-law/)

A primitive earns its place when it exposes or reduces one of the costs that
limit scaling:

- serialization
- contention
- coordination
- synchronization
- device transfer
- memory pressure
- queueing or batching limits

That means a trace span should make these costs inspectable when practical:

```text
span name
start and end time
device
batch size
token count
memory estimate
host/device copies
synchronization points
backend-specific notes
```

Reason: if a primitive cannot help with evidence, correctness, or scaling
visibility, it should probably stay a local implementation detail.

## Primitive Promotion Test

A concept should become a named primitive only if most of these are true:

- It crosses multiple modules.
- It survives MLX and PyTorch without changing meaning.
- It affects what a report can claim.
- It helps isolate performance or correctness.
- It exposes serialization, contention, synchronization, transfer, or memory
  cost.
- Hiding it would make debugging harder.

If not, keep it concrete and local until repetition proves otherwise.

## Practical First Contract

The first implementation target should be smaller than a generic runtime API:

```text
TraceRequest
Trace
TraceSpan
TraceArtifact
RuntimeCapability
```

Their job:

- `TraceRequest`: declares the model input and requested observations.
- `Trace`: records what happened and what was observed.
- `TraceSpan`: records one stage such as tokenize, prefill, forward, decode,
  observe, intervene, score, or render.
- `TraceArtifact`: records typed evidence produced or consumed by spans.
- `RuntimeCapability`: records what artifact kinds the executor can produce.

This is intentionally lower commitment than a `ModelRuntime.forward()` contract.
The executor can be MLX, Torch, MAX, or something else. The stable surface is
the trace.

## Inference Stack Shape

The stack should be close to the real inference path:

```text
load
  -> tokenize
  -> batch
  -> prefill
  -> forward
  -> observe
  -> decode
  -> score
  -> emit evidence
```

Not every run uses every span. A black-box endpoint may only emit text and
timing artifacts. MLX may emit layer activations and intervention artifacts.
Torch/CUDA may initially emit logits, device, and profile artifacts, then grow
toward activation artifacts.

This is the core epistemic rule: missing artifacts narrow the claim instead of
silently pretending parity.

## Capability Semantics

Capabilities should be artifact-oriented:

```text
can_emit_text
can_emit_tokens
can_emit_logits
can_emit_logprobs
can_emit_activations
can_apply_interventions
can_inspect_weights
can_mutate_weights
can_profile_device_memory
can_profile_sync_points
```

Reason: this aligns runtime support with report evidence. The report does not
care that a backend is "Torch" in the abstract. It cares whether the trace
contains the artifacts needed for a behavioral profile, distributional diff,
activation diagnostic, or model-change audit.

## Claim Mapping

The report layer should read trace artifacts and capabilities:

```text
text only
  -> behavioral profile or black-box behavioral diff

text + paired outputs
  -> black-box behavioral diff

logits or logprobs
  -> distributional diff

activations
  -> activation diagnostic

weights or base/transformed weights
  -> weight diff or model-change audit

intervention artifacts
  -> stronger causal evidence, bounded by the intervention actually run
```

The trace therefore becomes the bridge between inference and epistemic claim
strength.

## MLX First

Implement MLX first because it is the current reference path.

The first MLX slice should not migrate everything. It should produce a trace for
one narrow path:

```text
prompt ids
  -> forward
  -> logits artifact
  -> optional activation artifacts
  -> profile spans
```

Reason: this gives us a regression target before GPU work. PyTorch/CUDA should
then implement the same trace contract, with its own honest capability
declaration and profile spans.

## Testing Strategy

Use contract tests around the trace, not around framework internals:

- A trace has stable top-level fields.
- Each span has stable stage metadata.
- Each artifact has a stable kind, producer span, and backend-independent
  metadata.
- Unsupported requested artifacts fail explicitly or are recorded as missing.
- MLX establishes the first reference behavior.
- Torch/CUDA is compared against MLX for artifact presence, shape metadata,
  report structure, and metric direction, not bit-identical tensors.

Reason: fixing PyTorch failures one by one before defining this trace contract
risks encoding accidental MLX behavior instead of Vauban behavior.

## Non-Goals

- Do not rewrite every pipeline mode at once.
- Do not add a large generic framework abstraction.
- Do not hardcode activation, intervention, or weight primitives before the
  trace proves they need to be named.
- Do not hide unsupported artifacts with silent fallbacks.
- Do not claim GPU acceleration until profile spans measure it.
- Do not claim MAX parity until MAX can emit comparable trace artifacts.

## Open Questions

- What is the smallest MLX path that can emit logits and one activation artifact
  deterministically?
- Should trace artifacts hold raw backend tensors, tensor summaries, or both?
- Which existing mode should consume traces first: behavior trace, measure,
  probe, or scan?
- How much profile metadata should always be collected versus opt-in?
- How should trace artifacts be serialized without copying large tensors by
  default?

## Handoff Summary

Build the next Vauban runtime layer around **Trace**.

The trace records inference execution, artifacts, capabilities, and scaling
costs. Artifact kinds can grow over time. MLX provides the first reference
implementation. PyTorch/CUDA and MAX should be evaluated by whether they can
emit equivalent trace artifacts, not by whether they mimic MLX APIs.

The design goal is ease of mind: one core primitive, layered evidence, explicit
capabilities, profileable spans, and reports whose epistemic strength follows
from what the inference stack actually observed.
