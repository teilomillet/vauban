<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# MPS Primitive Boundary

Custom MPS work should begin at one primitive boundary: activation projection
and intervention inside the trace path. The reason is that this primitive is
central to Vauban's behavior reports, has clear tensor inputs and outputs, and
can be checked against `run_torch_activation_primitive()` without changing
product semantics.

## Boundary

The primitive owns only this local computation:

- input activation tensor with shape `(batch, tokens, d_model)`;
- one direction vector or subspace tensor on the same device;
- layer index and prompt/token metadata for evidence labeling;
- projection mode such as scalar projection or subspace projection;
- optional intervention mode such as subtract, add, clamp, or conditional steer.

It returns:

- projection summaries for the selected layer and tokens;
- an optional intervened activation tensor with the same shape and dtype policy;
- `TraceArtifact` metadata that names the layer, tensor shape, dtype, and
  device;
- `StageProfile` timing/memory fields emitted by the caller.

## Non-Goals

The MPS kernel must not own model loading, tokenization, prompt formatting,
behavior scoring, report rendering, or backend selection. Those are Vauban
orchestration concerns. The kernel is an accelerator for one evidence primitive,
not a second product surface.

## Candidate Kernels

Start with the smallest kernels that can be compared directly to
`run_torch_activation_primitive()`:

- direction projection: `activation @ direction`;
- subspace projection and removal;
- conditional steering gate based on projection threshold;
- batched projection summaries for behavior trace profile sweeps.

Each candidate must have a Torch reference path and shape/dtype/device tests
before benchmark work begins.

## Verification Gate

Before an MPS kernel can become a default implementation:

- compare outputs against the Torch reference on CPU and CUDA in CI or local
  smoke tests;
- validate on an Apple Silicon MPS host with the same behavior trace config;
- confirm emitted `TraceArtifact` and `StageProfile` records are stable;
- run a small behavior suite before and after enabling the kernel;
- report performance only from measured MPS benchmark data, with variance.

If a kernel improves speed but changes evidence semantics, keep it experimental.
