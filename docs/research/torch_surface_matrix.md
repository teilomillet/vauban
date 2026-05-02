<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Torch Surface Matrix

Vauban's default runtime surface is Torch-first. The reason is portability:
the same behavior-auditing config should run on CPU, CUDA, and MPS without
changing product semantics. MLX remains a legacy/reference backend, but new
default-facing examples and report workflows should not require MLX.

This matrix is a contract, not a performance claim. A row marked `full` means
the Torch runtime declares and tests that the primitive can produce the same
kind of Vauban evidence as the rest of the system expects. Performance claims
still require device-specific measurement.

| Primitive | Support | Evidence artifact | Required tests | Runtime proof |
|-----------|---------|-------------------|----------------|---------------|
| `load_model` | full | `LoadedModel` with Torch capabilities | `tests/test_model_io.py`, `tests/test_backend.py` | CPU/CUDA smoke; MPS pending host |
| `tokenize` | full | `TokenizeResult` | `tests/test_runtime_primitives.py` | CPU/CUDA smoke; MPS pending host |
| `forward` | full | `ForwardResult` | `tests/test_runtime_primitives.py` | CPU/CUDA smoke; MPS pending host |
| `runtime_trace` | full | `ForwardTrace` and `TraceArtifact` records | `tests/test_runtime_trace_types.py`, `tests/test_runtime_primitives.py` | CPU/CUDA smoke; MPS pending host |
| `generation` | full | generated text observations | `tests/test_direct_generation_cuda.py`, behavior trace tests | CPU/CUDA smoke; MPS pending host |
| `logits` | full | runtime logits evidence | `tests/test_runtime_primitives.py` | CPU/CUDA smoke; MPS pending host |
| `logprobs` | full | runtime token logprobs evidence | `tests/test_runtime_primitives.py` | CPU/CUDA smoke; MPS pending host |
| `activations` | full | activation trace and projection artifacts | `tests/test_runtime_trace_types.py`, `tests/test_runtime_primitives.py`, `tests/test_runtime_activation_primitive.py`, `tests/test_behavior_trace_toml.py` | CPU/CUDA smoke; MPS pending host |
| `interventions` | full | intervention records with primitive metadata | `tests/test_runtime_primitives.py`, `tests/test_runtime_activation_primitive.py`, `tests/test_behavior_trace_toml.py` | CPU/CUDA smoke; MPS pending host |
| `kv_cache` | full | declared runtime capability | `tests/test_torch_surface_docs.py` | CUDA source/runtime path; MPS pending host |
| `weight_access` | full | declared runtime capability | `tests/test_torch_surface_docs.py` | CPU/CUDA smoke; MPS pending host |
| `mutable_weights` | full | declared runtime capability | `tests/test_torch_surface_docs.py` | CPU/CUDA smoke; MPS pending host |
| `safetensors_io` | full | weight export/load artifacts | export and LoRA tests | CPU/CUDA source path; MPS device-neutral |
| `peft_lora_export` | full | PEFT-format adapter artifacts | LoRA export tests | CPU/CUDA source path; MPS device-neutral |
| `device_profile_cuda` | full | `StageProfile` with CUDA sync/memory fields | CUDA smoke and alignment tests | validated on RTX 4070 Ti |
| `device_profile_mps` | full | `StageProfile` with MPS sync/memory fields | `tests/test_torch_surface_docs.py`, profiling tests | source contract; MPS hardware validation pending |
| `profile_sweep` | full | behavior trace runtime profile sidecars | behavior trace/diff tests | CPU/CUDA smoke; MPS pending host |

## Enforcement

The matrix is enforced by `tests/test_torch_surface_docs.py`:

- every primitive above must remain present in this document;
- Torch capabilities must declare `full` for the evidence-producing runtime
  capabilities used by reports;
- Torch device kinds must include `cpu`, `cuda`, and `mps`;
- default-facing docs and examples must not reintroduce MLX-only model IDs,
  imports, or MLX runtime defaults.
- `[behavior_trace.activation_primitive]` must keep projection evidence explicit
  in TOML and in emitted trace artifacts.

## Claim Boundary

This document does not claim Torch is always faster than MLX. It claims Torch is
the portable default surface for Vauban evidence collection. Any future MPS
custom kernel must preserve the same trace artifacts and report evidence before
it can replace the reference Torch implementation.
