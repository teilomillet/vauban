# Reasoning Direction via Weight-Diff SVD

**Research question:** Is reasoning capability encoded as a low-rank direction in weight space, analogous to how refusal is?

**Method:** Apply LoX-style weight-diff SVD to reasoning model pairs (DeepSeek-R1-Distill-Qwen-7B vs Qwen2.5-7B), extract top-k directions, analyze spectral structure, probe, and steer.

**Prior art:** ThinkEdit (EMNLP 2025) proved reasoning-length is editable via the same projection-removal math as abliteration. RAIN-Merging (ICLR 2026) showed reasoning and instruction task vectors are nearly orthogonal. Nobody has published the SVD spectrum of a reasoning task vector.

See `docs/research/reasoning_mech_interp.md` for full literature review.

---

## Experiments

### Phase 1: Spectral analysis (does a reasoning direction exist?)

| Config | What it does |
|---|---|
| `01_reasoning_diff_7b.toml` | SVD of `W_R1-Distill - W_Qwen2.5-7B` — extract reasoning directions |
| `02_alignment_diff_7b.toml` | SVD of `W_Qwen2.5-7B-Instruct - W_Qwen2.5-7B` — extract alignment directions (control) |

Compare:
- Singular value spectrum shape (concentrated vs flat)
- Explained variance per layer (where does reasoning live?)
- Effective rank of the diff

### Phase 2: Probing (does the direction correlate with reasoning behavior?)

| Config | What it does |
|---|---|
| `03_probe_reasoning.toml` | Probe R1-Distill-Qwen-7B with reasoning vs factual prompts using the extracted direction |

### Phase 3: Steering (can we inject/suppress reasoning?)

| Config | What it does |
|---|---|
| `04_steer_inject.toml` | Amplify reasoning direction in Qwen2.5-7B-Instruct |
| `05_steer_suppress.toml` | Suppress reasoning direction in R1-Distill-Qwen-7B |

### Phase 4: 32B validation (if 7B works)

| Config | What it does |
|---|---|
| `06_reasoning_diff_32b.toml` | SVD of `W_QwQ-32B - W_Qwen2.5-32B` |

---

## Memory requirements

| Experiment | Models loaded | Estimated peak RAM |
|---|---|---|
| 01-02 (7B diff) | Two 7B bf16 (~15 GB each) | ~35 GB |
| 03 (7B probe) | One 7B 4-bit (~4 GB) + direction from disk | ~8 GB |
| 04-05 (7B steer) | One 7B 4-bit (~4 GB) + direction from disk | ~8 GB |
| 06 (32B diff) | Two 32B bf16 (~65 GB each) | ~140 GB |

For 01-02: requires 64+ GB unified memory (M2 Pro/Max or higher).
For 06: requires 192 GB (M2 Ultra or Mac Studio).

Alternative: use 4-bit models (auto-dequantized) — ~18 GB peak for 7B, ~40 GB for 32B. Introduces quantization noise in the diff but is practical on 32 GB Macs.
