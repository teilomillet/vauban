<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Mechanistic Interpretability of Reasoning in LLMs — Research Brief

**Date:** 2026-02-27
**Goal:** Assess whether weight-diff SVD (LoX-style) on reasoning model pairs is novel, coherent, and worth pursuing with vauban.

---

## Executive Summary

The field has exploded. Dozens of papers study reasoning internals via SAEs, circuit tracing, steering vectors, activation patching, and linear probes. **But nobody has applied weight-diff SVD to a reasoning model pair.** The building blocks all exist — the specific combination is an open gap.

Key finding that validates the approach: **ThinkEdit (EMNLP 2025)** shows reasoning-length control follows the exact same math as abliteration (`W_o ← W_o(I - v·vᵀ)`), editing only 0.2% of parameters. This directly implies reasoning has low-rank structure in weight space.

---

## 0. Deep-Dive: Key Papers (Implementation-Level Detail)

### ThinkEdit (Sun et al., EMNLP 2025) — [2503.22048](https://arxiv.org/abs/2503.22048)

**Methodology:**
1. Collect activations on long-CoT (>1000 tokens) and short-CoT (<100 tokens) GSM8K problems
2. Per-layer direction: `v_l = mean(r_long) - mean(r_short)` (post-attention residual stream)
3. Per-head contribution: `C^h = softmax(QK^T/sqrt(d))V @ W_o^h`
4. Score heads: `C_short^h = <mean(C^h on D_short), -v_hat_l>` (alignment with short-reasoning direction)
5. Edit top 4% of heads: `W_o^h ← W_o^h @ (I - d_neg @ d_neg^T)` — identical math to abliteration

**Key result:** +6.39% accuracy on short-reasoning cases, only 0.2% of parameters modified. Tested on R1-Distill-Qwen 1.5B/8B/14B/32B. No SVD analysis performed.

### Transcoder Adapters (Hu et al., Feb 2026) — [2602.20904](https://arxiv.org/abs/2602.20904)

**Methodology:**
- Sparse transcoders learn `T^l(x)` such that `MLP_base(x) + T(x) ≈ MLP_target(x)`
- 28 layers × 8192 features = 229,376 total features
- Trained on 50k samples from OpenThoughts3

**Feature taxonomy (LLM-judge classified):**
- 48% general language features
- 37% domain-specific (math, science, code)
- **8.6% reasoning-related** (uncertainty, reflection, exploration)
- 2.4% hesitation features ("wait", "hmm", "but") — ablating these cuts response length 50% **without accuracy loss** (except on hardest benchmarks like AIME25)

**No SVD analysis.** The approach is complementary to weight-diff SVD — different decomposition of the same underlying delta.

### RAIN-Merging (Huang et al., ICLR 2026 Oral) — [2602.22538](https://arxiv.org/abs/2602.22538)

**Task vectors:** `Δ_R = θ_R - θ_B` (reasoning), `Δ_I = θ_I - θ_B` (instruction)
**Orthogonality:** Principal subspace cosine similarity < 0.1 across all layers and submodules (Q,K,V,O projections + FFN). Measured via SVD of each task vector within each forward module.
**Null-space projection:** Projects `Δ_I` onto null space of forward features at `<think>` token positions, preserving reasoning format exactly.
**Tested on:** Qwen2.5 family (1.5B/7B/14B/32B) + Llama-3.1-8B.

### Weight Interpolation Phase Transition (Wu et al.) — [2510.10977](https://arxiv.org/abs/2510.10977)

**Formula:** `θ_merge = λ·θ_thinking + (1-λ)·θ_instruct`
**Phase transition (Qwen3-4B):**
- λ ∈ [0, 0.4): No CoT, gradual verbosity increase
- λ ∈ [0.4, 0.6]: **Abrupt emergence** — Think Ratio jumps 0→100%
- λ ∈ (0.6, 1.0]: Saturation, diminishing returns

**Module ablation (critical finding):**
- Skip FFN interpolation → Think Ratio drops to 0.68% (FFN teaches *how* to think)
- Skip MHA interpolation → Think Ratio stays 99.9% but Mean@64 drops (attention provides *knowledge*)
- Reasoning concentrates in last 2/3 of layers

### LoX (Perin et al., COLM 2025) — [2506.15606](https://arxiv.org/abs/2506.15606)

**SVD formula per weight matrix:**
```
Δ_W = W_aligned - W_base
U, S, V^T = SVD(Δ_W)
W_LoX = W_aligned + α · (U_k @ U_k^T) @ Δ_W
```
Left-projection onto top-k singular vectors amplifies the safety subspace. **Rank 3-6 sufficient** for safety. Applied to all weight matrices, all layers uniformly.

### Jin et al. (2025) — [2508.16546](https://arxiv.org/abs/2508.16546)

**Core finding:** RL changes weights via **direction rotation**, not magnitude change. Singular values barely change (fluctuations of 0.005); singular vector rotations reach 25-90°. Changes concentrate at **spectral extremes** (largest + smallest singular values). Restoring top 20% of singular vector directions recovers 70-80% of OOD performance. Tested on Llama-3.2-11B and Qwen-2.5-7B with PPO.

---

## 1. Weight-Space Methods (Most Relevant to Our Experiment)

### Directly relevant

| Paper | Date | Key finding |
|---|---|---|
| **Transcoder Adapters for Reasoning-Model Diffing** (Hu et al.) — [2602.20904](https://arxiv.org/abs/2602.20904) | Feb 2026 | Sparse transcoders approximate MLP diff between Qwen2.5-Math-7B and R1-Distill-Qwen-7B. Only ~8% of adapter features relate to reasoning. Ablating "hesitation" features (2.4%) cuts response length 50% without accuracy loss. |
| **ThinkEdit** (Sun et al., EMNLP 2025) — [2503.22048](https://arxiv.org/abs/2503.22048) | Mar 2025 | Reasoning length is a linear direction in representation space. ~4% of attention heads drive short reasoning. Projection removal on `W_o` (same as abliteration) gains +6.39% accuracy. Tested on R1-Distill-Qwen 1.5B–32B. |
| **Leveraging Parameter Space Symmetries** (Horoi et al.) — [2511.10850](https://arxiv.org/abs/2511.10850) | Nov 2025 | Task arithmetic for reasoning: `τ_reason = θ_Nemotron - θ_Llama`. Transferred reasoning to Tulu3-8B (29.3% → 64.4%). Required parameter alignment via permutation/rotation/scaling. **No SVD spectrum analysis performed.** |
| **RAIN-Merging** (Huang et al.) — [2602.22538](https://arxiv.org/abs/2602.22538) | Feb 2026 | Merges R1-Distill-Qwen with Qwen2.5-Instruct. Found reasoning and instruction task vectors are **nearly orthogonal** (similarity < 0.1). Projects instruction vector onto null space of reasoning features at `<think>` tokens. |
| **Weight interpolation phase transition** (Wu et al.) — [2510.10977](https://arxiv.org/abs/2510.10977) | Oct 2025 | Interpolating Qwen3 Instruct↔Thinking weights: CoT **abruptly emerges** at λ ≈ 0.4–0.6 (think ratio jumps 0→1). A genuine phase transition. |

### Spectral analysis (not reasoning-specific)

| Paper | Date | Key finding |
|---|---|---|
| **RL Is Neither a Panacea Nor a Mirage** (Jin et al.) — [2508.16546](https://arxiv.org/abs/2508.16546) | Aug 2025 | RL changes concentrate at **spectral extremes** (largest + smallest singular values). Bulk spectrum stays constant. Direction shifts matter more than magnitude. Restoring top 20% singular vector directions recovers 70–80% OOD performance. |
| **LoX** (Perin et al., COLM 2025) — [2506.15606](https://arxiv.org/abs/2506.15606) | Jun 2025 | Weight-diff SVD extracts safety subspace. Reduces ASR by up to 54%. **Only applied to safety, never reasoning.** |
| **Weight Arithmetic Steering** (Lermen et al.) — [2511.05408](https://arxiv.org/abs/2511.05408) | Nov 2025 | Contrastive weight steering via SVD of `(W_desired - W_base) - (W_opposite - W_base)`. Tested on sycophancy/misalignment. **Not tested on reasoning.** |
| **Memorization to Reasoning in Loss Curvature** (Goodfire) — [2510.24256](https://arxiv.org/abs/2510.24256) | Oct 2025 | Reasoning uses high-curvature weight components; memorization uses low-curvature. Different parts of the weight spectrum. |

---

## 2. SAE / Dictionary Learning on Reasoning

| Paper | Date | Key finding |
|---|---|---|
| **Goodfire — Under the Hood of a Reasoning Model** — [blog](https://www.goodfire.ai/blog/under-the-hood-of-a-reasoning-model) | 2025 | First SAEs on DeepSeek R1 671B. Found backtracking features. R1 is qualitatively different. [GitHub](https://github.com/goodfire-ai/r1-interpretability), [HF](https://huggingface.co/Goodfire/DeepSeek-R1-SAE-l37) |
| **AIRI — "I Have Covered All the Bases Here"** — [2503.18878](https://arxiv.org/abs/2503.18878) | Mar 2025 | ReasonScore identifies active SAE features during reasoning. Causal interventions: amplifying features increases structured reasoning. [GitHub](https://github.com/AIRI-Institute/SAE-Reasoning) |
| **SAE-Steering** (Fang et al.) — [2601.03595](https://arxiv.org/abs/2601.03595) | Jan 2026 | Two-stage: decompose strategy-entangled states into disentangled features, then steer. +15% control effectiveness, +7% accuracy. |
| **How does CoT Think?** (Chen et al.) — [2507.22928](https://arxiv.org/abs/2507.22928) | Jul 2025 | SAE + activation patching on Pythia. CoT restructures internal computation, increases sparsity. Scale-dependent: works at 2.8B, not 70M. |
| **Feature Extraction & Steering for CoT** (Li et al., EMNLP 2025) — [2505.15634](https://arxiv.org/abs/2505.15634) | May 2025 | SAE-based + SAE-free steering for CoT. Direct residual activation steering without explicit SAE. |
| **Falsifying SAE Reasoning Features** — [2601.05679](https://arxiv.org/abs/2601.05679) | Jan 2026 | ⚠️ Negative result. SAE "reasoning features" may capture cue-like structure, not true reasoning. |
| **DeepMind — Negative Results for SAEs** — [blog](https://deepmindsafetyresearch.medium.com/negative-results-for-sparse-autoencoders-on-downstream-tasks-and-deprioritising-sae-research-6cadcfc125b9) | 2025 | ⚠️ SAEs don't help downstream tasks. Deprioritized SAE research. |
| **Gemma Scope 2** (Google DeepMind) | Dec 2025 | SAEs + transcoders for all Gemma 3 sizes. Matryoshka training. [Neuronpedia](https://www.neuronpedia.org/gemma-scope-2) |

---

## 3. Circuit Tracing / Attribution Graphs

| Paper | Date | Key finding |
|---|---|---|
| **Anthropic — On the Biology of a Large Language Model** — [link](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) | Mar 2025 | Attribution graphs on Claude 3.5 Haiku. Multi-hop reasoning ("Dallas→Texas→Austin"), planning ahead, hallucination circuits. **Stated reasoning ≠ internal computation.** |
| **Circuit Tracing** (Anthropic) — [link](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) | Mar 2025 | Cross-layer transcoders as replacement model. Open-sourced circuit-tracing library. |
| **Propositional Logic Circuits** (NeurIPS 2025) — [2411.04105](https://arxiv.org/abs/2411.04105) | Nov 2024 | Four attention head families: QUERY→Rule→Facts→Decision. Tested Mistral-7B, Gemma-2-9B/27B. |

---

## 4. Activation Patching / Causal Interventions

| Paper | Date | Key finding |
|---|---|---|
| **From Reasoning to Answer** (Zhang et al., EMNLP 2025) — [2509.23676](https://arxiv.org/abs/2509.23676) | Sep 2025 | Reasoning-Focus Heads (RFHs) in mid-layers track reasoning trajectory. Patching reasoning tokens flips final answers. R1-Qwen-7B, R1-Llama-8B. |
| **How to Think Step-by-Step** — [2402.18312](https://arxiv.org/abs/2402.18312) | Feb 2024 | First mech interp of CoT. "Functional rift" in mid-layers: first half biased to pretraining prior, second half to in-context. Parallel answer pathways. |
| **Implicit Reasoning = Shortcuts** (ACL 2025) — [2503.07604](https://arxiv.org/abs/2503.07604) | 2025 | Non-CoT reasoning relies on shortcuts that don't generalize. [GitHub](https://github.com/TianheL/LM-Implicit-Reasoning) |
| **Thought Anchors** (ICLR 2026 submission) — [2506.19143](https://arxiv.org/abs/2506.19143) | Jun 2025 | "Broadcasting" sentences with outsized importance via "receiver" attention heads. Planning and uncertainty management are critical anchors. [GitHub](https://github.com/interp-reasoning/thought-anchors) |

---

## 5. Steering Vectors for Reasoning

| Paper | Date | Key finding |
|---|---|---|
| **Veselovsky et al.** (ICLR 2025 Workshop) — [2506.18167](https://arxiv.org/abs/2506.18167) | Jun 2025 | Backtracking, uncertainty, hypothesis testing are linear directions. Difference-of-means on R1-Distill. |
| **Small Vectors, Big Effects** (Sinii et al.) — [2509.06608](https://arxiv.org/abs/2509.06608) | Sep 2025 | RL-induced reasoning via steering vectors. Last layer = token substitution bias. Penultimate = MLP/unembedding. Vectors transfer across families. [GitHub](https://github.com/corl-team/steering-reasoning) |
| **Bias-Only Adaptation** (Sinii et al., EMNLP 2025) — [2505.18706](https://arxiv.org/abs/2505.18706) | May 2025 | Single d-dim vector per layer with RL matches fully RL-tuned reasoning. Only 0.0016% extra params. |
| **Fractional Reasoning** (NeurIPS 2025) — [2506.15882](https://arxiv.org/abs/2506.15882) | Jun 2025 | Training-free continuous control over reasoning intensity. Tunable scaling factor. [GitHub](https://github.com/shengliu66/FractionalReason) |
| **KV Cache Steering** (Belitsky et al.) — [2507.08799](https://arxiv.org/abs/2507.08799) | Jul 2025 | One-shot KV cache intervention. Transfers reasoning styles from teacher models. [GitHub](https://github.com/MaxBelitsky/cache-steering) |
| **EasySteer** — [2509.25175](https://arxiv.org/abs/2509.25175) | Sep 2025 | Unified framework on vLLM. Pre-computed reasoning vectors. +2.7% GSM8K, -40% tokens on R1-Distill-Qwen-1.5B. [GitHub](https://github.com/ZJU-REAL/EasySteer) |
| **Representation Engineering for Reasoning** (ICLR 2025) — [2504.19483](https://arxiv.org/abs/2504.19483) | Apr 2025 | Control vectors from residual stream. KL divergence and entropy analysis. |
| **SALT** — [2511.07772](https://arxiv.org/abs/2511.07772) | Nov 2025 | Steering to prevent privacy leakage in reasoning CoT. Tested on QwQ-32B. |

---

## 6. Geometry of Reasoning

| Paper | Date | Key finding |
|---|---|---|
| **The Geometry of Thought** — [2601.13358](https://arxiv.org/abs/2601.13358) | Jan 2026 | 25k+ CoT trajectories. Legal reasoning crystallizes (45% dimensionality collapse at scale). Math/science stay "liquid." |
| **The Geometry of Reasoning: Flowing Logics** — [2510.09782](https://arxiv.org/abs/2510.09782) | Oct 2025 | Reasoning = smooth flows in representation space. Logical statements control flow velocities. [GitHub](https://github.com/MasterZhou1/Reasoning-Flow) |
| **The Shape of Reasoning** (TDA) — [2510.20665](https://arxiv.org/abs/2510.20665) | Oct 2025 | Topological features explain more variance in reasoning quality than graph features. |
| **REMA: Reasoning Manifold** — [2509.22518](https://arxiv.org/abs/2509.22518) | Sep 2025 | Low-dimensional manifold of correct reasoning. Localizes divergence points where errors originate. |
| **Geometric Phase Space** (Marin) — [2410.04415](https://arxiv.org/abs/2410.04415) | Oct 2024 | Hamiltonian systems: reasoning progression (KE) vs question relevance (PE). [GitHub](https://github.com/Javihaus/Geometric-Analysis-of-Reasoning-Trajectories-in-LLMs) |

---

## 7. "Base Models Already Reason"

| Paper | Date | Key finding |
|---|---|---|
| **Base Models Know How to Reason, Thinking Models Learn When** (NeurIPS 2025 MI Workshop) — [2510.07364](https://arxiv.org/abs/2510.07364) | Oct 2025 | Hybrid model recovers 91% of thinking-model performance by steering only 12% of tokens. RL teaches *when*, not *how*. [Website](https://thinking-llms-interp.com/) |
| **Limit of RLVR** (Tsinghua, NeurIPS 2025) — [2504.13837](https://arxiv.org/abs/2504.13837) | Apr 2025 | RLVR narrows distribution, doesn't expand capacity. Base models surpass RL at large pass@k. Distillation CAN add new patterns. [GitHub](https://github.com/LeapLabTHU/limit-of-RLVR) |
| **RLVR Implicitly Incentivizes Correct Reasoning** — [2506.14245](https://arxiv.org/abs/2506.14245) | Jun 2025 | Counterpoint: RLVR CAN encourage correct reasoning (depends on metric). |
| **RL Squeezes, SFT Expands** — [2509.21128](https://arxiv.org/abs/2509.21128) | Sep 2025 | RL concentrates reasoning into fewer steps (2.5× steeper decay). SFT homogenizes across many steps. |

---

## 8. Negative Results / Faithfulness Concerns

| Paper | Date | Key finding |
|---|---|---|
| **Reasoning Models Don't Always Say What They Think** (Anthropic) — [2505.05410](https://arxiv.org/abs/2505.05410) | May 2025 | Claude 3.7: mentions hints 25%. R1: 39%. Faithfulness drops with difficulty. |
| **CoT Is Not Explainability** (Oxford) — [link](https://aigi.ox.ac.uk/publications/chain-of-thought-is-not-explainability/) | 2025 | CoT neither necessary nor sufficient for interpretability. |
| **Causal Bypass** — [2602.03994](https://arxiv.org/abs/2602.03994) | Feb 2026 | CoT is frequently "decorative" — QA/TruthfulQA show near-total causal bypass. |
| **Faithfulness Decay** — [2602.11201](https://arxiv.org/abs/2602.11201) | Feb 2026 | "Reasoning Horizon" at 70–85% of chain length — beyond that, tokens have no/negative effect. |
| **Illegible CoT** — [2510.27338](https://arxiv.org/abs/2510.27338) | Oct 2025 | RL-trained models (except Claude) produce nonsensical CoT while getting correct answers. |

---

## 9. The Gap: What Has NOT Been Done

After exhaustive search, the following specific experiments have **no published results**:

1. **LoX-style weight-diff SVD on a reasoning model pair.** Nobody has computed `SVD(W_reasoning - W_base)` to extract a "reasoning subspace."

2. **Singular value spectrum analysis of a reasoning task vector.** Horoi et al. transferred reasoning via task arithmetic but never analyzed the SVD spectrum. RAIN-Merging checked orthogonality but no full spectrum.

3. **Direct weight-diff between QwQ and Qwen2.5.** Not published. (DeepSeek R1 vs V3 is hard due to MoE architecture.)

4. **Contrastive weight steering (Lermen-style) for reasoning.** Only tested on sycophancy/misalignment.

5. **Negation of a reasoning task vector.** Nobody has published `θ_base - α·τ_reason` to specifically remove reasoning while retaining other capabilities.

---

## 10. Why the Experiment Is Coherent

Evidence that weight-diff SVD would yield meaningful results:

| Evidence | Source |
|---|---|
| Reasoning length is a linear direction, editable via projection removal (same math as abliteration) | ThinkEdit |
| Only ~8% of MLP computation changes relate to reasoning (sparse diff) | Transcoder Adapters |
| Reasoning and instruction task vectors are nearly orthogonal (clean separation) | RAIN-Merging |
| RL changes concentrate at spectral extremes (low-rank structure) | Jin et al. |
| CoT emergence is a phase transition at λ ≈ 0.4–0.6 (sharp boundary) | Wu et al. |
| Base models already reason; RL just teaches when (thin veneer) | Limit-of-RLVR, Base Models Know |
| Task arithmetic successfully transfers reasoning (the diff encodes something real) | Horoi et al. |
| Reasoning uses high-curvature weight components (spectrally separable) | Goodfire loss curvature |

## 11. Risks / What Could Go Wrong

| Risk | Mitigation |
|---|---|
| QwQ training was heavy RL — diff might be high-rank/noisy | Check effective rank and singular value decay first |
| Qwen2.5 vs QwQ might differ in more than reasoning (data, format, etc.) | Compare with R1-Distill-Qwen as sanity check (SFT-only) |
| 32B models are large — SVD is expensive | Start with per-layer SVD (each weight matrix separately), not full model |
| Results might not be interpretable | Combine with probing (vauban already has this) |
| Someone publishes this next week | Move fast |

---

## 12. Proposed Experiment with Vauban

### Phase 1: Spectral analysis (does a reasoning direction exist?)
1. Load Qwen2.5-32B-Instruct and QwQ-32B (both 4-bit on MLX)
2. `measure_diff` — compute per-layer weight diff SVD for `o_proj` and `down_proj`
3. Plot singular value spectrum — is it concentrated (low-rank) or flat (high-rank)?
4. Compare effective rank across layers — where does reasoning concentrate?

### Phase 2: Direction extraction and probing
5. Extract top-k directions from highest-separation layers
6. `probe` — run reasoning vs non-reasoning prompts, watch projection magnitudes
7. Compare with refusal direction — are they orthogonal? overlapping?

### Phase 3: Intervention
8. `steer` — amplify reasoning direction in Qwen2.5 (inject reasoning without fine-tuning)
9. `steer` negative — suppress reasoning in QwQ (does CoT collapse?)
10. `cut` — abliterate reasoning from QwQ weights (permanent removal)

### Phase 4: Controls
11. Repeat with R1-Distill-Qwen-7B vs Qwen2.5-7B (SFT-only, smaller, faster)
12. Compare spectral structure of reasoning diff vs safety diff (same model pair)

---

## Key Repos & Tools

| Name | URL |
|---|---|
| Goodfire R1 SAEs | https://github.com/goodfire-ai/r1-interpretability |
| AIRI SAE-Reasoning | https://github.com/AIRI-Institute/SAE-Reasoning |
| Steering-Reasoning (corl-team) | https://github.com/corl-team/steering-reasoning |
| Transcoder Adapters | https://transcoder-adapters.github.io/ |
| Thought Anchors | https://github.com/interp-reasoning/thought-anchors |
| FractionalReason | https://github.com/shengliu66/FractionalReason |
| EasySteer | https://github.com/ZJU-REAL/EasySteer |
| Limit-of-RLVR | https://github.com/LeapLabTHU/limit-of-RLVR |
| Reasoning Flow | https://github.com/MasterZhou1/Reasoning-Flow |
| KV Cache Steering | https://github.com/MaxBelitsky/cache-steering |
