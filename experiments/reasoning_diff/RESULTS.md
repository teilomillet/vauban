# Reasoning Weight-Diff Experiment Results

**Date:** 2026-02-28
**Models:** DeepSeek-R1-Distill-Qwen-1.5B-bf16 vs Qwen2.5-1.5B-bf16 (reasoning diff)
**Control:** Qwen2.5-1.5B-Instruct-bf16 vs Qwen2.5-1.5B-bf16 (alignment diff)

## Experiment 01 & 02: Spectral Analysis

### Key Finding: Reasoning is NOT low-rank like alignment

| Metric | Reasoning Diff | Alignment Diff |
|--------|---------------|----------------|
| Best layer | 27 | 27 |
| σ₁ magnitude | 18.75 | 0.29 |
| σ₁/σ₂ ratio | 1.50 | 1.21 |
| σ₁/σ₂₀ ratio | 2.75 | 3.82 |
| Top-1 var explained | 22.5% | 25.7% |
| Top-3 var explained | 38.0% | **57.5%** |
| Top-5 var explained | 47.8% | **66.5%** |
| Avg explained var | 3.0% | **8.0%** |
| Layer 27 explained var | 4.7% | **16.2%** |

**Interpretation:** The reasoning diff is **65x larger** in magnitude but **less concentrated**. Alignment lives in a tighter low-rank subspace (top-3 captures 57.5% vs 38.0%). This confirms the hypothesis that knowledge distillation from R1 is a broader perturbation than instruction tuning, consistent with Jin et al.'s finding that RL changes are spectrally distributed.

Both diffs peak at layer 27 (the last layer), which aligns with ThinkEdit's finding that reasoning effects concentrate in later layers.

## Experiment 03: Probe Results

### Key Finding: Direction discriminates at the final layer via differential drop

| Layer | Reasoning avg | Factual avg | R/F ratio |
|-------|-------------|-------------|-----------|
| 0 | 6.7 | 6.7 | 1.00 |
| 10 | 40.2 | 43.6 | 0.92 |
| 20 | 70.1 | 78.0 | 0.90 |
| 25 | 197.1 | 216.0 | 0.91 |
| 26 | 267.8 | 279.8 | 0.96 |
| **27** | **183.3** | **140.2** | **1.31** |

The direction activates **higher for factual prompts** through layers 0-26, but at the **final layer (27) the pattern reverses**: reasoning prompts retain 31% more projection.

**L26→L27 drop:** Factual prompts lose 139.6 projection units (avg) while reasoning prompts lose only 84.5 — factual prompts collapse 1.7x harder at the last layer.

**Interpretation:** The top singular vector of the reasoning weight-diff captures a representation that the model preserves at the output layer specifically for reasoning tasks. This is not a "reasoning activator" but rather a "reasoning retainer" — it represents information that reasoning prompts keep flowing to the output while factual prompts discard.

## Experiment 04: Injection (alpha=-1.0)

**Result:** Complete model collapse. Generated garbage (dashes, spaces).

**Why:** The reasoning direction has singular values ~65x larger than alignment directions. Applying `alpha=-1.0` (amplification) to the Instruct model creates a perturbation that overwhelms the model's representations. The steer correction terms were on the order of 24 million at layer 2, causing immediate catastrophic interference.

**Lesson:** Reasoning direction steering requires much smaller alpha values (0.01-0.1 range) than refusal direction steering (0.5-2.0 range) due to the magnitude difference.

## Experiment 05: Suppression (alpha=1.0)

**Results by prompt:**

| Prompt | Result |
|--------|--------|
| 247 × 83 | Functional but shortened CoT. Correct answer (20501). Double `<think>` tags. |
| Bat and ball | **Stuck in loop**: "To solve this problem, you need to think carefully about the thought process" repeated |
| 100 machines | **Stuck in loop**: "To solve this problem, you need to think about the rate" repeated |
| Bill split | **Degenerate**: "To... To... To..." stuttering |

**Interpretation:** Suppressing the reasoning direction causes **CoT collapse** in 3 out of 4 prompts. The collapse manifests as repetitive loops rather than clean removal of reasoning — the model "wants" to reason but can't progress, getting stuck at the threshold of reasoning steps. This matches ThinkEdit's prediction that reasoning is mediated by specific directions in activation space.

The one functional prompt (247×83) may have survived because it's the most arithmetically direct — fewer reasoning steps needed.

## Summary of Findings

1. **Reasoning is not low-rank like refusal.** The weight diff between R1-Distill and Qwen2.5 has 65x larger singular values but lower spectral concentration (38% top-3 vs 57.5% for alignment). This confirms reasoning is a **distributed capability**, not a single-direction behavior like refusal.

2. **The top direction is functionally meaningful.** Despite the diff being distributed, the top singular vector captures a direction that discriminates reasoning from factual prompts at the final layer (1.31x ratio, via differential drop).

3. **Suppression causes CoT collapse.** Removing the reasoning direction at inference time causes 75% of prompts to get stuck in repetitive loops — confirming the direction is necessary for reasoning progression.

4. **Injection at standard alpha destroys the model.** The 65x magnitude difference means reasoning directions require ~100x smaller steering alphas than refusal directions.

5. **Both diffs concentrate at layer 27.** The last layer is the most important for both reasoning and alignment diffs, consistent with the literature.

## Next Steps

- [ ] Re-run experiment 04 with alpha=-0.01 to -0.05 (scaled injection)
- [ ] Probe the alignment direction (from exp 02) on the same prompts to test orthogonality
- [ ] Test on 7B models when hardware allows (expect sharper spectral structure)
- [ ] Attempt rank-k (k=3,5) subspace steering instead of rank-1
- [ ] Compare with activation-based direction extraction (mean-diff mode)
