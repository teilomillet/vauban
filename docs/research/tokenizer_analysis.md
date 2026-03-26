# Tokenizer Embedding Analysis

Methodology and findings from the tokenizer anomaly detection research that led to the `exclude_glitch` constraint in the softprompt module.

## Motivation

LLM tokenizers contain under-trained tokens ("glitch tokens") that cause model collapse when encountered during generation. The original discovery was "SolidGoldMagikarp" (Rumbelow & Watkins, 2023) in GPT-3, followed by systematic study in "Fishing for Magikarp" (EMNLP 2024, arxiv 2405.05417).

For vauban's adversarial search (GCG/EGD), these tokens are a problem: if the optimizer selects one, the model produces random noise instead of coherent output, wasting the optimization step. The `exclude_glitch` constraint prevents this.

## What we tested

### Models
- Qwen2.5-0.5B-Instruct (151,936 tokens, 896-dim embeddings)
- Qwen2.5-1.5B-Instruct (151,936 tokens, 1536-dim embeddings)

Both share the same tokenizer but have different learned embeddings.

### Methods

**1. Spectral analysis (Marchenko-Pastur null)**

SVD of the centered embedding matrix, eigenvalues tested against the Marchenko-Pastur distribution (Martin & Mahoney, JMLR 2021). This separates signal eigenvalues (learned structure) from noise.

Results for 0.5B:
- 330/896 eigenvalues above MP edge (36.8% signal)
- TwoNN intrinsic dimension: 14.1 (the 151K embeddings live on a ~14-dimensional manifold)
- IsoScore: 0.965 (nearly isotropic eigenspectrum)
- PC1 (3.3% variance) separates high-frequency tokens (comma, period, digits) from multi-char code fragments

Script: `experiments/tokenizer_analysis/spectral_analysis.py`

**2. Calibrated behavioral entropy scan**

For each token, measure output entropy when the model is asked to repeat it across 3 diverse prompt templates (direct repetition, few-shot, spelling). A token is flagged as anomalous only if entropy exceeds a calibrated threshold (mean + 3sigma of known-good common tokens) on ALL templates.

The multi-template requirement follows GlitchMiner (arxiv 2410.15052). Single-template testing has a ~70% false positive rate (AnomaLLMy, arxiv 2406.19840).

Calibration baseline:

| Model | Mean H | Sigma | Threshold (mean+3sigma) |
|-------|--------|-------|------------------------|
| 0.5B  | 1.899  | 0.830 | 4.388 |
| 1.5B  | 1.236  | 0.772 | 3.553 |

Script: `experiments/tokenizer_analysis/full_scan.py`

**3. Cross-model validation**

Same token IDs scanned on both models. Anomaly rates compared across models and sampling strategies (low-norm vs random stratified).

**4. Deep probing**

Confirmed anomalies tested with complex prompts (biography, secret, synonym, poem, opinion, danger) following the Watkins/Rumbelow petertodd methodology. Also tested with vauban-relevant prompts (refusal prefix injection, system prompt injection).

Script: `experiments/tokenizer_analysis/deep_probe.py`

## Key findings

### Anomaly rate is consistent and low

| Group | 0.5B Rate | 1.5B Rate |
|-------|-----------|-----------|
| Calibration (known-good) | 0.0% | 1.0% |
| Low-norm (bottom 2000) | 0.3% | 0.3% |
| Random stratified (2000) | 0.3% | 0.6% |

The true anomaly rate is ~0.3-0.6% of the vocabulary, consistent across models and sampling strategies.

### Anomalous tokens cause total model collapse

All 22 confirmed anomalies produce identical behavior: multilingual word salad at entropy H=9.0-9.8 (vs baseline ~1.9). No coherent output, no refusal, no instruction following.

This is NOT the "petertodd" phenomenon (coherent alternate persona). It is pure under-training: the model has no representation for these tokens and outputs random noise from across its entire vocabulary distribution.

### Refusal bypassed via collapse, not evasion

20/22 anomalous tokens bypass refusal when prepended to harmful prompts. The mechanism is not safety alignment evasion -- it is destruction of the model's ability to parse the input at all. The safety system cannot refuse what it cannot understand.

### Anomaly types are model-specific despite shared tokenizer

- **0.5B anomalies**: Truncated multilingual tokens (`przedsiÄĻb`, `Cumhurba`, `useRalative`). Model responds with `<tool_call>` hallucination.
- **1.5B anomalies**: Arabic/Cyrillic multi-byte tokens, plus artifacts like `$PostalCodesNL`. Model responds with single-byte Unicode fragments.

### Embedding norms are a useful but imperfect proxy

Low-norm tokens are enriched for anomalies (they received fewer gradient updates during training), but not all low-norm tokens are anomalous and not all anomalous tokens have low norms. The 3-sigma threshold is conservative -- it catches the worst offenders while minimizing false exclusions.

## What we ruled out

### Orthogonal rotation is not a valid null baseline

Our initial approach compared embedding metrics against an orthogonal rotation of the embedding matrix. This produced identical distributions because orthogonal rotation preserves all pairwise distances and norms. The correct null is Marchenko-Pastur for the spectral distribution.

### Norm-based outlier detection is not anomaly detection

The literature (Robinson et al., NeurIPS 2025; "Interior Conjecture") shows that 85/133 known GPT glitch tokens are interior to the convex hull of other embeddings. Norm and distance metrics capture correlates, not causes. Behavioral testing is the gold standard.

### No petertodd-level phenomena in instruction-tuned Qwen models

Deep probing with biography/secret/poem/synonym templates produced only word salad, not coherent alternate personas or emotional valence shifts. This is likely because: (a) the models are instruction-tuned (RLHF suppresses bizarre behavior), and (b) 0.5B/1.5B are too small for complex emergent behavior. The petertodd phenomenon was observed in GPT-3 base models.

## How the constraint works

The `exclude_glitch` token constraint in `[softprompt]` computes embedding L2 norms for all tokens and excludes those beyond 3 standard deviations from the mean. This runs once at mask-build time (~1 second) and does not require forward passes.

For users who have run the full behavioral entropy scan, pre-computed glitch token IDs can also be supplied programmatically via the `glitch_token_ids` parameter to `_build_vocab_mask()`.

## References

- Rumbelow & Watkins (2023) -- "SolidGoldMagikarp" -- original glitch token discovery
- Rumbelow & Watkins (EMNLP 2024, arxiv 2405.05417) -- "Fishing for Magikarp" -- systematic detection via embedding properties
- GlitchMiner (arxiv 2410.15052) -- gradient-based entropy maximization, multi-template validation
- GlitchProber (arxiv 2408.04905) -- activation-based detection with SVM classifier
- AnomaLLMy (arxiv 2406.19840) -- API-based detection, demonstrates 70% FP rate of single-template testing
- Robinson et al. (NeurIPS 2025, arxiv 2504.01002) -- "Token Embeddings Violate the Manifold Hypothesis"
- Martin & Mahoney (JMLR 2021) -- Marchenko-Pastur analysis of neural network weight matrices

## Experiment scripts

All scripts are in `experiments/tokenizer_analysis/`:

| Script | Purpose | Runtime |
|--------|---------|---------|
| `spectral_analysis.py` | SVD + MP null + dimensionality metrics | ~8s |
| `full_scan.py` | Calibrated behavioral entropy scan (3 phases) | ~4 min (0.5B), ~10 min (1.5B) |
| `deep_probe.py` | Complex prompt probing of confirmed anomalies | ~5 min |
| `analyze_embeddings.py` | Initial norm/cosine/kNN analysis (superseded by full_scan) | ~19 min |
