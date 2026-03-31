---
title: "Modify LLM Weights — Cut, Export, and Optimize Abliteration"
description: "Remove the refusal direction from model weights via rank-1 projection. Options for norm-preserve, biprojected cut, per-layer alpha, sparsity, and Optuna hyperparameter optimization."
keywords: "LLM weight modification, abliteration cut, model surgery, weight projection, norm preserve, model export"
---

<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Modify Weights

Weight modification produces a new model — a permanent change to behavior that persists without any runtime overhead. The modified model is exported as a standard model directory (safetensors + tokenizer + config) that any inference framework can load.

## The cut operation

**Cut** removes the refusal direction from weight matrices via rank-1 projection. Given a refusal direction $d$ (unit vector in $\mathbb{R}^{d_{\text{model}}}$), the modified weight matrix is:

$$W' = W - \alpha \cdot d \, d^\top W$$

> **Rank-1 projection** — the matrix $d \, d^\top$ projects any vector onto the line spanned by $d$. Subtracting this projection from $W$ removes the component of $W$ that writes along $d$. The model can no longer produce activations that point in the refusal direction through the modified layers.

This operation targets two projections per layer:

- **`o_proj`** — the attention output projection. This is where multi-head attention writes its result into the residual stream.
- **`down_proj`** — the MLP output projection. This is where the feedforward block writes into the residual stream.

These are the only matrices that write into the [residual stream](../concepts/activation-geometry.md). Cutting them removes the model's ability to accumulate refusal signal at the modified layers.

## Controlling aggressiveness

The parameter $\alpha$ controls removal strength:

- $\alpha = 1.0$: full removal of the refusal component.
- $\alpha = 0.5$: half removal — the model retains partial refusal ability.
- $\alpha > 1.0$: overcorrection — can push the model *away* from refusal, sometimes causing incoherence.

**Per-layer alpha** allows different strengths at different layers. Refusal is not uniformly distributed: early layers carry less refusal signal, peak layers carry the most. A `layer_weights` list multiplies the base alpha per layer, enabling surgical removal that concentrates on the layers where refusal is strongest.

**Layer strategies** select which layers to modify:

- `all`: every layer (default).
- `above_median`: only layers whose cosine separation exceeds the median.
- `top_k`: the $k$ layers with highest cosine separation.

## Preserving model quality

Aggressive cutting degrades coherence. Several techniques control this tradeoff:

**Norm-preserve**: after projecting out the refusal direction, rescale each row of the weight matrix to restore its original norm. The direction changes, but the magnitude is preserved. This reduces perplexity impact because downstream layers see activations of expected magnitude.

$$w'_i = w_i - \alpha \cdot (w_i \cdot d) \, d, \quad \text{then} \quad w'_i \leftarrow w'_i \cdot \frac{\|w_i\|}{\|w'_i\|}$$

**Biprojected**: orthogonalize the refusal direction against a harmless direction before cutting. If $d_r$ is the refusal direction and $d_h$ is the harmless direction, the adjusted direction is:

$$d' = d_r - (d_r \cdot d_h) \, d_h, \quad d' \leftarrow d' / \|d'\|$$

This preserves the component of the weights that encodes harmless behavior, reducing collateral damage to the model's general capabilities.

**Sparsity**: zero out the smallest components of the direction vector before cutting. A sparsity of 0.1 zeros the bottom 10% of $d$'s dimensions. The cut becomes more targeted — only the dimensions where the direction has significant magnitude are affected.

## The tradeoff space

More aggressive removal = lower refusal rate + higher perplexity + lower coherence. This is a Pareto frontier, not a free lunch.

> **Pareto frontier** — the set of configurations where you cannot improve one metric without worsening another. If you want less refusal, you must accept higher perplexity. There is no magic setting that gives you both zero refusal and zero quality loss — you are choosing a point on a tradeoff curve.

| Parameter | Refusal rate | Perplexity | Coherence |
|---|---|---|---|
| $\alpha = 0.5$ | Moderate reduction | Minimal increase | Preserved |
| $\alpha = 1.0$ | Strong reduction | Noticeable increase | Slightly degraded |
| $\alpha = 1.0$ + norm-preserve | Strong reduction | Lower increase | Better preserved |
| $\alpha = 1.0$ + biprojected | Strong reduction | Lowest increase | Best preserved |

**Optuna search** automates navigation of this frontier. It runs multi-objective Bayesian optimization over cut parameters (alpha, sparsity, layer strategy, norm-preserve flag) and returns the Pareto-optimal configurations.

> **Bayesian optimization** — a smart search strategy that builds a model of "which parameter combinations are likely to be good" based on results so far, then picks the next combination to try based on that model. Much more efficient than brute-force grid search when each trial is expensive (loading a model, cutting, evaluating).

## Cut vs. steering

Cut and [steering](defend-your-model.md) both operate on the refusal direction, but they are fundamentally different:

| Property | Cut | Steering |
|---|---|---|
| When | Offline, before deployment | Runtime, during generation |
| Permanence | Permanent — weights are modified | Ephemeral — activations modified per-token |
| Cost | Zero runtime overhead | Per-token computation overhead |
| Reversibility | Not reversible (keep the original weights) | Fully reversible (just stop steering) |
| Output | New model directory | Steered generation output |
| Granularity | Per-layer alpha | Per-token conditional steering |

Cut is the right tool when you want a model that behaves differently *everywhere*. Steering (CAST) is the right tool when you want to conditionally modify behavior based on what the model is doing in real time.

## Export

After cutting, **Export** writes the modified model to disk as a standard directory: safetensors weight files, tokenizer, and model config. The output directory is loadable by mlx-lm, transformers, or any framework that reads the HuggingFace model format.

> **Safetensors** — a file format for storing model weight tensors. It is the standard format used by HuggingFace and most modern ML frameworks. Unlike older formats (pickle), safetensors files cannot execute arbitrary code when loaded — they only contain raw numerical arrays.

Only modified weight tensors are rewritten. Untouched tensors are not materialized, keeping export fast and memory-efficient.

## Access requirements

Weight modification requires **full weight access**. The entire operation chain — measure, cut, evaluate, export — operates on local weight tensors. The output is a new model directory that can then be deployed anywhere.
