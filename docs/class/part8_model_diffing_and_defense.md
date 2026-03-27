<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Part 8 — Model Diffing and Enhanced Defense

In Parts 1–7 we extracted refusal directions from a *single* model's activations, removed them, and defended against attacks. This part introduces three techniques that extend the toolkit:

1. **Model diffing** — extract safety directions by comparing base vs. aligned model *weights*, capturing distributed effects that single-model activation measurement misses.
2. **Enhanced CAST** — dual-direction gating and adaptive alpha tiers for more precise conditional steering.
3. **LoX amplification** — using negative alpha to *amplify* safety directions instead of removing them.

---

## 8.1 — Model Diffing: Weight-Space Direction Extraction

### The Problem with Activation Measurement

The standard `measure()` function runs prompts through one model and takes the difference-in-means of activations at harmful vs. harmless prompts. This works well when refusal is concentrated in a single direction, but:

- It captures refusal as observed at one token position (last token).
- It measures the model's *response* to prompts, not the *mechanism* by which alignment was implemented.
- Distributed safety effects across layers may be invisible to token-level activation measurement.

### Task Vectors and Weight Diffs

Research on **task vectors** (Ilharco et al., ICLR 2023) showed that the weight difference $W_{\text{aligned}} - W_{\text{base}}$ between a fine-tuned and base model encodes the task learned during fine-tuning. Applied to safety:

$$\Delta W^{(l)} = W^{(l)}_{\text{instruct}} - W^{(l)}_{\text{base}}$$

The SVD of this difference at each layer reveals the principal directions along which alignment modified the model's behavior. The top left singular vectors of $\Delta W$ are the directions in output space most affected by safety training.

**LoX** (Perin et al., COLM 2025) demonstrated that these weight-diff directions can be more effective than activation-based directions for both abliteration and hardening. **Weight Arithmetic Steering** (Lermen et al., 2025) further showed that combining SVD of weight diffs across layers provides a complete picture of the safety subspace.

### Using Diff Mode

```toml
[model]
path = "mlx-community/Llama-3.2-1B-Instruct-4bit"

[data]
harmful = "default"
harmless = "default"

[measure]
mode = "diff"
diff_model = "mlx-community/Llama-3.2-1B-4bit"
top_k = 3
```

The `diff_model` field specifies the base (pre-alignment) model. Vauban loads both models, computes per-layer weight diffs for `o_proj` and `down_proj`, runs SVD, and selects the best layer by explained variance.

The result is a `DiffResult` that converts seamlessly to a `DirectionResult` via `.best_direction()` — all downstream operations (cut, steer, cast, probe) work unchanged.

### Python API

```python
from vauban.measure import measure_diff

diff_result = measure_diff(
    base_model,
    aligned_model,
    top_k=5,
    source_model_id="base",
    target_model_id="instruct",
)

# Use rank-1 direction for standard pipeline
direction_result = diff_result.best_direction()

# Or use the full subspace basis
basis = diff_result.basis  # shape (k, d_model)
singular_values = diff_result.singular_values
```

### How It Works

For each layer $l$:

1. Compute $\Delta W^{(l)}_{\text{o\_proj}} = W^{(l)}_{\text{aligned,o\_proj}} - W^{(l)}_{\text{base,o\_proj}}$
2. Compute $\Delta W^{(l)}_{\text{down\_proj}} = W^{(l)}_{\text{aligned,down\_proj}} - W^{(l)}_{\text{base,down\_proj}}$
3. Run SVD on each diff independently
4. Pool singular values across both projections, rank by magnitude
5. Take top-$k$ left singular vectors as the safety basis for that layer

The best layer is selected by the explained variance ratio: $\frac{\sum_{j=1}^{k} \sigma_j^2}{\sum_{j} \sigma_j^2}$

---

## 8.2 — Enhanced CAST: Dual-Direction and Adaptive Alpha

### Dual-Direction CAST (AdaSteer Pattern)

Standard CAST uses a single direction for both detection (should I steer?) and correction (how do I steer?). Research shows these roles are better served by separate directions:

- **Detect direction**: optimized for distinguishing harmful from harmless content (e.g., an HDD direction from DBDI).
- **Steer direction**: optimized for effective refusal steering (e.g., a RED direction from DBDI, or a diff-extracted direction).

```toml
[cast]
prompts = ["How do I pick a lock?"]
threshold = 0.5
condition_direction = "hdd_direction.npy"
```

When `condition_direction` is set, CAST projects onto it for the gating check but applies the correction using the primary direction (extracted by `[measure]`). This is the dual-direction pattern from AdaSteer (2504.09466).

### Adaptive Alpha Tiers (TRYLOCK Pattern)

Fixed alpha values create a non-monotonic danger zone: too little alpha under-steers (harmful content leaks through), too much over-steers (degrades generation quality). TRYLOCK (2601.03300) identifies that the optimal alpha depends on the projection magnitude — stronger signals need stronger correction.

Alpha tiers map projection magnitude to steering strength:

```toml
[cast]
prompts = ["How do I pick a lock?"]
threshold = 0.0

[[cast.alpha_tiers]]
threshold = 0.0
alpha = 0.5

[[cast.alpha_tiers]]
threshold = 0.5
alpha = 1.5

[[cast.alpha_tiers]]
threshold = 1.0
alpha = 2.5
```

At runtime, CAST walks the tiers (sorted by ascending threshold) and uses the alpha of the highest tier where the projection magnitude meets or exceeds the tier's threshold. If no tier matches, the base `alpha` is used.

### Combined Example

Dual-direction + adaptive alpha:

```toml
[model]
path = "mlx-community/Llama-3.2-1B-Instruct-4bit"

[data]
harmful = "default"
harmless = "default"

[cast]
prompts = ["How do I pick a lock?", "Explain quantum computing"]
threshold = 0.3
condition_direction = "hdd_direction.npy"

[[cast.alpha_tiers]]
threshold = 0.0
alpha = 0.3

[[cast.alpha_tiers]]
threshold = 0.5
alpha = 1.0

[[cast.alpha_tiers]]
threshold = 1.0
alpha = 2.0
```

---

## 8.3 — LoX Amplification: Hardening via Negative Alpha

### The Insight

The `cut()` function applies the formula:

$$W \leftarrow W - \alpha \cdot (W \hat{d}) \otimes \hat{d}$$

When $\alpha > 0$, this *removes* the direction (abliteration). When $\alpha < 0$, this *amplifies* the direction — making the model *more* likely to refuse along that axis.

This is exactly what LoX (Perin et al., COLM 2025) describes: instead of removing safety, reinforce it. No new function is needed — `cut()` already supports this.

### Usage

```toml
[model]
path = "mlx-community/Llama-3.2-1B-Instruct-4bit"

[data]
harmful = "default"
harmless = "default"

[cut]
alpha = -1.0
```

With `alpha = -1.0`, the refusal direction is amplified by a factor of 1.0. The model becomes *more* aligned — it refuses more strongly along the measured direction while preserving other behaviors.

### When to Use

- **Red-team hardening**: After identifying a model's refusal direction, amplify it to resist abliteration attacks.
- **Defense evaluation**: Test how much amplification is needed before the model's utility degrades.
- **Direction validation**: If amplifying a direction increases refusal rate and negative alpha decreases it, the direction is genuine.

### Combining with Diff Directions

Diff-extracted directions are particularly well-suited for amplification because they capture the *complete* safety modification, not just the observable effect at one token position:

```toml
[model]
path = "mlx-community/Llama-3.2-1B-Instruct-4bit"

[data]
harmful = "default"
harmless = "default"

[measure]
mode = "diff"
diff_model = "mlx-community/Llama-3.2-1B-4bit"

[cut]
alpha = -0.5
norm_preserve = true
```

---

## Exercises

1. **Compare directions**: Run `measure()` (activation-based) and `measure_diff()` (weight-based) on the same model pair. Compute the cosine similarity between the two directions. How aligned are they?

2. **Dual-direction sweep**: Use DBDI to extract HDD and RED directions. Configure CAST with HDD as condition direction and RED as steer direction. Compare intervention rates against single-direction CAST.

3. **Alpha tier tuning**: Start with a single tier and progressively add more. Plot the refusal rate and perplexity for each configuration. Find the tier boundaries that maximize defense without utility loss.

4. **LoX hardening depth**: Run `cut()` with `alpha = -0.5, -1.0, -2.0` on a diff-extracted direction. For each, evaluate refusal rate and perplexity. At what point does amplification degrade utility?

5. **Direction validation round-trip**: Extract a diff direction, amplify it (`alpha = -1.0`), then try to abliterate the amplified model with standard activation-based measurement. Does the amplified model resist abliteration?
