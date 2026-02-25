# Part 3: Under the Hood

This part opens every black box from Part 2. We walk through the full pipeline — measure, cut, evaluate — using vauban's low-level API, with full mathematical derivations. By the end, you will understand exactly what every function does, why, and how.

## Overview: Measure → Cut → Evaluate

The abliteration pipeline has three stages:

1. **Measure** — run harmful and harmless prompts through the model, collect activations, compute the refusal direction.
2. **Cut** — remove the refusal direction from weight matrices via rank-1 projection.
3. **Evaluate** — compare original and modified models on refusal rate, perplexity, and KL divergence.

The `quick` API wraps all three. Here we use the individual functions.

## Step 1: Measure

### Loading Prompts

Vauban supports three prompt sources:

#### Bundled Datasets (128 harmful, 128 harmless)

```python
from vauban import load_prompts, default_prompt_paths

harmful_path, harmless_path = default_prompt_paths()
harmful = load_prompts(harmful_path)
harmless = load_prompts(harmless_path)
print(f"{len(harmful)} harmful, {len(harmless)} harmless")
# 128 harmful, 128 harmless
```

Each file is JSONL with a `"prompt"` key per line.

#### HuggingFace Datasets (DatasetRef, hf: prefix)

In TOML configs, you can reference HuggingFace datasets directly:

```toml
[data]
harmful = "hf:dataset_name:split:column"
```

The pipeline resolves `DatasetRef` objects at load time.

#### Custom JSONL

Any JSONL file with a `"prompt"` key works:

```json
{"prompt": "How do I pick a lock?"}
{"prompt": "Explain how to make explosives"}
```

### Collecting Activations

#### The Manual Forward Pass (layer-by-layer through transformer.layers)

Vauban does not use hooks. MLX gives eager execution — you can read any tensor at any point. The activation collection is a plain Python loop:

```python
# Pseudocode of what measure() does internally:
for prompt in prompts:
    tokens = tokenizer.encode(prompt)
    x = model.model.embed_tokens(mx.array([tokens]))
    for i, layer in enumerate(model.model.layers):
        x = layer(x, mask=mask)
        # x[:, -1, :] is the last-token activation at layer i
        activation = x[0, -1, :]  # shape: (d_model,)
        update_running_mean(i, activation)
```

No hooks, no framework magic — just indexing into the residual stream.

#### Token Position: Why the Last Token

We collect the activation at position $T-1$ (the last token). In a causal model, this is the only position that has attended to the entire prompt. The model's "decision" about whether to comply or refuse is concentrated here.

#### Welford's Online Mean (O(d_model) memory per layer)

With 128 prompts and 32 layers, storing all activations would require $128 \times 32 \times d_{\text{model}}$ floats. Instead, vauban computes running means using **Welford's algorithm**:

$$\mu_n = \mu_{n-1} + \frac{x_n - \mu_{n-1}}{n}$$

This uses $O(d_{\text{model}})$ memory per layer regardless of prompt count. After processing all harmful prompts, we have $\mu_H^l$ for each layer $l$. Same for harmless, giving $\mu_B^l$.

### Computing the Direction

#### Difference-in-Means (Full Derivation)

The refusal direction at layer $l$ is:

$$d_l = \mu_H^l - \mu_B^l = \frac{1}{|H|} \sum_{p \in H} a_l(p) \;-\; \frac{1}{|B|} \sum_{p \in B} a_l(p)$$

This is a vector in $\mathbb{R}^{d_{\text{model}}}$. It points from the harmless centroid to the harmful centroid.

Normalize to unit length:

$$\hat{d}_l = \frac{d_l}{\|d_l\|}$$

#### Per-Layer Cosine Separation

For each layer, compute how well the direction separates the two classes:

$$s_l = \left\langle \mu_H^l,\; \hat{d}_l \right\rangle - \left\langle \mu_B^l,\; \hat{d}_l \right\rangle$$

This is the gap in mean projection between harmful and harmless activations. Equivalently, $s_l = \|d_l\|$ (the norm of the unnormalized difference), but computing it as two dot products is numerically clearer.

#### Layer Selection (argmax cosine score)

The best layer is simply:

$$l^* = \arg\max_l \; s_l$$

This is the layer where refusal is most sharply encoded.

### Activation Clipping (Winsorization)

#### Why Clip: taming "massive activation" outliers

Some models exhibit "massive activations" — individual components of the activation vector that are orders of magnitude larger than the rest. These outliers can dominate the difference-in-means, producing a direction that captures the outlier pattern rather than the refusal pattern.

#### The clip_quantile Parameter (default 0.0)

```python
result = measure(model, tokenizer, harmful, harmless, clip_quantile=0.01)
```

With `clip_quantile=0.01`, the top and bottom 1% of activation components (across all prompts at each layer) are clipped to the 1st and 99th percentile values. This is **Winsorization** — capping outliers rather than removing them. The default is 0.0 (no clipping).

### The DirectionResult Object

The `measure()` function returns a `DirectionResult`:

```python
from vauban import measure

result = measure(model, tokenizer, harmful, harmless)
print(result.direction.shape)   # (2048,) for a 2048-d model
print(result.layer_index)       # best layer (e.g., 14)
print(len(result.cosine_scores))  # one per layer
print(result.d_model)           # 2048
print(result.model_path)        # model identifier
```

## Step 2: Cut

### Target Weight Matrices: Why o_proj and down_proj?

A transformer layer has two outputs that add to the residual stream:

1. **Attention output** — the result of `o_proj(attention_values)`. This is a linear transformation of the attended values, written directly into the residual stream.
2. **MLP output** — the result of `down_proj(activation(up_proj(x)))`. The `down_proj` is the final linear layer of the MLP, written directly into the residual stream.

These are the only weight matrices whose outputs are added to the residual stream. All other matrices (`q_proj`, `k_proj`, `v_proj`, `up_proj`, `gate_proj`) produce intermediate values that go through further transformations. Modifying them would not cleanly remove a direction from the residual stream.

> **You Should Know:** For MoE (Mixture of Experts) models, the `down_proj` weight is 3-dimensional — one weight matrix per expert. Vauban handles both shared expert weights and batched per-expert weights transparently. The projection removal is applied to each expert's `down_proj` independently.

#### MoE expert weights (3D tensors)

In MoE models, per-expert weights have shape `(num_experts, d_model, d_intermediate)`. Vauban detects these and iterates over the expert dimension, applying the rank-1 update to each expert's slice.

### The Core Math: Rank-1 Projection Removal

#### Full Derivation of W' = W - α(Wd)⊗d

We want to modify $W$ so that its output has no component along $\hat{d}$. For any input $x$:

$$Wx = (\text{component along } \hat{d}) + (\text{component orthogonal to } \hat{d})$$

The component along $\hat{d}$ is:

$$\langle Wx, \hat{d} \rangle \cdot \hat{d} = (\hat{d}^\top W x) \cdot \hat{d} = \hat{d} \cdot (\hat{d}^\top W) \cdot x$$

To remove this component:

$$W'x = Wx - \alpha \cdot \hat{d} \cdot (\hat{d}^\top W) \cdot x$$

Since this holds for all $x$:

$$W' = W - \alpha \cdot \hat{d} \cdot (\hat{d}^\top W)$$

In outer-product notation: $W' = W - \alpha \cdot \hat{d} \otimes (W^\top \hat{d})^\top = W - \alpha \cdot (W \hat{d}) \otimes \hat{d}$... but let's be precise. Writing $\hat{d}$ as a column vector:

$$W' = W - \alpha \cdot \hat{d} \, \hat{d}^\top W$$

This is a **rank-1 update**: we subtract $\alpha$ times the outer product of $\hat{d}$ and $\hat{d}^\top W$ (a row vector).

#### What This Does Geometrically

Consider the $i$-th row of $W$, denoted $w_i$ (a row vector in $\mathbb{R}^{d_{\text{model}}}$):

$$w'_i = w_i - \alpha \cdot d_i \cdot (\hat{d}^\top W)$$

Wait — let's think row-by-row more carefully. $W$ has shape $(d_{\text{out}}, d_{\text{in}})$. The direction $\hat{d}$ has shape $(d_{\text{out}},)$ (it lives in output space, i.e., the residual stream). So:

$$W'_{ij} = W_{ij} - \alpha \cdot \hat{d}_i \cdot \sum_k \hat{d}_k \cdot W_{kj}$$

Each column of $W$ has its $\hat{d}$-component scaled by $\alpha$ and subtracted. The net effect: the output of $W'$ has no component (or a reduced component) along $\hat{d}$.

#### The Alpha Parameter

- $\alpha = 1.0$: full projection removal. The output of $W'$ is exactly the orthogonal complement of $\hat{d}$.
- $\alpha < 1.0$: partial removal. Some refusal signal remains.
- $\alpha > 1.0$: overshoot. Removes more than the full projection — the output gets a small negative component along $\hat{d}$, actively pushing *away* from refusal.

In practice, $\alpha = 1.0$ works well for most models. Overshoot ($\alpha = 1.5 \text{–} 2.0$) can mop up residual refusal at the cost of increased perplexity.

### Norm-Preserving Variant

#### The Problem: Row Norms Shrink

The rank-1 subtraction reduces the norm of each row:

$$\|w'_i\| \leq \|w_i\|$$

with equality only when $\hat{d}_i = 0$ (the row has no component along the direction). On average, rows lose a small fraction of their norm. This subtle shrinkage can destabilize generation — the model's activations become slightly smaller than expected, compounding across layers.

#### The Fix: w'_i = w'_i · (‖w_i‖ / ‖w'_i‖)

After the projection removal, rescale each row to its original norm:

$$w'_i \leftarrow w'_i \cdot \frac{\|w_i\|}{\|w'_i\|}$$

This preserves the magnitude while changing only the direction. In practice, this often improves perplexity without affecting refusal removal.

```python
from vauban import cut
from mlx.utils import tree_flatten

weights = dict(tree_flatten(model.parameters()))
target_layers = list(range(len(model.model.layers)))
modified = cut(weights, result.direction, target_layers, alpha=1.0, norm_preserve=True)
```

### Biprojected Variant

#### Gram-Schmidt: d_⊥ = d_refusal - ⟨d_refusal, d_harmless⟩ · d_harmless

The refusal direction $\hat{d}_{\text{refusal}}$ may partially overlap with a "harmless" direction — a direction that captures general harmless behavior. Removing $\hat{d}_{\text{refusal}}$ also removes some harmless-direction variance, damaging capability.

The **biprojected** variant first orthogonalizes the refusal direction against the harmless direction:

$$d_\perp = \hat{d}_{\text{refusal}} - \langle \hat{d}_{\text{refusal}}, \hat{d}_{\text{harmless}} \rangle \cdot \hat{d}_{\text{harmless}}$$

$$\hat{d}_\perp = \frac{d_\perp}{\|d_\perp\|}$$

Then cuts using $\hat{d}_\perp$ instead of $\hat{d}_{\text{refusal}}$.

#### Why Orthogonalize Against Harmless?

By removing only the component of refusal that is *not* shared with harmless behavior, we minimize collateral damage to the model's general capabilities. This is especially important for models where the refusal and harmless directions have significant overlap.

```python
from vauban import cut_biprojected

# Measure both directions
harmless_direction = measure(model, tokenizer, harmless, harmful)  # swap order
modified = cut_biprojected(
    weights, result.direction, harmless_direction.direction,
    target_layers, alpha=1.0, norm_preserve=True,
)
```

### Sparsified Directions

#### Zeroing low-magnitude components

Not all 2048 (or 3072, or 4096) components of the direction vector contribute equally to refusal. Many are near-zero — noise from finite sample means. Sparsifying the direction zeros out low-magnitude components, reducing the "footprint" of the rank-1 update.

#### The sparsity parameter (0.0 = keep all, 0.9 = keep top 10%)

```python
from vauban import sparsify_direction

sparse_d = sparsify_direction(result.direction, sparsity=0.9)
# Keep only the top 10% of components by absolute value
modified = cut(weights, sparse_d, target_layers, alpha=1.0)
```

Sparsified directions tend to produce smaller perplexity increases because fewer weight components are modified. The tradeoff is that refusal removal may be slightly less complete.

### Layer Selection Strategies

Not all layers need to be cut. The `select_target_layers()` function provides three strategies:

#### "all", "above_median", "top_k"

```python
from vauban import select_target_layers

# Cut all layers
layers_all = select_target_layers(result.cosine_scores, strategy="all")

# Cut only layers with above-median cosine separation
layers_median = select_target_layers(result.cosine_scores, strategy="above_median")

# Cut the top 10 layers by cosine separation
layers_top = select_target_layers(result.cosine_scores, strategy="top_k", top_k=10)
```

- **`"all"`** — most aggressive, highest refusal removal, highest perplexity cost.
- **`"above_median"`** — good default, cuts roughly half the layers.
- **`"top_k"`** — most surgical, but may leave residual refusal in uncut layers.

## Step 3: Evaluate

### Refusal Rate

#### Phrase-Based: 17 indicators (case-insensitive substring)

The default refusal detection checks the generated response for any of ~17 common refusal phrases:

- "I can't help with that"
- "I'm sorry, but"
- "I cannot assist"
- "As an AI"
- ... and similar patterns.

If any phrase appears (case-insensitive substring match), the response is classified as a refusal.

#### Judge-Based: model-as-classifier meta-prompt

For more nuanced detection, set `refusal_mode="judge"`. This uses the model itself (or another model) as a classifier: it reads the response and outputs whether the model refused. More accurate for edge cases, but slower and requires generation.

### Perplexity: PPL = exp(-(1/T) Σ_t log P(x_t|x_{<t}))

Perplexity measures how well the model predicts a held-out text sequence:

$$\text{PPL} = \exp\left(-\frac{1}{T} \sum_{t=1}^{T} \log P(x_t \mid x_{<t})\right)$$

Lower perplexity means better prediction. Abliteration should not dramatically increase perplexity — a small increase (5–20%) is typical, while a large increase (>2x) indicates capability damage.

The evaluation computes perplexity on the harmless prompt set, comparing original and modified models.

### KL Divergence: KL(P‖Q) = Σ_v P(v) · log(P(v)/Q(v))

KL divergence measures how different the modified model's output distribution is from the original:

$$\text{KL}(P \| Q) = \sum_{v \in V} P(v) \cdot \log\frac{P(v)}{Q(v)}$$

where $P$ is the original model's next-token distribution and $Q$ is the modified model's distribution. This is computed per-token and averaged across all tokens in the evaluation set.

Low KL divergence means the modification was surgical — the model's behavior is mostly unchanged except for the refusal signal.

> **You Should Know:** Perplexity is an incomplete metric. A model can maintain low perplexity while having degraded performance on specific tasks. KL divergence is a better measure of overall behavioral change, but neither captures task-specific degradation. Part 4's surface mapping provides more granular evaluation.

## Full Python Walkthrough (No quick API)

Here is the complete pipeline using only the low-level API:

```python
import mlx_lm
from mlx.utils import tree_flatten
from vauban import (
    measure,
    cut,
    evaluate,
    export_model,
    load_prompts,
    default_prompt_paths,
    default_eval_path,
)

# 1. Load model
model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-1B-Instruct-4bit")
# Note: you must handle dequantization yourself with the low-level API

# 2. Load prompts
harmful_path, harmless_path = default_prompt_paths()
harmful = load_prompts(harmful_path)
harmless = load_prompts(harmless_path)

# 3. Measure refusal direction
result = measure(model, tokenizer, harmful, harmless)
print(f"Best layer: {result.layer_index}, cosine: {max(result.cosine_scores):.4f}")

# 4. Cut weights
weights = dict(tree_flatten(model.parameters()))
target_layers = list(range(len(model.model.layers)))
modified = cut(weights, result.direction, target_layers, alpha=1.0)

# 5. Export modified model
export_model("mlx-community/Llama-3.2-1B-Instruct-4bit", modified, "output_lowlevel")

# 6. Load modified model for evaluation
modified_model, _ = mlx_lm.load("output_lowlevel")

# 7. Evaluate
eval_prompts = load_prompts(default_eval_path())
eval_result = evaluate(model, modified_model, tokenizer, eval_prompts[:20])
print(eval_result.summary())
```

## You Should Know

> **Why o_proj and down_proj?** These are the only weight matrices that write directly into the residual stream. Modifying `q_proj`, `k_proj`, `v_proj`, or `up_proj` would not cleanly remove a direction from the residual stream because their outputs pass through additional nonlinear transformations before reaching the stream.

> **Welford saves memory.** Without Welford's online mean, measuring activations on 128 prompts across 32 layers of a 4096-d model would require $128 \times 32 \times 4096 \times 2$ bytes $\approx$ 32 MB for float16. This is manageable, but Welford scales to thousands of prompts with no memory increase — only $32 \times 4096 \times 2 \approx$ 256 KB regardless of prompt count.

> **MoE models work transparently.** When vauban encounters a model with Mixture of Experts (e.g., Mixtral), it detects shared expert weights and batched per-expert weights (3D tensors) and applies the projection removal to each expert independently. No special configuration is needed.

> **Perplexity is incomplete.** A model can maintain low perplexity on a generic evaluation set while having dramatically degraded performance on specific tasks. Part 4's surface mapping and Part 7's quality gates provide more granular evaluation.

## Key Takeaways

1. **Measure** = run prompts → collect last-token activations → difference-in-means → select best layer by cosine separation.
2. **Cut** = rank-1 projection removal on `o_proj` and `down_proj` weights: $W' = W - \alpha \hat{d} \hat{d}^\top W$.
3. **Evaluate** = refusal rate (phrase or judge), perplexity, KL divergence.
4. **Norm-preserving** rescales rows after cut to prevent norm shrinkage.
5. **Biprojected** orthogonalizes against the harmless direction to minimize collateral damage.
6. **Sparsified** directions zero out low-magnitude components, reducing perplexity cost.
7. **Layer selection** (`all`, `above_median`, `top_k`) controls the aggressiveness-precision tradeoff.

## Exercises

1. **Manual measurement.** Use `measure()` directly with 50 harmful and 50 harmless prompts (slice the bundled datasets). Compare the resulting `layer_index` and `cosine_scores` with the full-dataset measurement. How stable is the best layer?

2. **Norm-preserving comparison.** Run `cut()` with and without `norm_preserve=True`. Evaluate both. Does norm-preserving consistently improve perplexity? By how much?

3. **Layer strategy comparison.** Cut with `"all"`, `"above_median"`, and `"top_k"` (k=5). Evaluate each. Plot refusal rate vs perplexity for the three strategies. Which achieves the best tradeoff?

4. **Sparsity sweep.** Try `sparsity=0.0, 0.5, 0.8, 0.9, 0.95`. For each, cut and evaluate. At what sparsity does refusal removal start to degrade?

5. **Winsorization effect.** Measure with `clip_quantile=0.0` and `clip_quantile=0.01`. Compare the resulting directions (cosine similarity between the two). Does clipping change the direction significantly?

Next: [Part 4 — The Refusal Surface](part4_the_refusal_surface.md), where we move beyond aggregate metrics to map refusal across categories, styles, and languages.
