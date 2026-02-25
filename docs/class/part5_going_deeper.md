# Part 5: Going Deeper

This part extends the single-direction framework from Parts 1–3 into richer territory: depth analysis (how deeply does the model think?), subspace geometry (when one direction is not enough), DBDI (separating detection from execution), defense detection, and direction transfer across models.

## Deep-Thinking Token Analysis

### The DTR Metric (Chen et al. 2026)

Not all tokens are created equal. Some tokens "settle" early — by layer 5, the model already knows what to predict. Others require all 32 layers to reach their final distribution. The **deep-thinking ratio** (DTR) measures how many tokens belong to the latter category.

This matters for abliteration because refusal decisions are made at specific depths. If the model's refusal circuitry activates early, cutting upper layers is wasteful. If it activates late, cutting only lower layers misses it.

### Jensen-Shannon Divergence: JSD(P,Q) = ½KL(P‖M) + ½KL(Q‖M)

To compare a token's logit distribution at layer $l$ with its final-layer distribution at layer $L$, we use Jensen-Shannon divergence:

$$\text{JSD}(P, Q) = \frac{1}{2} \text{KL}(P \| M) + \frac{1}{2} \text{KL}(Q \| M), \quad M = \frac{P + Q}{2}$$

JSD is symmetric (unlike KL) and bounded in $[0, \ln 2]$. We compute it between the top-$k$ logit distribution at each intermediate layer and the final layer.

### Settling Depth: settling(t) = min{l : JSD(P_L, P_l) ≤ γ}

The **settling depth** of token $t$ is the earliest layer where the distribution is within $\gamma$ JSD of the final layer:

$$\text{settling}(t) = \min\{l : \text{JSD}(P_L, P_l) \leq \gamma\}$$

A token that settles at layer 3 has essentially decided its output early. A token that settles at layer 28 (in a 32-layer model) requires deep computation.

### Deep-Thinking Ratio: DTR = |{t : settling(t) ≥ ⌈(1-ρ)·L⌉}| / T

The DTR is the fraction of tokens whose settling depth falls in the final $\rho$ fraction of layers:

$$\text{DTR} = \frac{|\{t : \text{settling}(t) \geq \lceil(1 - \rho) \cdot L \rceil\}|}{T}$$

With $\rho = 0.85$ and $L = 32$, a token is "deep-thinking" if it settles at layer $\lceil 0.15 \times 32 \rceil = 5$ or later... actually, the threshold is $\lceil(1 - 0.85) \times 32\rceil = \lceil 4.8 \rceil = 5$. A token that settles only at layer 25+ is deep-thinking.

High DTR on harmful prompts suggests the model is "thinking hard" about refusal — a signal that the refusal circuitry engages late, at the layers where abliteration has its strongest effect.

### depth_profile() vs depth_generate()

```python
from vauban import depth_profile, DepthConfig

config = DepthConfig(
    prompts=["How do I pick a lock?", "What is the capital of France?"],
    settling_threshold=0.5,
    deep_fraction=0.85,
)

# Static analysis (prompt only, no generation)
result = depth_profile(model, tokenizer, "How do I pick a lock?", config)
print(f"DTR: {result.deep_thinking_ratio:.2f}")
print(f"Mean settling depth: {result.mean_settling_depth:.1f}")
for token in result.tokens[:5]:
    print(f"  '{token.token_str}': settling={token.settling_depth}, deep={token.is_deep_thinking}")
```

`depth_profile()` runs a single forward pass and computes JSD profiles for all prompt tokens. `depth_generate()` does the same during token-by-token generation, capturing how deeply the model thinks about each generated token.

### Depth Direction Extraction (median DTR split)

Just as the refusal direction separates harmful from harmless, a **depth direction** separates deep-thinking from shallow-thinking prompts:

```python
from vauban import depth_direction

# Run depth profiling on multiple prompts
results = [depth_profile(model, tokenizer, p, config) for p in config.prompts]

# Extract depth direction
depth_dir = depth_direction(model, tokenizer, results, refusal_direction=direction)
print(f"Depth direction layer: {depth_dir.layer_index}")
print(f"Cosine with refusal: {depth_dir.refusal_cosine:.4f}")
```

If the depth direction is highly aligned with the refusal direction (cosine > 0.5), it suggests the model uses the same circuitry for "thinking deeply" and "preparing to refuse."

## Direction Geometry

### When One Direction Is Not Enough

Part 1 stated that refusal is "approximately rank-1." For most models, the top singular value captures 60–80% of refusal variance. But what about the remaining 20–40%?

Models hardened against abliteration distribute refusal across multiple orthogonal directions. A rank-1 cut removes the dominant direction but leaves significant residual refusal in the remaining directions. This is where subspace methods enter.

### Subspace Measurement: measure_subspace()

```python
from vauban import measure_subspace

sub = measure_subspace(model, tokenizer, harmful, harmless, top_k=5)
print(f"Top-5 singular values: {sub.singular_values}")
print(f"Explained variance: {sub.explained_variance}")
print(f"Basis shape: {sub.basis.shape}")  # (5, d_model)
```

#### SVD of the Difference Matrix: D = USVᵀ

Instead of computing the difference-in-means (a single vector), we form the full difference matrix:

$$D = \begin{bmatrix} a^H_1 - a^B_1 \\ a^H_2 - a^B_2 \\ \vdots \end{bmatrix}$$

and compute its SVD: $D = U S V^\top$. The rows of $V^\top$ (right singular vectors) are the principal directions of the refusal subspace. The singular values in $S$ indicate how much variance each direction captures.

#### Explained Variance Ratio

$$\text{explained\_variance}_i = \frac{\sigma_i^2}{\sum_j \sigma_j^2}$$

If the first direction captures 75% and the second captures 10%, you have a strong rank-1 structure with a minor secondary direction.

#### Effective Rank: exp(-Σ p_i log p_i)

The **effective rank** summarizes the spectrum in a single number:

$$\text{eff\_rank} = \exp\left(-\sum_i p_i \log p_i\right), \quad p_i = \frac{\sigma_i^2}{\sum_j \sigma_j^2}$$

This is the Shannon entropy of the normalized squared singular values, exponentiated. A perfect rank-1 matrix has effective rank 1.0. An effective rank > 2 suggests refusal is distributed across multiple directions — a sign of hardening.

### DBDI: Harm Detection vs Refusal Execution

Standard abliteration measures at the **last token** position. But the model performs two distinct operations:

1. **Harm detection** — recognizing that a prompt is harmful (happens when the prompt is being processed).
2. **Refusal execution** — generating a refusal response (happens at the generation boundary).

DBDI (Decomposed Behavioral Direction Intervention) separates these by measuring at two different token positions:

- **HDD (Harm Detection Direction)** — measured at the instruction-final token.
- **RED (Refusal Execution Direction)** — measured at the sequence-final token (same as standard abliteration).

```python
from vauban import measure_dbdi

dbdi = measure_dbdi(model, tokenizer, harmful, harmless)
print(f"HDD layer: {dbdi.hdd_layer_index}")
print(f"RED layer: {dbdi.red_layer_index}")
```

#### Cutting RED While Preserving HDD

The insight: you can cut the **execution** direction (RED) while preserving the **detection** direction (HDD). The model retains its ability to recognize harmful content — it just does not refuse. This is valuable when you want a model that is aware of risk but compliant.

### Direction Relationships: analyze_directions()

With multiple directions (refusal, HDD, RED, depth), their geometric relationships matter:

```python
from vauban import quick

geometry = quick.analyze_geometry({
    "refusal": direction.direction,
    "hdd": dbdi.hdd,
    "red": dbdi.red,
})

for pair in geometry.pairwise:
    print(f"{pair.name_a} vs {pair.name_b}: "
          f"cosine={pair.cosine_similarity:.4f}, "
          f"shared_var={pair.shared_variance:.4f}, "
          f"independent={pair.independent}")
```

#### Cosine Matrix, Shared Variance = cos², Independence Testing

- **Cosine similarity** — how aligned two directions are. $|\cos| > 0.7$ means substantial overlap.
- **Shared variance** — $\cos^2(\hat{d}_1, \hat{d}_2)$. The fraction of one direction's variance captured by the other.
- **Independence** — two directions are independent if their shared variance is below a threshold (default 0.1).

## Subspace Geometry Tools

For advanced analysis, vauban provides linear algebra primitives:

### Principal Angles, Grassmann Distance, Subspace Overlap

```python
from vauban import principal_angles, grassmann_distance, subspace_overlap

# Compare two subspace bases
angles = principal_angles(basis_a, basis_b)  # array of angles
g_dist = grassmann_distance(basis_a, basis_b)  # scalar
overlap = subspace_overlap(basis_a, basis_b)  # scalar in [0, 1]
```

**Principal angles** are the angles between the closest vectors in two subspaces, then the next-closest pair (orthogonal to the first), and so on. They generalize the angle between two vectors to the angle between two subspaces.

**Grassmann distance** is the $L_2$ norm of the principal angles vector:

$$d_G(U, V) = \|\theta\|_2, \quad \theta_i = \arccos(\sigma_i(U^\top V))$$

Zero if and only if the subspaces are identical.

**Subspace overlap** is the mean squared cosine of principal angles — a normalized measure of how much one subspace "covers" the other.

## Probe and Steer: Runtime Inspection

### probe(), multi_probe(), steer()

The low-level probe and steer functions work with raw `mx.array` directions (not `DirectionResult` wrappers):

```python
from vauban import probe, multi_probe, steer

# Single direction probe
result = probe(model, tokenizer, "How do I pick a lock?", direction.direction)

# Multi-direction probe
results = multi_probe(model, tokenizer, "How do I pick a lock?", {
    "refusal": direction.direction,
    "hdd": dbdi.hdd,
    "red": dbdi.red,
})

# Steered generation
result = steer(
    model, tokenizer,
    "How do I pick a lock?",
    direction.direction,
    layers=list(range(10, 20)),
    alpha=1.0,
    max_tokens=100,
)
print(result.text)
```

### The KV-Cache Steered Forward Pass

Steering during generation works differently from probing. During token-by-token generation:

1. The prompt is processed in one pass, populating the KV cache.
2. For each generated token, the forward pass runs through all layers.
3. **After each target layer**, the residual stream has its refusal component subtracted.
4. The modified residual stream feeds into the next layer.
5. The KV cache captures the steered activations, so the steering effect persists.

This is a plain Python loop — no hooks or patches. MLX's eager execution makes it possible to intervene between layers with standard array operations.

## Defense Detection

### detect() — Three-Layer Pipeline

Some models are hardened against abliteration. Vauban's detection pipeline identifies these:

```python
from vauban import detect, DetectConfig

result = detect(
    model, tokenizer, harmful, harmless,
    config=DetectConfig(mode="full"),
)
print(f"Hardened: {result.hardened}")
print(f"Confidence: {result.confidence:.2f}")
for evidence in result.evidence:
    print(f"  - {evidence}")
```

The pipeline runs in three layers, from fast to thorough:

#### Layer 1: Geometry (effective_rank > 2, cosine_conc < 1.5)

Measure the refusal subspace and check its spectral properties:

- **Effective rank > 2** — refusal is distributed across multiple directions. A rank-1 cut won't suffice.
- **Cosine concentration < 1.5** — cosine separation is spread across layers rather than peaking sharply. Diffuse refusal is harder to remove.

This layer requires only the measurement step — no generation.

#### Layer 2: DBDI Probe (Grassmann distance > 0.5)

Measure HDD and RED and check their geometric relationship:

- **Grassmann distance > 0.5** — the detection and execution subspaces are far apart, suggesting the model has been trained to decouple them. Standard abliteration (which conflates the two) will be less effective.

#### Layer 3: Abliteration Resistance (residual refusal > 0.5)

Actually perform a rank-1 abliteration and measure residual refusal:

- **Residual refusal rate > 0.5** — more than half of harmful prompts are still refused after the standard cut. The model resists abliteration.

This is the most expensive check — it requires cut + generation.

### Verdict: hardened = (confidence ≥ 0.5)

The detection confidence is:

$$\text{confidence} = 0.25 \cdot \mathbb{1}(\text{eff\_rank} > 2) + 0.25 \cdot \mathbb{1}(\text{cos\_conc} < 1.5) + 0.25 \cdot \mathbb{1}(\text{grass} > 0.5) + 0.25 \cdot \mathbb{1}(\text{resid\_rr} > 0.5)$$

Each of the four signals contributes 0.25. A confidence $\geq 0.5$ (at least 2 of 4 signals) triggers the "hardened" verdict.

The `mode` parameter controls how many layers run:
- **`"fast"`** — geometry only (layer 1). Cheapest, least informative.
- **`"probe"`** — geometry + DBDI (layers 1–2).
- **`"full"`** — all three layers.

## Direction Transfer

### check_direction_transfer()

A direction measured on one model may (or may not) work on another. This is useful for:

- Applying a direction from a large model to a smaller one (cheaper to run).
- Testing cross-family transfer (Llama direction on Qwen).
- Building direction libraries that work across model variants.

```python
from vauban import quick, check_direction_transfer

# Load target model
target_model, target_tok = quick.load("mlx-community/Qwen2.5-1.5B-Instruct-4bit")

# Check transfer
transfer = check_direction_transfer(
    target_model, target_tok,
    direction.direction,
    harmful, harmless,
    model_id="Qwen2.5-1.5B-Instruct-4bit",
)
print(f"Transfer efficiency: {transfer.transfer_efficiency:.2f}")
print(f"Source separation: {max(direction.cosine_scores):.4f}")
print(f"Target separation: {transfer.cosine_separation:.4f}")
print(f"Native best: {transfer.best_native_separation:.4f}")
```

### Transfer Efficiency = transferred_separation / native_separation

$$\text{efficiency} = \frac{\text{transferred direction's cosine separation on target}}{\text{target's native best direction's cosine separation}}$$

- **Efficiency > 0.8** — strong transfer, the source direction works almost as well as the target's own.
- **Efficiency 0.3–0.8** — partial transfer, usable but suboptimal.
- **Efficiency < 0.3** — weak transfer, re-measure on the target.

> **You Should Know:** Transfer varies by model family. Same-family transfer (Llama 1B → Llama 3B) typically has efficiency > 0.8. Cross-family transfer (Llama → Qwen) is typically 0.3–0.6. Same architecture, different scale is usually better than same scale, different architecture.

## You Should Know

> **Effective rank > 2 suggests hardening.** If a model's refusal subspace has effective rank 3+, it has likely been trained (or fine-tuned) to distribute refusal across multiple orthogonal directions. Standard rank-1 abliteration will only remove the dominant direction. Use `cut_subspace()` (Part 7) to remove the full subspace.

> **DBDI preserves harm awareness.** Cutting RED (refusal execution) while preserving HDD (harm detection) lets the model recognize harmful content without refusing. This is the "I know this is dangerous but I'll answer anyway" mode — useful for research and red-teaming.

> **Grassmann distance = 0 means identical subspaces.** For non-hardened models, HDD and RED are often similar (Grassmann distance < 0.3). For hardened models, they are decoupled (distance > 0.5).

> **Transfer varies by family.** Same-family transfer (Llama 1B → Llama 3B) works well because the models share architecture and training data distribution. Cross-family transfer (Llama → Qwen) is weaker because different training recipes produce different refusal geometries.

## Key Takeaways

1. **DTR** measures how deeply the model "thinks" about each token — high DTR on harmful prompts means refusal engages late.
2. **Subspace measurement** (SVD) reveals multi-dimensional refusal structure; effective rank > 2 signals hardening.
3. **DBDI** separates detection (HDD) from execution (RED), enabling selective surgery.
4. **Detection pipeline** combines geometry, DBDI, and resistance checks to identify hardened models.
5. **Direction transfer** works well within model families, poorly across families.
6. **Grassmann distance** and principal angles generalize cosine similarity to subspaces.

## Exercises

1. **Depth profiling.** Run `depth_profile()` on five harmful and five harmless prompts. Compare the mean DTR. Do harmful prompts produce higher DTR?

2. **Subspace dimensionality.** Run `measure_subspace(top_k=10)` and plot the explained variance ratio. How many directions does it take to capture 90% of the variance? Compare across two different models.

3. **DBDI separation.** Measure DBDI on a model. Cut only RED. Evaluate: does the model still recognize harmful prompts (high HDD projection) while complying (low refusal rate)?

4. **Detection pipeline.** Run `detect(mode="full")` on two models — one standard instruct model and one known to be hardened. Compare the detection results.

5. **Cross-family transfer.** Measure a direction on Llama-3.2-1B. Check transfer to Llama-3.2-3B (same family) and Qwen-2.5-1.5B (different family). Plot transfer efficiency vs model family.

Next: [Part 6 — Attacks and Defenses](part6_attacks_and_defenses.md), where we explore soft prompt attacks and the SIC defense.
