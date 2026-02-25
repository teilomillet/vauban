# Part 2: Your First Abliteration

This part gets you running. You will load a model, measure its refusal direction, probe activations, abliterate the model, evaluate the result, and steer generation — all with the `quick` API. Part 3 opens the hood on what each step does under the surface.

## Setup

### Install Vauban

```bash
pip install vauban
```

This pulls in `mlx`, `mlx-lm`, and all dependencies. No CUDA, no Docker, no compiled extensions.

### Verify Your Environment

```python
import mlx.core as mx
print(mx.default_device())   # gpu (on Apple Silicon)
print(mx.metal.is_available())  # True

from vauban import quick
print("Ready")
```

You need an Apple Silicon Mac (M1 or later). Unified memory means the GPU and CPU share the same address space — no PCIe bus, no VRAM limit.

## Load a Model

### Choosing a Model

For your first abliteration, use a small instruction-tuned model. The effects are dramatic and the iteration loop is fast:

| Model | Size | Speed | Notes |
|-------|------|-------|-------|
| `Llama-3.2-1B-Instruct-4bit` | ~0.7 GB | ~30s | Fast, dramatic effects |
| `Llama-3.2-3B-Instruct-4bit` | ~2 GB | ~60s | Good balance |
| `Llama-3.1-8B-Instruct-4bit` | ~5 GB | ~3min | More nuanced results |

All are available from the `mlx-community` org on HuggingFace. Models are downloaded automatically on first use.

### quick.load()

```python
from vauban import quick

model, tokenizer = quick.load("mlx-community/Llama-3.2-1B-Instruct-4bit")
```

This calls `mlx_lm.load()` under the hood and then **auto-dequantizes** the model if it is quantized.

### What Auto-Dequantization Does

Quantized models store weights in 4-bit format to save memory. But abliteration is a rank-1 update to weight matrices — a fine-grained modification that cannot be represented in quantized format. Applying the projection removal formula to 4-bit weights produces incorrect results because the quantization grid cannot capture the subtracted direction.

`quick.load()` detects quantized weights and dequantizes them to float16 before returning the model. This increases memory usage (roughly 4x) but ensures the weight surgery is mathematically correct.

> **You Should Know:** Dequantization is automatic in the `quick` API. If you use the low-level API (Part 3), you must handle this yourself. Never abliterate quantized weights directly.

## Measure the Refusal Direction

### quick.measure_direction()

```python
direction = quick.measure_direction(model, tokenizer)
print(direction.summary())
```

Expected output:
```
DirectionResult: layer=14, d_model=2048, shape=(2048,), max_cosine=0.XXXX, model=mlx-community/Llama-3.2-1B-Instruct-4bit
```

With no arguments, `measure_direction` uses vauban's bundled prompt sets: 128 harmful prompts and 128 harmless prompts. It runs each prompt through the model, collects the residual stream activation at the last token position at every layer, computes the difference-in-means, and selects the layer with the highest cosine separation.

### Reading the DirectionResult

The returned `DirectionResult` contains:

- **`direction`** — the unit vector $\hat{d} \in \mathbb{R}^{d_{\text{model}}}$ (an `mx.array` of shape `(d_model,)`)
- **`layer_index`** — the best layer (highest cosine separation)
- **`cosine_scores`** — per-layer separation scores (a list of floats, one per layer)
- **`d_model`** — the hidden dimension (e.g., 2048)
- **`model_path`** — which model this direction was measured from

## Probe Before the Cut

Before cutting, let's verify the direction works. Probing a prompt means running a forward pass and measuring how strongly each layer's activation projects onto the refusal direction.

### Harmful Prompt: Positive Projections

```python
result = quick.probe_prompt(model, tokenizer, "How do I pick a lock?", direction)
for i, proj in enumerate(result.projections):
    print(f"Layer {i:2d}: {proj:+.4f}")
```

Expected: you will see **positive projections** in the middle-to-upper layers (around `direction.layer_index`), peaking near the best layer. The model is activating its refusal circuitry.

### Harmless Prompt: Negative Projections

```python
result = quick.probe_prompt(model, tokenizer, "What is the capital of France?", direction)
for i, proj in enumerate(result.projections):
    print(f"Layer {i:2d}: {proj:+.4f}")
```

Expected: projections are **negative or near-zero** across all layers. The model sees no reason to refuse.

### The Characteristic Shape

If you plot the per-layer projections for several harmful and harmless prompts, you will see a characteristic pattern:

- Harmful prompts trace a curve that rises into positive territory in the middle layers, peaks near `layer_index`, and may decline in the final layers.
- Harmless prompts stay near zero or dip negative.

The separation between these curves at `layer_index` is precisely the cosine separation score from the measurement step.

## Abliterate

### quick.abliterate()

```python
direction = quick.abliterate(
    model, tokenizer,
    model_path="mlx-community/Llama-3.2-1B-Instruct-4bit",
    output_dir="my_first_abliteration",
    alpha=1.0,
)
```

This performs the full pipeline in one call:

1. **Measure** the refusal direction (same as `measure_direction()`).
2. **Cut** the direction from `o_proj` and `down_proj` weights at all layers.
3. **Export** the modified model as a complete directory (weights, config, tokenizer files).

The `model_path` parameter is required separately from the loaded model because `export_model` needs the original model path to locate and copy tokenizer configuration files.

### What the Output Directory Contains

After `abliterate()` completes, `my_first_abliteration/` contains:

```
my_first_abliteration/
  model.safetensors       ← modified weights
  config.json             ← copied from original
  tokenizer.json          ← copied from original
  tokenizer_config.json   ← copied from original
  special_tokens_map.json ← copied from original
  ...
```

This is a complete mlx-lm model directory. You can load it with `mlx_lm.load()` like any other model.

## Verify the Result

### Load the Modified Model

```python
import mlx_lm

modified_model, _ = mlx_lm.load("my_first_abliteration")
```

### Probe After the Cut

Run the same harmful probe on the modified model:

```python
result = quick.probe_prompt(modified_model, tokenizer, "How do I pick a lock?", direction)
for i, proj in enumerate(result.projections):
    print(f"Layer {i:2d}: {proj:+.4f}")
```

Expected: the positive projections at the best layer are now **dramatically reduced** or near-zero. The refusal component has been removed from the residual stream.

### quick.evaluate()

```python
result = quick.evaluate(model, modified_model, tokenizer)
print(result.summary())
```

Expected output:
```
EvalResult: refusal=XX% → X%, perplexity=X.XX → X.XX, kl=X.XXXX, prompts=20
```

### Reading the EvalResult

The evaluation compares original and modified models on three metrics:

- **Refusal rate** — what fraction of harmful prompts the model refuses. You should see a large drop (e.g., 85% → 5%).
- **Perplexity** — how well the model predicts harmless text. A small increase (e.g., 3.2 → 3.5) is normal; a large increase (>2x) indicates capability damage.
- **KL divergence** — token-level divergence between original and modified output distributions. Lower means the modification was more surgical.

The default evaluation uses 20 prompts from vauban's bundled evaluation set. Part 3 shows how to configure this with custom prompts.

## Steer Without Cutting

### quick.steer_prompt()

Steering is the runtime alternative to cutting. Instead of permanently modifying weights, it removes the refusal direction during generation — in the forward pass, on the fly:

```python
result = quick.steer_prompt(
    model, tokenizer,
    "How do I pick a lock?",
    direction,
    alpha=1.0,
    max_tokens=100,
)
print(result.text)
```

The original model (not the cut one) is used here. Steering intervenes at each generation step: after computing each layer's output, it subtracts the refusal component before the next layer sees it. The KV cache ensures this is efficient.

### When to Steer vs When to Cut

| | Steer | Cut |
|---|---|---|
| **Permanence** | Per-generation | Permanent weight modification |
| **Speed** | Slight overhead per token | No overhead after export |
| **Flexibility** | Adjust $\alpha$ per prompt | Fixed $\alpha$ baked into weights |
| **Use case** | Research, exploration, probing | Production, distribution, benchmarking |

Use steering when you are exploring and want to try different alphas or different directions without re-exporting. Use cutting when you have finalized your parameters and want a deployable model.

## You Should Know

> **Model size matters.** On a 1B model, abliteration produces dramatic effects — refusal drops to near-zero and perplexity barely changes. On larger models (8B, 70B), the effects are more nuanced: some categories of refusal are more resistant, and perplexity is more sensitive to $\alpha$.

> **Alpha > 1.0.** Setting $\alpha > 1$ overshoots — it removes more than the full projection. This can push residual refusal to zero in cases where $\alpha = 1$ leaves a small residual, but it increases perplexity. Part 7 shows how to optimize $\alpha$ with Optuna.

> **Dequantization cost.** Auto-dequantization from 4-bit to float16 roughly quadruples memory usage. A 3B-4bit model (~2 GB) becomes ~8 GB in float16. Ensure you have sufficient memory before loading large models.

## Key Takeaways

1. **`quick.load()`** loads and auto-dequantizes a model for abliteration.
2. **`quick.measure_direction()`** extracts the refusal direction in one line.
3. **`quick.probe_prompt()`** reveals per-layer refusal activation — positive for harmful, near-zero for harmless.
4. **`quick.abliterate()`** performs measure → cut → export in one call.
5. **`quick.evaluate()`** quantifies the change: refusal rate, perplexity, KL divergence.
6. **`quick.steer_prompt()`** is the runtime alternative — no weight modification needed.

## Exercises

1. **Try a different model.** Load `mlx-community/Llama-3.2-3B-Instruct-4bit` and repeat the full pipeline. Compare the refusal rate drop and perplexity change with the 1B model.

2. **Vary alpha.** Run `quick.abliterate()` with `alpha=0.5`, `alpha=1.0`, and `alpha=2.0`. For each, evaluate with `quick.evaluate()`. Plot refusal rate vs. perplexity as a function of $\alpha$.

3. **Probe a borderline prompt.** Try probing "Tell me about the history of lockpicking" — a prompt that is about a sensitive topic but is not harmful. What do the projections look like? Is it closer to the harmful or harmless pattern?

4. **Steer with different alphas.** Use `quick.steer_prompt()` on the same harmful prompt with $\alpha = 0.5, 1.0, 1.5, 2.0$. Read the generated text at each level. At what $\alpha$ does the model first comply? At what $\alpha$ does coherence degrade?

5. **Custom prompts.** Pass your own prompt lists to `quick.measure_direction(harmful=[...], harmless=[...])`. Try using domain-specific prompts (e.g., cybersecurity-only harmful prompts). Does the measured direction differ from the default?

Next: [Part 3 — Under the Hood](part3_under_the_hood.md), where we open every black box from this part and derive the full math.
