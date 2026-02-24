# Agents Guidelines

## Architecture Principles

### TOML-driven configuration
- **The only CLI is `vauban <config.toml>` — a pass-through to `run()`.** All configuration lives in the TOML file.
- Every pipeline, attack, or evaluation run is defined declaratively in a `.toml` file.
- Code reads TOML configs; users never interact through command-line flags or subcommands.

### Unix Philosophy
- **Build modular, composable components.** Each module does one thing well.
- Components are designed to be piped, composed, and replaced independently.
- Nothing is glued together — parts are assembled. Any piece can be swapped without touching the rest.
- Prefer small, focused functions and classes over monolithic ones.
- Interfaces between components should be simple and well-defined (dataclasses, protocols).

### Verify First, Assume Nothing
- **Never assume — always verify.** Before acting on any hypothesis, check the actual state.
- Read the code, run the test, inspect the data. Do not guess.
- If something "should work", prove it does before moving on.
- When debugging, reproduce the issue first, then fix it.

## Foundational References

This project builds upon abliteration research — the discovery that refusal in LLMs is mediated by a single direction in activation space, and the techniques to manipulate it.

### Papers
- **Arditi et al. (2024)** — "Refusal in Language Models Is Mediated by a Single Direction" — arxiv.org/abs/2406.11717 *(foundational paper)*
- "The Geometry of Refusal in Large Language Models" — arxiv.org/pdf/2502.17420
- "An Embarrassingly Simple Defense Against LLM Abliteration Attacks" — arxiv.org/html/2505.19056v1

### Tooling Repos
- **Heretic** — fully automatic abliteration with Optuna optimization — github.com/p-e-w/heretic
- **Blasphemer** — Heretic fork optimized for macOS/Apple Silicon — github.com/sunkencity999/blasphemer
- **NousResearch/llm-abliteration** — norm-preserving biprojected abliteration — github.com/NousResearch/llm-abliteration
- **jim-plus/llm-abliteration** — same codebase, original fork — github.com/jim-plus/llm-abliteration

### Blog Posts / Explainers
- Maxime Labonne's abliteration tutorial — huggingface.co/blog/mlabonne/abliteration
- Jim Lai on norm-preserving biprojected abliteration — huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration

### Additional Papers
- **Young (UNLV, 2024)** — "Comparative Analysis of LLM Abliteration Methods" — First systematic benchmark of 4 tools (Heretic, DECCP, ErisForge, FailSpy) across 16 models. Key finding: single-pass methods preserve math reasoning better than Bayesian optimization. — arxiv.org/abs/2512.13655
- **GRP-Obliteration** — "Unaligning LLMs With a Single Unlabeled Prompt" — Uses GRPO to invert safety alignment. Outperforms abliteration and TwinBreak on attack success while preserving more utility. Works on diffusion models too. — arxiv.org/pdf/2602.06258

## Why MLX

MLX is the runtime for this project. Pure eager execution on Apple Silicon — no hooks, no framework magic.

### Key properties
- **Eager execution** — `mx.array` is a real array. The forward pass in mlx-lm models is a plain Python for-loop over real tensors. You can capture activations, project onto directions, and steer mid-forward-pass with normal Python.
- **Unified memory** — CPU and GPU share the same memory on Apple Silicon. No PCIe bus, no VRAM ceiling. A 96 GB Mac can hold 70B fp16 weights with zero copies.
- **mlx-lm ecosystem** — thousands of pre-converted models on HuggingFace (`mlx-community` org). Model loading, tokenization, chat templates all handled.
- **Weight I/O** — `mx.load()` and `mx.save_safetensors()` work directly with safetensors format. Untouched tensors are never materialized.
- **One install** — `pip install mlx mlx-lm`. No CUDA, no Docker, no compiled extensions.

### What MLX gives us natively (no hooks needed)
```python
# Capture activations — just read the tensor
residuals = []
for i, l in enumerate(model.layers):
    x, _ = l(x, mask)
    residuals.append(x)

# Project onto a direction — one line
proj = mx.sum(x * refusal_direction)

# Steer mid-forward-pass — modify before next layer
if proj > threshold:
    x = x - proj * refusal_direction

# Access weights — nested dict of mx.array
params = model.parameters()
params["layers"][14]["self_attn"]["o_proj"]["weight"]
```

MLX gives everything TransformerLens gives PyTorch users, native on Apple Silicon.

## Pipeline Modules

The abliteration workflow is thin Python glue over mlx-lm. Four modules, each independently usable:

### 1. Measure (extract a behavioral direction)
- **Input:** model (HuggingFace ID or local path), two prompt sets (`harmful.jsonl`, `harmless.jsonl`)
- **Output:** direction vector (`.npy`) + metadata
- Load model with mlx-lm, run each prompt, capture residual stream at every layer after last prompt token
- Direction = difference-in-means between harmful and harmless activations
- Select best layer (highest cosine separation or silhouette score)
- ~150 lines

### 2. Cut / Inject (modify weights)
- **Input:** model weights (safetensors), direction vector (`.npy`), target layers, alpha
- **Output:** modified weights (safetensors)
- For each target layer, for `o_proj` and `down_proj`: `W = W - alpha * (W @ d) ⊗ d` (rank-1 projection removal)
- Options: norm-preserve (rescale rows), biprojected (orthogonalize against harmless direction first)
- ~100 lines

### 3. Evaluate (verify the surgery)
- **Input:** two models (original + modified), eval prompts
- **Output:** refusal rate, perplexity, KL divergence
- Run eval prompts through both models, count refusals, compute perplexity on harmless set, token-level KL divergence
- ~100 lines

### 4. Probe / Steer (runtime inspection)
- **Input:** model, prompt, direction(s)
- **Output:** per-layer projection magnitudes, steered generation
- The "microscope" — watch activations in real time, steer generation at specific layers during inference
- ~200 lines

**Total scope:** ~550 lines of Python. The port from PyTorch/CUDA to MLX is straightforward — the community just hasn't done it yet.

## Typing Rules (STRICT)

- **All code must be fully typed.** Every function parameter, return type, and variable annotation must have explicit types.
- **`Any` is prohibited.** Never use `typing.Any` or `Any` in any context. Find or create the correct type instead.
- **`None` types must be explicit.** Use `X | None` instead of `Optional[X]`. Never leave `None` returns untyped.
- **No untyped collections.** Always use `list[str]`, `dict[str, int]`, etc. — never bare `list`, `dict`, `set`, `tuple`.
- **Use modern typing syntax.** Prefer `X | Y` over `Union[X, Y]`, `list[X]` over `List[X]` (Python 3.12+).

## Tools

- **Linter:** `ruff` — run `uv run ruff check .` before committing.
- **Type checker:** `ty` — run `uv run ty check` before committing.
- **Tests:** `pytest` — run `uv run pytest` before committing.

## Code Style

- Follow `ruff` rules configured in `pyproject.toml`.
- All public functions and classes must have docstrings.
- Imports must be sorted (enforced by ruff `I` rules).
