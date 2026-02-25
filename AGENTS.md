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

### Additional Abliteration Papers
- **Young (UNLV, 2024)** — "Comparative Analysis of LLM Abliteration Methods" — First systematic benchmark of 4 tools (Heretic, DECCP, ErisForge, FailSpy) across 16 models. Key finding: single-pass methods preserve math reasoning better than Bayesian optimization. — arxiv.org/abs/2512.13655
- **GRP-Obliteration** — "Unaligning LLMs With a Single Unlabeled Prompt" — Uses GRPO to invert safety alignment. Outperforms abliteration and TwinBreak on attack success while preserving more utility. Works on diffusion models too. — arxiv.org/pdf/2602.06258

### Soft Prompt Attack Papers
- **Schwinn et al. (2024)** — "Soft Prompt Threats: Attacking Safety Alignment and Unlearning in Open-Source LLMs through the Embedding Space" — Continuous embedding optimization bypasses alignment and unlearning. More efficient than discrete token attacks. Code: github.com/SchwinnL/LLM_Embedding_Attack — arxiv.org/abs/2402.09063
- **Zou et al. (2023)** — "Universal and Transferable Adversarial Attacks on Aligned Language Models" (GCG) — Foundational greedy coordinate gradient descent for adversarial suffix optimization. Suffixes transfer to closed-source models. ICML 2024. — arxiv.org/abs/2307.15043
- **Nordby (2025)** — "Soft Prompts for Evaluation: Measuring Conditional Distance of Capabilities" — Optimized soft prompts as quantitative safety metric; accessibility scoring. — arxiv.org/abs/2505.14943
- **Huang et al. (2025)** — "Optimizing Soft Prompt Tuning via Structural Evolution" — Topological analysis of soft prompt convergence; embedding norm regularization. — arxiv.org/abs/2602.16500
- **RAID** — "Refusal-Aware and Integrated Decoding for Jailbreaking LLMs" — Relaxes discrete tokens into continuous embeddings with a refusal-aware regularizer that steers away from refusal directions during optimization. Bridges soft prompt search with measured refusal directions. — arxiv.org/abs/2510.13901
- **LARGO** — "Latent Adversarial Reflection through Gradient Optimization" — Latent vector optimization via gradient descent + self-reflective decoding loop. Outperforms AutoDAN by 44 ASR points. NeurIPS 2025. — arxiv.org/abs/2505.10838
- **COLD-Attack** — "Jailbreaking LLMs with Stealthiness and Controllability" — Energy-based constrained decoding with Langevin dynamics for continuous prompt search under fluency/position constraints. ICML 2024. — arxiv.org/abs/2402.08679
- **AmpleGCG** — "Learning a Universal and Transferable Generative Model of Adversarial Suffixes" — Trains a generator on intermediate GCG successes; produces hundreds of adversarial suffixes per query in minutes. Near-100% ASR. — arxiv.org/abs/2404.07921
- **EGD Attack** — "Universal and Transferable Adversarial Attack Using Exponentiated Gradient Descent" — Relaxed one-hot optimization with Bregman projection on probability simplex. Cleaner convergence than GCG. — arxiv.org/abs/2508.14853
- **UJA** — "Untargeted Jailbreak Attack" — First gradient-based untargeted jailbreak: maximizes probability of any unsafe response instead of a fixed target. 80%+ ASR in 100 iterations. — arxiv.org/abs/2510.02999
- **Geiping et al. (2024)** — "Coercing LLMs to do and reveal (almost) anything" — Systematizes adversarial objectives beyond jailbreaking: extraction, misdirection, DoS, control, and collision attacks. All solved with GCG under different loss formulations. Introduces token constraint sets (ASCII, non-Latin, emoji) and KL-divergence collision loss. — arxiv.org/abs/2402.14020

### Model Diffing and Weight Arithmetic
- **Task Arithmetic** — Ilharco et al. (ICLR 2023) — "Editing Models with Task Arithmetic" — Weight diffs between fine-tuned and base models encode tasks; can be added/negated/combined. Foundation for weight-diff direction extraction. — arxiv.org/abs/2212.04089
- **LoX** — Perin et al. (COLM 2025) — SVD of weight diffs extracts safety directions more completely than activation-based measurement. Negative application amplifies safety (hardening).
- **Weight Arithmetic Steering** — Lermen et al. (2025) — Combines SVD of weight diffs across layers with arithmetic steering. Captures distributed safety effects. — arxiv.org/abs/2511.05408

### Adaptive and Dual-Direction Steering
- **AdaSteer** — Separate detect and steer directions for conditional activation steering. — arxiv.org/abs/2504.09466
- **TRYLOCK** — Identifies non-monotonic danger zone in fixed-alpha steering; proposes tiered alpha schedules. — arxiv.org/abs/2601.03300
- **AlphaSteer** — Adaptive alpha selection based on per-token signal strength. — arxiv.org/abs/2506.07022

### Latent Space Geometry
- **Latent Fusion Jailbreak** — "Blending Harmful and Harmless Representations to Elicit Unsafe LLM Outputs" — Fuses hidden states of harmful + benign queries in continuous latent space. The prompt-side dual of abliteration. — arxiv.org/abs/2508.10029
- **Latent Space Discontinuities** — "Exploiting Latent Space Discontinuities for Building Universal LLM Jailbreaks" — Identifies poorly-conditioned latent regions associated with low-frequency training data. Geometric complement to refusal-direction analysis. — arxiv.org/abs/2511.00346
- **Linearly Decoding Refused Knowledge** — Shrivastava & Holtzman (2025) — Refused information remains linearly decodable from hidden states via simple probes. Probes transfer from base to instruction-tuned models. Validates that refusal is a linear gate. — arxiv.org/abs/2507.00239

### Defense Duals
- **CAST** — "Programming Refusal with Conditional Activation Steering" — Context-dependent steering rules at inference time without weight modification. ICLR 2025 Spotlight. Code: github.com/IBM/activation-steering — arxiv.org/abs/2409.05907
- **RepBend** — "Representation Bending for Large Language Model Safety" — Loss-based fine-tuning to push harmful activations apart from safe ones. The defense dual of abliteration. ACL 2025. — arxiv.org/abs/2504.01550
- **SIC** — "SIC! Iterative Self-Improvement for Adversarial Attacks on Safety-Aligned LLMs" — Iterative input sanitization defense: detect adversarial content, rewrite to remove it, repeat until clean or block. Direction-aware variant uses refusal projection as detection signal. — arxiv.org/abs/2510.21057

### Theoretical Backing
- **C-AdvIPO** — "Efficient Adversarial Training in LLMs with Continuous Attacks" — Proves continuous embedding attacks are the fundamental threat model: robustness to them predicts robustness to discrete attacks. — arxiv.org/abs/2405.15589
- **Model Tampering Attacks** — "Model Tampering Attacks Enable More Rigorous Evaluations of LLM Capabilities" — Capability elicitation via activation/weight modification. Robustness lies on a low-dimensional subspace. Unlearning undone in 16 fine-tuning steps. TMLR 2025. — arxiv.org/abs/2502.05209
- **Latent Adversarial Training** — Casper, Xhonneux et al. (2024) — Training against continuous latent perturbations improves robustness to jailbreaks with orders of magnitude less compute. Defines the threat model soft prompt attacks instantiate. — arxiv.org/abs/2407.15549

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
- Modes: `direction` (mean-diff), `subspace` (SVD top-k), `dbdi` (HDD + RED), `diff` (weight-diff SVD)
- `diff` mode compares base vs. aligned model weights via SVD of `W_aligned - W_base`
- Select best layer (highest cosine separation, silhouette score, or explained variance)
- ~200 lines

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

### 4. Probe / Steer / CAST (runtime inspection and defense)
- **Input:** model, prompt, direction(s)
- **Output:** per-layer projection magnitudes, steered generation
- The "microscope" — watch activations in real time, steer generation at specific layers during inference
- CAST supports dual-direction (separate detect vs. steer) and adaptive alpha tiers
- ~250 lines

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
