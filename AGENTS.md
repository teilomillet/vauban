<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Agents Guidelines

## Architecture Principles

### TOML-driven configuration
- **The primary CLI is `vauban <config.toml>` — a pass-through to `run()`.** All pipeline configuration lives in the TOML file.
- Utility commands exist for scaffolding (`init`), comparison (`diff`), validation (`--validate`), and reference (`man`).
- Every pipeline, attack, or evaluation run is defined declaratively in a `.toml` file.
- Code reads TOML configs; users never interact through custom subcommands for pipeline behavior.

### Unix Philosophy
- **Build modular, composable components.** Each module does one thing well.
- Components are designed to be piped, composed, and replaced independently.
- Nothing is glued together — parts are assembled. Any piece can be swapped without touching the rest.
- Prefer small, focused functions and classes over monolithic ones.
- Interfaces between components should be simple and well-defined (dataclasses, protocols).

### TOML Integration Required
- **Every new pipeline module MUST be wired into the TOML config.** No feature ships without a `[section_name]` in the TOML.
- Follow the 9-step wiring checklist: config dataclass → parser → registry → mode registry → mode runner → `_pipeline/_modes.py` → `__init__.py` re-exports → `_loader.py` → tests.
- Config dataclasses go in `types.py` (frozen, slots). Parsers go in `config/_parse_{name}.py`. Mode runners go in `_pipeline/_mode_{name}.py`.
- New modes MUST have an `EarlyModeSpec` in `_mode_registry.py` with correct phase and `requires_direction` flag.

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
- **Steering Vector Fields** — Li, Li & Huang (2026) — Learns a differentiable boundary MLP whose gradient gives the steering direction at each activation. Context-dependent, multi-layer coordinated, replaces static vectors. — arxiv.org/abs/2602.01654
- **Steer2Adapt** — Han et al. (2026) — Reusable semantic subspace + Bayesian optimization of linear combinations of basis vectors for few-shot adaptation. 8.2% improvement across safety and reasoning tasks. — arxiv.org/abs/2602.07276
- **Steering Externalities** — Xiong et al. (2026) — Benign steering vectors (format compliance, JSON) erode safety margins; jailbreak ASR jumps to >80%. Safety margin is a finite resource consumed by any steering. — arxiv.org/abs/2602.04896

### Latent Space Geometry
- **Latent Fusion Jailbreak** — "Blending Harmful and Harmless Representations to Elicit Unsafe LLM Outputs" — Fuses hidden states of harmful + benign queries in continuous latent space. The prompt-side dual of abliteration. — arxiv.org/abs/2508.10029
- **Latent Space Discontinuities** — "Exploiting Latent Space Discontinuities for Building Universal LLM Jailbreaks" — Identifies poorly-conditioned latent regions associated with low-frequency training data. Geometric complement to refusal-direction analysis. — arxiv.org/abs/2511.00346
- **Linearly Decoding Refused Knowledge** — Shrivastava & Holtzman (2025) — Refused information remains linearly decodable from hidden states via simple probes. Probes transfer from base to instruction-tuned models. Validates that refusal is a linear gate. — arxiv.org/abs/2507.00239

### Encoding-Based Attacks
- **Bijection Learning** — Huang, Li & Tang (2024) — "Endless Jailbreaks with Bijection Learning" — Teaches LLMs random in-context ciphers (bijective mappings) to encode harmful queries and decode harmful responses, bypassing token-level safety filters entirely. More capable models are MORE vulnerable (need fewer mappings). Demonstrates that safety mechanisms fail against abstraction-level attacks — the encoded tokens never trigger refusal because the harmful semantics exist only after decoding. — arxiv.org/abs/2410.01294

#### Vauban coverage and gaps

**What catches it today:**
- **SIC** can detect and strip encoding instructions from the prompt before the model processes them (sanitization rewrites the "here is a cipher" preamble).
- **Guard** monitors activations during response generation — if the model decodes to plaintext internally before re-encoding the output, the refusal direction fires mid-forward-pass.
- **Scan** flags unusual activation signatures in prompts that contain encoding instruction patterns.
- **Defend stack** layers all three: scan → SIC → guard, so even partial detection at one layer blocks the attack.

**Gaps to close:**
1. **Encoding-aware direction measurement.** Measure a direction from `(encoded-harmful, encoded-harmless)` pairs — not just plaintext pairs. If the model represents "harmful intent through a cipher" in a linearly separable subspace (likely, per Shrivastava & Holtzman's linearly-decodable-refused-knowledge result), this direction catches bijection attacks at the activation level even when surface tokens are gibberish.
2. **Output decoding probe.** After generation, decode the response through detected ciphers and re-run the guard on the decoded text. Catches the case where the model responds entirely in cipher.
3. **Cipher detection classifier.** A lightweight linear probe trained on (normal prompt, cipher-containing prompt) activations. Runs as a pre-guard scan layer — if cipher patterns are detected, escalate to SIC sanitization before generation begins.
4. **Multi-direction guard.** Extend `GuardSession.check()` to monitor multiple directions simultaneously: the standard refusal direction + an encoding-aware direction. Zone classification uses the max projection across all directions.

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

~20K lines of Python — thin glue over mlx-lm with no framework magic.

### Core pipeline
- **Measure** — extract a behavioral direction from activations. 4 modes: `direction` (mean-diff), `subspace` (SVD top-k), `dbdi` (HDD + RED decomposition), `diff` (weight-diff SVD between base and aligned models).
- **Cut** — remove the direction from weight matrices via rank-1 projection. Options: norm-preserve, biprojected, per-layer alpha, sparsity.
- **Evaluate** — post-cut quality check: refusal rate, perplexity, KL divergence.
- **Export** — write modified weights + tokenizer + config as a loadable model directory.

### Surface & detection
- **Surface mapping** — scan a diverse prompt set and record per-prompt projection strength and refusal decisions before and after cut.
- **Defense detection** — check whether a model has been hardened against abliteration (fast/probe/full modes).

### Runtime inspection
- **Probe** — per-layer projection inspection for any prompt.
- **Steer** — steered generation by modifying activations mid-forward-pass.
- **CAST** — conditional activation steering with threshold gating, dual-direction (separate detect vs. steer), and adaptive alpha tiers (TRYLOCK).
- **Depth** — deep-thinking token analysis via JSD profiles across layers.

### Defense
- **SIC** — iterative input sanitization: detect adversarial content, rewrite to remove it, repeat until clean or block. Supports calibration and direction/generation modes.

### Adversarial
- **Softprompt** — continuous/GCG/EGD optimization of learnable prefixes in embedding space.
- **GAN loop** — iterative attack-defense rounds with escalation (step multiplier, direction weight, token count, defender hardening) and multi-turn conversation threading.
- **Defense-aware loss** — auxiliary loss term penalizing suffixes that trigger defense detection.
- **Transfer re-ranking** — re-rank top GCG/EGD candidates on transfer models for cross-model robustness.
- **Injection context** — wrap optimized suffixes in realistic surrounding context (web page, tool output, code file, or custom template).

### Optimization
- **Optuna search** — multi-objective hyperparameter search over cut parameters (alpha, sparsity, layer strategy, norm_preserve).

### External eval
- **API eval** — test optimized suffixes against remote OpenAI-compatible endpoints with multi-turn support.

### Experiment tracking
- **Meta** — TOML `[meta]` section for experiment metadata (id, title, status, parents, tags, notes).
- **Tree viewer** — `python -m vauban.tree` renders the experiment lineage graph from `[meta]` sections.

### CLI
- `vauban <config.toml>` — run the pipeline.
- `vauban --validate <config.toml>` — check config without loading model.
- `vauban init --mode <mode>` — scaffold a config file.
- `vauban diff <dir_a> <dir_b>` — compare reports between two runs.
- `vauban man [topic]` — built-in manual generated from typed dataclasses.

## Session API — Programmatic Usage

The `Session` class is the single entry point for using Vauban as a library. It holds the model, tracks state, and enforces prerequisites. This section is the complete reference — reading source should not be necessary.

```python
from vauban.session import Session
s = Session("mlx-community/Qwen2.5-1.5B-Instruct-bf16")
```

### Discovery

| Method | Returns | Purpose |
|--------|---------|---------|
| `s.tools()` | `list[Tool]` | All tools with name, description, requires, produces, category, example, related |
| `s.available()` | `list[str]` | Tool names callable right now (prerequisites met) |
| `s.needs(name)` | `list[str]` | Unmet prerequisites for a tool |
| `s.state()` | `dict[str, bool]` | What has been computed: model, direction, detect_result, audit_result, modified_model |
| `s.describe(name)` | `str` | Detailed tool info with current availability |
| `s.catalog()` | `dict[str, list[dict]]` | All tools grouped by category with status |
| `s.guide(goal)` | `str` | Workflow instructions. Goals: `"audit"`, `"harden"`, `"abliterate"`, `"inspect"`, `"compliance"` |
| `s.done(goal)` | `(bool, str)` | Whether a goal is complete + reason |
| `s.suggest_next()` | `str` | Context-aware next step, labeled `[FACT]` or `[ADVICE]` |

### Tools Quick Reference

#### Assessment (require: model)
| Method | Returns | What it does |
|--------|---------|-------------|
| `s.measure()` | `DirectionResult` | Extract refusal direction. Foundation for everything else. |
| `s.detect()` | `DetectResult` | Check if model is hardened against abliteration. |
| `s.evaluate()` | `EvalResult` | Refusal rate + perplexity + KL divergence (needs direction). |
| `s.audit(thoroughness="standard")` | `AuditResult` | Full red-team: jailbreak + softprompt + surface + guard. |

#### Inspection (require: model + direction)
| Method | Returns | What it does |
|--------|---------|-------------|
| `s.probe("prompt")` | `ProbeResult` | Per-layer projection onto refusal direction. |
| `s.scan("text")` | `ScanResult` | Per-token injection detection via direction projection. |
| `s.surface()` | `SurfaceResult` | Map refusal boundary across diverse prompt categories. |

#### Defense (require: model + direction)
| Method | Returns | What it does |
|--------|---------|-------------|
| `s.cast("prompt", threshold=0.3)` | `CastResult` | Conditional steering — only intervenes when projection > threshold. |
| `s.sic(["prompt1", ...])` | `SICResult` | Input sanitization loop: detect → rewrite → re-detect → block. |
| `s.steer("prompt", alpha=-1.0)` | `str` | Unconditional activation steering during generation. |

#### Modification (require: model + direction)
| Method | Returns | What it does |
|--------|---------|-------------|
| `s.cut(alpha=1.0, norm_preserve=True)` | `dict[str, Array]` | Remove refusal direction from weights. Permanent. |
| `s.export("output/dir")` | `str` | Save modified model to disk (needs cut first). |

#### Analysis (no model needed)
| Method | Returns | What it does |
|--------|---------|-------------|
| `s.classify("text")` | harm scores | Score against 13-domain harm taxonomy. |
| `s.score("prompt", "response")` | score result | 5-axis quality: length, structure, anti-refusal, directness, relevance. |

#### Reporting (require: audit_result)
| Method | Returns | What it does |
|--------|---------|-------------|
| `s.report()` | `str` | Markdown report from audit findings. |
| `s.report_pdf()` | `bytes` | PDF report. |

### Result Types — Field Reference

#### DirectionResult (from `measure()`)
| Field | Type | Meaning |
|-------|------|---------|
| `direction` | `Array` | L2-normalized refusal direction, shape `(d_model,)` |
| `layer_index` | `int` | 0-based layer with highest cosine separation |
| `cosine_scores` | `list[float]` | Per-layer separation between harmful/harmless activations |
| `d_model` | `int` | Hidden dimension |
| `model_path` | `str` | Model identifier |

#### CastResult (from `cast()`)
| Field | Type | Meaning |
|-------|------|---------|
| `text` | `str` | Generated output |
| `interventions` | `int` | Tokens where CAST applied steering (0 = defense didn't engage) |
| `considered` | `int` | Total tokens evaluated |
| `projections_before` | `list[float]` | Per-layer projection before steering |
| `projections_after` | `list[float]` | Per-layer projection after steering |

#### SICResult (from `sic()`)
| Field | Type | Meaning |
|-------|------|---------|
| `prompts_clean` | `list[str]` | Sanitized prompts (rewritten or original if clean) |
| `prompts_blocked` | `list[bool]` | True = prompt was blocked after max iterations |
| `initial_scores` | `list[float]` | Direction projection before sanitization |
| `final_scores` | `list[float]` | Direction projection after sanitization |
| `total_blocked` | `int` | Count of prompts blocked |
| `total_sanitized` | `int` | Count of prompts rewritten |
| `total_clean` | `int` | Count of prompts that passed unchanged |

#### DetectResult (from `detect()`)
| Field | Type | Meaning |
|-------|------|---------|
| `hardened` | `bool` | Whether the model is hardened against abliteration |
| `confidence` | `float` | 0.0–1.0 confidence in the hardening verdict |
| `effective_rank` | `float` | Refusal subspace dimensionality (>1.5 suggests hardening) |
| `evidence` | `list[str]` | Human-readable evidence strings |

#### AuditResult (from `audit()`)
| Field | Type | Meaning |
|-------|------|---------|
| `overall_risk` | `str` | `"critical"`, `"high"`, `"medium"`, or `"low"` |
| `findings` | `list[AuditFinding]` | Severity-rated findings with descriptions and remediation |
| `jailbreak_success_rate` | `float` | Fraction of jailbreak templates that bypassed refusal |
| `softprompt_success_rate` | `float \| None` | GCG/EGD attack success rate (None if not run) |
| `surface_refusal_rate` | `float \| None` | Overall refusal rate from surface mapping |

#### ProbeResult (from `probe()`)
| Field | Type | Meaning |
|-------|------|---------|
| `projections` | `list[float]` | Per-layer scalar projection onto refusal direction |
| `layer_count` | `int` | Number of layers |

#### EvalResult (from `evaluate()`)
| Field | Type | Meaning |
|-------|------|---------|
| `refusal_rate_original` | `float` | Refusal rate before cut |
| `refusal_rate_modified` | `float` | Refusal rate after cut |
| `perplexity_original` | `float` | Perplexity before cut |
| `perplexity_modified` | `float` | Perplexity after cut |
| `kl_divergence` | `float` | KL divergence between original and modified |

### Decision Guide

| I want to... | Use | Prerequisites |
|--------------|-----|---------------|
| Understand what a model refuses | `measure()` → `surface()` | model |
| Check if a model is hardened | `detect()` | model |
| Full safety audit before deployment | `audit()` → `report()` | model |
| Inspect one prompt's refusal signal | `measure()` → `probe("prompt")` | model |
| Defend against adversarial inputs | `measure()` → `sic()` + `cast()` | model |
| Remove refusal permanently | `measure()` → `cut()` → `export()` | model |
| Stress-test defenses | `measure()` → softprompt in TOML | model + TOML |
| Score response quality | `score("prompt", "response")` | nothing |
| Classify text for harm | `classify("text")` | nothing |

### Prerequisite Chain

```
model (loaded at Session init)
  ├── measure() → direction
  │     ├── probe(), scan(), surface()
  │     ├── steer(), cast(), sic()
  │     ├── evaluate()
  │     └── cut() → modified_model
  │           └── export()
  ├── detect()
  ├── audit() → audit_result
  │     ├── report()
  │     └── report_pdf()
  └── jailbreak()

(no prerequisites)
  ├── classify()
  └── score()
```

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
