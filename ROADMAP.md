# Vauban Roadmap

Three phases, sequenced so each builds on the last.
A (foundation) enables C (composability) enables B (defense expansion).

---

## Current State (March 2026)

| Metric | Value |
|--------|-------|
| Core LOC | 51,837 across 220 Python modules |
| Test LOC | 42,431 (0.82x ratio) |
| Types | 171 frozen dataclasses, 9 protocols, 0 `Any` in typed code |
| Mode handlers | 24 early-mode handlers plus the default pipeline |
| Experiment logging | `[meta]` lineage + per-run `experiment_log.jsonl` |
| Backends | MLX (primary), PyTorch (secondary) |
| Local gates | `ruff`, `ty`, and `pytest` all green |

### Strengths
- Every tool is typed, immutable, and deterministic
- Weight I/O is decoupled from model objects
- Backend abstraction (MLX/torch) is clean — duck typing via protocols
- Config validation catches typos and invalid values before model load
- Real-model integration coverage exists for the core pipeline and is opt-in via `VAUBAN_INTEGRATION=1`
- Experiment lineage and per-run metrics are already recorded via `[meta]` and `experiment_log.jsonl`

### Weaknesses
- Real-model integration tests are opt-in and not part of the default gate
- Tracking is append-only JSONL, not a queryable experiment store
- Sequential hotspots remain in transfer re-ranking, GAN rounds, and API eval
- Softprompt still re-exports many internal helpers that should be private
- Mixed calling convention persists: some functions take config objects, others take direct args
- Paper-inspired modules exist for RepBend, COLD, LARGO, AmpleGCG, and Fusion, but maturity and benchmark coverage are still uneven

---

## Phase A: Research Instrument

**Goal**: Make vauban reproducible, observable, and trustworthy.
Every experiment should be trackable, every result reproducible, every module tested.

### A.1 Test Coverage Hardening

**Target**: Keep the suite near 0.9x test/code ratio while promoting more
coverage from "exists locally" to "enforced automatically."

#### Priority 1 — CI Enforcement And Integration Promotion
The repository now has fast unit coverage plus opt-in real-model integration
tests. The next step is to tighten the gate, not to invent it from scratch.

```
.github/workflows/ci.yml
├── uv run ruff check .
├── uv run ty check
└── uv run pytest -q

tests/test_integration.py
├── measure → probe → cut → evaluate on a real model
├── harmful vs harmless projection contrast
├── weight modification and projection shrinkage
└── perplexity/refusal sanity checks
```

Next up:
- keep the default CI path fast on macOS runners
- split real-model coverage into smoke vs. full integration tiers
- decide which integration slice should run on every PR versus scheduled/manual

#### Priority 2 — Module-Level Test Coverage

| Module | LOC | What to test |
|--------|-----|-------------|
| `_serializers.py` | 373 | Round-trip: construct result → serialize → deserialize → assert equality |
| `_suggestions.py` | 325 | Known typos flagged, unknown sections warned, numeric ranges enforced |
| `_diff.py` | 335 | Two report dirs → diff output matches expected deltas |
| `_forward.py` | 212 | embed_and_mask shapes, cache creation, lm_head forward |
| `__main__.py` | 304 | CLI arg parsing (mock sys.argv), help text, error messages |
| `_init.py` | 139 | Scaffold output for each mode is valid TOML |
| `_ops_torch.py` | 242 | Mirror every MLX op test with torch equivalent |
| `_compose.py` | 68 | load_bank → compose_direction → verify shape and norm |
| `_model_io.py` | 61 | load_model returns correct protocol types |

#### Priority 3 — Softprompt Deep Coverage

The softprompt subpackage has the worst test ratio (0.19x).
Split `test_softprompt.py` into:

```
tests/softprompt/
├── test_gcg.py          # GCG-specific: beam search, candidate scoring, re-ranking
├── test_egd.py          # EGD-specific: Bregman projection, temperature, warm-start
├── test_continuous.py   # Continuous: Adam, cosine schedule, embed regularization
├── test_loss.py         # Every loss function: targeted, untargeted, defensive, externality
├── test_gan.py          # GAN loop: escalation, multi-turn, defense evaluation
├── test_generation.py   # Prefill/decode, attack evaluation, multi-turn generation
├── test_utils.py        # Token constraints, prompt encoding, injection context
└── test_dispatcher.py   # Mode dispatch, config to attack function routing
```

### A.2 Experiment Tracking

Extend the current TOML `[meta]` + `experiment_log.jsonl` flow into structured,
queryable tracking.

#### Local Tracking (SQLite)

```python
# New module: vauban/tracking.py

@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    id: str                          # UUID or user-provided
    config_hash: str                 # SHA256 of resolved config
    model_hash: str                  # SHA256 of model weights
    started_at: str                  # ISO 8601
    finished_at: str | None
    status: str                      # running | completed | failed
    config_snapshot: str             # Full resolved TOML
    metrics: dict[str, float]        # Key metrics (loss, refusal_rate, etc.)
    artifacts: list[str]             # Paths to output files
    parent_ids: list[str]            # Lineage from [meta].parents
    tags: list[str]                  # From [meta].tags
    notes: str                       # From [meta].notes

def init_db(path: Path) -> None: ...
def log_experiment(record: ExperimentRecord) -> None: ...
def query_experiments(filters: dict[str, str]) -> list[ExperimentRecord]: ...
def compare_experiments(id_a: str, id_b: str) -> dict[str, float]: ...
```

Storage: `~/.vauban/experiments.db` (SQLite). One table, JSON columns for nested fields.
CLI: `vauban experiments [list|show|compare|export]`.

#### W&B Integration (Optional)

```toml
# pyproject.toml
[project.optional-dependencies]
tracking = ["wandb>=0.16"]
```

```python
# vauban/tracking.py (extended)

def _wandb_available() -> bool:
    try:
        import wandb
        return wandb.api.api_key is not None
    except ImportError:
        return False

def log_to_wandb(record: ExperimentRecord) -> None:
    """Push experiment record to W&B if available and configured."""
    ...
```

Opt-in via `[tracking]` TOML section or `VAUBAN_WANDB=1` env var.
No W&B dependency in core install.

### A.3 Reproducibility

#### Seed Management

Currently seeds are set per-module. Centralize:

```python
# vauban/_seed.py

def set_global_seed(seed: int) -> None:
    """Set seed for all backends (MLX, Python random, numpy)."""
    import random
    random.seed(seed)
    # MLX doesn't have global seed — seed is per-key
    # numpy seed for np.load/save determinism
    import numpy as np
    np.random.seed(seed)
```

Add `seed` to `PipelineConfig`. Pass through to all stochastic operations.

#### Config Snapshot

After `run()` resolves all paths and defaults, write the fully-resolved config:

```python
# In run(), after config loading:
resolved_path = config.output_dir / "resolved_config.toml"
write_resolved_config(config, resolved_path)
```

This captures the *actual* config used, including defaults and path resolution.

#### Model Fingerprint

Hash the model weights at load time:

```python
def model_fingerprint(model: CausalLM) -> str:
    """SHA256 of flattened weight tensor checksums."""
    ...
```

Store in experiment record. Catches silent model changes between runs.

### A.4 Structured Logging

Replace ad-hoc `_log()` calls with structured JSON logging:

```python
# vauban/_logging.py

import logging
import json

class StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        if hasattr(record, "metrics"):
            entry["metrics"] = record.metrics
        return json.dumps(entry)
```

Configurable via `[verbose]` TOML section:

```toml
[verbose]
level = "info"        # debug | info | warning | error
format = "structured" # structured | human
file = "vauban.log"   # optional log file path
```

### A.5 Analysis Automation

Automate the analysis patterns from MEMORY.md:

```python
# vauban/analysis.py

def compute_effect_sizes(
    results: list[ExperimentRecord],
    grouping: str,  # "position", "loss_mode", "has_gan", etc.
    metric: str,    # "cast_score", "final_loss", "interventions"
) -> EffectSizeResult: ...

def compute_eta_squared(
    results: list[ExperimentRecord],
    factors: list[str],
    outcome: str,
) -> VarianceDecomposition: ...

def find_dominant_factor(
    results: list[ExperimentRecord],
    outcome: str,
) -> FactorRanking: ...
```

CLI: `vauban analyze --metric cast_score --group position experiments/*.toml`

---

## Phase C: Composable Library

**Goal**: Make vauban's components independently importable and embeddable.
Anyone should be able to `pip install vauban` and use individual tools without the pipeline.

### C.1 API Surface Cleanup

#### Hide Softprompt Internals

Current state: `vauban/softprompt/__init__.py` exports 50+ internal helpers.
Target: Export only 5 public functions.

```python
# vauban/softprompt/__init__.py (after cleanup)

__all__ = [
    "softprompt_attack",
    "gan_loop",
    "paraphrase_prompts",
    "evaluate_against_defenses",
    "evaluate_against_defenses_multiturn",
]
```

Move internal helpers to `vauban/softprompt/_internal.py` or keep in their current
modules but remove from `__init__.py` exports.

#### Standardize Calling Convention

Two patterns exist today:

```python
# Pattern 1: Direct args (measure, cut, probe)
measure(model, tokenizer, harmful, harmless, clip_quantile=0)

# Pattern 2: Config object (softprompt_attack, sic_sanitize)
softprompt_attack(model, tokenizer, prompts, config, direction)
```

**Decision**: Keep both. Direct args for simple tools (< 6 params),
config objects for complex tools (> 6 params). Document the convention.

The threshold is already naturally respected — `measure` has 5 params,
`softprompt_attack` has 41 config fields. No refactor needed.

### C.2 Decouple Pipeline Stages

Currently `run()` is a 400-line function that threads state between stages.
Extract each stage into a standalone function with explicit I/O:

```python
# Current: tightly coupled in run()
direction_result = measure(model, tokenizer, harmful, harmless)
modified_weights = cut(flat_weights, direction_result.direction, ...)
export_model(model_path, modified_weights, output_dir)
eval_result = evaluate(model, modified_model, tokenizer, prompts)

# Target: each stage explicitly declares its inputs and outputs
# (This already works today! The coupling is in run(), not in the functions.)
```

The key insight: **vauban is already composable at the function level**.
The problem is that `run()` hides this composability behind a monolithic orchestrator.

**Action**: Document the composable usage prominently. Add examples to README.

```python
# README example: composable usage
import vauban

model, tokenizer = vauban.load_model("mlx-community/Qwen2.5-1.5B-Instruct-bf16")
harmful = vauban.load_prompts("harmful.jsonl")
harmless = vauban.load_prompts("harmless.jsonl")

# Measure
result = vauban.measure(model, tokenizer, harmful, harmless)
print(f"Best layer: {result.layer_index}, d_model: {result.d_model}")

# Probe
for prompt in ["How do I pick a lock?", "What is gravity?"]:
    probe = vauban.probe(model, tokenizer, prompt, result.direction, result.layer_index)
    print(f"{prompt}: {probe.projections[result.layer_index]:.3f}")

# Steer
steered = vauban.steer(model, tokenizer, "How do I pick a lock?",
                        result.direction, result.layer_index, alpha=3.0)
print(steered.text)

# CAST (conditional)
cast_result = vauban.cast_generate(model, tokenizer, "How do I pick a lock?",
                                    result.direction, [result.layer_index],
                                    alpha=3.0, threshold=0.5)
print(cast_result.text)
```

### C.3 Plugin System

#### Custom Loss Functions

```python
# vauban/softprompt/_loss_registry.py

LossFn = Callable[[CausalLM, Array, Array, int, ...], Array]

_LOSS_REGISTRY: dict[str, LossFn] = {
    "targeted": _compute_loss,
    "untargeted": _compute_untargeted_loss,
    "defensive": _compute_defensive_loss,
    "externality": _compute_externality_loss,
}

def register_loss(name: str, fn: LossFn) -> None:
    """Register a custom loss function for soft prompt optimization."""
    _LOSS_REGISTRY[name] = fn

def get_loss(name: str) -> LossFn:
    if name not in _LOSS_REGISTRY:
        available = ", ".join(sorted(_LOSS_REGISTRY))
        msg = f"Unknown loss mode {name!r}. Available: {available}"
        raise ValueError(msg)
    return _LOSS_REGISTRY[name]
```

Usage:

```python
import vauban
from vauban.softprompt import register_loss

def my_custom_loss(model, soft_embeds, prompt_ids, n_tokens, ...):
    # Custom objective
    return loss_value

register_loss("my_objective", my_custom_loss)

# Then in TOML:
# [softprompt]
# loss_mode = "my_objective"
```

#### Custom Early Modes

```python
# vauban/config/_mode_registry.py (extended)

def register_early_mode(
    section: str,
    mode_name: str,
    phase: str,  # "before_prompts" | "after_measure"
    requires_direction: bool,
    predicate: Callable[[PipelineConfig], bool],
    runner: Callable[[_EarlyModeContext], None],
) -> None:
    """Register a custom early-return mode."""
    ...
```

### C.4 Packaging

```toml
# pyproject.toml (restructured)
[project]
name = "vauban"
version = "0.1.0"
requires-python = ">=3.12"

# Minimal core: types, config, ops abstraction
dependencies = []

[project.optional-dependencies]
mlx = ["mlx>=0.22", "mlx-lm>=0.21"]        # Apple Silicon
torch = ["torch>=2.0", "transformers>=4.36", "accelerate>=0.26"]  # CUDA/CPU
attack = ["vauban[mlx]"]                     # Soft prompt optimization
defense = ["vauban[mlx]"]                    # CAST, SIC, scan, defend
optimize = ["optuna>=4.0"]                   # Hyperparameter search
tracking = ["wandb>=0.16"]                   # W&B integration
all = ["vauban[mlx,attack,defense,optimize,tracking]"]
dev = ["pytest", "ruff", "ty", "matplotlib"]
```

**Note**: The `mlx` dependency makes `vauban[core]` Apple Silicon-only.
The `torch` extra enables cross-platform use.
A future `vauban[core]` with zero backend deps would only export types and config parsing.

### C.5 Documentation

#### API Reference

Auto-generate from docstrings using `pdoc` or similar:

```bash
pdoc vauban --output docs/api/
```

Every public function already has docstrings. Just need to build and host.

#### Tutorial Notebooks

```
docs/tutorials/
├── 01_measure_and_probe.ipynb    # Extract direction, inspect projections
├── 02_cut_and_evaluate.ipynb     # Abliterate and measure quality
├── 03_cast_steering.ipynb        # Runtime conditional steering
├── 04_softprompt_attack.ipynb    # GCG/EGD optimization
├── 05_defense_stack.ipynb        # SIC + CAST + policy composition
├── 06_gan_adversarial.ipynb      # Attack-defense iteration
└── 07_experiment_tracking.ipynb  # W&B integration
```

---

## Phase B: Defense Toolkit

**Goal**: Make vauban the standard tool for LLM safety evaluation and hardening.
Both sides of the wall: attack to find vulnerabilities, defend to fix them.

### B.1 New Defense Techniques

#### RepBend — Representation Bending

Loss-based fine-tuning that pushes harmful activations away from safe ones
(Patel et al., ACL 2025). The defense dual of abliteration.

```python
# vauban/repbend.py

@dataclass(frozen=True, slots=True)
class RepBendConfig:
    layers: list[int] | None = None
    learning_rate: float = 1e-4
    n_steps: int = 100
    separation_weight: float = 1.0
    preservation_weight: float = 0.5

@dataclass(frozen=True, slots=True)
class RepBendResult:
    initial_separation: float
    final_separation: float
    loss_history: list[float]
    modified_layers: list[int]

def harden_repbend(
    model: CausalLM,
    tokenizer: Tokenizer,
    harmful: list[str],
    harmless: list[str],
    config: RepBendConfig,
    direction: Array | None = None,
) -> RepBendResult:
    """Fine-tune model to increase activation separation between harmful and harmless.

    Unlike cut (which removes directions), RepBend pushes representations apart
    via a contrastive loss on the activation space. This makes the safety boundary
    wider and harder to abliterate.
    """
    ...
```

Integration: `[repbend]` TOML section, early mode after measure.

#### Latent Adversarial Training (LAT)

Train against continuous latent perturbations (Casper et al., 2024).
Uses the soft prompt threat model as the attack surface during training.

```python
# vauban/lat.py

@dataclass(frozen=True, slots=True)
class LATConfig:
    n_epochs: int = 5
    attack_steps: int = 10      # PGD steps per training step
    attack_lr: float = 0.01
    defense_lr: float = 1e-5
    epsilon: float = 0.1        # Perturbation budget (L2 norm)
    layers: list[int] | None = None

def train_lat(
    model: CausalLM,
    tokenizer: Tokenizer,
    harmful: list[str],
    harmless: list[str],
    config: LATConfig,
) -> LATResult:
    """Adversarial training against continuous embedding perturbations.

    Inner loop: optimize worst-case perturbation (PGD on embeddings)
    Outer loop: update model weights to be robust to perturbation
    """
    ...
```

Integration: `[lat]` TOML section, standalone early mode.

### B.2 Automated Red/Blue Benchmarking

#### Tournament Mode

Pit attack configs against defense configs automatically:

```python
# vauban/tournament.py

@dataclass(frozen=True, slots=True)
class TournamentConfig:
    attack_configs: list[Path]     # TOML configs with [softprompt]
    defense_configs: list[Path]    # TOML configs with [cast] or [sic] or [defend]
    prompts: Path                  # Shared evaluation prompts
    n_seeds: int = 3               # Repeat each matchup for variance

@dataclass(frozen=True, slots=True)
class MatchupResult:
    attack_config: str
    defense_config: str
    attack_success_rate: float
    defense_block_rate: float
    mean_interventions: float
    seed_variance: float

def run_tournament(
    model: CausalLM,
    tokenizer: Tokenizer,
    config: TournamentConfig,
) -> TournamentResult:
    """Run all attack x defense combinations and report results.

    Each matchup:
    1. Run attack config → get optimized suffix
    2. Evaluate suffix against defense config → get block rate
    3. Repeat with different seeds for variance estimation
    """
    ...
```

CLI: `vauban tournament attacks/ defenses/ --prompts eval.jsonl`

Output: Markdown table of attack x defense win rates, Elo rankings.

#### Standardized Benchmarks

Bundle benchmark prompt sets and report format:

```python
# vauban/benchmarks.py

BENCHMARK_SUITES = {
    "harmbench": "HarmBench standard prompts (Mazeika et al. 2024)",
    "advbench": "AdvBench harmful behaviors (Zou et al. 2023)",
    "safety_surface": "Vauban 6-axis surface coverage",
}

def run_benchmark(
    model: CausalLM,
    tokenizer: Tokenizer,
    suite: str,
    defense_config: Path | None = None,
) -> BenchmarkResult:
    """Run a standardized safety benchmark and report results."""
    ...
```

### B.3 Runtime Defense

#### Production CAST/SIC Deployment

Extract CAST and SIC into a lightweight inference wrapper:

```python
# vauban/runtime.py

class SafetyGuard:
    """Lightweight inference wrapper with CAST/SIC defense stack.

    Designed for production deployment — no training deps, minimal memory.
    """

    def __init__(
        self,
        model: CausalLM,
        tokenizer: Tokenizer,
        direction: Array,
        cast_config: CastConfig,
        sic_config: SICConfig | None = None,
    ) -> None: ...

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
    ) -> GuardedResult:
        """Generate with defense stack active.

        Returns generation text plus defense telemetry
        (interventions, projections, SIC iterations, block decisions).
        """
        ...

    def is_safe(self, prompt: str) -> bool:
        """Quick safety check without full generation."""
        ...
```

#### Streaming Inference

Add async/streaming support for real-time defense:

```python
# vauban/runtime.py (extended)

async def generate_stream(
    self,
    prompt: str,
    max_tokens: int = 100,
) -> AsyncIterator[StreamChunk]:
    """Stream tokens with per-token defense telemetry."""
    ...
```

#### Defense Cost Analysis

Measure the latency and throughput impact of defense layers:

```python
# vauban/cost.py

@dataclass(frozen=True, slots=True)
class DefenseCostResult:
    baseline_tokens_per_second: float
    defended_tokens_per_second: float
    overhead_percent: float
    mean_intervention_latency_ms: float
    peak_memory_mb: float

def measure_defense_cost(
    model: CausalLM,
    tokenizer: Tokenizer,
    prompts: list[str],
    defense_config: CastConfig | SICConfig,
    direction: Array,
    n_warmup: int = 5,
    n_measure: int = 20,
) -> DefenseCostResult:
    """Benchmark defense overhead on representative prompts."""
    ...
```

---

## Sequencing and Dependencies

```
Phase A (Foundation)
├── A.1 Test hardening              ← no deps, start immediately
├── A.2 Experiment tracking         ← needs A.1 for test coverage
├── A.3 Reproducibility             ← needs A.2 for experiment records
├── A.4 Structured logging          ← independent, can parallel with A.1
└── A.5 Analysis automation         ← needs A.2 for experiment data

Phase C (Composable Library)
├── C.1 API surface cleanup         ← needs A.1 (tests protect refactor)
├── C.2 Pipeline decoupling         ← needs C.1
├── C.3 Plugin system               ← needs C.1
├── C.4 Packaging                   ← needs C.1, C.2
└── C.5 Documentation               ← needs C.1, C.2, C.4

Phase B (Defense Toolkit)
├── B.1 RepBend + LAT               ← needs C.3 (plugin system for new modes)
├── B.2 Tournament + benchmarks     ← needs A.2 (tracking), C.2 (composable stages)
├── B.3 Runtime defense             ← needs C.1 (clean API), C.4 (packaging)
└── B.4 Defense cost analysis       ← needs B.3
```

## Milestones

### M1: Tested Foundation (Phase A.1 + A.4)
- 0.9x test/code ratio
- Structured JSON logging
- All 16 untested modules covered
- Softprompt test split complete

### M2: Trackable Research (Phase A.2 + A.3 + A.5)
- SQLite experiment database
- Optional W&B integration
- Seed management + config snapshots
- `vauban experiments` CLI
- `vauban analyze` CLI

### M3: Clean Library (Phase C.1 + C.2 + C.3)
- Softprompt internals hidden
- Composable usage documented in README
- Loss function plugin registry
- Early mode plugin registry

### M4: Published Package (Phase C.4 + C.5)
- PyPI publish with extras (`vauban[mlx]`, `vauban[torch]`, etc.)
- API reference auto-generated
- 7 tutorial notebooks
- Migration guide

### M5: Defense Platform (Phase B.1 + B.2)
- RepBend hardening
- Latent adversarial training
- Tournament mode
- Standardized benchmark suites

### M6: Production Runtime (Phase B.3 + B.4)
- SafetyGuard inference wrapper
- Streaming defense
- Cost analysis tooling
- Deployment guide

---

## Non-Goals

Things we explicitly **will not** do:

- **Multi-GPU / distributed training** — Vauban is a single-machine research tool.
  Cloud-scale training is a different problem with different tools.
- **Model training from scratch** — Vauban operates on pre-trained models.
  We measure, cut, steer, and attack — we don't pre-train.
- **Web UI / dashboard** — CLI + notebooks + W&B is sufficient.
  Building a custom UI is a maintenance burden with low ROI.
- **Support models < 100M params** — Too small for meaningful refusal geometry.
  Our test models (135M SmolLM2) are already at the practical minimum.
- **Backwards compatibility with Python < 3.12** — Modern typing syntax
  (`X | Y`, `list[X]`) is non-negotiable.
