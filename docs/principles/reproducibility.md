---
title: "Reproducibility — Traceable LLM Safety Experiments"
description: "TOML configs capture complete experiment definitions. Frozen dataclass results, experiment lineage via meta sections, vauban tree for DAG visualization, and no hidden state."
keywords: "reproducible AI experiments, experiment tracking, TOML experiment config, experiment lineage, frozen dataclass"
---

<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Reproducibility

Every experiment in Vauban is defined by a single TOML file, produces immutable results, and can be traced through an experiment lineage graph. Reproducibility is not an add-on --- it is a consequence of the design.

## The TOML file is the experiment

A Vauban config file captures the complete specification of a pipeline run: the model, the measurement mode, the cut parameters, the prompt set, the defense configuration, the attack settings. There are no hidden arguments, no environment-dependent defaults, no global state leaking in.

Two people with the same TOML file and the same model checkpoint will get the same results. The config is sufficient to reproduce the experiment.

> **TOML** --- Tom's Obvious Minimal Language. A configuration file format designed to be human-readable. Vauban uses it because it maps cleanly to typed dataclasses: each `[section]` corresponds to a frozen dataclass, each key to a typed field.

## Validation before execution

```
vauban --validate config.toml
```

This checks the config against the typed schema without loading the model. It catches typos, missing required fields, invalid enum values, and type mismatches in roughly 50 milliseconds instead of the minutes it takes to load a model.

The validation is the same type checking the pipeline runner performs --- not a separate schema. If validation passes, the config will parse correctly at runtime. If it fails, you get a precise error pointing to the problematic field.

!!! tip "Fail fast"
    Validate before committing a long run. A typo in `[softprompt]` discovered after 3 minutes of model loading and 20 minutes of optimization is a typo that could have been caught in 50ms.

## Immutable results

Every pipeline module produces a frozen dataclass as its result. `MeasureResult`, `CutResult`, `EvalResult`, `SurfaceReport`, `SoftpromptResult` --- all frozen, all typed, all serializable to JSON.

Frozen means the result cannot be mutated after creation. When the pipeline produces a direction and hands it to three downstream consumers (Cut, CAST, Probe), all three receive the same immutable object. There is no risk of one module silently modifying a shared result.

> **Frozen dataclass** --- a Python dataclass declared with `@dataclass(frozen=True)`. Any attempt to modify a field after construction raises `FrozenInstanceError`. Combined with `slots=True`, this also prevents accidental creation of new attributes.

The result dataclasses also carry their own provenance: which model was used, which config produced them, and what parameters were active. The result file is a self-contained record.

## Experiment lineage

The `[meta]` section in a TOML config tracks experiment metadata:

```toml
[meta]
id = "exp-061"
title = "Infix GCG with PPL and paraphrase"
status = "complete"
parents = ["exp-042", "exp-055"]
tags = ["infix", "gcg", "ppl", "paraphrase"]
notes = "Breakthrough config: 85.7% ASR, CAST=0.00"
```

This section has no effect on the pipeline --- it is pure metadata. But it connects experiments into a directed acyclic graph: each experiment knows which earlier experiments it builds on.

The `vauban tree` command renders this graph from a directory of TOML configs:

```
exp-042 (baseline direction)
  |
  +-- exp-055 (infix position test)
  |     |
  |     +-- exp-061 (+ PPL + paraphrase)
  |     +-- exp-062 (+ PPL, no paraphrase)
  |
  +-- exp-060 (DAW isolation test)
```

This is not version control --- it is experiment lineage. The graph shows how conclusions were reached, which experiments informed which design decisions, and where branches of investigation diverged.

## No hidden state

The Session API makes state explicit. After any sequence of operations, `s.state()` returns a dictionary describing what has been computed, what parameters were used, and what results are available. There is no implicit global state, no module-level caches that persist between runs, no ambient configuration.

This matters for reproducibility because it means the state of the system is always inspectable and always derivable from the sequence of operations performed. Two Sessions that have performed the same operations on the same model are in the same state.

## Backend determinism

The `_ops` and `_array` abstraction layers ensure that pipeline code expresses computation in terms of mathematical operations (mean, SVD, projection, cosine similarity) rather than backend-specific API calls. The same mathematical operations should produce the same results regardless of whether the backend is MLX or PyTorch.

!!! warning "Floating-point reality"
    Numerical results will not be bit-identical across backends due to differences in floating-point accumulation order, reduction algorithms, and hardware. "Same results" means statistically equivalent --- same direction (cosine similarity > 0.99), same refusal rates, same qualitative conclusions. Not identical mantissa bits.

Vauban does not claim perfect numerical reproducibility across backends. It claims that the experiment definition (TOML config), the result structure (frozen dataclasses), and the experiment lineage (`[meta]` graph) are fully reproducible and backend-independent.

## The reproducibility chain

A fully traceable experiment has:

1. **A TOML config** that defines every parameter.
2. **A model checkpoint** identified by name and revision.
3. **A prompt dataset** referenced by path or HuggingFace identifier.
4. **Frozen result files** produced by the pipeline.
5. **A `[meta]` section** connecting this experiment to its predecessors.

Given items 1--3, anyone can reproduce the run. Items 4--5 provide the audit trail showing that the run was performed and how it relates to the broader investigation.

## Related pages

- [Composability](composability.md) --- how TOML configs wire modules together
- [Attack-Defense Duality](attack-defense-duality.md) --- why both attack and defense runs need the same traceability
- [Last-Mile Reliability](last-mile-reliability.md) --- how reproducible audits support deployment decisions
