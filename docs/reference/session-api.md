---
title: "Session API — Programmatic Entry Point for Vauban"
description: "Session wraps a loaded model with self-describing tool discovery, prerequisite tracking, guided workflows, and typed results. The API an AI assistant reads first."
keywords: "vauban session API, programmatic LLM safety, AI agent API, tool discovery, model safety automation"
---

<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Session API

`vauban.session.Session` is the programmatic entry point. It holds the model, tokenizer, and accumulated state. Every Vauban capability is available as a method. This is the page an AI assistant reads first when using Vauban as a library.

## What Session is

A Session wraps a loaded model and tracks the results of every operation. Tools declare prerequisites (`measure` must run before `probe`), and the Session enforces this automatically — if you call `probe()` before `measure()`, it measures first.

```python
from vauban.session import Session

s = Session("mlx-community/Qwen2.5-1.5B-Instruct-bf16")
```

The constructor loads the model and tokenizer, dequantizes if needed, and loads default harmful/harmless prompt sets. Custom prompts can be passed via `harmful_prompts` and `harmless_prompts` keyword arguments.

> **Tokenizer** — the component that converts text into tokens (numbers) and back. The model processes numbers, not letters. The tokenizer handles the translation in both directions. When Session loads a model, it loads the tokenizer too, since they must match.

> **Dequantize** — convert a compressed (quantized) model back to full-precision floating point. Quantized models use fewer bits per number to save memory, but some operations (like extracting precise refusal directions) work better at full precision. Session handles this automatically.

## Discovery

Four methods for understanding what is available:

**`s.tools()`** — returns a list of `Tool` dataclasses, each with `name`, `description`, `requires`, `produces`, and `category`. This is the complete capability inventory.

**`s.available()`** — returns the names of tools whose prerequisites are currently met. At session start, this includes tools that only require `model` (measure, detect, audit, jailbreak) and tools with no requirements (classify, score).

**`s.needs(tool_name)`** — returns the unmet prerequisites for a specific tool. `s.needs("probe")` returns `["direction"]` if `measure()` has not been called.

**`s.state()`** — returns a dict of booleans showing what has been computed: `model`, `direction`, `detect_result`, `audit_result`, `modified_model`.

## Guided workflows

**`s.guide(goal)`** — returns plain-text instructions for a named workflow. Available goals:

| Goal | What it covers |
|---|---|
| `"audit"` | Red-team assessment with findings and report |
| `"compliance"` | EU AI Act compliance assessment |
| `"harden"` | Improve model safety via CAST/SIC |
| `"abliterate"` | Remove refusal via measure, cut, export |
| `"inspect"` | Understand model behavior via measure, probe |

Calling `s.guide()` with no argument lists all available workflows.

**`s.done(goal)`** — returns `(is_done: bool, reason: str)`. Call after each step to know when to stop.

**`s.suggest_next()`** — context-aware recommendations based on current state and findings. Each suggestion is labeled `[FACT]` (based on measured data) or `[ADVICE]` (heuristic). Use `s.done(goal)` rather than `suggest_next()` to determine completion — suggestions always recommend more work.

## Tool categories

### Assessment

| Method | Description | Requires | Produces |
|---|---|---|---|
| `s.measure()` | Extract the refusal direction from activations | model | `DirectionResult` |
| `s.detect()` | Check if model is hardened against abliteration | model | `DetectResult` |
| `s.evaluate()` | Baseline metrics: refusal rate, perplexity | model, direction | `EvalResult` |
| `s.audit(...)` | Full red-team assessment with severity-rated findings | model | `AuditResult` |

`audit()` accepts `company_name`, `system_name`, and `thoroughness` ("quick", "standard", "deep").

### Inspection

| Method | Description | Requires | Produces |
|---|---|---|---|
| `s.probe(prompt)` | Per-layer projection onto refusal direction | direction | `ProbeResult` |
| `s.scan(content)` | Per-token injection detection | direction | `ScanResult` |

`probe()` returns `ProbeResult` with a `projections` list (one float per layer), `layer_count`, and `prompt`.

`scan()` returns `ScanResult` with `injection_probability`, `overall_projection`, `spans`, `per_token_projections`, and `flagged`.

### Defense

| Method | Description | Requires | Produces |
|---|---|---|---|
| `s.steer(prompt, alpha=1.0)` | Generate with unconditional activation steering | direction | generation result |
| `s.cast(prompt, alpha=1.0, threshold=0.0)` | Generate with conditional activation steering | direction | `CastResult` |
| `s.sic(prompts)` | Iterative input sanitization | direction | `SICResult` |

`cast()` returns `CastResult` with the generated text and intervention count. `sic()` accepts a list of prompts and returns `SICResult` with per-prompt results and aggregate statistics.

### Modification

| Method | Description | Requires | Produces |
|---|---|---|---|
| `s.cut(alpha=1.0, norm_preserve=False)` | Remove refusal direction from weights | direction | modified weight dict |
| `s.export(output_dir)` | Save modified weights to disk | modified_model | path string |

`cut()` modifies `o_proj` and `down_proj` weights across all layers. `export()` writes a standard model directory (safetensors + tokenizer + config).

### Analysis

| Method | Description | Requires | Produces |
|---|---|---|---|
| `s.classify(text)` | Score text against 13-domain harm taxonomy | none | harm scores |
| `s.score(prompt, response)` | 5-axis quality assessment (length, structure, anti-refusal, directness, relevance) | none | score result |

These are static methods — they work without a loaded model and without a measured direction.

### Reporting

| Method | Description | Requires | Produces |
|---|---|---|---|
| `s.report(fmt="markdown")` | Generate report from audit results | audit_result | markdown or JSON string |
| `s.report_pdf()` | Generate PDF report from audit results | audit_result | PDF bytes |

`report()` accepts `"markdown"` or `"dict"` format. `report_pdf()` returns raw PDF bytes.

## Prerequisite chain

The dependency graph between tools:

```
model (always available)
  |
  +-- measure() --> direction
  |     |
  |     +-- probe(prompt)
  |     +-- scan(content)
  |     +-- steer(prompt)
  |     +-- cast(prompt)
  |     +-- sic(prompts)
  |     +-- evaluate()
  |     +-- cut() --> modified_model
  |           |
  |           +-- export(path)
  |
  +-- detect()
  +-- audit() --> audit_result
  |     |
  |     +-- report()
  |     +-- report_pdf()
  |
  +-- jailbreak()

(no prerequisites)
  +-- classify(text)
  +-- score(prompt, response)
```

If a tool's prerequisite is not met, calling it will auto-trigger the prerequisite. For example, `s.probe("test")` will call `s.measure()` first if no direction exists.

> **Prerequisite tracking** — the Session knows which operations depend on which results. You do not need to manually manage the order — if you call a tool that needs a direction, the Session measures one first. This makes programmatic use forgiving: you can call what you need, and dependencies resolve automatically.

## Authoritative source

This page is a reference summary. The code and its docstrings in `vauban/session.py` are the authoritative source for method signatures, parameter types, return types, and behavioral details. When this page and the code disagree, the code is correct.
