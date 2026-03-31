---
title: "Refusal Surface Mapping — Where Your LLM Refuses and Why"
description: "Map the multi-dimensional refusal boundary across topics, styles, and languages. Surface mapping reveals where a model refuses correctly, where it over-refuses, and where it under-refuses."
keywords: "refusal surface, LLM refusal mapping, safety coverage, refusal boundary, model behavior analysis, safety audit"
---

<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Surface Mapping

Systematically discovering where a model refuses and where it does not.

## The refusal surface

A model's refusal behavior is not a single number. It varies across multiple dimensions:

- **Category** — the model may refuse drug synthesis but comply with social engineering.
- **Style** — a direct request may trigger refusal while a roleplay framing does not.
- **Language** — refusal may be strong in English but weak in other languages.
- **Turn depth** — the model may comply after several turns of conversation that gradually shift context.
- **Framing** — academic framing, hypothetical framing, or fictional framing each produce different refusal rates.

The combination of these dimensions forms the **refusal surface** — a multi-dimensional landscape where each point represents a specific type of request and the model's disposition toward refusing or complying with it.

> **What is a "refusal surface" intuitively?** — Think of a topographic map. The height at each point represents how strongly the model refuses that type of request. Mountains are strong refusal (the model consistently declines). Valleys are weak or absent refusal (the model complies). The terrain is not uniform — it has ridges, cliffs, and plains. Surface mapping is the process of surveying this terrain to understand its shape before you modify it.

## What surface mapping does

Surface mapping runs a diverse prompt set through the model and records two quantities per prompt:

1. **Projection** — the scalar projection of the last-token activation onto the [refusal direction](refusal-direction.md). This measures the model's *internal* refusal signal.
2. **Refusal decision** — whether the model actually refuses (detected by checking generated output for refusal phrases). This measures the model's *external* behavior.

The gap between these is informative. A prompt with high projection but no refusal suggests the model is close to refusing but does not — a fragile boundary. A prompt with low projection but refusal suggests refusal driven by something other than the measured direction.

> **False positive vs. false negative** — a false positive is when the model refuses a harmless request (it "sees danger" where there is none). A false negative is when the model complies with a harmful request (it misses real danger). Surface mapping reveals both: high projection on harmless prompts suggests false-positive risk, and low projection on harmful prompts suggests false-negative risk.

## Before and after

Surface mapping is most valuable as a **comparison**. Run the same prompt set before and after an intervention ([abliteration](abliteration.md), [steering](steering.md), defense hardening) and compare:

- Which categories lost refusal? Which retained it?
- Did the intervention create new refusals (false positives)?
- Where are the remaining refusal boundaries?

This comparison turns a binary "did refusal rate drop?" into a detailed map of what changed and what did not.

## Two modes

### Full mode (generate=True)

Probes activations **and** generates a response for each prompt. Refusal is detected by checking the generated text for known refusal phrases.

Cost: ~61 forward passes per prompt (1 probe + ~60 generation tokens). Gives ground-truth refusal decisions.

### Fast recon (generate=False)

Probes activations only. No generation, no refusal detection. Records per-prompt projection values without the cost of generation.

Cost: 1 forward pass per prompt. Gives a projection landscape without behavioral confirmation. Useful for quick exploration or when you only need the [activation geometry](activation-geometry.md) signal.

## Coverage score

Not all prompt sets cover the surface equally. A set of 200 prompts about drug synthesis tells you nothing about social engineering or multilingual behavior. The **coverage score** measures how thoroughly a prompt set spans the theoretical surface grid:

$$\text{coverage} = \frac{|\text{observed cells}|}{|\text{total cells}|}$$

where cells are defined by the Cartesian product of category, style, language, turn depth, and framing buckets.

> **Cartesian product** — all possible combinations of items from multiple lists. If you have 3 categories, 2 styles, and 2 languages, the Cartesian product is 3 x 2 x 2 = 12 unique combinations (cells). Coverage measures how many of those 12 cells your prompt set actually tests.

Higher coverage means more confidence that the surface map represents the model's true refusal landscape rather than a narrow slice of it. Low coverage is a warning that unmapped regions may contain surprises.

## Grouping and aggregation

Surface mapping results are grouped along each dimension:

- **By category**: refusal rate and mean projection for each harm category.
- **By style**: how framing (direct, roleplay, academic, hypothetical) affects refusal.
- **By language**: cross-lingual refusal consistency.

Each group reports count, refusal rate, mean projection, and standard deviation. This reveals asymmetries — categories where the model is confident (high projection, consistent refusal) versus categories where it is borderline (moderate projection, inconsistent refusal).

## Surface mapping in the pipeline

Surface mapping is a **discovery** tool. It does not modify the model — it observes it. The typical workflow:

1. **Map** the original model's refusal surface.
2. **Intervene** (cut, steer, harden).
3. **Map again** to see what changed.
4. **Compare** the two maps to evaluate the intervention.

This is the empirical counterpart to [measurement](measurement.md): measurement extracts a direction, surface mapping shows what that direction means in practice across the full space of inputs.

## Relationship to defense

Surface mapping informs [defense design](defense-complementarity.md). If the map reveals that infix-positioned requests in a specific category consistently evade CAST, that is a signal to strengthen SIC for that category or adjust CAST thresholds.

The refusal surface is not static — it changes with every intervention. Continuous mapping is how you track the model's behavioral envelope over time.

!!! tip "Discovery before action"
    Map the surface before cutting or steering. A direction may look clean by cosine separation but produce unexpected refusal patterns across categories. Surface mapping catches asymmetries that aggregate metrics miss.
