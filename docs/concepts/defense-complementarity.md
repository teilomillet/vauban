---
title: "Defense Complementarity — Why Layered LLM Defenses Work"
description: "No single defense catches all attacks. CAST and SIC are negatively correlated (r=-0.52) — combined they block 88% of attacks vs 71% individually. Learn why defense composition matters."
keywords: "LLM defense, defense in depth, CAST SIC complementarity, layered defense, safety alignment defense, jailbreak prevention"
---

<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Defense Complementarity

Why no single defense catches everything, and how layering independent defenses achieves coverage that none can alone.

## The empirical finding

Vauban's adversarial research produced a counterintuitive result: the two primary defenses — CAST (activation [steering](steering.md)) and SIC (input sanitization) — are **negatively correlated** in what they catch.

| Metric | Value |
|---|---|
| CAST blocks | 70.6% of attacks |
| SIC blocks | 23.5% of attacks |
| **Combined** blocks | **88.2%** of attacks |
| Correlation (r) | -0.52 |
| Cohen's kappa | -0.35 |

> **Negative correlation** — when two things tend to move in opposite directions. A correlation of -0.52 between CAST and SIC means that when CAST succeeds at blocking an attack, SIC tends to fail on that same attack, and vice versa. This is actually *good* for defense — it means they cover each other's weaknesses rather than failing on the same things.

> **Cohen's kappa** — a statistic that measures agreement between two classifiers, adjusted for chance. Negative kappa means the two defenses disagree more than random chance would predict — further confirming they catch different attack types.

The combined rate (88.2%) is not the sum of the individual rates because there is some overlap. But the negative correlation is the key: attacks that evade CAST tend to be caught by SIC, and vice versa. They fail on different things.

> **What is "defense in depth"?** — A security principle borrowed from military engineering (and Vauban's namesake). Instead of one strong wall, build multiple independent layers of defense. If an attacker breaches one layer, the next layer catches them. The layers do not need to be individually perfect — they need to cover each other's blind spots. In network security: firewall + intrusion detection + application-level controls. In Vauban: SIC + CAST + Guard.

> **Why can't one defense catch everything?** — Each defense operates on different information. SIC examines the input text. CAST monitors internal activations. Guard checks generated output. An attack crafted to look benign at the surface level (bypassing SIC) may still trigger high refusal-direction projection (caught by CAST). An attack that manipulates activations to stay below CAST thresholds may contain detectable adversarial patterns in the input text (caught by SIC). No single signal is sufficient because adversarial attacks can optimize against any one signal.

## The three defense layers

### SIC: input sanitization (before generation)

SIC (Iterative Self-Improvement for Adversarial Attacks) operates **before the model generates a response**. It examines the input and applies iterative sanitization:

1. **Detect** adversarial content in the prompt (via direction-based projection scoring or phrase matching).
2. **Rewrite** the prompt to remove detected adversarial content.
3. **Repeat** until the prompt is clean or a maximum iteration count is reached.
4. **Block** if sanitization cannot clean the prompt.

SIC catches attacks that embed adversarial tokens in the prompt — soft prompt suffixes, GCG-optimized strings, encoded instructions. It operates on the input representation and can neutralize attacks before they ever reach the model's forward pass.

**Blind spot:** attacks that do not alter the surface-level prompt in a detectable way. Infix-positioned adversarial content surrounded by natural-looking context can pass SIC because the sanitization heuristics do not flag it.

### CAST: activation steering (during generation)

[CAST](steering.md) operates **during the forward pass**, token by token. It monitors the projection of activations onto the [refusal direction](refusal-direction.md) and steers when the projection crosses a threshold.

CAST catches attacks that affect the model's internal state — even if the prompt looks benign on the surface. If an adversarial input causes high refusal-direction activation, CAST detects and counteracts it in real time.

**Blind spot:** infix-positioned attacks. Research shows that adversarial content placed in the middle of a prompt (between natural context) reduces the number of CAST interventions by 6x compared to suffix placement. The activation signature is diluted by surrounding context, keeping projection below the steering threshold.

### Guard: output monitoring (after generation)

Guard operates **after generation completes**. It checks the model's output against safety criteria. If the output violates a policy, Guard can trigger a cache rewind — discarding the generated tokens and re-generating with stronger steering or a modified prompt.

> **Cache rewind** — the model keeps a checkpoint of its internal state at "safe" points during generation. If harmful content is detected in the output, Guard rolls back to the last safe checkpoint and tries generating again. Think of it like an undo button — it erases the harmful output and gets a second chance.

Guard is the last line of defense. It catches cases where both SIC and CAST were insufficient — the attack passed input sanitization, stayed below activation thresholds, but the resulting output is still harmful.

## Why the negative correlation exists

The blind spots are structurally complementary:

| Attack position | SIC | CAST |
|---|---|---|
| **Suffix** (appended tokens) | Catches — anomalous tokens at prompt end | Catches — strong activation signature |
| **Prefix** (prepended tokens) | Catches — anomalous tokens at prompt start | Catches — early activation disturbance |
| **Infix** (embedded in context) | Often misses — tokens blend with context | Often misses — diluted activation signature |

All CAST-blind attacks in the research dataset are infix-positioned. All SIC-blind attacks are prefix- or suffix-positioned. The defenses fail on different geometries of the same attack type.

This is not a coincidence — it follows from what each defense observes:

- SIC looks at **surface tokens**. Infix attacks are surrounded by natural tokens that mask the adversarial content.
- CAST looks at **activation magnitude**. Infix attacks produce weaker projection because the adversarial signal is averaged with context activations in the residual stream.

## Assembly, not monolith

The defense stack is **composed**, not built as a single system. Each layer:

- Has its own configuration section in TOML
- Can be enabled or disabled independently
- Operates on different data (input text, activations, output text)
- Has independent failure modes

This means you can run SIC alone, CAST alone, Guard alone, or any combination. The defense-in-depth property emerges from composition, not from a single complex module.

```
Input --> [SIC] --> Forward pass --> [CAST] --> Output --> [Guard]
          scan       steer per-token           check + rewind
```

Adding or removing a layer does not require modifying the others. This follows Vauban's Unix philosophy: each component does one thing, and they are assembled rather than glued.

## Measured coverage

From the adversarial research campaign (N=34 attack configurations against Qwen2.5-1.5B):

| Defense | Evasion rate | Blocks |
|---|---|---|
| CAST only | 29.4% | 70.6% |
| SIC only | 76.5% | 23.5% |
| CAST + SIC | 11.8% | 88.2% |
| CAST + SIC + Guard | ~0% | ~100% |

The residual 11.8% that evades both CAST and SIC are the dual evaders — attacks that are infix-positioned *and* optimized against both detection signals. Guard catches these at the output level.

!!! warning "No defense is absolute"
    These numbers are from a specific adversarial campaign against a specific model. A sufficiently resourced attacker can always find new evasion strategies. The defense stack reduces the attack surface — it does not eliminate it. Continuous [surface mapping](surface-mapping.md) and red-teaming are necessary to maintain coverage.
