---
title: "Attack-Defense Duality — Why Both Sides Live in One Tool"
description: "Attack and defense are two operations on the same geometric object. The refusal direction powers both abliteration and CAST. You cannot defend what you cannot break."
keywords: "attack defense duality, LLM security, red team blue team, abliteration defense, Vauban philosophy"
---

<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Attack-Defense Duality

Vauban includes tools to both break and harden model safety alignment. This is not a design compromise --- it is the core design principle. You cannot defend what you cannot break, because attack and defense are two operations on the same geometric object.

## One direction, two operations

Refusal in language models is mediated by a single direction in activation space ([Arditi et al., 2024](https://arxiv.org/abs/2406.11717)). Once you have measured that direction, two things become possible:

- **Remove it from the weights.** This is abliteration --- the model stops refusing. This is the attack.
- **Monitor it at runtime.** This is [CAST](../concepts/steering.md) --- the model's refusal behavior is enforced or modulated dynamically. This is the defense.

The measurement is the same. The direction is the same. The difference is what you do with the knowledge.

> **Direction** --- a unit vector in activation space that points along a behavioral axis. The "refusal direction" separates activations produced by harmful prompts from those produced by harmless ones. Projecting an activation onto this direction tells you how much the model "wants to refuse."

## The same geometry everywhere

This duality recurs at every level of the tool:

| Knowledge | Attack use | Defense use |
|-----------|-----------|-------------|
| Refusal direction | Cut it from weights (abliteration) | Monitor it at runtime (CAST, Guard) |
| Softprompt optimization | Find adversarial tokens that bypass defenses | Reveal which input regions defenses miss |
| Surface mapping | Identify prompts the model refuses incorrectly | Audit refusal coverage before deployment |
| SIC sanitization | Study what adversarial patterns look like | Rewrite adversarial inputs before they reach the model |
| Weight-diff measurement | Extract what alignment training added | Verify alignment was applied correctly |

The geometric object does not change. The intent does.

## Why both must live together

A defense that has never been tested against attacks is a defense you hope works. A safety audit that does not include adversarial stress-testing is an audit that measures the easy cases.

Concretely:

**Softprompt attacks reveal defense blind spots.** Running GCG or EGD optimization against a CAST-defended model produces adversarial tokens. Those tokens are diagnostic --- they show exactly which regions of activation space the defense fails to cover. The [defense-aware loss](../capabilities/stress-test-defenses.md) makes this explicit: the optimizer is penalized for triggering detection, so the tokens it finds are precisely the ones that slip through.

**Defense composition requires attack diversity.** CAST and SIC are complementary defenses with a negative correlation (r = -0.52). CAST catches what SIC misses, and vice versa. You only discover this by running diverse attacks and observing which defense catches which. A tool that only defends cannot tell you where its defense fails.

**Measurement is neutral.** Extracting a refusal direction from a model is neither attack nor defense. It is understanding. What you do with that understanding is a decision made in the TOML config, not in the code.

> **Abliteration** --- the process of removing a model's refusal behavior by projecting out the refusal direction from its weight matrices. The term combines "ablation" (surgical removal) with "obliteration."

## The historical parallel

Sebastien Le Prestre de Vauban was Louis XIV's chief military engineer. He directed 53 sieges and built or redesigned 300 fortresses. His fortification designs were superior precisely because he understood siege warfare from the attacker's perspective. He did not build walls and hope they held --- he built walls that addressed the specific ways he knew walls could fall.

The same logic applies to model safety. A defense informed by real attacks is qualitatively different from a defense built on assumptions about what attacks look like.

!!! warning "Responsible use"
    The attack capabilities exist to improve defenses, not to circumvent them in deployed systems. Vauban is a research instrument --- the understanding it produces is meant to make AI systems more reliable, not less.

## How this shapes the tool

Every pipeline module in Vauban can be understood through this lens:

- **Measure** produces knowledge (the direction).
- **Cut** and **Softprompt** consume that knowledge offensively.
- **CAST**, **SIC**, and **Guard** consume it defensively.
- **Probe**, **Surface**, and **Audit** inspect the knowledge itself.
- **Evaluate** checks the consequences of any operation.

The TOML config determines which combination runs. The code makes no judgment about which side you are on --- it provides the geometry and lets you decide.

## Related pages

- [Composability](composability.md) --- how these pieces fit together
- [Last-Mile Reliability](last-mile-reliability.md) --- why this matters for production
- [Reproducibility](reproducibility.md) --- how experiments stay traceable across attack and defense runs
