---
title: "Stress-Test LLM Defenses — GCG, EGD, GAN Loop Attacks"
description: "Adversarial optimization to find defense blind spots. GCG discrete tokens, EGD continuous relaxation, GAN attack-defense loops, defense-aware loss, and transfer re-ranking."
keywords: "LLM stress test, adversarial attack LLM, GCG attack, EGD optimization, red team AI, jailbreak testing, defense evasion"
---

<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Stress-Test Defenses

Defenses that are never attacked are untested assumptions. Vauban includes a full adversarial toolkit for probing defense boundaries — the same tools used to build defenses are available to break them.

## Gradient-based token optimization

The **Softprompt** module optimizes learnable token sequences that bypass model defenses. Two core algorithms:

**GCG** (Greedy Coordinate Gradient): discrete token optimization. At each step, compute the gradient of the attack loss with respect to every candidate token substitution, score the top-$k$ candidates, and keep the best. The search operates directly in token space — each candidate is a valid token sequence.

> **GCG** — Greedy Coordinate Gradient descent (Zou et al., 2023). Iteratively substitutes tokens to minimize a target loss. The "greedy" part: only one position is changed per step. The "coordinate" part: all vocab tokens are scored at the candidate position via a gradient.

**EGD** (Exponentiated Gradient Descent): continuous relaxation on the probability simplex. Instead of discrete token swaps, EGD maintains a distribution over the vocabulary at each position and updates via Bregman projection. Smoother optimization landscape, sometimes finds solutions GCG misses.

> **Gradient** — a vector that points in the direction of steepest increase of a function. In optimization, gradients tell you "which small change would improve the result the most." GCG and EGD both use gradients of a loss function to figure out which token substitutions are most promising.

> **Probability simplex** — the set of all valid probability distributions over vocabulary tokens. Each position sums to 1.0 (100%). EGD works with "soft" token assignments (e.g., 60% token A, 30% token B, 10% token C) rather than hard picks, giving the optimizer a smoother landscape to search.

Both algorithms support:

- **Token position**: prefix, suffix, or infix. Position determines where the adversarial tokens are placed relative to the user's prompt.
- **Defense-aware loss**: an auxiliary loss term that penalizes tokens triggering defense detection. The attack explicitly minimizes SIC score and CAST intervention count alongside the primary jailbreak objective.
- **Transfer re-ranking**: score top candidates on one or more transfer models and keep those that generalize. Adversarial tokens that work only on the optimization model are less interesting than those that transfer.
- **Paraphrase augmentation**: rephrase attack prompts across optimization steps to prevent overfitting to exact wording.

## The GAN loop

The **GAN loop** is an iterative attack-defense protocol. Each round: the attacker optimizes a token sequence, the defender (SIC + CAST) tries to block it, and both sides receive feedback for the next round.

> **GAN loop** — not a generative adversarial network in the neural sense. The name describes the iterative dynamic: attacker generates, defender evaluates, attacker adapts. Each round warm-starts from the previous round's best result.

Escalation dynamics across rounds:

- Attack parameters can increase: more optimization steps, higher direction weight, more tokens.
- Defense parameters can harden: lower thresholds, more SIC iterations.
- Multi-turn threading: the conversation history accumulates across rounds, so each round optimizes against the full prior context.

Empirical findings from 24 GAN configurations: **the GAN loop never degrades attack performance** (0/24 configs showed degradation). The average improvement is a CAST score reduction of -0.10, with infix placement benefiting most (delta = -0.121).

## Position dominance

The single most important variable in adversarial token optimization is **where the tokens are placed**.

| Position | CAST evasion rate | Effect size |
|---|---|---|
| Infix | 82% | Baseline |
| Prefix | 12% | $d = -2.93$ vs infix (CAST) |
| Suffix | 4% | $d = -4.69$ vs infix (interventions) |

> **Infix** — adversarial tokens inserted *within* the user's prompt (e.g., between sentences), rather than prepended (prefix) or appended (suffix). Infix tokens reduce CAST interventions by 6x compared to suffix tokens because they are processed in the middle of the prompt's semantic context, making them harder for activation-based defenses to isolate.

Position explains 67.2% of CAST score variance ($\eta^2 = 0.672$). The next strongest factor (GAN loop) explains 19.2%. This is a decisive result: **suffix optimization is a dead end for CAST evasion** (0/15 configs evaded), while infix optimization succeeds reliably.

> **$\eta^2$ (eta-squared)** — the fraction of total variation in an outcome that is explained by a single factor. $\eta^2 = 0.672$ means position alone accounts for 67.2% of why some attacks evade CAST and others do not. This is a very large effect — in social science, $\eta^2 > 0.14$ is considered "large."

## Additional attack vectors

**Jailbreak templates**: prompt-level attacks that do not require gradients. DAN, hypothetical framing, reasoning chain exploitation, role-play instructions. These work at any access level, including endpoint-only.

**Fusion**: latent-space blending of harmful and harmless prompt representations. At a target layer, the hidden states of a harmful prompt and a benign prompt are interpolated: $h_{\text{fused}} = \lambda \cdot h_{\text{harmful}} + (1-\lambda) \cdot h_{\text{benign}}$. This is the prompt-side dual of abliteration — instead of removing the refusal direction from weights, it dilutes the harmful signal in the activation.

**COLD-Attack**: energy-based constrained decoding with Langevin dynamics. Searches for adversarial prompts in continuous space under fluency and position constraints.

**LARGO**: latent vector optimization via gradient descent combined with a self-reflective decoding loop. Operates in latent space rather than token space.

**AmpleGCG**: trains a generator on intermediate GCG successes, then produces hundreds of adversarial suffixes per query in minutes. Trades optimization time for coverage.

## Key research findings

Beyond position dominance, several findings shape how to interpret stress-test results:

- **Defense complementarity reverses by model size.** On Qwen3.5-0.8B, SIC blocks 100% but CAST blocks 0%. On Qwen3.5-4B, CAST blocks 100% but SIC is weaker. Always test on your target model.
- **Escalation hurts.** Enabling defense escalation in the GAN loop *reduces* attack effectiveness ($d = +1.53$). Stable parameters outperform escalating ones.
- **Fast convergence predicts failure.** Attack configurations that converge quickly (low loss early) have a 22% evasion rate. Slow-start configurations that take longer to find their footing achieve 80% evasion. Early success is a local minimum trap.
- **Loss and evasion are misaligned.** Configurations with the lowest optimization loss have the *worst* defense evasion. The loss function measures proximity to a target output; the defense measures activation geometry. These are different axes.

## Access requirements

Gradient-based attacks (GCG, EGD, fusion, COLD, LARGO) require **full weight access** — the gradients must be computed through the model's forward and backward passes.

Jailbreak templates and API evaluation work with **endpoint access** — they operate entirely at the prompt level.

See [Access Levels](access-levels.md) for the complete matrix.
