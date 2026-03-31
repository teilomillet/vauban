---
title: "Papers — Academic Foundations for LLM Safety Research"
description: "Bibliography of foundational papers on abliteration, soft prompt attacks, defense mechanisms, weight arithmetic, adaptive steering, and latent space geometry."
keywords: "LLM safety papers, abliteration papers, Arditi et al, GCG paper, CAST paper, activation steering research"
---

<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Papers

Academic foundations organized by topic area. Each entry: authors, year, title, one-sentence summary, link.

---

## Foundational

**Arditi et al. (2024)** — "Refusal in Language Models Is Mediated by a Single Direction." The paper that started it all: refusal is a linear direction in activation space, and removing it disables refusal. [arXiv:2406.11717](https://arxiv.org/abs/2406.11717)

**Anonymous (2025)** — "The Geometry of Refusal in Large Language Models." Extends Arditi with geometric analysis of the refusal manifold across model families. [arXiv:2502.17420](https://arxiv.org/pdf/2502.17420)

**Anonymous (2025)** — "An Embarrassingly Simple Defense Against LLM Abliteration Attacks." Shows that abliteration can be defended against with straightforward techniques. [arXiv:2505.19056](https://arxiv.org/html/2505.19056v1)

**Young (2024)** — "Comparative Analysis of LLM Abliteration Methods." First systematic benchmark of 4 tools (Heretic, DECCP, ErisForge, FailSpy) across 16 models. Single-pass methods preserve math reasoning better than Bayesian optimization. [arXiv:2512.13655](https://arxiv.org/abs/2512.13655)

## Soft prompt attacks

**Zou et al. (2023)** — "Universal and Transferable Adversarial Attacks on Aligned Language Models" (GCG). Foundational greedy coordinate gradient descent for adversarial suffix optimization. Suffixes transfer to closed-source models. ICML 2024. [arXiv:2307.15043](https://arxiv.org/abs/2307.15043)

**Schwinn et al. (2024)** — "Soft Prompt Threats: Attacking Safety Alignment and Unlearning in Open-Source LLMs through the Embedding Space." Continuous embedding optimization bypasses alignment more efficiently than discrete token attacks. [arXiv:2402.09063](https://arxiv.org/abs/2402.09063)

**Nordby (2025)** — "Soft Prompts for Evaluation: Measuring Conditional Distance of Capabilities." Optimized soft prompts as a quantitative safety metric; accessibility scoring. [arXiv:2505.14943](https://arxiv.org/abs/2505.14943)

**Huang et al. (2025)** — "Optimizing Soft Prompt Tuning via Structural Evolution." Topological analysis of soft prompt convergence; embedding norm regularization. [arXiv:2602.16500](https://arxiv.org/abs/2602.16500)

**EGD — Anonymous (2025)** — "Universal and Transferable Adversarial Attack Using Exponentiated Gradient Descent." Relaxed one-hot optimization with Bregman projection on the probability simplex. Cleaner convergence than GCG. [arXiv:2508.14853](https://arxiv.org/abs/2508.14853)

**RAID — Anonymous (2025)** — "Refusal-Aware and Integrated Decoding for Jailbreaking LLMs." Relaxes discrete tokens into continuous embeddings with a refusal-aware regularizer that steers away from refusal directions during optimization. [arXiv:2510.13901](https://arxiv.org/abs/2510.13901)

**LARGO — Anonymous (2025)** — "Latent Adversarial Reflection through Gradient Optimization." Latent vector optimization via gradient descent + self-reflective decoding loop. Outperforms AutoDAN by 44 ASR points. NeurIPS 2025. [arXiv:2505.10838](https://arxiv.org/abs/2505.10838)

**COLD-Attack — Anonymous (2024)** — "Jailbreaking LLMs with Stealthiness and Controllability." Energy-based constrained decoding with Langevin dynamics for continuous prompt search under fluency/position constraints. ICML 2024. [arXiv:2402.08679](https://arxiv.org/abs/2402.08679)

**AmpleGCG — Anonymous (2024)** — "Learning a Universal and Transferable Generative Model of Adversarial Suffixes." Trains a generator on intermediate GCG successes; produces hundreds of adversarial suffixes per query in minutes. [arXiv:2404.07921](https://arxiv.org/abs/2404.07921)

**UJA — Anonymous (2025)** — "Untargeted Jailbreak Attack." First gradient-based untargeted jailbreak: maximizes probability of *any* unsafe response instead of a fixed target. [arXiv:2510.02999](https://arxiv.org/abs/2510.02999)

**Geiping et al. (2024)** — "Coercing LLMs to do and reveal (almost) anything." Systematizes adversarial objectives beyond jailbreaking: extraction, misdirection, DoS, control, and collision attacks, all solved with GCG under different loss formulations. [arXiv:2402.14020](https://arxiv.org/abs/2402.14020)

## Defense

**CAST — Anonymous (2025)** — "Programming Refusal with Conditional Activation Steering." Context-dependent steering rules at inference time without weight modification. ICLR 2025 Spotlight. [arXiv:2409.05907](https://arxiv.org/abs/2409.05907)

**RepBend — Anonymous (2025)** — "Representation Bending for Large Language Model Safety." Loss-based fine-tuning to push harmful activations apart from safe ones. The defense dual of abliteration. ACL 2025. [arXiv:2504.01550](https://arxiv.org/abs/2504.01550)

**SIC — Anonymous (2025)** — "SIC! Iterative Self-Improvement for Adversarial Attacks on Safety-Aligned LLMs." Iterative input sanitization: detect, rewrite, repeat. Direction-aware variant uses refusal projection as detection signal. [arXiv:2510.21057](https://arxiv.org/abs/2510.21057)

**Casper, Xhonneux et al. (2024)** — "Latent Adversarial Training." Training against continuous latent perturbations improves robustness to jailbreaks with orders of magnitude less compute. [arXiv:2407.15549](https://arxiv.org/abs/2407.15549)

## Weight arithmetic

**Ilharco et al. (2023)** — "Editing Models with Task Arithmetic." Weight diffs between fine-tuned and base models encode tasks; can be added, negated, or combined. Foundation for weight-diff direction extraction. ICLR 2023. [arXiv:2212.04089](https://arxiv.org/abs/2212.04089)

**Perin et al. (2025)** — LoX. SVD of weight diffs extracts safety directions more completely than activation-based measurement. Negative application amplifies safety (hardening). COLM 2025.

**Lermen et al. (2025)** — "Weight Arithmetic Steering." Combines SVD of weight diffs across layers with arithmetic steering. Captures distributed safety effects. [arXiv:2511.05408](https://arxiv.org/abs/2511.05408)

## Adaptive and dual-direction steering

**AdaSteer — Anonymous (2025)** — Separate detect and steer directions for conditional activation steering. [arXiv:2504.09466](https://arxiv.org/abs/2504.09466)

**TRYLOCK — Anonymous (2025)** — Identifies non-monotonic danger zones in fixed-alpha steering; proposes tiered alpha schedules. [arXiv:2601.03300](https://arxiv.org/abs/2601.03300)

**AlphaSteer — Anonymous (2025)** — Adaptive alpha selection based on per-token signal strength. [arXiv:2506.07022](https://arxiv.org/abs/2506.07022)

**Li, Li & Huang (2026)** — "Steering Vector Fields." Learns a differentiable boundary MLP whose gradient gives the steering direction at each activation. Context-dependent, multi-layer coordinated, replaces static vectors. [arXiv:2602.01654](https://arxiv.org/abs/2602.01654)

**Han et al. (2026)** — "Steer2Adapt." Reusable semantic subspace + Bayesian optimization of linear combinations of basis vectors for few-shot adaptation. [arXiv:2602.07276](https://arxiv.org/abs/2602.07276)

**Xiong et al. (2026)** — "Steering Externalities." Benign steering vectors (format compliance, JSON) erode safety margins; jailbreak ASR jumps to >80%. Safety margin is a finite resource consumed by any steering. [arXiv:2602.04896](https://arxiv.org/abs/2602.04896)

## Latent space geometry

**Anonymous (2025)** — "Blending Harmful and Harmless Representations to Elicit Unsafe LLM Outputs" (Latent Fusion Jailbreak). Fuses hidden states of harmful + benign queries in continuous latent space. The prompt-side dual of abliteration. [arXiv:2508.10029](https://arxiv.org/abs/2508.10029)

**Anonymous (2025)** — "Exploiting Latent Space Discontinuities for Building Universal LLM Jailbreaks." Identifies poorly-conditioned latent regions associated with low-frequency training data. [arXiv:2511.00346](https://arxiv.org/abs/2511.00346)

**Shrivastava & Holtzman (2025)** — "Linearly Decoding Refused Knowledge." Refused information remains linearly decodable from hidden states via simple probes. Probes transfer from base to instruction-tuned models. Validates that refusal is a linear gate, not information erasure. [arXiv:2507.00239](https://arxiv.org/abs/2507.00239)

## Encoding-based attacks

**Huang, Li & Tang (2024)** — "Endless Jailbreaks with Bijection Learning." Teaches LLMs random in-context ciphers to encode harmful queries, bypassing token-level safety filters entirely. More capable models are *more* vulnerable. [arXiv:2410.01294](https://arxiv.org/abs/2410.01294)

## Theoretical

**C-AdvIPO — Anonymous (2024)** — "Efficient Adversarial Training in LLMs with Continuous Attacks." Proves continuous embedding attacks are the fundamental threat model: robustness to them predicts robustness to discrete attacks. [arXiv:2405.15589](https://arxiv.org/abs/2405.15589)

**Anonymous (2025)** — "Model Tampering Attacks Enable More Rigorous Evaluations of LLM Capabilities." Capability elicitation via activation/weight modification. Robustness lies on a low-dimensional subspace. Unlearning undone in 16 fine-tuning steps. TMLR 2025. [arXiv:2502.05209](https://arxiv.org/abs/2502.05209)

**GRP-Obliteration — Anonymous (2025)** — "Unaligning LLMs With a Single Unlabeled Prompt." Uses GRPO to invert safety alignment. Outperforms abliteration and TwinBreak on attack success while preserving more utility. [arXiv:2602.06258](https://arxiv.org/pdf/2602.06258)
