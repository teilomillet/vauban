# References

A consolidated bibliography of all papers, tooling repositories, and blog posts cited across the Spinning Up in Abliteration series. Entries are organized by topic and listed alphabetically within each section.

---

## Foundational

**Arditi, A., Ballard, O., Chakraborty, A., Heimersheim, S., Nanda, N.** (2024). "Refusal in Language Models Is Mediated by a Single Direction." *arXiv:2406.11717*. [arxiv.org/abs/2406.11717](https://arxiv.org/abs/2406.11717)
— The foundational paper. Demonstrates that refusal behavior in instruction-tuned LLMs is controlled by a single direction in activation space, and that removing it via rank-1 projection eliminates refusal while preserving general capabilities. Cited in Parts 1–7.

**"The Geometry of Refusal in Large Language Models."** (2025). *arXiv:2502.17420*. [arxiv.org/pdf/2502.17420](https://arxiv.org/pdf/2502.17420)
— Extends Arditi et al. with geometric analysis of the refusal surface across categories, styles, and languages. Establishes that refusal is not uniform — specific prompt axes exhibit distinct geometric signatures. Cited in Parts 1, 4.

**"An Embarrassingly Simple Defense Against LLM Abliteration Attacks."** (2025). *arXiv:2505.19056v1*. [arxiv.org/html/2505.19056v1](https://arxiv.org/html/2505.19056v1)
— Shows that distributing refusal across multiple orthogonal directions defeats standard rank-1 abliteration. Motivates the detection pipeline in Part 5. Cited in Part 5.

---

## Comparative Analysis

**Young, C.** (UNLV, 2024). "Comparative Analysis of LLM Abliteration Methods." *arXiv:2512.13655*. [arxiv.org/abs/2512.13655](https://arxiv.org/abs/2512.13655)
— First systematic benchmark of four abliteration tools (Heretic, DECCP, ErisForge, FailSpy) across 16 models. Key finding: single-pass methods preserve mathematical reasoning better than Bayesian optimization. Cited in Parts 5, 7.

**"GRP-Obliteration: Unaligning LLMs With a Single Unlabeled Prompt."** (2025). *arXiv:2602.06258*. [arxiv.org/pdf/2602.06258](https://arxiv.org/pdf/2602.06258)
— Uses GRPO to invert safety alignment. Outperforms abliteration and TwinBreak on attack success while preserving more utility. Works on diffusion models too. Cited in References.

---

## Soft Prompt Attacks

**Schwinn, L., Dobre, D., Xhonneux, S., Pfreundt, F., Günther, M., Hein, M.** (2024). "Soft Prompt Threats: Attacking Safety Alignment and Unlearning in Open-Source LLMs through the Embedding Space." *arXiv:2402.09063*. [arxiv.org/abs/2402.09063](https://arxiv.org/abs/2402.09063)
— Continuous embedding optimization bypasses alignment and unlearning. More efficient than discrete token attacks. Foundational reference for vauban's continuous soft prompt mode. Cited in Part 6.

**Zou, A., Wang, Z., Kolter, J. Z., Fredrikson, M.** (2023). "Universal and Transferable Adversarial Attacks on Aligned Language Models" (GCG). *ICML 2024. arXiv:2307.15043*. [arxiv.org/abs/2307.15043](https://arxiv.org/abs/2307.15043)
— Foundational greedy coordinate gradient descent for adversarial suffix optimization. Suffixes transfer to closed-source models. Reference for vauban's GCG mode. Cited in Part 6.

**Nordby, S.** (2025). "Soft Prompts for Evaluation: Measuring Conditional Distance of Capabilities." *arXiv:2505.14943*. [arxiv.org/abs/2505.14943](https://arxiv.org/abs/2505.14943)
— Optimized soft prompts as a quantitative safety metric; defines the accessibility score $A = \exp(-L_{\text{final}})$. Cited in Part 6.

**Huang, Y., et al.** (2025). "Optimizing Soft Prompt Tuning via Structural Evolution." *arXiv:2602.16500*. [arxiv.org/abs/2602.16500](https://arxiv.org/abs/2602.16500)
— Topological analysis of soft prompt convergence; embedding norm regularization improves stability. Cited in Part 6.

**"RAID: Refusal-Aware and Integrated Decoding for Jailbreaking LLMs."** (2025). *arXiv:2510.13901*. [arxiv.org/abs/2510.13901](https://arxiv.org/abs/2510.13901)
— Relaxes discrete tokens into continuous embeddings with a refusal-aware regularizer that steers away from refusal directions during optimization. Bridges soft prompt search with measured refusal directions. Cited in Part 6.

**"LARGO: Latent Adversarial Reflection through Gradient Optimization."** (2025). *NeurIPS 2025. arXiv:2505.10838*. [arxiv.org/abs/2505.10838](https://arxiv.org/abs/2505.10838)
— Latent vector optimization via gradient descent plus a self-reflective decoding loop. Outperforms AutoDAN by 44 ASR points. Cited in References.

**"COLD-Attack: Jailbreaking LLMs with Stealthiness and Controllability."** (2024). *ICML 2024. arXiv:2402.08679*. [arxiv.org/abs/2402.08679](https://arxiv.org/abs/2402.08679)
— Energy-based constrained decoding with Langevin dynamics for continuous prompt search under fluency and position constraints. Cited in References.

**"AmpleGCG: Learning a Universal and Transferable Generative Model of Adversarial Suffixes."** (2024). *arXiv:2404.07921*. [arxiv.org/abs/2404.07921](https://arxiv.org/abs/2404.07921)
— Trains a generator on intermediate GCG successes; produces hundreds of adversarial suffixes per query in minutes. Near-100% attack success rate. Cited in References.

**"EGD Attack: Universal and Transferable Adversarial Attack Using Exponentiated Gradient Descent."** (2025). *arXiv:2508.14853*. [arxiv.org/abs/2508.14853](https://arxiv.org/abs/2508.14853)
— Relaxed one-hot optimization with Bregman projection on the probability simplex. Cleaner convergence than GCG. Reference for vauban's EGD mode. Cited in Part 6.

**"UJA: Untargeted Jailbreak Attack."** (2025). *arXiv:2510.02999*. [arxiv.org/abs/2510.02999](https://arxiv.org/abs/2510.02999)
— First gradient-based untargeted jailbreak: maximizes probability of any unsafe response instead of a fixed target. 80%+ attack success rate in 100 iterations. Cited in Part 6.

**Geiping, J., Stein, A., Shu, M., Saadatpanah, K., Goldblum, M., Goldstein, T.** (2024). "Coercing LLMs to do and reveal (almost) anything." *arXiv:2402.14020*. [arxiv.org/abs/2402.14020](https://arxiv.org/abs/2402.14020)
— Systematizes adversarial objectives beyond jailbreaking: extraction, misdirection, DoS, control, and collision attacks. All solved with GCG under different loss formulations. Introduces token constraint sets and KL-divergence collision loss. Cited in Part 6.

---

## Latent Space Geometry

**"Latent Fusion Jailbreak: Blending Harmful and Harmless Representations to Elicit Unsafe LLM Outputs."** (2025). *arXiv:2508.10029*. [arxiv.org/abs/2508.10029](https://arxiv.org/abs/2508.10029)
— Fuses hidden states of harmful and benign queries in continuous latent space. The prompt-side dual of abliteration. Cited in References.

**"Exploiting Latent Space Discontinuities for Building Universal LLM Jailbreaks."** (2025). *arXiv:2511.00346*. [arxiv.org/abs/2511.00346](https://arxiv.org/abs/2511.00346)
— Identifies poorly-conditioned latent regions associated with low-frequency training data. Geometric complement to refusal-direction analysis. Cited in References.

**Shrivastava, A., Holtzman, A.** (2025). "Linearly Decoding Refused Knowledge." *arXiv:2507.00239*. [arxiv.org/abs/2507.00239](https://arxiv.org/abs/2507.00239)
— Refused information remains linearly decodable from hidden states via simple probes. Probes transfer from base to instruction-tuned models. Validates that refusal is a linear gate, not information erasure. Cited in Part 1.

---

## Defense

**"CAST: Programming Refusal with Conditional Activation Steering."** (2025). *ICLR 2025 Spotlight. arXiv:2409.05907*. [arxiv.org/abs/2409.05907](https://arxiv.org/abs/2409.05907)
— Context-dependent steering rules at inference time without weight modification. Cited in References.

**"RepBend: Representation Bending for Large Language Model Safety."** (2025). *ACL 2025. arXiv:2504.01550*. [arxiv.org/abs/2504.01550](https://arxiv.org/abs/2504.01550)
— Loss-based fine-tuning to push harmful activations apart from safe ones. The defense dual of abliteration. Cited in References.

**"SIC! Iterative Self-Improvement for Adversarial Attacks on Safety-Aligned LLMs."** (2025). *arXiv:2510.21057*. [arxiv.org/abs/2510.21057](https://arxiv.org/abs/2510.21057)
— Iterative input sanitization defense: detect adversarial content, rewrite to remove it, repeat until clean or block. Direction-aware variant uses refusal projection as detection signal. Cited in Part 6.

---

## Theoretical Foundations

**"C-AdvIPO: Efficient Adversarial Training in LLMs with Continuous Attacks."** (2024). *arXiv:2405.15589*. [arxiv.org/abs/2405.15589](https://arxiv.org/abs/2405.15589)
— Proves continuous embedding attacks are the fundamental threat model: robustness to them predicts robustness to discrete attacks. Theoretical justification for soft prompt attack research. Cited in Part 6.

**"Model Tampering Attacks Enable More Rigorous Evaluations of LLM Capabilities."** (2025). *TMLR 2025. arXiv:2502.05209*. [arxiv.org/abs/2502.05209](https://arxiv.org/abs/2502.05209)
— Capability elicitation via activation and weight modification. Robustness lies on a low-dimensional subspace. Unlearning undone in 16 fine-tuning steps. Cited in References.

**Casper, S., Xhonneux, L., et al.** (2024). "Latent Adversarial Training." *arXiv:2407.15549*. [arxiv.org/abs/2407.15549](https://arxiv.org/abs/2407.15549)
— Training against continuous latent perturbations improves robustness to jailbreaks with orders of magnitude less compute. Defines the threat model soft prompt attacks instantiate. Cited in References.

---

## Model Diffing and Weight Arithmetic

**Ilharco, G., Ribeiro, M. T., Wortsman, M., Gururangan, S., Schmidt, L., Hajishirzi, H., Farhadi, A.** (2023). "Editing Models with Task Arithmetic." *ICLR 2023*. [arxiv.org/abs/2212.04089](https://arxiv.org/abs/2212.04089)
— Introduces task vectors: the weight difference between a fine-tuned and pre-trained model encodes the task. Task vectors can be added, negated, and combined arithmetically. Foundation for weight-diff direction extraction. Cited in Part 8.

**Perin, G., et al.** (2025). "LoX: Understanding and Fixing Broken LLM Safety via Logit Extraction." *COLM 2025*.
— Demonstrates that SVD of weight diffs between base and aligned models extracts safety directions more completely than activation-based measurement. Negative application (amplification) hardens models against abliteration. Cited in Part 8.

**Lermen, S., et al.** (2025). "Weight Arithmetic Steering." *arXiv:2511.05408*. [arxiv.org/abs/2511.05408](https://arxiv.org/abs/2511.05408)
— Combines SVD of weight diffs across layers with arithmetic steering. Shows that weight-space directions capture distributed safety effects invisible to token-level activation probes. Cited in Part 8.

## Adaptive and Dual-Direction Steering

**AdaSteer.** (2025). *arXiv:2504.09466*. [arxiv.org/abs/2504.09466](https://arxiv.org/abs/2504.09466)
— Proposes separate detect and steer directions for conditional activation steering. The detect direction gates whether to intervene; the steer direction applies the correction. Cited in Part 8.

**TRYLOCK.** (2025). *arXiv:2601.03300*. [arxiv.org/abs/2601.03300](https://arxiv.org/abs/2601.03300)
— Identifies the non-monotonic danger zone in fixed-alpha steering: optimal alpha depends on projection magnitude. Proposes tiered alpha schedules. Cited in Part 8.

**AlphaSteer.** (2025). *arXiv:2506.07022*. [arxiv.org/abs/2506.07022](https://arxiv.org/abs/2506.07022)
— Adaptive alpha selection for activation steering based on per-token signal strength. Complements TRYLOCK's tier-based approach. Cited in Part 8.

---

## Depth Analysis

**Chen, Y., et al.** (2026). "Deep-Thinking in Large Language Models." *arXiv:2602.13517*. [arxiv.org/abs/2602.13517](https://arxiv.org/abs/2602.13517)
— Introduces the deep-thinking ratio (DTR) metric: measures how many tokens require all layers to settle. Establishes Jensen-Shannon divergence profiles across layers as a diagnostic tool. Cited in Part 5.

---

## Blog Posts and Tutorials

**Labonne, M.** (2024). "Abliteration Tutorial." *Hugging Face Blog*. [huggingface.co/blog/mlabonne/abliteration](https://huggingface.co/blog/mlabonne/abliteration)
— Step-by-step tutorial walking through abliteration with code. Accessible introduction for practitioners.

**Lai, J.** (2024). "Norm-Preserving Biprojected Abliteration." *Hugging Face Blog*. [huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration](https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration)
— Explains why naive projection removal shrinks weight norms and introduces the biprojected + norm-preserving fix. Cited in Part 3.

---

## Tooling Repositories

**Heretic** — Fully automatic abliteration with Optuna optimization.
[github.com/p-e-w/heretic](https://github.com/p-e-w/heretic)
— Reference for vauban's multi-objective optimization pipeline. Cited in Part 7.

**Blasphemer** — Heretic fork optimized for macOS / Apple Silicon.
[github.com/sunkencity999/blasphemer](https://github.com/sunkencity999/blasphemer)

**NousResearch/llm-abliteration** — Norm-preserving biprojected abliteration.
[github.com/NousResearch/llm-abliteration](https://github.com/NousResearch/llm-abliteration)
— Reference implementation for norm-preserving and biprojected cut variants. Cited in Part 3.

**jim-plus/llm-abliteration** — Original fork of the above.
[github.com/jim-plus/llm-abliteration](https://github.com/jim-plus/llm-abliteration)
