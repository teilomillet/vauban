<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Glossary

Quick-reference definitions for terms used throughout the Spinning Up in Abliteration series. Each entry notes which Part introduces the concept.

---

**Abliteration.** The technique of removing a model's refusal behavior by projecting out a direction in activation space from its weight matrices. The name is a portmanteau of "ablation" and "obliteration." Introduced in [Part 1](part1_what_is_abliteration.md).

**Accessibility score.** A scalar measuring how easy it is to bypass a model's safety alignment via soft prompt optimization: $A = \exp(-L_{\text{final}})$, where $L_{\text{final}}$ is the final attack loss. Higher means less safe. From Nordby (2025). Introduced in [Part 6](part6_attacks_and_defenses.md).

**Activation.** The vector produced by a layer (or the residual stream) at a specific token position during a forward pass. In vauban, activations are `mx.array` tensors of shape `(d_model,)`. Introduced in [Part 1](part1_what_is_abliteration.md).

**Alpha ($\alpha$).** The scaling factor for projection removal in the cut step. $\alpha = 1.0$ removes the full projection; $\alpha > 1.0$ overshoots (more aggressive removal at the cost of higher perplexity). Introduced in [Part 2](part2_your_first_abliteration.md).

**Biprojected.** A cut variant where the refusal direction is first orthogonalized against the harmless direction via Gram-Schmidt before projection removal. Preserves harmless-direction variance that standard cut may damage. Introduced in [Part 3](part3_under_the_hood.md).

**Causal LM.** A language model that generates tokens left-to-right, where each token attends only to previous tokens. All models in this series are causal LMs (GPT-style). Introduced in [Part 1](part1_what_is_abliteration.md).

**Cosine separation.** The difference in mean projection onto a direction between harmful and harmless activations: $s_l = \langle \mu_H^l, \hat{d}_l \rangle - \langle \mu_B^l, \hat{d}_l \rangle$. Used to select the best layer for cutting. Introduced in [Part 1](part1_what_is_abliteration.md).

**Coverage score.** The fraction of the theoretical 5D surface grid (category $\times$ style $\times$ language $\times$ turn_depth $\times$ framing) that contains at least one observed prompt. Measures how thoroughly a prompt set covers the refusal surface. Introduced in [Part 4](part4_the_refusal_surface.md).

**Cut.** The weight surgery step: removing a direction from `o_proj` and `down_proj` weight matrices via rank-1 projection. Formally: $W' = W - \alpha \cdot d \cdot d^\top W$. Introduced in [Part 2](part2_your_first_abliteration.md), derived in [Part 3](part3_under_the_hood.md).

**$d_{\text{model}}$.** The hidden dimension of the model (e.g., 2048 for Llama-3.2-1B, 3072 for Llama-3.2-3B). All directions, activations, and weight matrix rows live in $\mathbb{R}^{d_{\text{model}}}$. Introduced in [Part 1](part1_what_is_abliteration.md).

**DBDI (Decomposed Behavioral Direction Intervention).** A technique that separates the refusal direction into two components: HDD (harm detection direction) at the instruction-final token, and RED (refusal execution direction) at the sequence-final token. Cutting RED while preserving HDD lets the model recognize harmful content without refusing. Introduced in [Part 5](part5_going_deeper.md).

**Deep-thinking ratio (DTR).** The fraction of tokens whose representations settle only in the final layers of the model: $\text{DTR} = |\{t : \text{settling}(t) \geq \lceil(1-\rho) \cdot L\rceil\}| / T$. High DTR means the model needs all layers. From Chen et al. (2026). Introduced in [Part 5](part5_going_deeper.md).

**Detect.** Vauban's three-layer pipeline for determining whether a model has been hardened against abliteration. Checks geometry, DBDI probe, and abliteration resistance. Introduced in [Part 5](part5_going_deeper.md).

**Difference-in-means.** The method for extracting a behavioral direction: $d_l = \frac{1}{|H|} \sum_{p \in H} a_l(p) - \frac{1}{|B|} \sum_{p \in B} a_l(p)$. Computes the vector pointing from the harmless cluster to the harmful cluster in activation space. Introduced in [Part 1](part1_what_is_abliteration.md).

**Direction.** A unit vector in $\mathbb{R}^{d_{\text{model}}}$ that represents a behavioral axis (e.g., refusal, harm detection, depth). Stored as `mx.array` of shape `(d_model,)`. Introduced in [Part 1](part1_what_is_abliteration.md).

**`down_proj`.** The second linear layer in a transformer MLP block. Its output adds directly to the residual stream, making it a natural target for projection removal. Introduced in [Part 3](part3_under_the_hood.md).

**Effective rank.** Shannon entropy of normalized squared singular values: $\text{eff\_rank} = \exp(-\sum_i p_i \log p_i)$. A rank-1 matrix has effective rank 1.0; higher values indicate a spread-out spectrum (hardened model). Introduced in [Part 5](part5_going_deeper.md).

**EGD (Exponentiated Gradient Descent).** A soft prompt optimization mode that maintains a probability distribution over the vocabulary for each token position, using Bregman projection onto the probability simplex. Introduced in [Part 6](part6_attacks_and_defenses.md).

**Embedding space.** The continuous vector space where token embeddings live ($\mathbb{R}^{d_{\text{model}}}$). Soft prompt attacks optimize directly in this space, bypassing discrete token constraints. Introduced in [Part 6](part6_attacks_and_defenses.md).

**GCG (Greedy Coordinate Gradient).** A discrete token optimization method that uses gradient information to identify promising token substitutions, then evaluates candidates in batches. From Zou et al. (2023). Introduced in [Part 6](part6_attacks_and_defenses.md).

**Grassmann distance.** The $L_2$ norm of principal angles between two subspaces: $d_G(U, V) = \|\theta\|_2$. Zero if and only if the subspaces are identical. Used to compare HDD and RED subspaces. Introduced in [Part 5](part5_going_deeper.md).

**HDD (Harm Detection Direction).** The component of DBDI extracted at the instruction-final token position. Represents the model's ability to recognize harmful content. Introduced in [Part 5](part5_going_deeper.md).

**JSD (Jensen-Shannon Divergence).** A symmetric divergence measure: $\text{JSD}(P, Q) = \frac{1}{2} \text{KL}(P \| M) + \frac{1}{2} \text{KL}(Q \| M)$, where $M = (P+Q)/2$. Bounded in $[0, \ln 2]$. Used to compare logit distributions across layers in depth analysis. Introduced in [Part 5](part5_going_deeper.md).

**KL divergence.** Kullback-Leibler divergence between token distributions of original and modified models: $\text{KL}(P \| Q) = \sum_v P(v) \cdot \log(P(v) / Q(v))$. Measures how much the modification changed the model's output distribution. Introduced in [Part 3](part3_under_the_hood.md).

**Layer selection.** The strategy for choosing which layers to cut. Options: `"all"` (every layer), `"above_median"` (layers with above-median cosine separation), `"top_k"` (the $k$ highest-scoring layers). Introduced in [Part 3](part3_under_the_hood.md).

**MLX.** Apple's machine learning framework for Apple Silicon. Pure eager execution with unified CPU/GPU memory. Vauban's runtime. Introduced in [Part 2](part2_your_first_abliteration.md).

**Norm-preserve.** A cut variant that rescales each modified weight row to its original norm after projection removal: $w'_i = w'_i \cdot (\|w_i\| / \|w'_i\|)$. Prevents the norm shrinkage caused by naive projection removal. Introduced in [Part 3](part3_under_the_hood.md).

**`o_proj`.** The output projection in a transformer attention block. Its output adds directly to the residual stream, making it a natural target for projection removal alongside `down_proj`. Introduced in [Part 3](part3_under_the_hood.md).

**Perplexity.** An evaluation metric measuring how well the model predicts a held-out text: $\text{PPL} = \exp\left(-\frac{1}{T} \sum_t \log P(x_t | x_{<t})\right)$. Lower is better. A large perplexity increase after cutting indicates capability damage. Introduced in [Part 3](part3_under_the_hood.md).

**Probe.** Running a forward pass and measuring per-layer projections onto a direction. Reveals where in the network a concept (refusal, harm, depth) is active. Introduced in [Part 2](part2_your_first_abliteration.md).

**Projection.** The component of an activation vector along a direction: $\text{proj}(a, d) = \langle a, d \rangle \cdot d$. Positive projection onto the refusal direction indicates the model is about to refuse. Introduced in [Part 1](part1_what_is_abliteration.md).

**RAID (Refusal-Aware and Integrated Decoding).** A direction-guided soft prompt loss that penalizes positive projection onto the refusal direction: $L_{\text{raid}} = w \cdot \max(0, \langle h, d \rangle)$. Bridges soft prompt attacks with measured refusal directions. Introduced in [Part 6](part6_attacks_and_defenses.md).

**RED (Refusal Execution Direction).** The component of DBDI extracted at the sequence-final token position. Represents the model's decision to refuse. Introduced in [Part 5](part5_going_deeper.md).

**Residual stream.** The main information highway through a transformer: each layer reads from and writes to this shared vector. At any point, it is a $d_{\text{model}}$-dimensional vector. Introduced in [Part 1](part1_what_is_abliteration.md).

**Settling depth.** The earliest layer at which a token's logit distribution is within $\gamma$ JSD of the final layer's distribution: $\text{settling}(t) = \min\{l : \text{JSD}(P_L, P_l) \leq \gamma\}$. Tokens with high settling depth are "deep-thinking" tokens. Introduced in [Part 5](part5_going_deeper.md).

**SIC (Iterative Self-Improvement for Adversarial Attacks).** An input sanitization defense: detect adversarial content, rewrite to remove it, repeat until clean or block. In vauban, uses either direction-based (projection score) or generation-based (phrase matching) detection. Introduced in [Part 6](part6_attacks_and_defenses.md).

**Silhouette score.** A clustering quality metric measuring how well harmful and harmless activations separate at each layer. Higher values indicate cleaner separation. Used as an alternative to cosine separation for layer selection. Introduced in [Part 3](part3_under_the_hood.md).

**Soft prompt.** A sequence of learnable embedding vectors prepended to the input. Unlike discrete prompts, soft prompts are optimized via gradient descent in continuous embedding space. Introduced in [Part 6](part6_attacks_and_defenses.md).

**Sparsity.** The fraction of direction components zeroed out before cutting: `sparsity=0.9` keeps only the top 10% by absolute value. Reduces the rank-1 update's footprint, potentially preserving more capability. Introduced in [Part 3](part3_under_the_hood.md).

**Steer.** Generating text while removing a direction at specified layers during inference. Unlike cut (which modifies weights permanently), steering is applied per-generation via KV-cache intervention. Introduced in [Part 2](part2_your_first_abliteration.md).

**Subspace.** A multi-dimensional generalization of a single direction. Extracted via SVD of the difference matrix between harmful and harmless activations. The top-$k$ right singular vectors form an orthonormal basis for the refusal subspace. Introduced in [Part 5](part5_going_deeper.md).

**Surface.** The 5D grid of prompt attributes (category, style, language, turn_depth, framing) over which refusal behavior is mapped. A surface scan reveals which combinations of attributes trigger refusal. Introduced in [Part 4](part4_the_refusal_surface.md).

**Welford's algorithm.** An online algorithm for computing running means in a single pass with $O(d_{\text{model}})$ memory per layer: $\mu_n = \mu_{n-1} + (x_n - \mu_{n-1}) / n$. Used during activation collection to avoid storing all activations. Introduced in [Part 3](part3_under_the_hood.md).

**Winsorization.** Clipping activations at a given quantile to tame outliers ("massive activations"). Controlled by the `clip_quantile` parameter. For example, `clip_quantile=0.01` clips the top and bottom 1% of activation components. Introduced in [Part 3](part3_under_the_hood.md).
