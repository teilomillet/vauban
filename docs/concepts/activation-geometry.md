---
title: "Activation Geometry — How LLM Behavior Lives in Vector Space"
description: "Activations encode model behavior as directions in high-dimensional space. Learn how the residual stream, linear representations, and geometric structure enable measuring and modifying LLM behavior."
keywords: "activation geometry, LLM activations, residual stream, activation space, transformer internals, mechanistic interpretability"
---

<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Activation Geometry

How behavior is encoded in the geometry of a neural network's internal representations.

## Every forward pass produces geometry

When a transformer processes a prompt, every token at every layer produces a vector in $\mathbb{R}^{d_{\text{model}}}$. For Qwen2.5-1.5B, that is a 1536-dimensional vector. For Llama-3.2-3B, 3072 dimensions. These vectors are **activations** --- the internal state of the model at a specific position and depth.

> **Activation** --- the vector a layer produces at a given token position during a forward pass. Think of it as a snapshot of everything the model "knows" at that point in the computation. In Vauban, activations are arrays of shape `(d_model,)`.

The collection of all possible activation vectors across all prompts forms the model's **activation space**. This space has geometric structure: prompts that the model treats similarly cluster together, and prompts it treats differently are separated.

## The residual stream

A transformer is a stack of layers. Each layer reads a vector, transforms it through attention and MLP blocks, and **adds** the result back:

$$h^{l+1} = h^l + \text{Attn}^l(h^l) + \text{MLP}^l(h^l)$$

The vector $h^l$ is the **residual stream** at layer $l$. It is the main information highway through the model --- a running sum that accumulates contributions from every layer below.

> **Residual stream** --- the shared vector that all layers read from and write to. Each layer's output is *added* to the stream rather than replacing it. This additive structure is what makes surgical interventions possible: you can modify what specific layers write without disrupting everything else.

Two components write into the residual stream at each layer:

- **`o_proj`** --- the attention output projection. Writes the result of multi-head attention.
- **`down_proj`** --- the MLP output projection. Writes the result of the feedforward block.

These are the intervention points for [abliteration](abliteration.md) and [steering](steering.md).

## Behavior is geometric

The central observation behind Vauban: **behavioral properties correspond to directions in activation space**.

When a model is about to refuse, its activations at the final token position have a large positive component along a specific direction. When it is about to comply, that component is small or negative. The direction does not encode *what* the model will say --- it encodes *whether it will refuse*.

This is not unique to refusal. Honesty, sentiment, style, and other high-level properties also correspond to directions. The [linear representation hypothesis](linear-representation.md) provides the theoretical backing.

## Why geometry matters

If behavior is geometric, you can:

| Operation | Geometric action | Vauban module |
|---|---|---|
| **Measure** it | Extract the direction via [difference-in-means](measurement.md) | `measure` |
| **Remove** it | Project the direction out of weight matrices | `cut` ([abliteration](abliteration.md)) |
| **Monitor** it | Compute projection magnitude during generation | `cast` ([steering](steering.md)) |
| **Steer** it | Add or subtract the direction from activations at runtime | `steer` |
| **Detect** it | Check projection as a signal for adversarial input | `sic` ([defense](defense-complementarity.md)) |
| **Map** it | Scan many prompts and record projection + refusal decisions | `surface` ([surface mapping](surface-mapping.md)) |

All of these are linear algebra operations on the same underlying geometric objects. The [refusal direction](refusal-direction.md) is the most studied instance, but the framework applies to any linearly represented concept.

## The last-token bottleneck

In a causal language model, each token attends only to tokens before it. The **last token position** is the only one that has seen the entire prompt. This is where the model's decision to comply or refuse is concentrated --- it is the bottleneck through which all prompt information flows before generation begins.

Formally, given a prompt of length $T$:

$$a_l = h^l_T \in \mathbb{R}^{d_{\text{model}}}$$

Most [measurement](measurement.md) modes collect activations at this position. The exception is DBDI, which separates the instruction-final token (where harm *detection* peaks) from the sequence-final token (where refusal *execution* peaks).

## MLX makes this native

MLX's eager execution model means activation capture is ordinary Python. There are no hooks, no framework abstractions, no registration APIs. The forward pass is a for-loop over layers, and you read, project, or modify tensors inline:

```python
for i, layer in enumerate(model.layers):
    x = layer(x, mask=mask)
    # x is a real mx.array --- project, steer, record, whatever you need
```

This is why Vauban uses MLX: the geometry is directly accessible at every point in the computation.
