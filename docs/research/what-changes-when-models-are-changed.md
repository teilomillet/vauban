---
title: "What Changes When Models Are Changed?"
description: "The Vauban thesis: model transformations are the object, access-aware auditing is the method, and Model Behavior Change Reports are the artifact."
keywords: "model transformations, behavioral diffing, model behavior change report, access-aware auditing, model diffing, Vauban"
---

<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# What Changes When Models Are Changed?

Models are no longer static artifacts. They are repeatedly transformed:
fine-tuned, merged, quantized, wrapped in prompt templates, steered at runtime,
post-trained, and replaced by new checkpoints.

Every one of those operations is a behavioral change event.

Vauban's thesis is:

> Model transformations are the object. Access-aware auditing is the method.
> Model Behavior Change Reports are the artifact.

The question is not just whether the target metric improved. The question is
what else changed.

## The Object: Model Transformations

Vauban should focus on the operations people actually perform on models:

- Fine-tuning a base model for a task.
- Running a post-training or reinforcement loop.
- Updating from checkpoint 1200 to checkpoint 2000.
- Merging adapters or averaging compatible weights.
- Quantizing a model for local or cheaper deployment.
- Changing a system prompt or prompt template.
- Applying controlled steering or runtime interventions.
- Replacing a production endpoint with a new release.

These transformations create value, but they also create behavioral risk. A
fine-tune can improve the benchmark and make the model more overconfident. A
quantization pass can preserve perplexity and still shift refusal behavior. A
prompt wrapper can improve tool use and weaken instruction hierarchy. A merged
model can inherit capabilities and side effects from both parents.

That is why the central Vauban question is:

> What changed when the model changed?

## The Method: Access-Aware Auditing

The hard part is that auditors rarely have perfect access.

Sometimes you have both the base and transformed weights. Sometimes you have one
local model. Sometimes you only have an API endpoint. Sometimes you only have
old logs and new logs. Sometimes logprobs are available; often they are not.

The report must not make the same claim from all of those situations.

| Access | What the report can support |
|---|---|
| One model or endpoint snapshot | A behavioral profile. |
| Two output traces or run reports | A black-box behavioral diff. |
| Endpoint with logprobs | A distributional diff. |
| Local weights and activations | Activation diagnostics. |
| Base plus transformed model | A stronger model-change audit. |

This is the no-base-model problem in practical form. It is not the whole topic.
It is the discipline that prevents overclaiming.

With only one model, Vauban can say: "under this suite, the model behaved this
way." With paired outputs, Vauban can say: "observed behavior changed." With
activations, Vauban can say: "this internal signal shifted alongside the
behavior." With base plus transformed weights, Vauban can say more about the
transformation itself.

That access ladder should be visible in the report.

## The Artifact: Model Behavior Change Reports

The output should be a readable report, not only a plot or a metric dump.

A Model Behavior Change Report should answer:

- What transformation was audited?
- What access was available?
- What suite or traces were used?
- What behavior changed?
- What side effects appeared?
- What internal evidence exists, if internals were available?
- What examples are safe to show?
- What limitations constrain the claim?
- What should a deployer do next?

The report should make deployment-relevant changes legible:

- Target-task performance improved.
- Refusal behavior changed.
- Over-refusal increased in ambiguous benign cases.
- Uncertainty expression decreased.
- The model became more assertive under underspecification.
- Activation diagnostics suggest a stronger refusal-associated direction in
  later layers.
- Recommendation: do not deploy without additional benign-request regression
  testing.

That is a behavioral changelog for a model update.

## Why Not Just "Model Diffing"?

Model diffing is the technical neighborhood, but it should not be the whole
brand.

Pure model diffing often asks: what internal differences exist between two
models? Vauban asks a more applied question:

> What changed behaviorally, where is there evidence for that change, and what
> can we responsibly claim from the access we have?

That keeps Vauban useful when deep mechanistic diffing is unavailable, too
expensive, or not yet mature enough for the deployment question at hand.

When internals are available, activation-space diagnostics should strengthen the
report. When they are not, the report should still be useful, but more modest.

## The Vauban Primitive Stack

The conceptual primitives are simple:

1. **Transformation** — the thing that changed: fine-tune, merge, quantization,
   checkpoint update, prompt wrapper, steering intervention, or endpoint update.
2. **Suite or trace** — the prompts, categories, logs, or examples used to
   observe behavior.
3. **Profile** — a structured description of one model's behavior under the
   suite.
4. **Diff** — a comparison between two profiles or run artifacts.
5. **Diagnostic** — optional internal evidence when weights or activations are
   available.
6. **Report** — the artifact that ties findings, limitations, reproducibility,
   and recommendations together.

The CLI remains TOML-first. The durable unit should be a shareable config and a
report artifact, not an ad hoc notebook.

```toml
[behavior_report]
title = "Model Behavior Change Report"
target_change = "checkpoint-1200 -> checkpoint-2000"
findings = [
  "Target-task performance improved.",
  "Over-refusal increased in ambiguous benign cases.",
  "Uncertainty expression decreased under underspecification.",
]
recommendation = "Run additional benign-request regression testing before deployment."
```

## What This Means for Vauban

Every major feature should serve one of these questions:

- What behavior changed?
- Under what conditions did it change?
- Where is there internal evidence of that change?
- Is the change stable across prompts, prompt families, models, or runs?
- Is the change desirable, undesirable, or ambiguous?
- Can the finding be reproduced?

If a feature does not improve the report, it is probably secondary.

That does not mean Vauban should abandon activation steering, refusal
directions, probes, guards, or stress tests. It means those tools should be
subordinate to the artifact. They are instruments for building better behavioral
reports.

The first-order identity is therefore:

> Vauban produces access-aware behavioral reports of model transformations.

