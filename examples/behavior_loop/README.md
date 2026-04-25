# Behavior Loop With Runtime Sidecars

This example demonstrates Vauban's local model-change report loop:

1. collect a baseline behavior trace,
2. collect a candidate behavior trace,
3. diff the traces into a Model Behavior Change Report,
4. include runtime sidecar coverage for logits, logprobs, and activation summaries.

Run from the repository root:

```bash
vauban examples/behavior_loop/baseline_trace.toml
vauban examples/behavior_loop/candidate_trace.toml
vauban examples/behavior_loop/diff.toml
```

The trace configs use a small shared safe suite and write outputs under
`output/examples/behavior_loop`. Runtime sidecars are opt-in and summary-only:
they record which prompts, logprobs, and activation-layer summaries are
available, not raw activation tensors.
