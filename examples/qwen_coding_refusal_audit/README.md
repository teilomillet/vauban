# Qwen Coding Refusal Audit

This example runs a safe, small behavior-diff loop for a local Qwen3.6-27B
model:

1. collect a baseline trace,
2. collect a candidate trace,
3. diff the traces into a Model Behavior Change Report.

The suite is intentionally safe and compact. It measures false-refusal and
response-shape signals around benign coding work; it does not certify coding
correctness. Use generated outputs as evidence for a follow-up unit-test suite
before claiming a coding-capability improvement.

```bash
vauban examples/qwen_coding_refusal_audit/baseline_trace.toml
vauban examples/qwen_coding_refusal_audit/candidate_trace.toml
vauban examples/qwen_coding_refusal_audit/diff.toml
```

The baseline uses `mlx-community/Qwen3.6-27B-OptiQ-4bit`, a local MLX 4-bit
language-stack quantization of `Qwen/Qwen3.6-27B`. Edit `candidate_trace.toml`
to point at the transformed or alternative model under audit.
