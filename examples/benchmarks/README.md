# Canonical Environment Benchmarks

This directory contains checked-in TOML configs for Vauban's built-in
scenario-backed environment benchmarks.

Each benchmark file:

- uses a named `[environment].scenario`
- includes baseline `[meta]` metadata for reproducibility
- uses a shared GCG softprompt baseline
- writes results to `output/benchmarks/<scenario>` at the repo root

Validate one before running it:

```bash
vauban --validate examples/benchmarks/share_doc.toml
```

Run it:

```bash
vauban examples/benchmarks/share_doc.toml
```

Available benchmarks:

- `examples/benchmarks/data_exfil.toml`
- `examples/benchmarks/fedex_phishing.toml`
- `examples/benchmarks/garage_door.toml`
- `examples/benchmarks/ignore_email.toml`
- `examples/benchmarks/salesforce_admin.toml`
- `examples/benchmarks/share_doc.toml`

If you want a fresh scaffold instead of the checked-in baseline pack:

```bash
vauban init --scenario share_doc --output share_doc.toml
```
