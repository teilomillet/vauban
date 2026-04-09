<!-- SPDX-FileCopyrightText: 2026 Teilo Millet -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Vauban Overview

```mermaid
flowchart LR
    subgraph Deployment["Deployment Under Test"]
        U["Users / Adversaries"]
        APP["App / Agent / Workflow"]
        TOOLS["Tools / RAG / Policies"]
        MODEL["Model"]
        U --> APP
        APP --> MODEL
        APP <--> TOOLS
    end

    subgraph Access["What Vauban Can Access"]
        API["API / prompts / outputs"]
        ACT["Activations / traces"]
        W["Weights / training"]
    end

    subgraph Vauban["Vauban Assurance Loop"]
        OBJ["Objective\nWhat must be preserved?\nWhat must be prevented?"]
        ATT["Adaptive attacks\nprompt / softprompt / transfer / tool misuse"]
        DEF["Defenses\nSIC / CAST / guard / policy / hardening"]
        JUDGE["Judging\nattack success\nbenign utility\nfalse refusals"]
        FRONTIER["Decision frontier\nsafe enough?\nwhat was lost?\nwhat survives?"]

        OBJ --> ATT
        OBJ --> DEF
        ATT --> JUDGE
        DEF --> JUDGE
        JUDGE --> FRONTIER
        FRONTIER -. iterate .-> ATT
        FRONTIER -. iterate .-> DEF
    end

    API --> Vauban
    ACT --> Vauban
    W --> Vauban

    MODEL -. exposes .-> API
    MODEL -. if available .-> ACT
    MODEL -. if owned .-> W

    FRONTIER --> REPORT["Deployment verdict\nreports / metrics / next actions"]
```

## Reading Guide

- `API` access enables prompt-level attack and evaluation.
- `Activations` enable measure, probe, scan, steering-style analysis, and deeper safety evidence.
- `Weights` enable full white-box work: cutting, export, fine-tuning, and the strongest adversarial loops.
- The center of Vauban is the same in all cases: define the objective, pressure it with adaptive attacks, defend it, and measure retained useful behavior.
