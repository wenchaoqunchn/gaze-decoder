# GazeDecoder

> Accompanying code and data for the paper:
> **"GazeDecoder: Context-Aware Usability Issue Detection from Eye-Tracking Data"**
> Submitted to *IEEE Transactions on Software Engineering (TSE)*.

---

## Overview

GazeDecoder is a multi-stream Transformer model that detects usability issues in
software user interfaces from eye-tracking data.
It jointly encodes raw gaze coordinate sequences and component-level semantic context
retrieved from a pre-computed knowledge base, enabling window-level binary classification
of usability issues without manual annotation overhead.

Key results (controlled study, 20 participants, 3 037 labelled gaze windows):

| Model                    | F1         | Precision | Recall | Scott–Knott Tier |
| ------------------------ | ---------- | --------- | ------ | ---------------- |
| **GazeDecoder**          | **0.9467** | 0.9531    | 0.9404 | **1**            |
| Best baseline (PatchTST) | 0.9242     | 0.9114    | 0.9406 | 2                |

---

## Repository Map

```
gaze-decoder/
├── app/          Stimulus web application (Vue 3 frontend + Flask backend)
├── dataset/      Eye-tracking dataset  (20 participants, 3 037 windows)
├── context/      Knowledge-base construction pipeline  (Contribution 1)
└── model/        GazeDecoder model, ablation study, baselines  (Contributions 2 & 3)
```

---

## Quick Start

### 1 — Run the stimulus application

```bash
# Backend (eye-tracker API)
cd app/backend
pip install -r requirements.txt
python run.py

# Frontend (in a new terminal)
cd app/frontend
npm install
npm run dev
```

See [`app/README.md`](app/README.md) for full setup instructions and eye-tracker
hardware requirements.

### 2 — Explore the dataset

The eye-tracking dataset is located in `dataset/`.
Each participant folder `P1/`–`P20/` contains raw gaze samples, AOI definitions,
view-switch logs, and derived AOI-level metrics.

See [`dataset/README.md`](dataset/README.md) for the data schema and collection
protocol.

### 3 — Train and evaluate GazeDecoder

```bash
cd model
pip install -r requirements.txt
jupyter notebook ablation.ipynb        # ablation study
jupyter notebook baselines.ipynb       # baseline comparisons
```

See [`model/README.md`](model/README.md) for the full training protocol and how to
reproduce the paper's Table III results.

### 4 — Rebuild the semantic knowledge base

```bash
cd context
python extract_context_data.py         # step 1: extract raw context
python build_context_features.py       # step 2: build AOI context tree
python process_context_logic.py        # step 3: embed with SentenceTransformer
```

See [`context/README.md`](context/README.md) for the complete pipeline.

---

## Paper Architecture Summary

```
Raw gaze (x, y)  ──► Positional Encoding ──► Transformer Encoder ──►┐
                                                                      ├──► IIB ──► CtxSA ──► OIB ──► Classification Head
Knowledge Base K ──► Context Embedding ──────────────────────────────┘
  anchor(c) = (id_c, loc_c)
  desc(c)   → (v_text_c , v_code_c)
```

- **IIB** (Input Injection Block): cross-attention between gaze sequence and context
- **CtxSA** (Context Self-Attention): refines context-conditioned representation
- **OIB** (Output Injection Block): produces the final context-aware binary readout

---

## Repository Contents at a Glance

| Directory  | What it contains                                | Paper section                      |
| ---------- | ----------------------------------------------- | ---------------------------------- |
| `app/`     | Library-portal Vue app + Flask eye-tracker API  | §4 (Toolkit)                       |
| `dataset/` | 20-participant gaze recordings + labels         | §5.1 (Dataset)                     |
| `context/` | KB construction: AOI → description → embeddings | §3.1 (Contribution 1)              |
| `model/`   | GazeDecoder model + 12 baselines + ablation     | §3.2–3.4, §5 (Contributions 2 & 3) |


---

## License

This project is released under the [MIT License](LICENSE).
