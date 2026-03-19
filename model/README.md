# model/ — GazeDecoder Model

This directory contains the GazeDecoder multi-stream Transformer model,
the full ablation study, and comparisons against 12 baselines.
It implements **Contributions 2 and 3** of the paper.

---

## Architecture

```
Gaze sequence  (T × 2)
      │
      ▼
Positional Encoding
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  Transformer Encoder  (L layers, d_model, h heads)      │
└─────────────────────────────────────────────────────────┘
      │                        ▲
      │                        │ Context vector  (768-d)
      ▼                        │   = Text(384) + Code(384)
┌─────────────────┐            │   from knowledge base K
│  IIB            │◄───────────┘
│  (Input         │  Cross-attention: Q=gaze, KV=context
│  Injection      │
│  Block)         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  CtxSA          │  Self-attention over context-conditioned
│  (Context       │  sequence representation
│  Self-Attention)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  OIB            │  Fuses context summary with temporal
│  (Output        │  readout; optionally injects Layer-2
│  Injection      │  macro-window features
│  Block)         │
└────────┬────────┘
         │
         ▼
 Classification Head → issue probability (0 / 1)
```

### Feature Vector Layout (786-d)

| Slice       | Dim | Content                                                   |
| ----------- | --- | --------------------------------------------------------- |
| `[0:2]`     | 2   | Normalised gaze (x, y) — spatial stream                   |
| `[2:770]`   | 768 | Text(384) + Code(384) — semantic context stream           |
| `[770:778]` | 8   | Layer-1: per-timestep micro-window behavioural stats      |
| `[778:786]` | 8   | Layer-2: whole-window macro behavioural stats (broadcast) |

---

## Directory Layout

```
model/
├── README.md              This file
├── requirements.txt       Python dependencies
├── environment.yml        Conda environment spec
├── ablation.ipynb         Ablation study: 7 GazeDecoder variants (IIB/OIB channels)
├── ablation_ctx_v2.ipynb  Context-channel ablation: text vs code vs combined
├── baselines.ipynb        12 baseline models (ML + DL)
└── shared/
    ├── README.md          Module-level API reference
    ├── __init__.py
    ├── config.py          Global hyper-parameters and path configuration
    ├── dataset.py         EyeSeqDataset — window builder + LOSO split
    ├── features.py        Layer-1 and Layer-2 behavioural feature extractors
    ├── models.py          All model definitions (GazeDecoder variants + baselines)
    ├── training.py        LOSO training loop, checkpoint, evaluation
    └── viz.py             Plotting utilities (Scott–Knott diagram, confusion matrix)
```

---

## Experiment Protocol

**Leave-One-Subject-Out (LOSO) cross-validation**

- 20 folds, one held-out participant per fold.
- For each fold, the remaining 19 participants' data are split 80/20 into
  train and validation.
- The validation split is used only for checkpoint selection (best epoch by F1).
- The held-out participant's data form the test set — never seen during training.

**Hyper-parameters** (see `shared/config.py`)

| Parameter          | Value                            |
| ------------------ | -------------------------------- |
| Window size        | 64 samples                       |
| Stride             | 32 samples                       |
| d_model            | 64                               |
| Transformer layers | 2                                |
| Attention heads    | 4                                |
| Batch size         | 64                               |
| Epochs             | 40                               |
| Optimiser          | AdamW                            |
| LR schedule        | Linear warmup + cosine annealing |
| Seed               | 42                               |

---

## Results (Table III)

| Model                                    | F1         | Precision | Recall | SK Tier |
| ---------------------------------------- | ---------- | --------- | ------ | ------- |
| **GazeDecoder** (Bchan_Spatial_L1_L2oib) | **0.9467** | 0.9531    | 0.9404 | **1**   |
| Bchan_Spatial_L1                         | 0.9388     | 0.9421    | 0.9356 | 2       |
| TimesBERT                                | 0.9229     | 0.9301    | 0.9158 | 2       |
| PatchTST                                 | 0.9187     | 0.9244    | 0.9131 | 2       |
| iTransformer                             | 0.9144     | 0.9198    | 0.9091 | 3       |
| … (8 more baselines)                     | …          | …         | …      | 3–4     |

Statistical test: Wilcoxon signed-rank with Holm–Bonferroni correction.
Effect size: Cohen's d = 1.643 (GazeDecoder vs best baseline, p = 0.010).

---

## Reproducing the Results

### 1. Install dependencies

```bash
pip install -r requirements.txt
# or, using conda:
conda env create -f environment.yml
conda activate gazedecoder
```

### 2. Set data paths

Edit `shared/config.py` and set:

```python
GAZE_DIR     = "/path/to/dataset"           # dataset/ in this repository
ECONTEXT_PATH = "/path/to/context/context_features/complete_econtext.json"
```

Alternatively, set environment variables `GAZE_DIR` and `ECONTEXT_PATH`.

### 3. Run the experiments

```bash
# Ablation study (GazeDecoder variants)
jupyter nbconvert --to notebook --execute ablation.ipynb

# Context-channel ablation
jupyter nbconvert --to notebook --execute ablation_ctx_v2.ipynb

# Baselines
jupyter nbconvert --to notebook --execute baselines.ipynb
```

Results are cached in `archive/` as JSON files for incremental execution.
Re-running a notebook with an existing cache skips already-computed folds.

### 4. Colab

The notebooks detect Google Colab automatically via `shared/config.py`.
Mount your Drive, copy the `dataset/` and `context/` directories, and update
`ROOT_DIR` in `config.py` before running.

---

## Dependencies

See `requirements.txt` for the full list. Core packages:

- `torch >= 2.0`
- `sentence-transformers >= 2.2`
- `scikit-learn >= 1.3`
- `xgboost >= 2.0`
- `lightgbm >= 4.0`
- `pandas`, `numpy`, `matplotlib`, `seaborn`
