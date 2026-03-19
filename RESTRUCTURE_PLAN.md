# GazeDecoder — Repository Restructuring Plan

> **Purpose**: This document describes the proposed restructuring of the `gaze-decoder/`
> repository for open-source release accompanying the TSE submission *"GazeDecoder:
> Context-Aware Usability Issue Detection from Eye-Tracking Data"*.
>
> **Goal**: One public repository that is immediately reproducible by anyone who has
> read the paper — clear layer separation, English-only documentation, no stale or
> duplicate artefacts.

---

## 1. Current Problems

| #   | Problem                                                                                                                                    | Location                                                                                        |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------- |
| 1   | Mixed Chinese / English filenames and doc strings                                                                                          | `analysis/context/软件功能文档.md`, comments in `routes.py`, `App.vue`, etc.                    |
| 2   | Hard-coded absolute paths pointing to a private machine                                                                                    | `build_context_features.py`, `extract_context_data.py`, `process_context_logic.py`, `config.py` |
| 3   | No top-level README that connects the four sub-systems                                                                                     | Root `gaze-decoder/`                                                                            |
| 4   | Model code (`GazeDecoderV3/`) and context-extraction code (`context/`) mixed under one `analysis/` folder without clear purpose separation | `analysis/`                                                                                     |
| 5   | `store_backup.js` (stale backup file) committed to repo                                                                                    | `frontend_en/src/`                                                                              |
| 6   | Dataset pre-processing scripts (`gaze/`) co-located with raw participant data in the same flat directory                                   | `gaze/`                                                                                         |
| 7   | No `requirements.txt` / `environment.yml` for the model training environment                                                               | `analysis/GazeDecoderV3/`                                                                       |
| 8   | `__pycache__/` directories committed                                                                                                       | multiple locations                                                                              |
| 9   | `backend/lib/dlls/` (binary DLLs) committed without explanation                                                                            | `backend/lib/dlls/`                                                                             |
| 10  | No LICENSE file                                                                                                                            | root                                                                                            |

---

## 2. Proposed Directory Layout

```text
gaze-decoder/                          ← repository root
│
├── README.md                          ★ entry point — paper abstract, repo map, quick-start
├── LICENSE                            ★ MIT (or IEEE open-access compatible)
├── .gitignore
│
├── app/                               ★ (renamed from frontend_en + backend)
│   │                                    The stimulus web application used in the study
│   ├── README.md                      ★ describes the app, how to run it, pages & AOIs
│   ├── frontend/                      (renamed from frontend_en/)
│   │   ├── README.md                  ★ Vue 3 setup, page inventory, AOI list
│   │   ├── index.html
│   │   ├── package.json
│   │   ├── vite.config.js
│   │   ├── public/
│   │   └── src/
│   │       ├── main.js
│   │       ├── App.vue
│   │       ├── style.css
│   │       ├── AOISelector.vue
│   │       ├── getAOIInfo.js
│   │       ├── assets/
│   │       ├── components/
│   │       ├── router/
│   │       └── views/
│   │
│   └── backend/                       (moved from backend/)
│       ├── README.md                  ★ Flask API, endpoints, eye-tracker setup
│       ├── requirements.txt
│       ├── run.py
│       ├── app/
│       │   ├── __init__.py
│       │   ├── config.py
│       │   └── routes.py
│       ├── lib/
│       │   ├── README.md              ★ explains DLLs and PyGazeAnalyser dependency
│       │   ├── __init__.py
│       │   ├── dlls/
│       │   └── PyGazeAnalyser/
│       └── utils/
│           ├── __init__.py
│           ├── data_analysis.py
│           ├── eye_tracking.py
│           └── session_tools.py
│
├── dataset/                           ★ (renamed / reorganised from gaze/)
│   │                                    Eye-tracking dataset — 20 participants, 3 037 windows
│   ├── README.md                      ★ dataset overview, collection protocol, ethics
│   ├── docs/
│   │   ├── index.md
│   │   ├── dataset-structure.md
│   │   └── data-dictionary.md
│   ├── scripts/                       ★ (moved pre-processing scripts out of data root)
│   │   ├── README.md                  ★ explains each script and execution order
│   │   ├── clean_calibration.py
│   │   ├── final_clean_split.py
│   │   ├── inspect_data.py
│   │   ├── process_images.py
│   │   ├── renumber_split.py
│   │   ├── check_counts.py
│   │   └── check_timestamps.py
│   ├── P1/
│   │   ├── AOI.csv
│   │   ├── metrics.csv
│   │   ├── raw_data.csv
│   │   ├── view_switch.csv
│   │   ├── img/
│   │   └── split_data/
│   ├── P2/ … P20/                     (same layout)
│   └── AOI.csv                        ★ global AOI definition (all views, all sessions)
│
├── context/                           ★ (extracted from analysis/context/)
│   │                                    Knowledge base construction pipeline (Contribution 1)
│   ├── README.md                      ★ explains anchor(c), desc(c), (v_text, v_code) pipeline
│   ├── app-function-spec.md           ★ (renamed from 软件功能文档.md, translated to English)
│   ├── add_src_index.py
│   ├── extract_context_data.py
│   ├── process_aoi.py
│   ├── distribute_aoi.py
│   ├── build_context_features.py
│   ├── process_context_logic.py
│   ├── dataset_loader_example.py
│   ├── dataset_loader_units.py
│   ├── user_aoi_labeled/
│   │   ├── README.md                  ★ column schema for AOI_S*.csv files
│   │   ├── AOI_S1.csv … AOI_S6.csv
│   ├── frontend_src/                  ★ Vue source snapshot used for code embedding
│   │   └── src/
│   └── context_features/              ★ pre-computed knowledge base K
│       ├── README.md                  ★ schema of complete_econtext.json
│       ├── context_extraction_raw.json
│       └── complete_econtext.json
│
└── model/                             ★ (renamed from analysis/GazeDecoderV3/)
    │                                    GazeDecoder model — Contributions 2 & 3
    ├── README.md                      ★ architecture overview, how to train, reproduce results
    ├── requirements.txt               ★ NEW — PyTorch, sentence-transformers, etc.
    ├── environment.yml                ★ NEW — conda environment spec
    ├── ablation.ipynb                 ★ GazeDecoder ablation study (IIB / CtxSA / OIB variants)
    ├── ablation_ctx_v2.ipynb          ★ context channel ablation
    ├── baselines.ipynb                ★ 12 baseline comparisons
    └── shared/
        ├── README.md                  ★ module-level API reference
        ├── __init__.py
        ├── config.py
        ├── dataset.py
        ├── features.py
        ├── models.py
        ├── training.py
        └── viz.py
```

> **Symbols**: ★ = new file to create or significantly revise.
> Files without ★ are moved/renamed from current locations.

---

## 3. File-by-File Migration Map

### 3.1 Moves & Renames

| Current path                                   | New path                       | Notes                      |
| ---------------------------------------------- | ------------------------------ | -------------------------- |
| `frontend_en/`                                 | `app/frontend/`                | Rename only                |
| `frontend_en/src/store_backup.js`              | *deleted*                      | Stale backup               |
| `backend/`                                     | `app/backend/`                 | Move                       |
| `gaze/` (data files `P*/`)                     | `dataset/`                     | Move data root up          |
| `gaze/*.py`                                    | `dataset/scripts/`             | Separate scripts from data |
| `gaze/docs/`                                   | `dataset/docs/`                | Move                       |
| `analysis/context/`                            | `context/`                     | Promote to top-level       |
| `analysis/context/软件功能文档.md`             | `context/app-function-spec.md` | Translate + rename         |
| `analysis/GazeDecoderV3/`                      | `model/`                       | Promote to top-level       |
| `analysis/GazeDecoderV3/ablation.ipynb`        | `model/ablation.ipynb`         | Move                       |
| `analysis/GazeDecoderV3/ablation_ctx_v2.ipynb` | `model/ablation_ctx_v2.ipynb`  | Move                       |
| `analysis/GazeDecoderV3/baselines.ipynb`       | `model/baselines.ipynb`        | Move                       |
| `analysis/GazeDecoderV3/shared/`               | `model/shared/`                | Move                       |

### 3.2 Files to Delete

| Path                                         | Reason            |
| -------------------------------------------- | ----------------- |
| `frontend_en/src/store_backup.js`            | Stale backup file |
| `backend/**/__pycache__/`                    | Build artefact    |
| `backend/lib/__pycache__/`                   | Build artefact    |
| `analysis/context/__pycache__/` (if present) | Build artefact    |
| `analysis/GazeDecoderV3/shared/__pycache__/` | Build artefact    |

### 3.3 New Files to Create

| New path                             | Content                                                              |
| ------------------------------------ | -------------------------------------------------------------------- |
| `README.md`                          | Top-level: paper abstract, repo map, quick-start for each sub-system |
| `LICENSE`                            | MIT license                                                          |
| `.gitignore`                         | Python, Node, Jupyter standard ignores                               |
| `app/README.md`                      | App overview, how to start frontend + backend together               |
| `app/frontend/README.md`             | Vue 3 setup guide, page inventory, AOI coordinates                   |
| `app/backend/README.md`              | Flask setup, API endpoint reference, eye-tracker config              |
| `app/backend/lib/README.md`          | DLL origins (Tobii SDK), PyGazeAnalyser attribution                  |
| `dataset/README.md`                  | Dataset overview: 20 participants, labeling protocol, file schema    |
| `dataset/scripts/README.md`          | Script execution order and purpose of each script                    |
| `context/README.md`                  | KB construction pipeline: anchor → desc → (v_text, v_code)           |
| `context/app-function-spec.md`       | English translation of `软件功能文档.md`                             |
| `context/user_aoi_labeled/README.md` | AOI_S*.csv column schema                                             |
| `context/context_features/README.md` | `complete_econtext.json` schema                                      |
| `model/README.md`                    | Architecture, training protocol, Scott-Knott results table           |
| `model/requirements.txt`             | Python dependency list for model training                            |
| `model/environment.yml`              | Conda environment spec                                               |
| `model/shared/README.md`             | Module API reference for `shared/` package                           |

### 3.4 Code Changes Required

| File                                | Change                                                                       |
| ----------------------------------- | ---------------------------------------------------------------------------- |
| `context/build_context_features.py` | Replace hard-coded Windows paths with `pathlib` relative paths or CLI args   |
| `context/extract_context_data.py`   | Same as above                                                                |
| `context/process_context_logic.py`  | Same as above                                                                |
| `model/shared/config.py`            | Replace Colab-specific hard paths with `os.environ` / relative-path defaults |
| `app/backend/app/routes.py`         | Translate inline Chinese comments to English                                 |
| `app/frontend/src/App.vue`          | Translate inline Chinese comments to English                                 |
| `app/frontend/src/views/**`         | Translate inline Chinese comments to English (non-blocking)                  |

---

## 4. README Content Outlines

### 4.1 Root `README.md`

```text
# GazeDecoder

> Accompanying code and data for the paper:
> "GazeDecoder: Context-Aware Usability Issue Detection from Eye-Tracking Data"
> Submitted to IEEE Transactions on Software Engineering (TSE).

## Repository Map
app/        – Stimulus web application (Vue 3 frontend + Flask backend)
dataset/    – Eye-tracking dataset (20 participants, 3 037 labelled gaze windows)
context/    – Knowledge base construction pipeline (Contribution 1)
model/      – GazeDecoder model, ablation study, baselines (Contributions 2 & 3)

## Quick Start
1. Run the stimulus app  →  see app/README.md
2. Explore the dataset   →  see dataset/README.md
3. Train / evaluate      →  see model/README.md
4. Rebuild the KB        →  see context/README.md
```

### 4.2 `app/README.md`

- System overview (library portal for eye-tracking experiments)
- Architecture: Vue 3 (Vite) + Flask + Tobii eye tracker
- How to run (two-terminal quick-start)
- Page list with view names (matches `dataset/docs/data-dictionary.md`)

### 4.3 `dataset/README.md`

- Study design: 20 participants, controlled lab setting
- Labelling protocol: issue / no-issue ground truth
- File schema (links to `docs/`)
- Ethics and anonymisation statement
- Citation

### 4.4 `context/README.md`

- Background: why semantic context matters (cite paper §3.1)
- Pipeline stages: AOI labelling → source indexing → context extraction → embedding
- Input/output of each script
- Schema of `complete_econtext.json`
- How to regenerate the knowledge base from scratch

### 4.5 `model/README.md`

- Architecture diagram (IIB → Transformer → CtxSA → OIB)
- Feature vector layout (786-d)
- Experiment protocol: participant-level 5-fold cross-validation
- Results table (F1, Precision, Recall for GazeDecoder and all 12 baselines)
- How to run: `jupyter nbconvert --execute model/ablation.ipynb`
- Dependency installation: `pip install -r model/requirements.txt`

---

## 5. `.gitignore` Additions

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
.env
*.egg-info/

# Jupyter
.ipynb_checkpoints/

# Node / Vue
node_modules/
dist/

# Model cache
archive/
*.pkl
*.pt
*.pth

# OS
.DS_Store
Thumbs.db
```

---

## 6. Execution Order

The steps below should be followed **in order** when restructuring the repository:

1. **Create new directory skeleton** — create all new empty directories
2. **Copy / move files** — follow the migration map in §3.1
3. **Delete stale files** — follow §3.2
4. **Fix hard-coded paths** — follow §3.4
5. **Create all new README / doc files** — follow §4
6. **Add `LICENSE`, `.gitignore`** — §3.3
7. **Add `model/requirements.txt` and `model/environment.yml`** — §3.3
8. **Verify notebooks run end-to-end** — `jupyter nbconvert --execute`
9. **Final commit** — `git add -A && git commit -m "chore: restructure for TSE open-source release"`

---

Document generated: 2026-03-12
