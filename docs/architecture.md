# GazeDecoder — Architecture Diagrams

This document contains two architecture diagrams for the GazeDecoder open-source tool:
one in **Mermaid** syntax and one in **PlantUML** syntax.

---

## Mermaid Diagram

```mermaid
flowchart TD
    %% ─────────────────────────────────────────────
    %% LAYER 0 — Experiment Application
    %% ─────────────────────────────────────────────
    subgraph APP["🖥️  app/  —  Stimulus Web Application"]
        direction LR
        FE["Frontend\nVue 3 + Vite\n28 page views\n(Element Plus UI)"]
        BE["Backend\nFlask REST API\nTobii Pro SDK\n(.NET CLR bridge)"]
        FE -->|"HTTP /init /switch /stop /analyze"| BE
    end

    %% ─────────────────────────────────────────────
    %% LAYER 1 — Dataset
    %% ─────────────────────────────────────────────
    subgraph DS["📦  dataset/  —  Eye-Tracking Dataset"]
        direction LR
        RAW["Raw Recordings\nP1–P20\nraw_data.csv\nview_switch.csv"]
        SCRIPTS["Preprocessing Scripts\ndataset/scripts/ (×7)\ngaze segmentation\nfixation detection\nAOI assignment"]
        PROCESSED["Processed Data\nAOI.csv  metrics.csv\nsplit_data/ (windows)"]
        RAW --> SCRIPTS --> PROCESSED
    end

    %% ─────────────────────────────────────────────
    %% LAYER 2 — Knowledge Base Pipeline
    %% ─────────────────────────────────────────────
    subgraph CTX["🗂️  context/  —  Knowledge Base Construction Pipeline"]
        direction LR
        AOI_SRC["AOI Label Files\nuser_aoi_labeled/\n+ Vue 3 Source"]
        PA["① process_aoi.py\nParse & normalise AOI"]
        ASI["② add_src_index.py\nAlign labels → source lines"]
        ECD["③ extract_context_data.py\nExtract code snippets"]
        BCF["④ build_context_features.py\nBuild pseudo-DOM tree\ngen. text descriptions"]
        PCL["⑤ process_context_logic.py\nEncode with\nall-MiniLM-L6-v2\n(384-d × 2)"]
        ECONTEXT["complete_econtext.json\n{componentInfo, src_path,\ntext_embed(384d),\ncode_embed(384d)}"]

        AOI_SRC --> PA --> ASI --> ECD --> BCF --> PCL --> ECONTEXT
    end

    %% ─────────────────────────────────────────────
    %% LAYER 3 — Model
    %% ─────────────────────────────────────────────
    subgraph MODEL["🧠  model/  —  GazeDecoder"]
        direction TB

        subgraph FEAT["Feature Extraction  (model/shared/features.py)"]
            direction LR
            SP["Spatial\n2-d\n(norm x, y)"]
            SEM["Semantic\n768-d\nText 384-d\n+ Code 384-d"]
            L1["Layer 1  8-d\nper-timestep stats\n(fixation_ratio,\nsaccade_amp,\nvelocity, …)"]
            L2["Layer 2  8-d\nwhole-window stats\n(path_length,\nentropy,\ncentroid_shift, …)"]
        end

        VEC["Input Tensor\n786-d per timestep\n[SP | SEM | L1 | L2]"]
        SP & SEM & L1 & L2 --> VEC

        subgraph BLOCKS["GazeDecoder Transformer  (model/shared/models.py)"]
            direction TB
            IIB["IIB — Input Integration Block\nMHA cross-attention\nQ = behavioral stream\nKV = context vector\n(ctx_proj: 768→d_model)"]
            TENC["Transformer Encoder\n(multi-layer self-attention)"]
            CTXSA["CtxSA — Context-Semantic Attention\nattend over KB context\nto re-weight hidden states"]
            OIB["OIB — Output Integration Block\ngating: ctx + L2 features\n→ binary logit"]
            IIB --> TENC --> CTXSA --> OIB
        end

        VEC --> IIB
        ECONTEXT -.->|"loaded via ECONTEXT_PATH"| IIB

        OUT["Binary Classification\n(Usability Issue: Yes / No)"]
        OIB --> OUT

        subgraph EVAL["Evaluation  (notebooks)"]
            direction LR
            ABL["ablation.ipynb\nablation_ctx_v2.ipynb\n7 ChronosX variants\n(vary b_in × OIB)"]
            BASE["baselines.ipynb\n12 baselines\n(XGBoost, RF, LightGBM,\nBiLSTM, 1D-CNN,\nPatchTST, Mamba, …)"]
        end
        OUT --> EVAL
    end

    %% ─────────────────────────────────────────────
    %% Cross-layer data flows
    %% ─────────────────────────────────────────────
    BE -->|"raw_data.csv\nview_switch.csv"| RAW
    PROCESSED -->|"split_data/ windows\nGAZE_DIR"| FEAT
    ECONTEXT -->|"context vectors\nECONTEXT_PATH"| BLOCKS

    %% ─────────────────────────────────────────────
    %% Results callout
    %% ─────────────────────────────────────────────
    RES["📊 LOSO CV Results\nF1 = 0.9467\nPrec = 0.9531  Rec = 0.9404\nScott–Knott Tier 1"]
    EVAL --> RES
```

---

## PlantUML Diagram

```plantuml
@startuml GazeDecoder_Architecture
skinparam backgroundColor #FAFAFA
skinparam componentStyle rectangle
skinparam defaultFontName Helvetica
skinparam defaultFontSize 12
skinparam ArrowColor #555555
skinparam PackageBorderColor #888888
skinparam PackageBackgroundColor #F0F4FF
skinparam ComponentBackgroundColor #FFFFFF
skinparam ComponentBorderColor #6688AA
skinparam NoteBackgroundColor #FFFFF0
skinparam NoteBorderColor #BBBB00

title GazeDecoder — System Architecture

' ─────────────────────────────────────────────
' PACKAGE 1 — Stimulus Web Application
' ─────────────────────────────────────────────
package "app/  ·  Stimulus Web Application" as APP #EEF5FF {

    package "app/frontend/" as FE_PKG {
        [Vue 3 + Vite\nElement Plus\n28 page views] as FE
    }

    package "app/backend/" as BE_PKG {
        [Flask REST API] as FLASK
        [EyeTracker\n(Tobii Pro SDK\n+ .NET CLR)] as TOBII
        [DataAnalyzer\nfixation / saccade\ndetection] as ANALYZER
        [SessionManager\nsession lifecycle] as SESSION
    }

    FE --> FLASK : HTTP\n/init /switch\n/stop /analyze
    FLASK --> TOBII
    FLASK --> ANALYZER
    FLASK --> SESSION
}

' ─────────────────────────────────────────────
' PACKAGE 2 — Eye-Tracking Dataset
' ─────────────────────────────────────────────
package "dataset/  ·  Eye-Tracking Dataset  (N=20)" as DS #EFF8EE {

    database "Raw Recordings\nP1 – P20\nraw_data.csv\nview_switch.csv" as RAW

    package "dataset/scripts/" as SCRIPTS_PKG {
        [Preprocessing Scripts\n× 7 (gaze segmentation,\nfixation detection,\nAOI assignment)] as SCRIPTS
    }

    database "Processed Data\nAOI.csv  metrics.csv\nsplit_data/ (3 037 windows)" as PROC

    RAW --> SCRIPTS : pipeline
    SCRIPTS --> PROC
}

' ─────────────────────────────────────────────
' PACKAGE 3 — Knowledge Base Pipeline
' ─────────────────────────────────────────────
package "context/  ·  KB Construction Pipeline" as CTX #FFF8EE {

    [① process_aoi.py\nParse & normalise AOI labels] as PA
    [② add_src_index.py\nAlign labels → Vue source lines] as ASI
    [③ extract_context_data.py\nExtract code snippets] as ECD
    [④ build_context_features.py\nBuild pseudo-DOM tree\n& text descriptions] as BCF
    [⑤ process_context_logic.py\nEncode with all-MiniLM-L6-v2\n(384-d text + 384-d code)] as PCL

    database "context_features/\ncomplete_econtext.json\n{componentInfo, src_path,\ntext_embed(384d), code_embed(384d)}" as ECONTEXT

    [AOI Label Files\n+ Vue 3 Source] as AOI_SRC

    AOI_SRC --> PA
    PA --> ASI
    ASI --> ECD
    ECD --> BCF
    BCF --> PCL
    PCL --> ECONTEXT
}

' ─────────────────────────────────────────────
' PACKAGE 4 — GazeDecoder Model
' ─────────────────────────────────────────────
package "model/  ·  GazeDecoder" as MODEL #FFF0F0 {

    package "model/shared/features.py  ·  Feature Extraction" as FEAT_PKG {
        [Spatial  2-d\n(norm x, y)] as SP
        [Semantic  768-d\nText-384d + Code-384d\n(from econtext)] as SEM
        [Layer 1  8-d  per-timestep\nfixation_ratio, saccade_amp\nvelocity, dispersion, …] as L1
        [Layer 2  8-d  whole-window\npath_length, entropy\ncentroid_shift, …] as L2
        [Input Tensor  786-d / timestep\n[SP | SEM | L1 | L2]] as VEC
        SP --> VEC
        SEM --> VEC
        L1 --> VEC
        L2 --> VEC
    }

    package "model/shared/models.py  ·  GazeDecoder Transformer" as ARCH_PKG {
        [IIB  ·  Input Integration Block\nMHA cross-attention\nQ = behavioral stream\nKV = context vector\nctx_proj: 768 → d_model] as IIB
        [Transformer Encoder\n(multi-layer self-attention\n+ position encoding)] as TENC
        [CtxSA  ·  Context-Semantic Attention\nre-weight hidden states\nover KB context] as CTXSA
        [OIB  ·  Output Integration Block\ngating: ctx + L2 features\n→ binary logit] as OIB
        IIB --> TENC
        TENC --> CTXSA
        CTXSA --> OIB
    }

    VEC --> IIB

    package "Evaluation Notebooks" as EVAL_PKG {
        [ablation.ipynb\nablation_ctx_v2.ipynb\n7 ChronosX variants\n(vary b_in × OIB conditioning)] as ABL
        [baselines.ipynb\n12 baseline models\nXGBoost, RF, LightGBM\nBiLSTM, 1D-CNN, PatchTST\niTransformer, Mamba …] as BASE
    }

    OIB --> ABL
    OIB --> BASE
}

' ─────────────────────────────────────────────
' Cross-package data flows
' ─────────────────────────────────────────────
TOBII --> RAW : raw_data.csv\nview_switch.csv
SESSION --> RAW

PROC --> FEAT_PKG : split_data/ windows\n(GAZE_DIR)

ECONTEXT --> SEM    : text & code embeddings
ECONTEXT --> IIB    : context vectors\n(ECONTEXT_PATH)

' ─────────────────────────────────────────────
' Results
' ─────────────────────────────────────────────
note right of EVAL_PKG
  **LOSO Cross-Validation Results**
  F1    = 0.9467
  Prec  = 0.9531
  Rec   = 0.9404
  Scott–Knott Tier 1
  (vs 12 baselines)
end note

@enduml
```
