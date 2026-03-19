# shared/ — GazeDecoder Shared Module

The `shared/` package contains all Python modules shared across the three
experiment notebooks (`ablation.ipynb`, `ablation_ctx_v2.ipynb`, `baselines.ipynb`).

Import pattern in notebooks:

```python
import sys
sys.path.insert(0, "..")   # add model/ to path
from shared.config import *
from shared.dataset import EyeSeqDataset
from shared.models import build_model
from shared.training import run_loso
```

---

## Module Reference

### `config.py` — Global Configuration

Central registry for all hyper-parameters and path resolution.

Key constants:

| Name            | Default                             | Description             |
| --------------- | ----------------------------------- | ----------------------- |
| `WINDOW_SIZE`   | 64                                  | Gaze samples per window |
| `STRIDE`        | 32                                  | Window stride (samples) |
| `FEAT_DIM`      | 786                                 | Total feature dimension |
| `SPATIAL_SLICE` | `[0:2]`                             | Normalised (x, y)       |
| `TEXT_SLICE`    | `[2:386]`                           | Text embedding (384-d)  |
| `CODE_SLICE`    | `[386:770]`                         | Code embedding (384-d)  |
| `L1_SLICE`      | `[770:778]`                         | Layer-1 features        |
| `L2_SLICE`      | `[778:786]`                         | Layer-2 features        |
| `EPOCHS`        | 40                                  | Training epochs         |
| `LR`            | 1e-3                                | Initial learning rate   |
| `BATCH_SIZE`    | 64                                  | Mini-batch size         |
| `SEED`          | 42                                  | Global random seed      |
| `DEVICE`        | `"cuda"` if available, else `"cpu"` | Compute device          |

Path configuration is resolved from environment variables `GAZE_DIR` and
`ECONTEXT_PATH`, falling back to relative defaults suitable for local execution.

---

### `dataset.py` — EyeSeqDataset

```python
class EyeSeqDataset(torch.utils.data.Dataset)
```

Builds the windowed feature dataset from `dataset/P*/` and
`context/context_features/complete_econtext.json`.

Constructor parameters:

| Parameter          | Type | Description                      |
| ------------------ | ---- | -------------------------------- |
| `participant_root` | str  | Path to `dataset/`               |
| `econtext_path`    | str  | Path to `complete_econtext.json` |
| `cache_dir`        | str  | Directory for `.pkl` cache files |
| `window_size`      | int  | Window length (default 64)       |
| `stride`           | int  | Window stride (default 32)       |

Each sample `x` is a `(window_size, FEAT_DIM)` float32 array.
Label `y` is 0 or 1.

Key method:

```python
def get_loso_splits(dataset) -> List[Tuple[Subset, Subset, Subset]]:
    """Returns 20 (train, val, test) Subset triples for LOSO CV."""
```

---

### `features.py` — Behavioural Feature Extraction

Implements Layer-1 and Layer-2 feature extractors.

#### Layer-1 (8-d, per-timestep micro-window)

Computed over a short sliding sub-window (~0.5 s) centred on each timestep:

| Index | Name               | Description                          |
| ----- | ------------------ | ------------------------------------ |
| 0     | `fixation_ratio`   | Fraction of low-velocity steps       |
| 1     | `saccade_amp`      | Mean saccade magnitude               |
| 2     | `saccade_std`      | Saccade amplitude standard deviation |
| 3     | `velocity`         | Mean velocity (normalised units/s)   |
| 4     | `dispersion_x`     | x-axis standard deviation            |
| 5     | `dispersion_y`     | y-axis standard deviation            |
| 6     | `direction_change` | Mean absolute angle change           |
| 7     | `acceleration`     | Mean absolute acceleration           |

#### Layer-2 (8-d, whole-window macro stats, broadcast)

Computed once per window and broadcast to all timesteps:

| Index | Name                | Description                                            |
| ----- | ------------------- | ------------------------------------------------------ |
| 0     | `total_path_length` | Sum of all step distances                              |
| 1     | `mean_velocity`     | Mean velocity across the window                        |
| 2     | `velocity_std`      | Velocity standard deviation                            |
| 3     | `x_range`           | max(x) − min(x)                                        |
| 4     | `y_range`           | max(y) − min(y)                                        |
| 5     | `direction_entropy` | 8-bin angular entropy                                  |
| 6     | `revisit_density`   | Fraction of second-half steps in first-half grid cells |
| 7     | `centroid_shift`    | First-half vs second-half centroid distance            |

---

### `models.py` — All Model Definitions

Contains the full GazeDecoder architecture plus all 12 baseline models.

**GazeDecoder variants** (ablation study):

| Variant name             | `b_in` channels                    | OIB condition              |
| ------------------------ | ---------------------------------- | -------------------------- |
| `Bchan_Spatial`          | Spatial (2-d)                      | mean(ctx)                  |
| `Bchan_L1`               | Layer-1 (8-d)                      | mean(ctx)                  |
| `Bchan_Spatial_L1`       | Spatial + L1 (10-d)                | mean(ctx)                  |
| `Bchan_Spatial_L1_L2bc`  | Spatial + L1 + L2-broadcast (18-d) | mean(ctx)                  |
| `Bchan_Spatial_L2oib`    | Spatial (2-d)                      | fused(ctx + L2)            |
| `Bchan_Spatial_L1_L2oib` | Spatial + L1 (10-d)                | fused(ctx + L2) ← **best** |
| `Bchan_Full_L2oib`       | Spatial + L1 + L2-broadcast (18-d) | fused(ctx + L2)            |

**Baseline models**:

- ML: XGBoost, RandomForest, LightGBM
- DL: BiLSTM, 1D-CNN, TransformerEnc, PatchTST, iTransformer, TimesNet,
  DLinear, Mamba, TimesBERT

---

### `training.py` — LOSO Training Framework

Implements the training loop with:

- Linear warmup + cosine annealing LR schedule
- Best-epoch checkpoint selection (by F1 on validation set)
- Per-fold JSON cache (avoids recomputation on notebook restart)
- Deterministic seeding via `set_fold_seed(fold)`

Key function:

```python
def run_loso(
    model_name: str,
    build_fn: Callable,
    dataset: EyeSeqDataset,
    archive_dir: Path,
) -> dict:
    """Run full LOSO CV and return aggregated metrics dict."""
```

---

### `viz.py` — Visualisation Utilities

- `plot_sk_diagram(results)` — Scott–Knott rank diagram (paper Figure 4)
- `plot_confusion_matrix(y_true, y_pred)` — Per-fold confusion matrix
- `plot_loso_boxplot(results)` — F1 distribution across folds
