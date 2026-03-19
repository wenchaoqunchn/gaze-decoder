# shared/config.py
# ─────────────────────────────────────────────────────────────────────────────
# GazeDecoder V3 — Global configuration
# Shared by both GazeDecoderV3ablation.ipynb and GazeDecoderV3baselines.ipynb
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import random
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

# ── Colab detection ───────────────────────────────────────────────────────────
try:
    from google.colab import drive  # noqa: F401

    IN_COLAB = True
except ImportError:
    IN_COLAB = False


# ── Drive mount (call once from notebook) ────────────────────────────────────
def mount_drive():
    if IN_COLAB:
        from google.colab import drive

        drive.mount("/content/drive")


# ── Paths ─────────────────────────────────────────────────────────────────────
# Resolution order (highest priority first):
#   1. Environment variables  GAZE_DIR  and  ECONTEXT_PATH
#   2. CLI argument  --root   (sets ROOT_DIR; sub-paths derived automatically)
#   3. Repository-relative defaults  (dataset/ and context/ at repo root)
#
# The Colab helper mount_drive() remains available but is no longer required
# for local execution.

import argparse as _argparse

_HERE = os.path.dirname(os.path.abspath(__file__))  # model/shared/
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))  # repo root

_cli = _argparse.ArgumentParser(add_help=False)
_cli.add_argument("--root", default=None, help="Override repo root path")
_cli_args, _ = _cli.parse_known_args()

if _cli_args.root:
    _REPO_ROOT = _cli_args.root

# Individual path overrides via environment variables
GAZE_DIR = os.environ.get(
    "GAZE_DIR",
    os.path.join(_REPO_ROOT, "dataset"),
)
_context_dir = os.path.join(_REPO_ROOT, "context")
ECONTEXT_PATH = os.environ.get(
    "ECONTEXT_PATH",
    os.path.join(_context_dir, "context_features", "complete_econtext.json"),
)

# Legacy aliases kept for backward compatibility
ROOT_DIR = _REPO_ROOT
CODE_DIR = os.path.join(_REPO_ROOT, "model")
CONTEXT_DIR = _context_dir

# ── Archive layout ────────────────────────────────────────────────────────────
# model/
#   archive/
#     ablation/          ← per-variant CV fold caches (ablation experiment)
#       <variant_name>/
#         fold_01.json … fold_NN.json
#         final_report.json
#     baselines/         ← per-model CV fold caches (baseline experiment)
#       <model_name>/
#         fold_01.json … fold_NN.json
#         final_report.json
#     dataset/           ← dataset .pkl caches
ARCHIVE_DIR = Path(CODE_DIR) / "archive"
ABL_DIR = ARCHIVE_DIR / "ablation"
BASE_DIR = ARCHIVE_DIR / "baselines"
DS_CACHE_DIR = ARCHIVE_DIR / "dataset"


def ensure_dirs():
    """Create all archive subdirectories (idempotent)."""
    for d in [ABL_DIR, BASE_DIR, DS_CACHE_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ── Hyperparameters ───────────────────────────────────────────────────────────
WINDOW_SIZE = 64  # gaze window length (timesteps)
STRIDE = 32  # sliding window stride
HIT_THRESHOLD = 10  # min gaze points inside AOI to count as a hit
BATCH_SIZE = 16
LR = 5e-5
EPOCHS = 40  # V3: train up to EPOCHS rounds, pick best-checkpoint by val F1
WARMUP_RATIO = 0.15  # fraction of total steps used for linear LR warmup
GRAD_CLIP = 1.0

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42


def set_global_seed(seed: int = SEED):
    """Fix all random sources globally."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_fold_seed(fold: int, base_seed: int = SEED):
    """
    Deterministic per-fold seed.
    formula: base_seed * 31 + fold
    Ensures different folds get different seeds while remaining fully
    reproducible across runs and models.
    """
    fold_seed = base_seed * 31 + fold
    random.seed(fold_seed)
    np.random.seed(fold_seed)
    torch.manual_seed(fold_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(fold_seed)


# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Feature dimension constants ───────────────────────────────────────────────
# V3 feature layout (786d total):
#   [0:2]    Spatial  : normalised gaze (x, y)
#   [2:770]  Semantic : Text-384 + Code-384 embeddings   (ctx stream only)
#   [770:778] Layer1  : 8d per-timestep micro-window stats
#   [778:786] Layer2  : 8d window-level macro stats (broadcast per sample)
FEAT_DIM = 786

# Primary slices
SPATIAL_SLICE = slice(0, 2)
EMBED_SLICE = slice(2, 770)  # text + code combined  (ctx stream)
TEXT_SLICE = slice(2, 386)
CODE_SLICE = slice(386, 770)
L1_SLICE = slice(770, 778)  # Layer1: 8d micro-window behavioral stats
L2_SLICE = slice(778, 786)  # Layer2: 8d window macro stats (broadcast)

# Legacy alias kept for baselines.ipynb (ML feature flattening)
BEHAV_SLICE = slice(770, 786)  # full behavioral = L1 + L2 = 16d

N_LAYER1 = 8
N_LAYER2 = 8
N_BEHAV = N_LAYER1 + N_LAYER2  # 16


def print_config():
    print(f"Device  : {DEVICE}")
    print(f"Repo    : {ROOT_DIR}")
    print(f"Gaze    : {GAZE_DIR}")
    print(f"Context : {ECONTEXT_PATH}")
    print(f"Archive : {ARCHIVE_DIR}")
    print(
        f"Config  : window={WINDOW_SIZE}, stride={STRIDE}, bs={BATCH_SIZE}, "
        f"lr={LR}, epochs={EPOCHS} (fixed, best-of), seed={SEED}"
    )
    print(f"Protocol: participant-level 5-fold CV")
