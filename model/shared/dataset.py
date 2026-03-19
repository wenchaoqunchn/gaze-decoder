# shared/dataset.py
# ─────────────────────────────────────────────────────────────────────────────
# GazeDecoder V3 — EyeSeqDataset & LOSO split utilities
# Shared by both ablation and baselines notebooks.
# ─────────────────────────────────────────────────────────────────────────────
import os
import glob
import json
import pickle
from bisect import bisect_right
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from shared.config import (
    WINDOW_SIZE,
    STRIDE,
    N_BEHAV,
    N_LAYER1,
    N_LAYER2,
    SPATIAL_SLICE,
    TEXT_SLICE,
    CODE_SLICE,
    BEHAV_SLICE,
    L1_SLICE,
    L2_SLICE,
    FEAT_DIM,
)
from shared.features import extract_layer1_features, extract_layer2_features


# ─────────────────────────────────────────────────────────────────────────────
# EyeSeqDataset
# ─────────────────────────────────────────────────────────────────────────────
class EyeSeqDataset(Dataset):
    """
    View-aware EyeSeq dataset loader.

    Scans all P* directories under `participant_root`.
    Builds samples: {x: ndarray[T, 786], y: float (0|1), p_id: str}

    Feature layout per sample (786d):
        [0:2]    Spatial  : normalised gaze (x, y)
        [2:770]  Semantic : Text(384) + Code(384) embeddings
        [770:778] Layer1  : 8d per-timestep micro-window stats
        [778:786] Layer2  : 8d window-level macro stats (broadcast)

    Cache is written to `cache_dir` with a filename that encodes key params.

    Parameters
    ----------
    participant_root : str   – path to gaze/ directory
    econtext_path   : str   – path to complete_econtext.json
    cache_dir       : str   – directory to store/load .pkl cache
    window_size     : int   – window length in gaze samples (default 64)
    stride          : int   – stride between windows (default 32)
    exclude_pids    : list  – participant IDs to skip
    min_attn_ratio  : float – minimum AOI-hit ratio per window
    issue_ratio_thr : float – minimum issue-label ratio to label window as 1
    """

    def __init__(
        self,
        participant_root: str,
        econtext_path: str,
        cache_dir: Optional[str] = None,
        window_size: int = WINDOW_SIZE,
        stride: int = STRIDE,
        exclude_pids: Optional[List[str]] = None,
        min_attn_ratio: float = 0.1,
        issue_ratio_thr: float = 0.1,
    ):
        self.root = participant_root
        self.window_size = window_size
        self.stride = stride
        self.exclude_pids = exclude_pids or []
        self.min_attn_ratio = min_attn_ratio
        self.issue_ratio_thr = issue_ratio_thr
        self.samples: list = []

        # ── Cache path ────────────────────────────────────────────────────────
        self.cache_path = None
        if cache_dir:
            excl_tag = (
                "_excl" + "".join(sorted(self.exclude_pids))
                if self.exclude_pids
                else ""
            )
            cache_name = (
                f"eyeseq_w{window_size}_s{stride}"
                f"_thr{int(issue_ratio_thr * 100)}_d{FEAT_DIM}{excl_tag}.pkl"
            )
            self.cache_path = os.path.join(cache_dir, cache_name)

        # ── Load econtext map ─────────────────────────────────────────────────
        print(f"📂 Loading econtext from {os.path.basename(econtext_path)} …")
        with open(econtext_path, "r", encoding="utf-8") as f:
            self._econtext = json.load(f)

        # ── Try cache first, then build ───────────────────────────────────────
        if self.cache_path and os.path.exists(self.cache_path):
            print(f"⚡ Loading from cache: {os.path.basename(self.cache_path)}")
            try:
                with open(self.cache_path, "rb") as f:
                    self.samples = pickle.load(f)
                print(f"   Loaded {len(self.samples):,} samples.")
                return
            except Exception as e:
                print(f"   Cache load failed ({e}), rebuilding …")

        self._build()

        if self.cache_path and self.samples:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump(self.samples, f)
            print(f"💾 Cache saved → {os.path.basename(self.cache_path)}")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_view_timings(self, p_dir: str):
        vs = os.path.join(p_dir, "view_switch.csv")
        if not os.path.exists(vs):
            return None, None
        df = pd.read_csv(vs).sort_values("time")
        return df["time"].values, df["view"].values

    def _view_at(self, ts, times, views):
        if times is None:
            return "HomePage"
        idx = bisect_right(times, ts)
        return views[0] if idx == 0 else views[idx - 1]

    @staticmethod
    def _bool_str(v) -> bool:
        return str(v).strip().lower() in ("true", "1", "yes", "t")

    def _hit_test(self, cx: float, cy: float, aoi_df: pd.DataFrame) -> dict:
        for _, row in aoi_df.iloc[::-1].iterrows():
            if row["x1"] <= cx <= row["x2"] and row["y1"] <= cy <= row["y2"]:
                label = (
                    1.0
                    if (
                        self._bool_str(row.get("is_designed_defect", False))
                        or self._bool_str(row.get("is_reported_by_user", False))
                    )
                    else 0.0
                )
                return {
                    "hit": True,
                    "label": label,
                    "info": row["componentInfo"],
                    "src": row["src_index"],
                }
        return {"hit": False, "label": 0.0}

    def _build(self):
        from tqdm import tqdm  # local import so module loads without tqdm

        p_dirs = sorted(glob.glob(os.path.join(self.root, "P*")))
        active = [
            d
            for d in p_dirs
            if not any(ex in os.path.basename(d) for ex in self.exclude_pids)
        ]
        print(
            f"🔍 Discovered {len(p_dirs)} participants, "
            f"processing {len(active)} (excluding {self.exclude_pids})"
        )

        temp = []
        for p_dir in tqdm(active, desc="Participants"):
            p_id = os.path.basename(p_dir)

            aoi_path = os.path.join(p_dir, "AOI.csv")
            if not os.path.exists(aoi_path):
                continue
            try:
                aoi_full = pd.read_csv(aoi_path)
            except Exception:
                continue

            aoi_by_view = {v: g for v, g in aoi_full.groupby("view")}
            sw_times, sw_views = self._load_view_timings(p_dir)
            split_files = glob.glob(os.path.join(p_dir, "split_data", "*.csv"))

            for fpath in split_files:
                try:
                    df = pd.read_csv(fpath)
                except Exception:
                    continue
                if len(df) < self.window_size:
                    continue

                # ── Raw columns ───────────────────────────────────────────────
                t_col = (
                    df["time"].values
                    if "time" in df.columns
                    else (
                        df["timestamp"].values
                        if "timestamp" in df.columns
                        else np.arange(len(df)) * 33.3
                    )
                )
                x_col = (
                    df["x"].values
                    if "x" in df.columns
                    else df.get("raw_x", pd.Series(np.zeros(len(df)))).values
                )
                y_col = (
                    df["y"].values
                    if "y" in df.columns
                    else df.get("raw_y", pd.Series(np.zeros(len(df)))).values
                )

                txy = np.stack([t_col, x_col / 1920.0, y_col / 1080.0], axis=1)

                # ── Layer1: per-timestep 8d micro-window stats ────────────────
                l1feats = extract_layer1_features(txy, micro_win=16)  # [T', 8]
                pad = len(txy) - len(l1feats)
                if pad > 0:
                    l1feats = np.concatenate(
                        [np.zeros((pad, N_LAYER1), dtype=np.float32), l1feats]
                    )
                if l1feats.shape[0] > 1:
                    mu, sigma = l1feats.mean(0), l1feats.std(0) + 1e-6
                    l1feats = (l1feats - mu) / sigma

                # ── Layer2: whole-split macro stats, z-score normalised ────────
                # Computed per sliding window below; placeholder zeros here.
                l2_placeholder = np.zeros(N_LAYER2, dtype=np.float32)

                # ── Per-gaze feature vector ───────────────────────────────────
                seq_feat, seq_label, seq_hit = [], [], []

                for seq_idx, (_, row) in enumerate(df.iterrows()):
                    cx = row.get("x", row.get("raw_x", 0))
                    cy = row.get("y", row.get("raw_y", 0))
                    ts = row.get("time", row.get("timestamp", 0))

                    view = self._view_at(ts, sw_times, sw_views)
                    aois = aoi_by_view.get(view, pd.DataFrame())
                    hit = self._hit_test(cx, cy, aois)

                    sp = [cx / 1920.0, cy / 1080.0]
                    te = [0.0] * 384
                    ce = [0.0] * 384
                    label = 0.0
                    is_h = 0

                    if hit["hit"]:
                        is_h = 1
                        key = f"{hit['info']}|{hit['src']}"
                        if key in self._econtext:
                            ctx = self._econtext[key]
                            te = ctx["embed_text"]
                            ce = ctx["embed_code"]
                        label = hit["label"]

                    b_l1 = (
                        l1feats[seq_idx].tolist()
                        if seq_idx < len(l1feats)
                        else [0.0] * N_LAYER1
                    )
                    # L2 is zeros here; overwritten per-window in sliding loop below
                    b_l2 = l2_placeholder.tolist()
                    seq_feat.append(sp + te + ce + b_l1 + b_l2)  # 2+384+384+8+8 = 786
                    seq_label.append(label)
                    seq_hit.append(is_h)

                arr_x = np.array(seq_feat, dtype=np.float32)  # [N, 786]
                arr_y = np.array(seq_label, dtype=np.float32)  # [N]
                arr_h = np.array(seq_hit, dtype=np.float32)  # [N]

                # ── Pre-compute all per-window L2 vectors, then batch z-score ─
                # Collect raw L2 for every valid window first, normalise together.
                win_l2_raw = []
                valid_wins = []
                n_wins = len(arr_x) - self.window_size + 1
                for i in range(0, n_wins, self.stride):
                    wl = arr_y[i : i + self.window_size]
                    wh = arr_h[i : i + self.window_size]
                    if wh.mean() < self.min_attn_ratio:
                        continue
                    txy_win = txy[i : i + self.window_size]
                    l2_raw = extract_layer2_features(txy_win)  # [8]
                    win_l2_raw.append(l2_raw)
                    valid_wins.append(i)

                # Z-score normalise L2 across all windows of this split
                if len(win_l2_raw) > 1:
                    l2_stack = np.stack(win_l2_raw, axis=0)  # [W, 8]
                    l2_mu = l2_stack.mean(axis=0)
                    l2_sig = l2_stack.std(axis=0) + 1e-6
                    win_l2_norm = (l2_stack - l2_mu) / l2_sig
                elif len(win_l2_raw) == 1:
                    win_l2_norm = np.zeros((1, N_LAYER2), dtype=np.float32)
                else:
                    continue  # no valid windows in this split

                # ── Sliding window ────────────────────────────────────────────
                for wi, i in enumerate(valid_wins):
                    wl = arr_y[i : i + self.window_size]
                    final_label = 1.0 if wl.mean() > self.issue_ratio_thr else 0.0

                    # Overwrite L2 slice with z-score normalised window L2
                    win_x = arr_x[i : i + self.window_size].copy()  # [T, 786]
                    win_x[:, L2_SLICE] = win_l2_norm[wi][np.newaxis, :]  # broadcast

                    temp.append(
                        {
                            "x": win_x,
                            "y": final_label,
                            "p_id": p_id,
                        }
                    )

        self.samples = temp
        n_pos = sum(s["y"] for s in self.samples)
        n_neg = len(self.samples) - n_pos
        print(
            f"✅ Built {len(self.samples):,} windows | "
            f"Pos={int(n_pos)}, Neg={int(n_neg)}, "
            f"Ratio={n_neg / max(n_pos, 1):.1f}:1"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (torch.from_numpy(s["x"]), torch.tensor(int(s["y"]), dtype=torch.long))


# ─────────────────────────────────────────────────────────────────────────────
# FeatureMaskedDataset — zeros out selected feature channels
# ─────────────────────────────────────────────────────────────────────────────
class FeatureMaskedDataset(Dataset):
    """
    Wraps an EyeSeqDataset and zeros out selected feature slices.
    Used for ablation experiments.
    """

    def __init__(self, base: EyeSeqDataset, zero_slices: List[slice]):
        self.base = base
        self.zero_slices = zero_slices

    @property
    def samples(self):
        return self.base.samples

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        x = x.clone()
        for sl in self.zero_slices:
            x[:, sl] = 0.0
        return x, y


# ─────────────────────────────────────────────────────────────────────────────
# LOSO split utility
# ─────────────────────────────────────────────────────────────────────────────
def get_loso_splits(
    ds: EyeSeqDataset,
) -> List[Tuple[List[int], List[int], str]]:
    """
    Return participant-level LOSO splits.

    Returns
    -------
    list of (train_indices, test_indices, heldout_pid)
    """
    pids = sorted(set(s["p_id"] for s in ds.samples))
    splits = []
    for pid in pids:
        test_idx = [i for i, s in enumerate(ds.samples) if s["p_id"] == pid]
        train_idx = [i for i, s in enumerate(ds.samples) if s["p_id"] != pid]
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        splits.append((train_idx, test_idx, pid))
    return splits


# ─────────────────────────────────────────────────────────────────────────────
# Numpy collection helper
# ─────────────────────────────────────────────────────────────────────────────
def collect_numpy(ds: Dataset, indices: List[int]):
    """Collect a Subset to numpy arrays (X, y, p_ids)."""
    X = np.stack([ds.samples[i]["x"] for i in indices]).astype(np.float32)
    y = np.array([ds.samples[i]["y"] for i in indices]).astype(np.int64)
    p = np.array([ds.samples[i]["p_id"] for i in indices])
    return X, y, p


# ─────────────────────────────────────────────────────────────────────────────
# Dataset factory
# ─────────────────────────────────────────────────────────────────────────────
def build_dataset(
    gaze_dir: str,
    econtext_path: str,
    cache_dir: str,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
    hit_threshold: int = 10,
    exclude_pids: Optional[List[str]] = None,
    min_attn_ratio: float = 0.1,
) -> EyeSeqDataset:
    """
    Build (or load cached) the default EyeSeqDataset (union label).

    Parameters
    ----------
    hit_threshold  : minimum fraction (×100) of a window's gaze points that
                     must be issue-labelled to assign label=1.
                     e.g. hit_threshold=10 → issue_ratio_thr=0.10
    min_attn_ratio : minimum fraction of window points that must land on any
                     AOI (attention filter).
    """
    issue_ratio_thr = hit_threshold / 100.0
    return EyeSeqDataset(
        participant_root=gaze_dir,
        econtext_path=econtext_path,
        cache_dir=cache_dir,
        window_size=window_size,
        stride=stride,
        exclude_pids=exclude_pids,
        min_attn_ratio=min_attn_ratio,
        issue_ratio_thr=issue_ratio_thr,
    )


# ─────────────────────────────────────────────────────────────────────────────
# EyeSeqDatasetV2 — multimodal context (contextV2 JSON)
# ─────────────────────────────────────────────────────────────────────────────
# Field name → vector length in complete_multimodal_context.json
_V2_FIELDS = {
    "embed_func": 384,  # G1: functional description (MiniLM)
    "embed_code": 768,  # G2: source code (CodeBERT) — 768d!
    "embed_text_src": 384,  # G3a: visible source text (multilingual-MiniLM)
    "embed_text_ocr": 384,  # G3b: OCR text from screenshot
    "embed_img_origin_patch": 384,  # G4: AOI patch (ViT-Small)
    "embed_img_origin_page": 384,  # G5: whole page (ViT-Small)
}
# embed_code is 768d (CodeBERT); all others are 384d.
# Available ctx_mode presets (controls which fields are concatenated → projected → 768d):
#   "full"      : all 6 fields  → concat dim = 384*5 + 768 = 2688d
#   "no_img"    : G1+G2+G3      → concat dim = 384*3 + 768 = 1920d   (text-only)
#   "img_only"  : G4+G5         → concat dim = 384*2 = 768d           (image-only, no proj needed)
#   "v1_compat" : G1+G2 (replicate original embed_text+embed_code layout) → 384+768=1152d
CTX_MODE_FIELDS = {
    "full": [
        "embed_func",
        "embed_code",
        "embed_text_src",
        "embed_text_ocr",
        "embed_img_origin_patch",
        "embed_img_origin_page",
    ],
    "no_img": ["embed_func", "embed_code", "embed_text_src", "embed_text_ocr"],
    "img_only": ["embed_img_origin_patch", "embed_img_origin_page"],
    "v1_compat": ["embed_func", "embed_code"],
}


def _ctx_concat_dim(fields: List[str]) -> int:
    return sum(_V2_FIELDS[f] for f in fields)


class EyeSeqDatasetV2(EyeSeqDataset):
    """
    Drop-in replacement for EyeSeqDataset that reads multimodal context from
    complete_multimodal_context.json (contextV2/).

    The semantic slot [2:770] of the 786d feature vector is filled with the
    multimodal context vector, **projected to 768d** so all existing models
    (ctx_proj: 768→d_model) work without any architecture change.

    Projection is a fixed (non-trained) random-orthogonal linear map computed
    once from the concatenated dimension → 768.  For ctx_mode="img_only" where
    concat dim = 768, the identity is used (no projection needed).

    Parameters
    ----------
    ctx_mode : one of {"full", "no_img", "img_only", "v1_compat"}
        Controls which modality fields are included.
    v2_econtext_path : str
        Path to complete_multimodal_context.json  (contextV2/).
    proj_seed : int
        Seed for the fixed random projection matrix.
    All other parameters are forwarded to EyeSeqDataset.
    """

    def __init__(
        self,
        participant_root: str,
        v2_econtext_path: str,
        cache_dir: Optional[str] = None,
        window_size: int = WINDOW_SIZE,
        stride: int = STRIDE,
        exclude_pids: Optional[List[str]] = None,
        min_attn_ratio: float = 0.1,
        issue_ratio_thr: float = 0.1,
        ctx_mode: str = "full",
        proj_seed: int = 42,
    ):
        if ctx_mode not in CTX_MODE_FIELDS:
            raise ValueError(
                f"ctx_mode must be one of {list(CTX_MODE_FIELDS)}, got '{ctx_mode}'"
            )

        # ── Set all instance attributes BEFORE calling any helper ─────────────
        self.root = participant_root
        self.window_size = window_size
        self.stride = stride
        self.exclude_pids = exclude_pids or []
        self.min_attn_ratio = min_attn_ratio
        self.issue_ratio_thr = issue_ratio_thr
        self.samples: list = []
        self._ctx_mode = ctx_mode
        self._v2_econtext_path = v2_econtext_path
        self._proj_seed = proj_seed
        self._active_fields = CTX_MODE_FIELDS[ctx_mode]
        self._in_dim = _ctx_concat_dim(self._active_fields)

        # ── Build fixed random-orthogonal projection matrix (in_dim → 768) ───
        # QR decomposition preserves norms better than a raw Gaussian map.
        # Stored as float32 numpy array; applied once per gaze point at build time.
        rng = np.random.default_rng(proj_seed)
        if self._in_dim == 768:
            self._proj_mat = None  # identity — no projection needed
        elif self._in_dim < 768:
            raw = rng.standard_normal((self._in_dim, 768)).astype(np.float32)
            self._proj_mat = raw / (np.linalg.norm(raw, axis=0, keepdims=True) + 1e-8)
        else:
            # Random orthonormal columns: QR of (in_dim × 768) random matrix
            raw = rng.standard_normal((self._in_dim, 768)).astype(np.float32)
            q, _ = np.linalg.qr(raw)  # q shape: (in_dim, 768) when in_dim≥768
            self._proj_mat = q[:, :768].astype(np.float32)

        # ── Cache path (includes ctx_mode tag to avoid V1 collision) ──────────
        self.cache_path = None
        if cache_dir:
            excl_tag = (
                "_excl" + "".join(sorted(self.exclude_pids))
                if self.exclude_pids
                else ""
            )
            cache_name = (
                f"eyeseq_w{window_size}_s{stride}"
                f"_thr{int(issue_ratio_thr * 100)}"
                f"_d{FEAT_DIM}_v2_{ctx_mode}{excl_tag}.pkl"
            )
            self.cache_path = os.path.join(cache_dir, cache_name)

        # ── Try cache first, then build ───────────────────────────────────────
        if self.cache_path and os.path.exists(self.cache_path):
            print(
                f"⚡ Loading V2 cache [{ctx_mode}]: {os.path.basename(self.cache_path)}"
            )
            try:
                with open(self.cache_path, "rb") as f:
                    self.samples = pickle.load(f)
                print(f"   Loaded {len(self.samples):,} samples.")
                return
            except Exception as e:
                print(f"   Cache load failed ({e}), rebuilding …")

        self._build()

        if self.cache_path and self.samples:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump(self.samples, f)
            print(f"💾 Cache saved → {os.path.basename(self.cache_path)}")

    # ── Project a raw concat vector to 768d ──────────────────────────────────
    def _project(self, vec: np.ndarray) -> List[float]:
        """vec: (in_dim,) float32 → list of 768 floats."""
        if self._proj_mat is None:
            return vec.tolist()
        return (vec @ self._proj_mat).tolist()

    # ── Override _build to use v2 econtext ───────────────────────────────────
    def _build(self):
        from tqdm import tqdm

        # Load V2 econtext
        print(f"📂 Loading V2 econtext (mode='{self._ctx_mode}') …")
        with open(self._v2_econtext_path, "r", encoding="utf-8") as f:
            v2_ctx = json.load(f)

        # Zero fallback vectors per field
        _zero = {f: [0.0] * _V2_FIELDS[f] for f in self._active_fields}
        _zero_proj = [0.0] * 768

        p_dirs = sorted(glob.glob(os.path.join(self.root, "P*")))
        active = [
            d
            for d in p_dirs
            if not any(ex in os.path.basename(d) for ex in self.exclude_pids)
        ]
        print(
            f"🔍 Discovered {len(p_dirs)} participants, "
            f"processing {len(active)} (excluding {self.exclude_pids})"
        )

        temp = []
        for p_dir in tqdm(active, desc="Participants"):
            p_id = os.path.basename(p_dir)

            aoi_path = os.path.join(p_dir, "AOI.csv")
            if not os.path.exists(aoi_path):
                continue
            try:
                aoi_full = pd.read_csv(aoi_path)
            except Exception:
                continue

            aoi_by_view = {v: g for v, g in aoi_full.groupby("view")}
            sw_times, sw_views = self._load_view_timings(p_dir)
            split_files = glob.glob(os.path.join(p_dir, "split_data", "*.csv"))

            for fpath in split_files:
                try:
                    df = pd.read_csv(fpath)
                except Exception:
                    continue
                if len(df) < self.window_size:
                    continue

                t_col = (
                    df["time"].values
                    if "time" in df.columns
                    else df.get(
                        "timestamp", pd.Series(np.arange(len(df)) * 33.3)
                    ).values
                )
                x_col = (
                    df["x"].values
                    if "x" in df.columns
                    else df.get("raw_x", pd.Series(np.zeros(len(df)))).values
                )
                y_col = (
                    df["y"].values
                    if "y" in df.columns
                    else df.get("raw_y", pd.Series(np.zeros(len(df)))).values
                )
                txy = np.stack([t_col, x_col / 1920.0, y_col / 1080.0], axis=1)

                from shared.features import (
                    extract_layer1_features,
                    extract_layer2_features,
                )

                l1feats = extract_layer1_features(txy, micro_win=16)
                pad = len(txy) - len(l1feats)
                if pad > 0:
                    l1feats = np.concatenate(
                        [np.zeros((pad, N_LAYER1), dtype=np.float32), l1feats]
                    )
                if l1feats.shape[0] > 1:
                    mu, sigma = l1feats.mean(0), l1feats.std(0) + 1e-6
                    l1feats = (l1feats - mu) / sigma

                l2_placeholder = np.zeros(N_LAYER2, dtype=np.float32)

                seq_feat, seq_label, seq_hit = [], [], []

                for seq_idx, (_, row) in enumerate(df.iterrows()):
                    cx = row.get("x", row.get("raw_x", 0))
                    cy = row.get("y", row.get("raw_y", 0))
                    ts = row.get("time", row.get("timestamp", 0))

                    view = self._view_at(ts, sw_times, sw_views)
                    aois = aoi_by_view.get(view, pd.DataFrame())
                    hit = self._hit_test(cx, cy, aois)

                    sp = [cx / 1920.0, cy / 1080.0]
                    label = 0.0
                    is_h = 0
                    ctx_768 = _zero_proj

                    if hit["hit"]:
                        is_h = 1
                        label = hit["label"]
                        # V2 JSON key format is identical to V1: "componentInfo|src_index"
                        # where src_index includes the line number suffix (e.g. "File.vue:3").
                        # AOI.csv src_index already has this format — use it directly.
                        key = f"{hit['info']}|{hit['src']}"
                        entry = v2_ctx.get(key)
                        if entry is not None:
                            parts = []
                            for field in self._active_fields:
                                vec = entry.get(field, _zero[field])
                                parts.extend(vec)
                            concat_vec = np.array(parts, dtype=np.float32)
                            ctx_768 = self._project(concat_vec)

                    b_l1 = (
                        l1feats[seq_idx].tolist()
                        if seq_idx < len(l1feats)
                        else [0.0] * N_LAYER1
                    )
                    b_l2 = l2_placeholder.tolist()
                    # Layout identical to V1: sp(2) + ctx_768(768) + L1(8) + L2(8) = 786
                    seq_feat.append(sp + ctx_768 + b_l1 + b_l2)
                    seq_label.append(label)
                    seq_hit.append(is_h)

                arr_x = np.array(seq_feat, dtype=np.float32)
                arr_y = np.array(seq_label, dtype=np.float32)
                arr_h = np.array(seq_hit, dtype=np.float32)

                win_l2_raw, valid_wins = [], []
                n_wins = len(arr_x) - self.window_size + 1
                for i in range(0, n_wins, self.stride):
                    wh = arr_h[i : i + self.window_size]
                    if wh.mean() < self.min_attn_ratio:
                        continue
                    txy_win = txy[i : i + self.window_size]
                    win_l2_raw.append(extract_layer2_features(txy_win))
                    valid_wins.append(i)

                if len(win_l2_raw) > 1:
                    l2_stack = np.stack(win_l2_raw, axis=0)
                    l2_mu, l2_sig = l2_stack.mean(0), l2_stack.std(0) + 1e-6
                    win_l2_norm = (l2_stack - l2_mu) / l2_sig
                elif len(win_l2_raw) == 1:
                    win_l2_norm = np.zeros((1, N_LAYER2), dtype=np.float32)
                else:
                    continue

                for wi, i in enumerate(valid_wins):
                    wl = arr_y[i : i + self.window_size]
                    final_label = 1.0 if wl.mean() > self.issue_ratio_thr else 0.0
                    win_x = arr_x[i : i + self.window_size].copy()
                    win_x[:, L2_SLICE] = win_l2_norm[wi][np.newaxis, :]
                    temp.append({"x": win_x, "y": final_label, "p_id": p_id})

        self.samples = temp
        n_pos = sum(s["y"] for s in self.samples)
        n_neg = len(self.samples) - n_pos
        print(
            f"✅ Built {len(self.samples):,} windows | "
            f"Pos={int(n_pos)}, Neg={int(n_neg)}, "
            f"Ratio={n_neg / max(n_pos, 1):.1f}:1  [ctx_mode={self._ctx_mode}]"
        )


def build_dataset_v2(
    gaze_dir: str,
    v2_econtext_path: str,
    cache_dir: str,
    ctx_mode: str = "full",
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
    hit_threshold: int = 10,
    exclude_pids: Optional[List[str]] = None,
    min_attn_ratio: float = 0.1,
    proj_seed: int = 42,
) -> "EyeSeqDatasetV2":
    """
    Build (or load cached) a multimodal EyeSeqDatasetV2.

    Parameters
    ----------
    v2_econtext_path : path to complete_multimodal_context.json
    ctx_mode         : one of {"full", "no_img", "img_only", "v1_compat"}
    proj_seed        : seed for fixed random-orthogonal projection (in_dim→768)
    """
    issue_ratio_thr = hit_threshold / 100.0
    ds = EyeSeqDatasetV2(
        participant_root=gaze_dir,
        v2_econtext_path=v2_econtext_path,
        cache_dir=cache_dir,
        window_size=window_size,
        stride=stride,
        exclude_pids=exclude_pids,
        min_attn_ratio=min_attn_ratio,
        issue_ratio_thr=issue_ratio_thr,
        ctx_mode=ctx_mode,
        proj_seed=proj_seed,
    )
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# EyeSeqDatasetV2_Wide — native wide-dimension multimodal dataset
#
# Design goal: avoid the random-projection information loss in EyeSeqDatasetV2.
# Instead of projecting (1152–2688)d → 768d, we keep all vectors at full
# dimension and append them after the original 768d semantic slot.
#
# Wide feature layout (WIDE_FEAT_DIM = 2322d):
#   [0:2]       Spatial                  (normalised gaze x, y)
#   [2:386]     embed_text  (384d)        old JSON (MiniLM, text description)
#   [386:770]   embed_code  (384d)        old JSON (sentence-transformer, code)
#   [770:1154]  embed_text_src  (384d)    new JSON G3a (multilingual-MiniLM)
#   [1154:1538] embed_text_ocr  (384d)    new JSON G3b (OCR)
#   [1538:1922] embed_img_origin_patch (384d) new JSON G4 (ViT-Small, patch)
#   [1922:2306] embed_img_origin_page  (384d) new JSON G5 (ViT-Small, page)
#   [2306:2314] Layer1  (8d)             per-timestep micro-window stats
#   [2314:2322] Layer2  (8d)             window-level macro stats
#
# WIDE_CTX_DIM = 768(old) + 4×384(G3a,G3b,G4,G5) = 2304
# WIDE_FEAT_DIM = 2 + 2304 + 8 + 8 = 2322
#
# New-JSON fields used: embed_text_src, embed_text_ocr,
#                       embed_img_origin_patch, embed_img_origin_page
# Old-JSON fields used: embed_text(384d), embed_code(384d)
# ─────────────────────────────────────────────────────────────────────────────

WIDE_CTX_DIM = 768 + 4 * 384  # 768(old) + 4×384(G3a,G3b,G4,G5) = 2304
WIDE_FEAT_DIM = 2 + WIDE_CTX_DIM + 8 + 8  # spatial + ctx + L1 + L2 = 2322

# Slice constants for the Wide layout
_W_SPATIAL = slice(0, 2)
_W_CTX = slice(2, 2 + WIDE_CTX_DIM)  # [2:2306]
_W_L1 = slice(2 + WIDE_CTX_DIM, 2 + WIDE_CTX_DIM + 8)  # [2306:2314]
_W_L2 = slice(2 + WIDE_CTX_DIM + 8, 2 + WIDE_CTX_DIM + 16)  # [2314:2322]

# New JSON fields that feed the extra channels in Wide layout
_WIDE_NEW_FIELDS = [
    "embed_text_src",  # G3a: 384d
    "embed_text_ocr",  # G3b: 384d
    "embed_img_origin_patch",  # G4:  384d
    "embed_img_origin_page",  # G5:  384d
]


class EyeSeqDatasetV2_Wide(EyeSeqDataset):
    """
    Wide-dimension multimodal dataset that avoids random projection.

    Reads BOTH the old econtext JSON (complete_econtext.json) and the new
    multimodal JSON (complete_multimodal_context.json).  The 768d old
    embedding is preserved at [2:770], and 4 new 384d channels are appended
    at [770:2306], giving a 2304d context block at [2:2306].

    Feature layout (2322d total):
        [0:2]      Spatial              normalised gaze (x, y)
        [2:386]    embed_text           384d  old JSON  (MiniLM, text description)
        [386:770]  embed_code           384d  old JSON  (sentence-transformer, code)
        [770:1154] embed_text_src       384d  new JSON  G3a (multilingual-MiniLM)
        [1154:1538] embed_text_ocr      384d  new JSON  G3b (OCR)
        [1538:1922] embed_img_origin_patch 384d new JSON G4 (ViT-Small, patch)
        [1922:2306] embed_img_origin_page  384d new JSON G5 (ViT-Small, page)
        [2306:2314] Layer1              8d   per-timestep micro-window stats
        [2314:2322] Layer2              8d   window-level macro stats

    Context slice for the model: x[:,:,2:2306] — 2304d, no projection.

    Parameters
    ----------
    econtext_path    : path to complete_econtext.json  (old V1 JSON)
    v2_econtext_path : path to complete_multimodal_context.json  (new V2 JSON)
    All other parameters forwarded to EyeSeqDataset.
    """

    def __init__(
        self,
        participant_root: str,
        econtext_path: str,
        v2_econtext_path: str,
        cache_dir: Optional[str] = None,
        window_size: int = WINDOW_SIZE,
        stride: int = STRIDE,
        exclude_pids: Optional[List[str]] = None,
        min_attn_ratio: float = 0.1,
        issue_ratio_thr: float = 0.1,
    ):
        self.root = participant_root
        self.window_size = window_size
        self.stride = stride
        self.exclude_pids = exclude_pids or []
        self.min_attn_ratio = min_attn_ratio
        self.issue_ratio_thr = issue_ratio_thr
        self.samples: list = []
        self._v2_econtext_path = v2_econtext_path

        # ── Cache path (tagged "wide" to avoid V1/V2 collisions) ───────────
        self.cache_path = None
        if cache_dir:
            excl_tag = (
                "_excl" + "".join(sorted(self.exclude_pids))
                if self.exclude_pids
                else ""
            )
            cache_name = (
                f"eyeseq_w{window_size}_s{stride}"
                f"_thr{int(issue_ratio_thr * 100)}"
                f"_d{WIDE_FEAT_DIM}_wide{excl_tag}.pkl"
            )
            self.cache_path = os.path.join(cache_dir, cache_name)

        # ── Load both JSON maps ─────────────────────────────────────────────
        print(f"📂 Loading old econtext from {os.path.basename(econtext_path)} …")
        with open(econtext_path, "r", encoding="utf-8") as f:
            self._econtext = json.load(f)

        # ── Try cache first, then build ─────────────────────────────────────
        if self.cache_path and os.path.exists(self.cache_path):
            print(f"⚡ Loading Wide cache: {os.path.basename(self.cache_path)}")
            try:
                with open(self.cache_path, "rb") as f:
                    self.samples = pickle.load(f)
                print(f"   Loaded {len(self.samples):,} samples.")
                return
            except Exception as e:
                print(f"   Cache load failed ({e}), rebuilding …")

        self._build_wide()

        if self.cache_path and self.samples:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump(self.samples, f)
            print(f"💾 Wide cache saved → {os.path.basename(self.cache_path)}")

    def _build_wide(self):
        """Build the 2322d wide feature vectors."""
        from tqdm import tqdm

        print(
            f"📂 Loading V2 econtext from {os.path.basename(self._v2_econtext_path)} …"
        )
        with open(self._v2_econtext_path, "r", encoding="utf-8") as f:
            v2_ctx = json.load(f)

        _zero_new = {f: [0.0] * _V2_FIELDS[f] for f in _WIDE_NEW_FIELDS}

        p_dirs = sorted(glob.glob(os.path.join(self.root, "P*")))
        active = [
            d
            for d in p_dirs
            if not any(ex in os.path.basename(d) for ex in self.exclude_pids)
        ]
        print(
            f"🔍 Discovered {len(p_dirs)} participants, "
            f"processing {len(active)} (excluding {self.exclude_pids})"
        )

        temp = []
        for p_dir in tqdm(active, desc="Participants"):
            p_id = os.path.basename(p_dir)

            aoi_path = os.path.join(p_dir, "AOI.csv")
            if not os.path.exists(aoi_path):
                continue
            try:
                aoi_full = pd.read_csv(aoi_path)
            except Exception:
                continue

            aoi_by_view = {v: g for v, g in aoi_full.groupby("view")}
            sw_times, sw_views = self._load_view_timings(p_dir)
            split_files = glob.glob(os.path.join(p_dir, "split_data", "*.csv"))

            for fpath in split_files:
                try:
                    df = pd.read_csv(fpath)
                except Exception:
                    continue
                if len(df) < self.window_size:
                    continue

                t_col = (
                    df["time"].values
                    if "time" in df.columns
                    else (
                        df["timestamp"].values
                        if "timestamp" in df.columns
                        else np.arange(len(df)) * 33.3
                    )
                )
                x_col = (
                    df["x"].values
                    if "x" in df.columns
                    else df.get("raw_x", pd.Series(np.zeros(len(df)))).values
                )
                y_col = (
                    df["y"].values
                    if "y" in df.columns
                    else df.get("raw_y", pd.Series(np.zeros(len(df)))).values
                )
                txy = np.stack([t_col, x_col / 1920.0, y_col / 1080.0], axis=1)

                l1feats = extract_layer1_features(txy, micro_win=16)
                pad = len(txy) - len(l1feats)
                if pad > 0:
                    l1feats = np.concatenate(
                        [np.zeros((pad, N_LAYER1), dtype=np.float32), l1feats]
                    )
                if l1feats.shape[0] > 1:
                    mu, sigma = l1feats.mean(0), l1feats.std(0) + 1e-6
                    l1feats = (l1feats - mu) / sigma

                l2_placeholder = np.zeros(N_LAYER2, dtype=np.float32)
                seq_feat, seq_label, seq_hit = [], [], []

                for seq_idx, (_, row) in enumerate(df.iterrows()):
                    cx = row.get("x", row.get("raw_x", 0))
                    cy = row.get("y", row.get("raw_y", 0))
                    ts = row.get("time", row.get("timestamp", 0))

                    view = self._view_at(ts, sw_times, sw_views)
                    aois = aoi_by_view.get(view, pd.DataFrame())
                    hit = self._hit_test(cx, cy, aois)

                    sp = [cx / 1920.0, cy / 1080.0]
                    te = [0.0] * 384  # embed_text (old JSON)
                    ce = [0.0] * 384  # embed_code (old JSON, 384d sentence-transformer)
                    new_vecs = {f: list(_zero_new[f]) for f in _WIDE_NEW_FIELDS}
                    label = 0.0
                    is_h = 0

                    if hit["hit"]:
                        is_h = 1
                        label = hit["label"]
                        key = f"{hit['info']}|{hit['src']}"

                        # Old JSON: embed_text + embed_code (both 384d)
                        if key in self._econtext:
                            ctx_old = self._econtext[key]
                            te = ctx_old["embed_text"]
                            ce = ctx_old["embed_code"]

                        # New JSON: G3a, G3b, G4, G5
                        entry = v2_ctx.get(key)
                        if entry is not None:
                            for f in _WIDE_NEW_FIELDS:
                                if f in entry:
                                    new_vecs[f] = entry[f]

                    b_l1 = (
                        l1feats[seq_idx].tolist()
                        if seq_idx < len(l1feats)
                        else [0.0] * N_LAYER1
                    )
                    b_l2 = l2_placeholder.tolist()

                    # Assemble 2322d vector: sp + te + ce + G3a + G3b + G4 + G5 + L1 + L2
                    wide_vec = (
                        sp
                        + te  # 384d old text
                        + ce  # 384d old code
                        + new_vecs["embed_text_src"]  # 384d G3a
                        + new_vecs["embed_text_ocr"]  # 384d G3b
                        + new_vecs["embed_img_origin_patch"]  # 384d G4
                        + new_vecs["embed_img_origin_page"]  # 384d G5
                        + b_l1  # 8d
                        + b_l2  # 8d
                    )  # total = 2 + 384*6 + 8 + 8 = 2322
                    seq_feat.append(wide_vec)
                    seq_label.append(label)
                    seq_hit.append(is_h)

                arr_x = np.array(seq_feat, dtype=np.float32)  # [N, 2322]
                arr_y = np.array(seq_label, dtype=np.float32)
                arr_h = np.array(seq_hit, dtype=np.float32)

                win_l2_raw, valid_wins = [], []
                n_wins = len(arr_x) - self.window_size + 1
                for i in range(0, n_wins, self.stride):
                    wh = arr_h[i : i + self.window_size]
                    if wh.mean() < self.min_attn_ratio:
                        continue
                    txy_win = txy[i : i + self.window_size]
                    win_l2_raw.append(extract_layer2_features(txy_win))
                    valid_wins.append(i)

                if len(win_l2_raw) > 1:
                    l2_stack = np.stack(win_l2_raw, axis=0)
                    l2_mu, l2_sig = l2_stack.mean(0), l2_stack.std(0) + 1e-6
                    win_l2_norm = (l2_stack - l2_mu) / l2_sig
                elif len(win_l2_raw) == 1:
                    win_l2_norm = np.zeros((1, N_LAYER2), dtype=np.float32)
                else:
                    continue

                for wi, i in enumerate(valid_wins):
                    wl = arr_y[i : i + self.window_size]
                    final_label = 1.0 if wl.mean() > self.issue_ratio_thr else 0.0
                    win_x = arr_x[i : i + self.window_size].copy()
                    win_x[:, _W_L2] = win_l2_norm[wi][np.newaxis, :]
                    temp.append({"x": win_x, "y": final_label, "p_id": p_id})

        self.samples = temp
        n_pos = sum(s["y"] for s in self.samples)
        n_neg = len(self.samples) - n_pos
        print(
            f"✅ Built {len(self.samples):,} Wide windows | "
            f"Pos={int(n_pos)}, Neg={int(n_neg)}, "
            f"Ratio={n_neg / max(n_pos, 1):.1f}:1  [WIDE_FEAT_DIM={WIDE_FEAT_DIM}]"
        )


def build_dataset_v2_wide(
    gaze_dir: str,
    econtext_path: str,
    v2_econtext_path: str,
    cache_dir: str,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
    hit_threshold: int = 10,
    exclude_pids: Optional[List[str]] = None,
    min_attn_ratio: float = 0.1,
) -> EyeSeqDatasetV2_Wide:
    """
    Build (or load cached) a wide-dimension multimodal EyeSeqDatasetV2_Wide.

    Uses both old and new context JSONs; no random projection is applied.
    The resulting feature tensor is 2322d.

    Parameters
    ----------
    econtext_path    : path to complete_econtext.json        (old V1, for embed_text+embed_code)
    v2_econtext_path : path to complete_multimodal_context.json (new V2, for G3a-G5)
    """
    issue_ratio_thr = hit_threshold / 100.0
    return EyeSeqDatasetV2_Wide(
        participant_root=gaze_dir,
        econtext_path=econtext_path,
        v2_econtext_path=v2_econtext_path,
        cache_dir=cache_dir,
        window_size=window_size,
        stride=stride,
        exclude_pids=exclude_pids,
        min_attn_ratio=min_attn_ratio,
        issue_ratio_thr=issue_ratio_thr,
    )
