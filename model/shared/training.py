# shared/training.py
# ─────────────────────────────────────────────────────────────────────────────
# GazeDecoder V3 — Training framework
#
# Key V3 changes vs V2:
#   1. EPOCHS=40, always train to completion, pick best-epoch checkpoint
#      (no early stopping — prevents premature termination at local optima)
#   2. Linear warmup + cosine annealing LR schedule for reliable convergence
#   3. set_fold_seed() called before every fold (reproducibility)
#   4. Per-fold JSON cache written immediately after each fold
#   5. Archive layout: archive/ablation/<variant>/ or archive/baselines/<model>/
# ─────────────────────────────────────────────────────────────────────────────
import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset

from shared.config import (
    BATCH_SIZE,
    DEVICE,
    EPOCHS,
    GRAD_CLIP,
    LR,
    SEED,
    WARMUP_RATIO,
    set_fold_seed,
)
from shared.dataset import EyeSeqDataset, collect_numpy, get_loso_splits
from shared.models import ModelSpec


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Binary classification metrics (positive class = 1 = Issue)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "p_issue": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "r_issue": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_issue": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def summarize_folds(fold_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    keys = list(fold_metrics[0].keys())
    return {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys}


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_loader(ds: Dataset, indices: List[int], shuffle: bool = False) -> DataLoader:
    return DataLoader(
        Subset(ds, indices), batch_size=BATCH_SIZE, shuffle=shuffle, drop_last=False
    )


def _class_weights(y: np.ndarray) -> torch.Tensor:
    y = y.astype(int)
    n1 = (y == 1).sum()
    n0 = len(y) - n1
    w1 = n0 / max(n1, 1)
    return torch.tensor([1.0, w1], dtype=torch.float32, device=DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# DL training — fixed epochs, best-epoch checkpoint
# ─────────────────────────────────────────────────────────────────────────────


def train_dl_one_fold(
    model: nn.Module,
    train_idx: List[int],
    val_idx: List[int],
    *,
    ds: Dataset,
    lr: float = LR,
    epochs: int = EPOCHS,
    grad_clip: float = GRAD_CLIP,
    warmup_ratio: float = WARMUP_RATIO,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Train for `epochs` rounds with linear-warmup + cosine-annealing LR.
    Best checkpoint = epoch with highest val F1(Issue).
    Returns {"model": trained_model, "history": {...}, "best_epoch": int}
    """
    model = model.to(DEVICE)
    train_loader = _make_loader(ds, train_idx, shuffle=True)
    val_loader = _make_loader(ds, val_idx, shuffle=False)

    _, y_train, _ = collect_numpy(ds, train_idx)
    criterion = nn.CrossEntropyLoss(weight=_class_weights(y_train))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # ── LR schedule: linear warmup → cosine annealing ─────────────────────────
    # Warmup for the first `warmup_ratio` fraction of total steps, then cosine
    # decay to lr/10.  This is step-level so it works regardless of batch count.
    total_steps = epochs * len(train_loader)
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    eta_min = lr / 10.0

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return eta_min / lr + (1.0 - eta_min / lr) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    global_step = 0

    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_f1 = -1.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())

    for ep in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        tr_losses = []
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.long().to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1
            tr_losses.append(loss.item())

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        va_losses, va_preds, va_labels = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.long().to(DEVICE)
                logits = model(xb)
                va_losses.append(criterion(logits, yb).item())
                va_preds.extend(logits.argmax(-1).cpu().numpy())
                va_labels.extend(yb.cpu().numpy())

        tr = float(np.mean(tr_losses))
        va = float(np.mean(va_losses))
        vf1 = float(f1_score(va_labels, va_preds, pos_label=1, zero_division=0))
        history["train_loss"].append(tr)
        history["val_loss"].append(va)
        history["val_f1"].append(vf1)

        if verbose and (ep == 1 or ep % 5 == 0 or ep == epochs):
            cur_lr = scheduler.get_last_lr()[0]
            print(
                f"  ep {ep:03d}/{epochs} | "
                f"lr={cur_lr:.2e}  "
                f"tr_loss={tr:.4f}  val_loss={va:.4f}  val_F1={vf1:.4f}"
                + (" ← best" if vf1 > best_f1 else "")
            )

        if vf1 > best_f1:
            best_f1 = vf1
            best_epoch = ep
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    if verbose:
        print(f"  ✓ best epoch={best_epoch}/{epochs}  best_val_F1={best_f1:.4f}")
    return {"model": model, "history": history, "best_epoch": best_epoch}


def predict_dl(
    model: nn.Module,
    ds: Dataset,
    indices: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    loader = _make_loader(ds, indices, shuffle=False)
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(DEVICE))
            y_pred.append(logits.argmax(-1).cpu().numpy())
            y_true.append(yb.numpy().astype(int))
    return np.concatenate(y_true), np.concatenate(y_pred)


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate features for ML baselines (mean + std over time axis)
# ─────────────────────────────────────────────────────────────────────────────


def aggregate_features(X: np.ndarray) -> np.ndarray:
    """[N, T, D] → [N, D*2] via mean+std (time axis)."""
    return np.concatenate([X.mean(axis=1), X.std(axis=1)], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────


def _save_fold(cache_dir: Path, fold_id: int, data: dict):
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / f"fold_{fold_id:02d}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_existing_folds(cache_dir: Path) -> Dict[int, dict]:
    done = {}
    for p in sorted(cache_dir.glob("fold_*.json")):
        fid = int(p.stem.split("_")[1])
        with open(p, "r", encoding="utf-8") as f:
            done[fid] = json.load(f)
    return done


def _save_final(cache_dir: Path, report: dict):
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / "final_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def _load_final(cache_dir: Path) -> Optional[dict]:
    p = cache_dir / "final_report.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Unified LOSO runner (resumable, per-fold cache)
# ─────────────────────────────────────────────────────────────────────────────


def run_loso(
    spec: ModelSpec,
    ds: Dataset,
    cache_dir: Path,
    *,
    verbose: bool = True,
) -> dict:
    """
    Run (or resume) LOSO for one model spec.

    Parameters
    ----------
    spec      : ModelSpec defining model name, kind and builder
    ds        : Dataset (full, LOSO splits applied internally)
    cache_dir : Path for per-fold + final_report caches
    verbose   : print progress

    Returns
    -------
    final_report dict with keys:
        model, protocol, n_folds, fold_metrics, summary, conf_mats,
        fold_info, histories (DL only)
    """
    # ── Already finished? ─────────────────────────────────────────────────────
    cached = _load_final(cache_dir)
    if cached is not None:
        if verbose:
            s = cached.get("summary", {})
            print(
                f"  ⏭ [cache] {spec.name}: " f"F1={s.get('f1_issue', float('nan')):.4f}"
            )
        return cached

    splits = get_loso_splits(ds)
    n_folds = len(splits)
    assert n_folds > 1, "Need ≥2 participants for LOSO."

    existing = _load_existing_folds(cache_dir)
    fold_metrics = []
    conf_mats = []
    fold_info = []
    histories = []  # DL only

    # Restore already-completed folds (preserving order)
    for fid in range(1, n_folds + 1):
        if fid in existing:
            fr = existing[fid]
            fold_metrics.append(fr["metric"])
            conf_mats.append(np.array(fr["conf_mat"]))
            fold_info.append(fr["fold_info"])
            if fr.get("history") is not None:
                histories.append(fr["history"])

    # Run missing folds
    for fid in range(1, n_folds + 1):
        if fid in existing:
            continue

        set_fold_seed(fid)  # ← V3: reproducible per-fold seed
        tr, te, pid = splits[fid - 1]
        if verbose:
            print(
                f"\n  ▶ {spec.name} | Fold {fid}/{n_folds} "
                f"heldout={pid} | train={len(tr)} test={len(te)}"
            )

        if spec.kind == "ml":
            # ── ML path ───────────────────────────────────────────────────────
            X_tr, y_tr, _ = collect_numpy(ds, tr)
            X_te, y_te, _ = collect_numpy(ds, te)
            X_tr = aggregate_features(X_tr)
            X_te = aggregate_features(X_te)
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
            clf = spec.build()
            n_pos = int((y_tr == 1).sum())
            n_neg = len(y_tr) - n_pos
            sw = np.where(y_tr == 1, n_neg / max(n_pos, 1), 1.0)
            try:
                clf.fit(X_tr, y_tr, sample_weight=sw)
            except TypeError:
                clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)
            metric = compute_metrics(y_te, y_pred)
            cm = confusion_matrix(y_te, y_pred)
            history = None

        else:
            # ── DL path ───────────────────────────────────────────────────────
            model = spec.build()
            out = train_dl_one_fold(model, tr, te, ds=ds, verbose=verbose)
            y_te_np, y_pred = predict_dl(out["model"], ds, te)
            metric = compute_metrics(y_te_np, y_pred)
            cm = confusion_matrix(y_te_np, y_pred)
            history = out["history"]

        fold_metrics.append(metric)
        conf_mats.append(cm)
        fi = {
            "fold": fid,
            "heldout_pid": pid,
            "n_test": int(len(te)),
            "n_train": int(len(tr)),
        }
        fold_info.append(fi)
        if history is not None:
            histories.append(history)

        _save_fold(
            cache_dir,
            fid,
            {
                "metric": metric,
                "conf_mat": cm.tolist(),
                "fold_info": fi,
                "history": history,
            },
        )

        if verbose:
            print(
                f"    P={metric['p_issue']:.4f}  R={metric['r_issue']:.4f}  "
                f"F1={metric['f1_issue']:.4f}  macroF1={metric['f1_macro']:.4f}"
            )

    final = {
        "model": spec.name,
        "protocol": "LOSO",
        "n_folds": n_folds,
        "fold_metrics": fold_metrics,
        "summary": summarize_folds(fold_metrics),
        "conf_mats": [
            cm.tolist() if isinstance(cm, np.ndarray) else cm for cm in conf_mats
        ],
        "fold_info": fold_info,
        "histories": histories if histories else None,
    }
    _save_final(cache_dir, final)

    if verbose:
        s = final["summary"]
        print(
            f"\n  ✅ {spec.name} DONE | "
            f"F1_issue={s['f1_issue']:.4f}  "
            f"P={s['p_issue']:.4f}  R={s['r_issue']:.4f}"
        )
    return final


# ─────────────────────────────────────────────────────────────────────────────
# Batch runners
# ─────────────────────────────────────────────────────────────────────────────


def run_all_models(
    registry: dict,
    ds: Dataset,
    archive_root: Path,
    *,
    names: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, dict]:
    """
    Run LOSO for all (or selected) models in `registry`.

    Parameters
    ----------
    registry     : dict[str, ModelSpec]
    ds           : full dataset
    archive_root : e.g. ABL_DIR or BASE_DIR  (per-model subdirs created inside)
    names        : subset of registry keys to run (None = all)
    """
    names = names or list(registry.keys())
    results = {}
    for name in names:
        spec = registry[name]
        print(f"\n{'═'*60}")
        print(f"  Model: {name}  ({spec.kind.upper()})")
        print(f"{'═'*60}")
        results[name] = run_loso(
            spec,
            ds,
            cache_dir=archive_root / name.replace("/", "-"),
            verbose=verbose,
        )
    return results
