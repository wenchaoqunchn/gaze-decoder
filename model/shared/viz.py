# shared/viz.py
# ─────────────────────────────────────────────────────────────────────────────
# GazeDecoder V3 — Visualization & Statistical Testing
# Shared by both ablation and baselines notebooks.
# ─────────────────────────────────────────────────────────────────────────────
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

from shared.training import summarize_folds


# ─────────────────────────────────────────────────────────────────────────────
# §A  Results → DataFrame
# ─────────────────────────────────────────────────────────────────────────────


def results_to_df(results: Dict[str, dict]) -> pd.DataFrame:
    """Convert a {model_name: final_report} dict to a tidy summary DataFrame."""
    rows = []
    for name, rep in results.items():
        s = rep.get("summary") or summarize_folds(rep["fold_metrics"])
        rows.append(
            {
                "model": name,
                "kind": rep.get("kind", "dl"),
                "precision_issue": s["p_issue"],
                "recall_issue": s["r_issue"],
                "f1_issue": s["f1_issue"],
                "f1_macro": s["f1_macro"],
                "acc": s["acc"],
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["f1_issue", "f1_macro"], ascending=False)
        .reset_index(drop=True)
    )


def fold_metric_vector(
    results: Dict[str, dict], model: str, metric: str = "f1_issue"
) -> np.ndarray:
    return np.array([fm[metric] for fm in results[model]["fold_metrics"]], dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# §B  Bar charts
# ─────────────────────────────────────────────────────────────────────────────


def plot_f1_leaderboard(
    df: pd.DataFrame,
    title: str = "F1-score (Issue class) — LOSO mean",
    save_path: Optional[str] = None,
):
    plt.figure(figsize=(10, 6))
    palette = {"dl": "#ED7D31", "ml": "#5B9BD5"}
    sns.barplot(
        data=df, y="model", x="f1_issue", hue="kind", dodge=False, palette=palette
    )
    plt.title(title)
    plt.xlim(0, 1)
    plt.grid(axis="x", alpha=0.3)
    plt.legend(title="kind")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
        print(f"✅ Saved: {save_path}")
    plt.show()


def plot_prf_grouped(
    df: pd.DataFrame,
    top_k: int = 10,
    title: str = "Precision / Recall / F1 (Issue) — Top-K models",
    save_path: Optional[str] = None,
):
    top = df.head(top_k).copy()
    melt = top.melt(
        id_vars=["model"],
        value_vars=["precision_issue", "recall_issue", "f1_issue"],
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(12, 5))
    sns.barplot(data=melt, x="model", y="score", hue="metric")
    plt.title(title)
    plt.ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
        print(f"✅ Saved: {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# §C  Confusion matrices
# ─────────────────────────────────────────────────────────────────────────────


def plot_mean_confmat(
    results: Dict[str, dict],
    model_name: str,
    title: Optional[str] = None,
):
    cms = results[model_name].get("conf_mats")
    if not cms:
        print(f"  [warn] No conf_mats found for {model_name}")
        return
    cm_mean = np.mean(np.stack([np.asarray(x) for x in cms]), axis=0)
    plt.figure(figsize=(3.6, 3.2))
    sns.heatmap(
        np.round(cm_mean).astype(int), annot=True, fmt="d", cmap="Blues", cbar=False
    )
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.title(title or f"{model_name} — mean confusion matrix (LOSO)")
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# §D  Training curves (DL only)
# ─────────────────────────────────────────────────────────────────────────────


def plot_training_curves(
    histories: List[Dict[str, List[float]]],
    title: str = "Fold-averaged training curves",
    save_path: Optional[str] = None,
):
    if not histories:
        print("  [warn] No histories to plot.")
        return
    max_len = max(len(h["train_loss"]) for h in histories)

    def pad(arr):
        return np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan)

    tr_m = np.nanmean([pad(h["train_loss"]) for h in histories], axis=0)
    va_m = np.nanmean([pad(h["val_loss"]) for h in histories], axis=0)
    vf_m = np.nanmean([pad(h["val_f1"]) for h in histories], axis=0)

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()
    ax1.plot(tr_m, label="train loss", color="#5B9BD5", linewidth=1.8)
    ax1.plot(va_m, label="val loss", color="#ED7D31", linewidth=1.8)
    ax2.plot(vf_m, label="val F1", color="#70AD47", linewidth=1.8, linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("Val F1 (Issue)")
    ax1.set_title(title)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax1.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
        print(f"✅ Saved: {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# §E  Statistical tests (identical to V2 §10)
# ─────────────────────────────────────────────────────────────────────────────


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    return float(d.mean() / (d.std(ddof=1) + 1e-12))


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    pool = np.sqrt(
        ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2) + 1e-12
    )
    d = (a.mean() - b.mean()) / pool
    J = 1 - 3 / (4 * (na + nb - 2) - 1)
    return float(d * J)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = sum(np.sum(ai > b) for ai in a)
    lt = sum(np.sum(ai < b) for ai in a)
    return float((gt - lt) / max(len(a) * len(b), 1))


def bootstrap_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, ci: float = 0.95, seed: int = 42
) -> Tuple[float, float]:
    """Bootstrap CI for mean(a−b) via percentile method."""
    rng = np.random.default_rng(seed)
    diff = np.asarray(a, float) - np.asarray(b, float)
    boots = [
        rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(n_boot)
    ]
    lo = np.percentile(boots, 100 * (1 - ci) / 2)
    hi = np.percentile(boots, 100 * (1 - (1 - ci) / 2))
    return float(lo), float(hi)


def paired_test(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    """Shapiro → paired-t or Wilcoxon + Bootstrap 95% CI."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a - b
    sh_p = float(stats.shapiro(diff).pvalue) if len(diff) >= 3 else float("nan")
    ci_lo, ci_hi = bootstrap_ci(a, b)
    if np.isnan(sh_p) or sh_p >= 0.05:
        t = stats.ttest_rel(a, b)
        return {
            "test": "paired_t",
            "p_value": float(t.pvalue),
            "ci95_lo": ci_lo,
            "ci95_hi": ci_hi,
            "effect": cohens_d(a, b),
            "effect_name": "cohens_d",
            "normality_p": sh_p,
        }
    else:
        w = stats.wilcoxon(a, b, zero_method="wilcox", correction=False)
        return {
            "test": "wilcoxon",
            "p_value": float(w.pvalue),
            "ci95_lo": ci_lo,
            "ci95_hi": ci_hi,
            "effect": cliffs_delta(a, b),
            "effect_name": "cliffs_delta",
            "normality_p": sh_p,
        }


def significance_table(
    results: Dict[str, dict],
    proposed: str,
    metric: str = "f1_issue",
) -> pd.DataFrame:
    """
    Compare `proposed` model against every other in `results`.
    Returns a DataFrame with p-value, Bootstrap 95% CI, effect size.
    """
    ref = fold_metric_vector(results, proposed, metric)
    rows = []
    for name in results:
        if name == proposed:
            continue
        vec = fold_metric_vector(results, name, metric)
        info = paired_test(ref, vec)
        rows.append(
            {
                "baseline": name,
                f"{proposed}_mean": float(ref.mean()),
                "baseline_mean": float(vec.mean()),
                "delta_mean": float((ref - vec).mean()),
                "ci95_lo": info["ci95_lo"],
                "ci95_hi": info["ci95_hi"],
                "ci_positive": info["ci95_lo"] > 0,
                "test": info["test"],
                "p_value": info["p_value"],
                info["effect_name"]: info["effect"],
                "n_folds": int(len(ref)),
            }
        )
    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# §F  Scott-Knott ESD ranking
# ─────────────────────────────────────────────────────────────────────────────


def _sk_split(
    vectors: List[np.ndarray],
    names: List[str],
    g_thr: float = 0.2,
) -> List[List[str]]:
    if len(names) <= 1:
        return [[n] for n in names]
    means = np.array([v.mean() for v in vectors])
    order = np.argsort(means)
    vectors = [vectors[i] for i in order]
    names = [names[i] for i in order]
    n = len(names)
    best_g, best_k = -1.0, 1
    for k in range(1, n):
        g = abs(hedges_g(np.concatenate(vectors[k:]), np.concatenate(vectors[:k])))
        if g > best_g:
            best_g, best_k = g, k
    if best_g < g_thr:
        return [names]
    return _sk_split(vectors[:best_k], names[:best_k], g_thr) + _sk_split(
        vectors[best_k:], names[best_k:], g_thr
    )


def scott_knott_esd(
    results: Dict[str, dict],
    metric: str = "f1_issue",
    g_thr: float = 0.2,
) -> pd.DataFrame:
    """
    Assign Scott-Knott ESD rank tiers.
    Tier 1 = best group, same tier = no meaningful difference (Hedges' g < g_thr).
    """
    names = list(results.keys())
    vectors = [fold_metric_vector(results, n, metric) for n in names]
    groups = list(reversed(_sk_split(vectors, names, g_thr)))
    rows = []
    for tier, group in enumerate(groups, start=1):
        for model in group:
            vec = fold_metric_vector(results, model, metric)
            rows.append(
                {
                    "rank_tier": tier,
                    "model": model,
                    f"{metric}_mean": float(vec.mean()),
                    f"{metric}_std": float(vec.std(ddof=1)),
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["rank_tier", f"{metric}_mean"], ascending=[True, False])
        .reset_index(drop=True)
    )


def plot_scott_knott(
    sk_df: pd.DataFrame,
    proposed: Optional[str] = None,
    metric: str = "f1_issue",
    title: str = "Scott-Knott ESD Ranking",
    save_path: Optional[str] = None,
):
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    palette = {}
    for m in sk_df["model"]:
        palette[m] = "#ED7D31" if m == proposed else "#5B9BD5"

    fig, ax = plt.subplots(figsize=(11, max(4, len(sk_df) * 0.45)))
    for _, row in sk_df.iterrows():
        ax.barh(
            row["model"],
            row[mean_col],
            color=palette[row["model"]],
            alpha=0.85,
            xerr=row[std_col],
            capsize=3,
            error_kw={"elinewidth": 1},
        )

    tier_grps = sk_df.groupby("rank_tier")["model"].apply(list)
    for tier, models in tier_grps.items():
        ys = [sk_df[sk_df["model"] == m].index[0] for m in models]
        ax.annotate(
            f"Tier {tier}",
            xy=(1.005, (min(ys) + max(ys)) / 2),
            xycoords=("axes fraction", "data"),
            fontsize=8,
            va="center",
            color="grey",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="grey", lw=0.7),
        )

    ax.set_xlabel(f"Mean {metric}")
    ax.set_title(f"{title}\nSame tier = no meaningful difference" " (Hedges' g < 0.2)")
    ax.set_xlim(0.5, 1.0)
    ax.grid(axis="x", alpha=0.3)

    handles = []
    if proposed:
        handles.append(mpatches.Patch(color="#ED7D31", label=f"{proposed} (proposed)"))
    handles.append(mpatches.Patch(color="#5B9BD5", label="Others"))
    ax.legend(handles=handles, loc="lower right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
        print(f"✅ Saved: {save_path}")
    plt.show()
