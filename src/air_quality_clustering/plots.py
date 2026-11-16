"""Unified figure production (300 dpi PNG).

Every figure carries units, sample size and method/parameters in its caption
or footer. EDA figures F01-F04 come from the real derived tables; candidate
figures F05-F07 come from the actual experiments table. No figure is produced
from simulated data.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.cluster.hierarchy import dendrogram  # noqa: E402

from .data import POLLUTANTS  # noqa: E402

DPI = 300
UNITS = "µg/m³"


def _save(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_f01_monthly_coverage(coverage: pd.DataFrame, path: str | Path) -> Path:
    """Monthly valid-hour fraction per pollutant and eligible-day fraction."""
    work = coverage.copy()
    work["month"] = work["date"].str[:7]
    count_cols = [f"n_valid_{c}" for c in POLLUTANTS]
    monthly = work.groupby("month")[count_cols].mean() / 24.0
    eligible_rate = work.groupby("month")["eligible"].mean()

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for column in count_cols:
        axes[0].plot(monthly.index, monthly[column], marker=".", label=column.replace("n_valid_", ""))
    axes[0].axhline(18 / 24, color="grey", linestyle="--", linewidth=1)
    axes[0].set_ylabel("valid-hour fraction")
    axes[0].set_title("F01 Monthly coverage (all 1,461 calendar days; dashed line = 18/24 h rule)")
    axes[0].legend(ncol=6, fontsize=8)
    axes[1].bar(eligible_rate.index, eligible_rate.to_numpy(), color="#4878a8")
    axes[1].set_ylabel("eligible-day fraction")
    axes[1].set_ylim(0, 1.05)
    step = max(1, len(monthly) // 16)
    axes[1].set_xticks(range(0, len(monthly), step))
    axes[1].set_xticklabels(monthly.index[::step], rotation=90, fontsize=7)
    fig.tight_layout()
    return _save(fig, path)


def fig_f02_distributions(daily: pd.DataFrame, path: str | Path) -> Path:
    """Raw vs log1p daily distributions: motivation for the transformation."""
    fig, axes = plt.subplots(2, 6, figsize=(16, 6), sharex="col")
    values = daily[POLLUTANTS].to_numpy(dtype="float64")
    for j, column in enumerate(POLLUTANTS):
        sns.histplot(values[:, j], bins=40, ax=axes[0, j], color="#4878a8")
        axes[0, j].set_title(f"{column} raw", fontsize=9)
        axes[0, j].set_xlabel(UNITS, fontsize=8)
        sns.histplot(np.log1p(values[:, j]), bins=40, ax=axes[1, j], color="#d1495b")
        axes[1, j].set_title(f"{column} log1p", fontsize=9)
        axes[1, j].set_xlabel("log1p(" + UNITS + ")", fontsize=8)
    fig.suptitle(f"F02 Daily concentration distributions (n={len(daily)} eligible days)")
    fig.tight_layout()
    return _save(fig, path)


def fig_f03_correlation(daily: pd.DataFrame, path: str | Path) -> Path:
    """Spearman correlation of raw daily concentrations."""
    corr = daily[POLLUTANTS].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, ax=ax)
    ax.set_title(f"F03 Spearman correlation, raw daily means (n={len(daily)})")
    fig.tight_layout()
    return _save(fig, path)


def fig_f04_time_series(
    daily: pd.DataFrame, coverage: pd.DataFrame, path: str | Path
) -> Path:
    """Daily means over the full calendar axis; excluded days stay as gaps."""
    full = pd.DataFrame({"date": coverage["date"]})
    merged = full.merge(daily, on="date", how="left")
    x = pd.to_datetime(merged["date"])
    fig, axes = plt.subplots(6, 1, figsize=(13, 11), sharex=True)
    for ax, column in zip(axes, POLLUTANTS):
        ax.plot(x, merged[column], linewidth=0.6, color="#4878a8")
        ax.set_ylabel(column, fontsize=9)
    axes[0].set_title(
        f"F04 Daily mean concentrations ({UNITS}); gaps = {int(coverage['eligible'].eq(False).sum())} excluded days"
    )
    axes[-1].set_xlabel("date (Asia/Shanghai local)")
    fig.tight_layout()
    return _save(fig, path)


def fig_f05_kmeans_metrics(
    experiments: pd.DataFrame, seed_aris: dict[int, list[float]], path: str | Path
) -> Path:
    """K-Means candidate diagnostics: inertia, metrics vs k, seed agreement."""
    km = experiments[experiments["algorithm"] == "kmeans"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    rep = km[km["seed"] == 100].sort_values("k")
    axes[0].plot(rep["k"], rep["inertia"], marker="o", color="#4878a8")
    axes[0].set_title("Inertia elbow (seed=100)")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("inertia")
    grouped = km.groupby("k")[["silhouette", "davies_bouldin", "calinski_harabasz"]].median()
    axes[1].plot(grouped.index, grouped["silhouette"], marker="o", label="silhouette (median)")
    axes[1].plot(grouped.index, grouped["davies_bouldin"], marker="s", label="DB (median)")
    ax2 = axes[1].twinx()
    ax2.plot(grouped.index, grouped["calinski_harabasz"], marker="^", color="green",
             label="CH (median)")
    ax2.set_ylabel("CH")
    axes[1].set_title("Internal metrics vs k (10-seed medians)")
    axes[1].set_xlabel("k")
    axes[1].legend(fontsize=7, loc="upper left")
    ks = sorted(seed_aris)
    axes[2].boxplot([seed_aris[k] for k in ks], tick_labels=[str(k) for k in ks])
    axes[2].set_title("Seed agreement: pairwise ARI (45 pairs/k)")
    axes[2].set_xlabel("k")
    axes[2].set_ylabel("ARI")
    fig.suptitle("F05 K-Means candidates E01 (70 settings, all retained)", y=1.02)
    fig.tight_layout()
    return _save(fig, path)


def fig_f06_dendrogram(Z: np.ndarray, path: str | Path, last_merges: int = 30) -> Path:
    """Truncated dendrogram from the SAME linkage used for the cuts."""
    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(Z, truncate_mode="lastp", p=last_merges, ax=ax, color_threshold=0,
               above_threshold_color="#4878a8")
    ax.set_title(
        f"F06 Ward dendrogram (euclidean, one tree; truncated display of last {last_merges} merges)"
    )
    ax.set_ylabel("merge distance")
    fig.tight_layout()
    return _save(fig, path)


def fig_f07_dbscan(
    grid: dict[int, dict], experiments: pd.DataFrame, path: str | Path
) -> Path:
    """k-distance curves with chosen eps quantiles + noise/coverage table."""
    db = experiments[experiments["algorithm"] == "dbscan"]
    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.3])
    for i, m in enumerate(sorted(grid)):
        ax = fig.add_subplot(gs[0, i])
        kdist = grid[m]["kdist_sorted"]
        ax.plot(np.arange(1, len(kdist) + 1), kdist, linewidth=0.8, color="#4878a8")
        for q, eps in grid[m]["eps_by_quantile"].items():
            ax.axhline(eps, linestyle="--", linewidth=0.8,
                       label=f"q{int(q * 100)}={eps:.2f}")
        ax.set_title(f"m={m}: sorted m-th NN distance (self included)")
        ax.set_xlabel("sample rank")
        ax.legend(fontsize=6)
    ax = fig.add_subplot(gs[0, 3])
    ax.axis("off")
    table_rows = []
    for _, row in db.iterrows():
        table_rows.append([
            row["experiment_id"], f"{row['eps']:.2f}" if pd.notna(row["eps"]) else "-",
            int(row["n_clusters"]) if pd.notna(row["n_clusters"]) else "-",
            f"{row['noise_fraction']:.2f}" if pd.notna(row["noise_fraction"]) else "-",
            row["status"],
        ])
    ax.table(cellText=table_rows, colLabels=["setting", "eps", "clusters", "noise", "status"],
             loc="center", cellLoc="center")
    ax.set_title("E03 outcomes (12 settings)")
    fig.suptitle("F07 DBSCAN density assumptions on the main representation", y=1.02)
    fig.tight_layout()
    return _save(fig, path)
