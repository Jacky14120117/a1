"""Knowledge interpretation tables (T09-T10), all on original units.

Cluster profiles are medians/IQRs of the RAW daily concentrations grouped by
the actual labels — log-space centroids are never back-transformed and called
means. Background (season/year/weather) joins by date AFTER the selection is
frozen; every proportion states its denominator. Descriptive only: no causal
claims, no significance tests on the clustering features themselves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .config import FEATURE_ORDER
from .evaluation import ari

WEATHER_COLUMNS = ["TEMP_mean", "PRES_mean", "DEWP_mean", "WSPM_mean", "RAIN_sum"]


def cluster_profiles(daily: pd.DataFrame, labels: np.ndarray, algorithm: str) -> pd.DataFrame:
    rows = []
    frame = daily.copy()
    frame["cluster"] = labels
    for cluster, group in frame.groupby("cluster"):
        for column in FEATURE_ORDER:
            values = group[column].to_numpy(dtype="float64")
            rows.append({
                "algorithm": algorithm, "cluster": int(cluster), "n_days": int(len(group)),
                "pollutant": column, "median": float(np.median(values)),
                "q1": float(np.quantile(values, 0.25)), "q3": float(np.quantile(values, 0.75)),
                "mean_original_units": float(values.mean()),
            })
    return pd.DataFrame(rows)


def display_mapping(profiles: pd.DataFrame, algorithm: str) -> dict[int, int]:
    """Display order by median PM2.5 ascending; original labels kept in data."""
    sub = profiles[(profiles["algorithm"] == algorithm) & (profiles["pollutant"] == "PM2.5")]
    ordered = sub.sort_values("median")["cluster"].tolist()
    return {int(c): i + 1 for i, c in enumerate(ordered)}


def contingency(labels_a: np.ndarray, labels_b: np.ndarray) -> pd.DataFrame:
    table = pd.crosstab(pd.Series(labels_a, name="kmeans"), pd.Series(labels_b, name="ward"))
    return table


def background_tables(
    daily: pd.DataFrame, context: pd.DataFrame, labels: np.ndarray, algorithm: str
) -> dict[str, pd.DataFrame]:
    frame = daily[["date"]].copy()
    frame["cluster"] = labels
    merged = frame.merge(context, on="date", how="left")

    season_counts = pd.crosstab(merged["cluster"], merged["season"])
    season_pct_of_cluster = season_counts.div(season_counts.sum(axis=1), axis=0)
    season_pct_of_season = season_counts.div(season_counts.sum(axis=0), axis=1)
    year_counts = pd.crosstab(merged["cluster"], merged["year"])

    weather_rows = []
    for cluster, group in merged.groupby("cluster"):
        for column in WEATHER_COLUMNS:
            values = group[column].dropna().to_numpy(dtype="float64")
            weather_rows.append({
                "algorithm": algorithm, "cluster": int(cluster), "variable": column,
                "n_valid_days": int(len(values)),
                "median": float(np.median(values)) if len(values) else None,
                "q1": float(np.quantile(values, 0.25)) if len(values) else None,
                "q3": float(np.quantile(values, 0.75)) if len(values) else None,
            })
    return {
        "season_counts": season_counts,
        "season_pct_of_cluster": season_pct_of_cluster,
        "season_pct_of_season": season_pct_of_season,
        "year_counts": year_counts,
        "weather_profiles": pd.DataFrame(weather_rows),
    }


def pca_display(X: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit ONCE on the full main matrix; display only, never model input."""
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(X)
    return coords, pca.explained_variance_ratio_


def sensitivity_branch(
    daily: pd.DataFrame,
    main_labels: dict[str, np.ndarray],
    ks: dict[str, int],
    cfg,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """raw+StandardScaler at frozen k; ARI vs main representation + sizes."""
    from .features import build_feature_matrix

    raw_matrix = build_feature_matrix(daily, cfg.sensitivity_representation)
    from sklearn.cluster import KMeans
    from scipy.cluster.hierarchy import cut_tree, linkage

    rows = []
    size_rows = []
    ward_cuts = None
    for algorithm, k in ks.items():
        if algorithm == "kmeans":
            lab = KMeans(
                n_clusters=k, n_init=int(cfg.algorithms["kmeans"]["n_init"]),
                max_iter=int(cfg.algorithms["kmeans"]["max_iter"]),
                algorithm=cfg.algorithms["kmeans"]["algorithm"],
                random_state=int(cfg.algorithms["kmeans"]["representative_seed"]),
            ).fit(raw_matrix.X).labels_
            ref_id = f"E01_k{k}_s{cfg.algorithms['kmeans']['representative_seed']}"
        else:
            if ward_cuts is None:
                Z = linkage(raw_matrix.X, method=cfg.algorithms["ward"]["linkage"],
                            metric=cfg.algorithms["ward"]["metric"])
                ward_cuts = {kk: cut_tree(Z, n_clusters=[kk])[:, 0] for kk in ks.values()}
            lab = ward_cuts[k]
            ref_id = f"E02_k{k}"
        agreement = ari(lab, main_labels[ref_id])
        unique, counts = np.unique(lab, return_counts=True)
        rows.append({
            "algorithm": algorithm, "k": k,
            "representation": cfg.sensitivity_representation,
            "ari_vs_main": agreement,
            "cluster_sizes": {int(c): int(n) for c, n in zip(unique, counts)},
        })
        for c, n in zip(unique, counts):
            size_rows.append({"algorithm": algorithm, "cluster": int(c), "n_days": int(n)})
    return pd.DataFrame(rows), pd.DataFrame(size_rows)
