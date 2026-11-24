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


def cluster_spread(daily: pd.DataFrame, labels: np.ndarray, algorithm: str) -> pd.DataFrame:
    """Within-cluster dispersion in ORIGINAL units (protocol §8: compare this first).

    IQR ratio = IQR / median, a scale-free read of how heterogeneous a cluster is.
    """
    frame = daily[FEATURE_ORDER].copy()
    frame["cluster"] = labels
    rows = []
    for cluster, group in frame.groupby("cluster"):
        for column in FEATURE_ORDER:
            values = group[column].to_numpy(dtype="float64")
            q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
            rows.append({
                "algorithm": algorithm, "cluster": int(cluster), "pollutant": column,
                "n_days": int(len(values)), "median": float(median),
                "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1),
                "iqr_over_median": float((q3 - q1) / median) if median > 0 else None,
                "p05": float(np.quantile(values, 0.05)),
                "p95": float(np.quantile(values, 0.95)),
                "min": float(values.min()), "max": float(values.max()),
            })
    return pd.DataFrame(rows)


def overlap_evidence(spread: pd.DataFrame, algorithm: str) -> pd.DataFrame:
    """Which pollutants separate the clusters and which overlap heavily.

    Overlap is the shared span of the two clusters' IQRs, expressed both in
    original units and as a fraction of the pooled IQR span.
    """
    sub = spread[spread["algorithm"] == algorithm]
    clusters = sorted(sub["cluster"].unique())
    rows = []
    for column in FEATURE_ORDER:
        data = sub[sub["pollutant"] == column].set_index("cluster")
        if len(clusters) < 2:
            continue
        a, b = data.loc[clusters[0]], data.loc[clusters[1]]
        lo = max(a["q1"], b["q1"])
        hi = min(a["q3"], b["q3"])
        shared = max(0.0, hi - lo)
        pooled = max(a["q3"], b["q3"]) - min(a["q1"], b["q1"])
        rows.append({
            "algorithm": algorithm, "pollutant": column,
            "cluster_a": int(clusters[0]), "cluster_b": int(clusters[1]),
            "iqr_overlap_low": float(lo) if shared > 0 else None,
            "iqr_overlap_high": float(hi) if shared > 0 else None,
            "iqr_overlap_width": float(shared),
            "overlap_fraction_of_pooled_iqr": float(shared / pooled) if pooled > 0 else None,
            "median_ratio_b_over_a": float(b["median"] / a["median"]) if a["median"] > 0 else None,
        })
    return pd.DataFrame(rows)


def disagreement_case(
    daily: pd.DataFrame, labels_km: np.ndarray, labels_ward: np.ndarray, top_n: int = 5
) -> dict:
    """Fixed-rule case study of the largest split/merge cell.

    Rule (protocol §7): pick the largest OFF-DIAGONAL contingency cell, then the
    earliest ``top_n`` dates inside it — never chosen by visual impression.
    """
    cont = contingency(labels_km, labels_ward)
    # Cluster numbers are arbitrary: the "agreement" cells are the optimal
    # one-to-one matching, NOT the numeric diagonal.
    from scipy.optimize import linear_sum_assignment

    matrix = cont.to_numpy(dtype=float)
    rows_idx, cols_idx = linear_sum_assignment(-matrix)
    matched = {(int(cont.index[i]), int(cont.columns[j])) for i, j in zip(rows_idx, cols_idx)}
    off = cont.copy().astype(float)
    for i in off.index:
        for j in off.columns:
            if (int(i), int(j)) in matched:
                off.loc[i, j] = np.nan
    values = off.to_numpy(dtype=float)
    if not np.isfinite(values).any() or np.nanmax(values) <= 0:
        return {"status": "no_off_diagonal_cell", "cell": None, "examples": [],
                "matched_pairs": sorted(matched)}
    stacked = off.stack()
    km_cell, ward_cell = stacked.idxmax()
    n_cell = int(cont.loc[km_cell, ward_cell])
    mask = (labels_km == int(km_cell)) & (labels_ward == int(ward_cell))
    dates = daily.loc[mask, "date"].tolist()[:top_n]
    example_rows = daily[daily["date"].isin(dates)][["date", *FEATURE_ORDER]]
    agreements = {
        "km_cluster_total": int((labels_km == int(km_cell)).sum()),
        "ward_cluster_total": int((labels_ward == int(ward_cell)).sum()),
        "intersection_days": n_cell,
        "km_only_days": int(((labels_km == int(km_cell)) & (labels_ward != int(ward_cell))).sum()),
        "ward_only_days": int(((labels_km != int(km_cell)) & (labels_ward == int(ward_cell))).sum()),
    }
    return {
        "status": "ok",
        "rule": "largest off-diagonal contingency cell, then earliest dates",
        "cell": {"kmeans_cluster": int(km_cell), "ward_cluster": int(ward_cell), "n_days": n_cell},
        "counts": agreements,
        "examples": example_rows.to_dict(orient="records"),
        "example_median_original_units": {
            c: float(example_rows[c].median()) for c in FEATURE_ORDER
        } if len(example_rows) else {},
    }


def season_coverage(coverage: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    """Eligible-day coverage per season with the actual calendar denominator."""
    merged = coverage[["date", "eligible"]].merge(context[["date", "season"]], on="date", how="left")
    rows = []
    for season, group in merged.groupby("season"):
        rows.append({
            "season": season,
            "calendar_days": int(len(group)),
            "eligible_days": int(group["eligible"].sum()),
            "coverage_fraction": float(group["eligible"].mean()),
        })
    return pd.DataFrame(rows)


def year_composition(context: pd.DataFrame, labels: np.ndarray, daily: pd.DataFrame,
                     algorithm: str) -> pd.DataFrame:
    """P(cluster | year) with the eligible-day count of each year as denominator."""
    frame = daily[["date"]].copy()
    frame["cluster"] = labels
    merged = frame.merge(context[["date", "year"]], on="date", how="left")
    counts = pd.crosstab(merged["year"], merged["cluster"])
    pct = counts.div(counts.sum(axis=1), axis=0)
    pct.columns = [f"pct_cluster_{int(c)}" for c in pct.columns]
    out = counts.copy()
    out.columns = [f"n_cluster_{int(c)}" for c in out.columns]
    out["eligible_days_in_year"] = counts.sum(axis=1).to_numpy()
    out = out.join(pct)
    out.insert(0, "algorithm", algorithm)
    return out.reset_index()


def sensitivity_labels(daily: pd.DataFrame, ks: dict[str, int], cfg) -> dict[str, np.ndarray]:
    """Partitions of the raw+StandardScaler branch at the frozen k values."""
    from scipy.cluster.hierarchy import cut_tree, linkage
    from sklearn.cluster import KMeans

    from .features import build_feature_matrix

    raw_matrix = build_feature_matrix(daily, cfg.sensitivity_representation)
    out: dict[str, np.ndarray] = {}
    ward_cuts = None
    for algorithm, k in ks.items():
        if algorithm == "kmeans":
            out[algorithm] = KMeans(
                n_clusters=k, n_init=int(cfg.algorithms["kmeans"]["n_init"]),
                max_iter=int(cfg.algorithms["kmeans"]["max_iter"]),
                algorithm=cfg.algorithms["kmeans"]["algorithm"],
                random_state=int(cfg.algorithms["kmeans"]["representative_seed"]),
            ).fit(raw_matrix.X).labels_
        else:
            if ward_cuts is None:
                Z = linkage(raw_matrix.X, method=cfg.algorithms["ward"]["linkage"],
                            metric=cfg.algorithms["ward"]["metric"])
                ward_cuts = {kk: cut_tree(Z, n_clusters=[kk])[:, 0] for kk in set(ks.values())}
            out[algorithm] = ward_cuts[k]
    return out


def sensitivity_branch(
    daily: pd.DataFrame,
    main_labels: dict[str, np.ndarray],
    ks: dict[str, int],
    cfg,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """raw+StandardScaler at frozen k; ARI vs main representation + sizes."""
    branch = sensitivity_labels(daily, ks, cfg)
    rows = []
    size_rows = []
    for algorithm, k in ks.items():
        lab = branch[algorithm]
        ref_id = (f"E01_k{k}_s{cfg.algorithms['kmeans']['representative_seed']}"
                  if algorithm == "kmeans" else f"E02_k{k}")
        unique, counts = np.unique(lab, return_counts=True)
        rows.append({
            "algorithm": algorithm, "k": k,
            "representation": cfg.sensitivity_representation,
            "ari_vs_main": ari(lab, main_labels[ref_id]),
            "cluster_sizes": {int(c): int(n) for c, n in zip(unique, counts)},
        })
        for c, n in zip(unique, counts):
            size_rows.append({"algorithm": algorithm, "cluster": int(c), "n_days": int(n)})
    return pd.DataFrame(rows), pd.DataFrame(size_rows)
