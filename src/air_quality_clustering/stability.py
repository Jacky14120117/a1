"""Temporal-block stability and frozen representative selection (T08).

Protocol v1 sections 4-6:
- blocks of 7 calendar days anchored at data.start (209 blocks here);
- 20 repetitions, seeds 100..119, floor(0.8 * n_blocks) blocks without
  replacement; eligible days inside retained blocks form the subset;
- scaler AND clustering are refit on each subset; ARI against the full-data
  reference restricted to common dates;
- shortlist: per-method silhouette within 0.02 of the eligible best;
  minimum cluster size max(10, ceil(0.01*n));
- representative: median ARI max, ties within 0.02 prefer smaller k, then
  higher silhouette, then ascending k; median ARI < 0.70 flags weak stability;
- DBSCAN representative needs >=2 non-noise clusters, min cluster size,
  coverage >= 0.70 and valid metrics; noise-set Jaccard reported separately.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree, linkage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

from .config import FEATURE_ORDER
from .evaluation import ari


def block_ids(dates: np.ndarray, anchor: date, block_days: int) -> np.ndarray:
    anchor_ts = pd.Timestamp(anchor)
    offsets = (pd.to_datetime(pd.Series(dates)) - anchor_ts).dt.days.to_numpy()
    return (offsets // block_days).astype("int64")


def draw_block_subsets(
    n_blocks: int, cfg, anchor: date
) -> list[dict[str, Any]]:
    """Deterministic per-seed block draws shared by every method."""
    st = cfg.stability
    retain = int(math.floor(float(st["retain_block_fraction"]) * n_blocks))
    subsets = []
    for seed in st["seeds"]:
        rng = np.random.default_rng(int(seed))
        chosen = np.sort(rng.choice(n_blocks, size=retain, replace=False))
        subsets.append({
            "seed": int(seed),
            "block_ids": chosen,
            "date_start": str(anchor + timedelta(days=int(chosen[0]) * int(st["block_days"]))),
            "date_end": str(anchor + timedelta(days=int(chosen[-1] + 1) * int(st["block_days"]) - 1)),
        })
    return subsets


def _refit_transform(raw: np.ndarray, representation: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transformed = np.log1p(raw) if representation == "log1p_standard_scaler" else raw
    scaler = StandardScaler()
    X = scaler.fit_transform(transformed)
    return X, scaler.mean_, scaler.scale_


def _valid_partition(labels: np.ndarray) -> bool:
    assigned = labels[labels != -1]
    return len(np.unique(assigned)) >= 2


def block_stability(
    raw: np.ndarray,
    dates: np.ndarray,
    ref_labels: dict[str, np.ndarray],
    km_ks: list[int],
    ward_ks: list[int],
    cfg,
    dbscan_setting: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Refit-on-subset ARI for every shortlisted k plus optional DBSCAN."""
    st = cfg.stability
    anchor = cfg.start_date
    bids = block_ids(dates, anchor, int(st["block_days"]))
    n_blocks = int(bids.max()) + 1
    subsets = draw_block_subsets(n_blocks, cfg, anchor)

    records: list[dict] = []
    db_records: list[dict] = []
    date_index = {d: i for i, d in enumerate(dates)}

    for sub in subsets:
        mask = np.isin(bids, sub["block_ids"])
        idx = np.flatnonzero(mask)
        sub_dates = dates[idx]
        X_sub, _, _ = _refit_transform(raw[idx], cfg.main_representation)
        n_sub = len(idx)

        for k in km_ks:
            model = KMeans(
                n_clusters=k, n_init=int(cfg.algorithms["kmeans"]["n_init"]),
                max_iter=int(cfg.algorithms["kmeans"]["max_iter"]),
                algorithm=cfg.algorithms["kmeans"]["algorithm"],
                random_state=int(cfg.algorithms["kmeans"]["representative_seed"]),
            ).fit(X_sub)
            common = sub_dates
            ref = ref_labels[f"E01_k{k}_s{cfg.algorithms['kmeans']['representative_seed']}"]
            ref_sub = ref[[date_index[d] for d in common]]
            ok = _valid_partition(model.labels_) and _valid_partition(ref_sub)
            records.append({
                "algorithm": "kmeans", "k": k, "seed": sub["seed"],
                "n_subset": n_sub, "n_common": len(common),
                "ari": ari(model.labels_, ref_sub) if ok else None,
                "status": "ok" if ok else "invalid_partition",
            })

        if ward_ks:
            Z = linkage(X_sub, method=cfg.algorithms["ward"]["linkage"],
                        metric=cfg.algorithms["ward"]["metric"])
            cuts = cut_tree(Z, n_clusters=ward_ks)
            for j, k in enumerate(ward_ks):
                lab = cuts[:, j]
                ref = ref_labels[f"E02_k{k}"]
                ref_sub = ref[[date_index[d] for d in sub_dates]]
                ok = _valid_partition(lab) and _valid_partition(ref_sub)
                records.append({
                    "algorithm": "ward", "k": k, "seed": sub["seed"],
                    "n_subset": n_sub, "n_common": len(sub_dates),
                    "ari": ari(lab, ref_sub) if ok else None,
                    "status": "ok" if ok else "invalid_partition",
                })

        if dbscan_setting is not None:
            m = int(dbscan_setting["min_samples"])
            eps = float(dbscan_setting["eps"])
            lab_sub = DBSCAN(eps=eps, min_samples=m, metric="euclidean").fit(X_sub).labels_
            ref = ref_labels[dbscan_setting["experiment_id"]]
            ref_sub = ref[[date_index[d] for d in sub_dates]]
            noise_a = lab_sub == -1
            noise_b = ref_sub == -1
            union = int((noise_a | noise_b).sum())
            inter = int((noise_a & noise_b).sum())
            jaccard = inter / union if union > 0 else None
            keep = ~(noise_a | noise_b)
            common_n = int(keep.sum())
            ok = _valid_partition(lab_sub[keep]) and _valid_partition(ref_sub[keep])
            db_records.append({
                "experiment_id": dbscan_setting["experiment_id"], "seed": sub["seed"],
                "n_subset": n_sub, "n_common_non_noise": common_n,
                "common_fraction": common_n / n_sub if n_sub else None,
                "ari": ari(lab_sub[keep], ref_sub[keep]) if ok else None,
                "noise_jaccard": jaccard,
                "jaccard_reason": None if union > 0 else "no_noise_in_both",
                "status": "ok" if ok else "invalid_partition",
            })

    rep_df = pd.DataFrame(records)
    db_df = pd.DataFrame(db_records)
    return rep_df, db_df, subsets


def aggregate_stability(rep_df: pd.DataFrame) -> pd.DataFrame:
    if rep_df.empty:
        return pd.DataFrame(columns=[
            "algorithm", "k", "valid_repetitions", "failed_repetitions",
            "median_ari", "q1_ari", "q3_ari",
        ])
    rows = []
    for (algorithm, k), group in rep_df.groupby(["algorithm", "k"]):
        valid = group["ari"].dropna()
        rows.append({
            "algorithm": algorithm, "k": int(k),
            "valid_repetitions": int(len(valid)),
            "failed_repetitions": int(len(group) - len(valid)),
            "median_ari": float(valid.median()) if len(valid) else None,
            "q1_ari": float(valid.quantile(0.25)) if len(valid) else None,
            "q3_ari": float(valid.quantile(0.75)) if len(valid) else None,
        })
    return pd.DataFrame(rows)


def select_representatives(
    experiments: pd.DataFrame,
    stability_agg: pd.DataFrame,
    cfg,
    n: int,
) -> dict[str, Any]:
    """Frozen rule chain; returns the full selection.json payload."""
    min_size = cfg.min_cluster_size(n)
    tol = float(cfg.selection["silhouette_shortlist_tolerance"])
    tie = float(cfg.selection["stability_tie_tolerance"])
    weak = float(cfg.selection["weak_stability_warning_below"])
    result: dict[str, Any] = {
        "protocol_version": "v1",
        "n": n,
        "minimum_cluster_size": min_size,
        "methods": {},
    }
    for algorithm in ("kmeans", "ward"):
        sub = experiments[experiments["algorithm"] == algorithm].copy()
        if algorithm == "kmeans":
            agg = sub.groupby("k").agg(
                silhouette=("silhouette", "median"),
                min_cluster_size=("min_cluster_size", "min"),
                status_ok=("status", lambda s: bool((s == "ok").all())),
            ).reset_index()
        else:
            agg = sub[["k", "silhouette", "min_cluster_size"]].copy()
            agg["status_ok"] = sub["status"].eq("ok").to_numpy()
        agg["eligible"] = agg["status_ok"] & (agg["min_cluster_size"] >= min_size)
        best = agg.loc[agg["eligible"], "silhouette"].max()
        agg["shortlisted"] = agg["eligible"] & (agg["silhouette"] >= best - tol) if pd.notna(best) else False
        method: dict[str, Any] = {
            "candidates": agg.to_dict(orient="records"),
            "best_eligible_silhouette": float(best) if pd.notna(best) else None,
            "representative": None,
        }
        short = agg[agg["shortlisted"]]
        if len(short):
            stab = (
                stability_agg[stability_agg["algorithm"] == algorithm].set_index("k")
                if not stability_agg.empty and "algorithm" in stability_agg.columns
                else pd.DataFrame(columns=["median_ari"])
            )
            scored = []
            for _, row in short.iterrows():
                k = int(row["k"])
                med = stab.loc[k, "median_ari"] if k in stab.index else None
                scored.append({"k": k, "silhouette": float(row["silhouette"]),
                               "median_ari": float(med) if med is not None and pd.notna(med) else None})
            with_ari = [s for s in scored if s["median_ari"] is not None]
            if with_ari:
                top = max(s["median_ari"] for s in with_ari)
                tied = [s for s in with_ari if top - s["median_ari"] <= tie]
                tied.sort(key=lambda s: (s["k"], -s["silhouette"], s["k"]))
                chosen = tied[0]
                method["representative"] = {
                    "k": chosen["k"],
                    "median_ari": chosen["median_ari"],
                    "silhouette": chosen["silhouette"],
                    "weak_stability": chosen["median_ari"] < weak,
                    "tie_break": "smaller_k_then_higher_silhouette_then_ascending_k",
                }
            method["stability_scores"] = scored
        result["methods"][algorithm] = method
    return result


def select_dbscan_representative(experiments: pd.DataFrame, cfg, n: int) -> dict[str, Any]:
    min_size = cfg.min_cluster_size(n)
    cov_min = float(cfg.selection["dbscan_minimum_coverage_for_representative"])
    tol = float(cfg.selection["silhouette_shortlist_tolerance"])
    sub = experiments[experiments["algorithm"] == "dbscan"].copy()
    sub["coverage"] = 1.0 - sub["noise_fraction"]
    sub["eligible"] = (
        sub["status"].eq("ok") & (sub["n_clusters"] >= 2)
        & (sub["min_cluster_size"] >= min_size)
        & (sub["coverage"] >= cov_min) & sub["silhouette"].notna()
    )
    out: dict[str, Any] = {
        "candidates": sub[["experiment_id", "min_samples", "eps", "n_clusters",
                           "coverage", "min_cluster_size", "silhouette", "eligible"]]
        .to_dict(orient="records"),
        "representative": None,
        "reason": None,
    }
    eligible = sub[sub["eligible"]]
    if not len(eligible):
        out["reason"] = "no_setting_meets_two_clusters_min_size_coverage_and_valid_metrics"
        return out
    best = eligible["silhouette"].max()
    short = eligible[eligible["silhouette"] >= best - tol].sort_values(
        ["coverage", "min_samples", "eps"], ascending=[False, True, True]
    )
    row = short.iloc[0]
    out["representative"] = {
        "experiment_id": row["experiment_id"], "min_samples": int(row["min_samples"]),
        "eps": float(row["eps"]), "silhouette": float(row["silhouette"]),
        "coverage": float(row["coverage"]),
    }
    return out
