"""Candidate experiments E01-E03 exactly as fixed in protocol v1.

E01 K-Means: k=2..8 x seeds 100..109 (70 runs, all retained).
E02 Ward: ONE euclidean linkage tree, cut at k=2..8 (7 cuts, not 7 fits).
E03 DBSCAN: m=5/10/20; eps = linear quantiles (0.70/0.80/0.90/0.95) of the
    m-th nearest-neighbour distance INCLUDING the sample itself (column m-1).
    12 settings; duplicate eps reuses one fit but keeps its own record;
    eps<=0 is recorded as invalid_eps, never nudged to a tiny positive value.

Every candidate keeps its labels aligned with the input date order; failed
or degenerate settings stay in the table with a status, never dropped.
"""

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import cut_tree, linkage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors

from .evaluation import internal_metrics


def _base_row(experiment_id: str, algorithm: str, representation: str, n_total: int) -> dict:
    return {
        "experiment_id": experiment_id,
        "algorithm": algorithm,
        "k": None,
        "eps": None,
        "min_samples": None,
        "seed": None,
        "representation": representation,
        "n_total": n_total,
        "n_scored": None,
        "n_clusters": None,
        "noise_fraction": 0.0,
        "min_cluster_size": None,
        "silhouette": None,
        "davies_bouldin": None,
        "calinski_harabasz": None,
        "inertia": None,
        "n_iter": None,
        "runtime_seconds": None,
        "status": None,
        "metric_reason": None,
        "notes": "",
    }


def run_kmeans_candidates(X: np.ndarray, cfg) -> tuple[list[dict], dict[str, np.ndarray]]:
    settings = cfg.algorithms["kmeans"]
    representation = cfg.main_representation
    rows: list[dict] = []
    labels: dict[str, np.ndarray] = {}
    n_total = int(X.shape[0])
    for k in settings["k"]:
        for seed in settings["seeds"]:
            experiment_id = f"E01_k{k}_s{seed}"
            row = _base_row(experiment_id, "kmeans", representation, n_total)
            row.update(k=int(k), seed=int(seed))
            start = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = KMeans(
                    n_clusters=int(k),
                    n_init=int(settings["n_init"]),
                    max_iter=int(settings["max_iter"]),
                    algorithm=settings["algorithm"],
                    random_state=int(seed),
                ).fit(X)
            row["runtime_seconds"] = time.perf_counter() - start
            lab = model.labels_
            labels[experiment_id] = lab
            unique, counts = np.unique(lab, return_counts=True)
            row["n_clusters"] = int(len(unique))
            row["min_cluster_size"] = int(counts.min())
            row["inertia"] = float(model.inertia_)
            row["n_iter"] = int(model.n_iter_)
            row.update(internal_metrics(X, lab).as_dict())
            convergence = [str(w.message) for w in caught]
            if convergence:
                row["notes"] = "convergence_warning: " + " | ".join(convergence)
            row["status"] = "ok" if len(unique) == int(k) else "degenerate_empty_cluster"
            rows.append(row)
    return rows, labels


def run_ward_candidates(
    X: np.ndarray, cfg
) -> tuple[list[dict], dict[str, np.ndarray], np.ndarray]:
    """One tree, seven cuts. The returned linkage matrix is the single source
    for both the labels and the dendrogram figure."""
    settings = cfg.algorithms["ward"]
    representation = cfg.main_representation
    n_total = int(X.shape[0])
    start = time.perf_counter()
    Z = linkage(X, method=settings["linkage"], metric=settings["metric"])
    fit_seconds = time.perf_counter() - start
    ks = [int(k) for k in settings["k"]]
    cuts = cut_tree(Z, n_clusters=ks)
    rows: list[dict] = []
    labels: dict[str, np.ndarray] = {}
    for j, k in enumerate(ks):
        experiment_id = f"E02_k{k}"
        row = _base_row(experiment_id, "ward", representation, n_total)
        row["k"] = k
        row["runtime_seconds"] = fit_seconds  # shared tree; noted below
        row["notes"] = "single linkage tree shared by all Ward cuts"
        lab = cuts[:, j]
        labels[experiment_id] = lab
        unique, counts = np.unique(lab, return_counts=True)
        row["n_clusters"] = int(len(unique))
        row["min_cluster_size"] = int(counts.min())
        row.update(internal_metrics(X, lab).as_dict())
        row["status"] = "ok" if len(unique) == k else "degenerate_empty_cluster"
        rows.append(row)
    return rows, labels, Z


def dbscan_eps_grid(X: np.ndarray, cfg) -> dict[int, dict[str, Any]]:
    """Per m: k-distance (self included) and the four protocol quantiles."""
    settings = cfg.algorithms["dbscan"]
    grid: dict[int, dict[str, Any]] = {}
    for m in settings["min_samples"]:
        m = int(m)
        neighbors = NearestNeighbors(n_neighbors=m).fit(X)
        distances, _ = neighbors.kneighbors(X)  # query=X -> self included
        kdist = distances[:, m - 1]
        eps_values = {
            float(q): float(np.quantile(kdist, q, method=settings["quantile_method"]))
            for q in settings["eps_k_distance_quantiles"]
        }
        grid[m] = {"kdist_sorted": np.sort(kdist), "eps_by_quantile": eps_values}
    return grid


def run_dbscan_candidates(
    X: np.ndarray, cfg, grid: dict[int, dict[str, Any]] | None = None
) -> tuple[list[dict], dict[str, np.ndarray], dict[int, dict[str, Any]]]:
    settings = cfg.algorithms["dbscan"]
    representation = cfg.main_representation
    n_total = int(X.shape[0])
    if grid is None:
        grid = dbscan_eps_grid(X, cfg)
    rows: list[dict] = []
    labels: dict[str, np.ndarray] = {}
    fit_cache: dict[tuple[int, float], np.ndarray] = {}
    for m in settings["min_samples"]:
        m = int(m)
        for q in settings["eps_k_distance_quantiles"]:
            q = float(q)
            eps = grid[m]["eps_by_quantile"][q]
            experiment_id = f"E03_m{m}_q{int(round(q * 100))}"
            row = _base_row(experiment_id, "dbscan", representation, n_total)
            row.update(min_samples=m, eps=eps)
            if eps <= 0.0:
                row["status"] = "invalid_eps"
                row["metric_reason"] = "invalid_eps"
                row["notes"] = "eps quantile is zero; not replaced by a tiny positive value"
                rows.append(row)
                continue
            key = (m, eps)
            start = time.perf_counter()
            if key in fit_cache:
                lab = fit_cache[key]
                row["notes"] = "duplicate eps for this m; fit reused"
            else:
                lab = DBSCAN(eps=eps, min_samples=m, metric="euclidean").fit(X).labels_
                fit_cache[key] = lab
            row["runtime_seconds"] = time.perf_counter() - start
            labels[experiment_id] = lab
            noise = int((lab == -1).sum())
            assigned = lab[lab != -1]
            unique, counts = np.unique(assigned, return_counts=True)
            row["n_clusters"] = int(len(unique))
            row["noise_fraction"] = noise / n_total
            row["min_cluster_size"] = int(counts.min()) if len(counts) else 0
            row.update(internal_metrics(X, lab, exclude_noise=True).as_dict())
            row["status"] = "ok"
            rows.append(row)
    return rows, labels, grid
