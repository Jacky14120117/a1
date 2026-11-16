"""Boundary tests for evaluation validity rules and candidate experiments.

Covers tests/README.md rows: DBSCAN all-noise / single-cluster handling,
ARI permutation invariance, eps self-inclusion, duplicate-eps bookkeeping,
Ward single-tree cuts and same-config reproducibility.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.cluster.hierarchy import cut_tree, linkage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from air_quality_clustering.clustering import (  # noqa: E402
    dbscan_eps_grid,
    run_dbscan_candidates,
    run_kmeans_candidates,
    run_ward_candidates,
)
from air_quality_clustering.config import load_config  # noqa: E402
from air_quality_clustering.evaluation import ari, internal_metrics  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return load_config(ROOT / "configs" / "project.yaml")


def two_blobs(n_per: int = 40, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=0.0, scale=0.3, size=(n_per, 6))
    b = rng.normal(loc=6.0, scale=0.3, size=(n_per, 6))
    return np.vstack([a, b])


# -- evaluation validity ----------------------------------------------------

def test_all_noise_metrics_null_with_reason():
    X = two_blobs()
    labels = np.full(len(X), -1)
    result = internal_metrics(X, labels, exclude_noise=True)
    assert result.silhouette is None
    assert result.davies_bouldin is None
    assert result.calinski_harabasz is None
    assert result.n_scored == 0
    assert result.metric_reason == "all_noise"


def test_single_cluster_plus_noise_not_two_clusters():
    X = two_blobs()
    labels = np.zeros(len(X), dtype=int)
    labels[::3] = -1  # one cluster + noise
    result = internal_metrics(X, labels, exclude_noise=True)
    assert result.silhouette is None
    assert result.metric_reason == "single_cluster"
    assert result.n_scored == int((labels != -1).sum())


def test_valid_partition_scores_finite():
    X = two_blobs()
    labels = np.array([0] * 40 + [1] * 40)
    result = internal_metrics(X, labels)
    assert result.silhouette is not None and result.silhouette > 0.5
    assert result.metric_reason == "ok"
    assert result.n_scored == 80


def test_ari_permutation_invariance():
    labels = np.array([0, 0, 1, 1, 2, 2])
    permuted = np.array([5, 5, 3, 3, 9, 9])
    assert ari(labels, permuted) == pytest.approx(1.0)
    # a genuinely different grouping (not a relabeling) must score below 1
    regrouped = np.array([0, 1, 0, 1, 0, 1])
    assert ari(labels, regrouped) < 1.0


# -- candidate grids ---------------------------------------------------------

def test_kmeans_grid_counts_and_labels(cfg):
    X = two_blobs()
    rows, labels = run_kmeans_candidates(X, cfg)
    assert len(rows) == 70
    assert len(labels) == 70
    for row in rows:
        lab = labels[row["experiment_id"]]
        assert len(lab) == len(X)
        assert row["n_total"] == len(X)
        assert row["status"] in ("ok", "degenerate_empty_cluster")
    k2 = [r for r in rows if r["k"] == 2 and r["seed"] == 100][0]
    assert k2["n_clusters"] == 2
    assert k2["silhouette"] is not None


def test_ward_single_tree_consistent_with_cuts(cfg):
    X = two_blobs()
    rows, labels, Z = run_ward_candidates(X, cfg)
    assert len(rows) == 7
    assert Z.shape == (len(X) - 1, 4)  # exactly one tree
    recomputed = cut_tree(Z, n_clusters=[2, 3, 4, 5, 6, 7, 8])
    for j, k in enumerate([2, 3, 4, 5, 6, 7, 8]):
        assert np.array_equal(labels[f"E02_k{k}"], recomputed[:, j])
    # cuts of the same tree are nested: k=2 partition refines every larger k
    coarse = labels["E02_k2"]
    for k in [3, 4, 5, 6, 7, 8]:
        fine = labels[f"E02_k{k}"]
        for cluster in np.unique(fine):
            assert len(np.unique(coarse[fine == cluster])) == 1


def test_dbscan_eps_includes_self(cfg):
    X = two_blobs()
    grid = dbscan_eps_grid(X, cfg)
    for m in (5, 10, 20):
        kdist = grid[m]["kdist_sorted"]
        assert len(kdist) == len(X)
        # self-inclusion: the smallest m-th distance can be ~0 only if
        # duplicates exist; here all points distinct so every value > 0
        assert (kdist > 0).all()
        eps_values = list(grid[m]["eps_by_quantile"].values())
        assert eps_values == sorted(eps_values)  # quantiles monotone


def test_dbscan_grid_12_rows_and_degenerate_status(cfg):
    X = two_blobs()
    rows, labels, grid = run_dbscan_candidates(X, cfg)
    assert len(rows) == 12
    for row in rows:
        assert row["status"] in ("ok", "invalid_eps")
        if row["status"] == "ok":
            lab = labels[row["experiment_id"]]
            assert len(lab) == len(X)
            noise = float((lab == -1).mean())
            assert row["noise_fraction"] == pytest.approx(noise)
            assigned = lab[lab != -1]
            expected_clusters = len(np.unique(assigned))
            assert row["n_clusters"] == expected_clusters
            if expected_clusters < 2:
                assert row["silhouette"] is None
                assert row["metric_reason"] in ("single_cluster", "all_noise")


def test_dbscan_duplicate_eps_keeps_records(cfg):
    # identical points -> all k-distances zero -> eps=0 -> invalid_eps records
    X = np.ones((30, 6))
    rows, labels, _ = run_dbscan_candidates(X, cfg)
    assert len(rows) == 12
    assert all(r["status"] == "invalid_eps" for r in rows)
    assert len(labels) == 0  # no fit performed for invalid eps


def test_same_config_reproducible(cfg):
    X = two_blobs(seed=7)
    rows_a, labels_a = run_kmeans_candidates(X, cfg)
    rows_b, labels_b = run_kmeans_candidates(X, cfg)
    for row_a, row_b in zip(rows_a, rows_b):
        eid = row_a["experiment_id"]
        assert ari(labels_a[eid], labels_b[eid]) == pytest.approx(1.0)
        if row_a["silhouette"] is not None:
            assert row_a["silhouette"] == pytest.approx(row_b["silhouette"], abs=1e-12)
