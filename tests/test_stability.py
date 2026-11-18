"""Boundary tests for block stability and the frozen selection rule chain."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from air_quality_clustering.config import load_config  # noqa: E402
from air_quality_clustering.stability import (  # noqa: E402
    aggregate_stability,
    block_ids,
    draw_block_subsets,
    select_dbscan_representative,
    select_representatives,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config(ROOT / "configs" / "project.yaml")


def test_block_ids_anchor_and_last_partial_block():
    dates = np.array(["2013-03-01", "2013-03-07", "2013-03-08", "2017-02-28"])
    bids = block_ids(dates, date(2013, 3, 1), 7)
    assert list(bids) == [0, 0, 1, 208]  # 209 blocks, last one partial


def test_subsets_deterministic_and_shared(cfg):
    a = draw_block_subsets(209, cfg, date(2013, 3, 1))
    b = draw_block_subsets(209, cfg, date(2013, 3, 1))
    assert len(a) == 20
    for sa, sb in zip(a, b):
        assert sa["seed"] == sb["seed"]
        assert np.array_equal(sa["block_ids"], sb["block_ids"])
        assert len(sa["block_ids"]) == 167
        assert len(np.unique(sa["block_ids"])) == 167  # without replacement


def _experiments_fixture() -> pd.DataFrame:
    rows = []
    # kmeans: k=2 best silhouette but tiny cluster -> excluded;
    # k=3 eligible best; k=4 within 0.02 -> shortlist; k=5 far -> out.
    for k, sil, minsize in [(2, 0.50, 5), (3, 0.40, 100), (4, 0.39, 100), (5, 0.30, 100)]:
        for seed in (100, 101):
            rows.append({"algorithm": "kmeans", "k": k, "seed": seed, "silhouette": sil,
                         "min_cluster_size": minsize, "status": "ok"})
    for k, sil in [(2, 0.45), (3, 0.41), (4, 0.40)]:
        rows.append({"algorithm": "ward", "k": k, "seed": None, "silhouette": sil,
                     "min_cluster_size": 100, "status": "ok"})
    return pd.DataFrame(rows)


def test_selection_min_size_shortlist_and_tie_break(cfg):
    exp = _experiments_fixture()
    agg = pd.DataFrame([
        {"algorithm": "kmeans", "k": 3, "median_ari": 0.80},
        {"algorithm": "kmeans", "k": 4, "median_ari": 0.79},  # within 0.02 tie -> smaller k
        {"algorithm": "ward", "k": 2, "median_ari": 0.90},
        {"algorithm": "ward", "k": 3, "median_ari": 0.88},
        {"algorithm": "ward", "k": 4, "median_ari": 0.60},
    ])
    sel = select_representatives(exp, agg, cfg, n=1333)
    km = sel["methods"]["kmeans"]
    assert [c["k"] for c in km["candidates"] if c["shortlisted"]] == [3, 4]
    assert km["representative"]["k"] == 3  # tie within 0.02 prefers smaller k
    ward = sel["methods"]["ward"]
    assert ward["representative"]["k"] == 2
    assert sel["minimum_cluster_size"] == 14


def test_selection_no_eligible_representative(cfg):
    exp = _experiments_fixture()
    exp["min_cluster_size"] = 1  # everything below the 14-day floor
    sel = select_representatives(exp, pd.DataFrame(), cfg, n=1333)
    assert sel["methods"]["kmeans"]["representative"] is None
    assert sel["methods"]["ward"]["representative"] is None


def test_weak_stability_flag(cfg):
    exp = _experiments_fixture()
    agg = pd.DataFrame([
        {"algorithm": "kmeans", "k": 3, "median_ari": 0.65},
        {"algorithm": "kmeans", "k": 4, "median_ari": 0.60},
    ])
    sel = select_representatives(exp, agg, cfg, n=1333)
    assert sel["methods"]["kmeans"]["representative"]["weak_stability"] is True


def test_dbscan_representative_rules(cfg):
    exp = pd.DataFrame([
        # eligible: 2 clusters, size ok, coverage .86, valid silhouette
        {"algorithm": "dbscan", "experiment_id": "E03_m5_q70", "min_samples": 5, "eps": 1.0,
         "n_clusters": 3, "noise_fraction": 0.14, "min_cluster_size": 100,
         "silhouette": 0.10, "status": "ok"},
        # single cluster -> ineligible
        {"algorithm": "dbscan", "experiment_id": "E03_m5_q80", "min_samples": 5, "eps": 1.2,
         "n_clusters": 1, "noise_fraction": 0.20, "min_cluster_size": 100,
         "silhouette": None, "status": "ok"},
        # coverage too low -> ineligible
        {"algorithm": "dbscan", "experiment_id": "E03_m10_q70", "min_samples": 10, "eps": 1.5,
         "n_clusters": 2, "noise_fraction": 0.50, "min_cluster_size": 100,
         "silhouette": 0.30, "status": "ok"},
    ])
    sel = select_dbscan_representative(exp, cfg, n=1333)
    assert sel["representative"]["experiment_id"] == "E03_m5_q70"
    none_sel = select_dbscan_representative(exp.iloc[1:], cfg, n=1333)
    assert none_sel["representative"] is None
    assert none_sel["reason"] is not None


def test_aggregate_counts_failures():
    reps = pd.DataFrame([
        {"algorithm": "kmeans", "k": 3, "ari": 0.8, "status": "ok"},
        {"algorithm": "kmeans", "k": 3, "ari": None, "status": "invalid_partition"},
        {"algorithm": "kmeans", "k": 3, "ari": 0.9, "status": "ok"},
    ])
    agg = aggregate_stability(reps)
    row = agg.iloc[0]
    assert row["valid_repetitions"] == 2
    assert row["failed_repetitions"] == 1
    assert row["median_ari"] == pytest.approx(0.85)
