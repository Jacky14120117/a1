"""Boundary tests for the interpretation analyses added for protocol completeness."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from air_quality_clustering.config import FEATURE_ORDER, load_config  # noqa: E402
from air_quality_clustering import discovery  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return load_config(ROOT / "configs" / "project.yaml")


def make_daily(n_low: int = 6, n_high: int = 4) -> pd.DataFrame:
    dates = pd.date_range("2013-03-01", periods=n_low + n_high).strftime("%Y-%m-%d")
    rows = []
    for i, date in enumerate(dates):
        base = 10.0 if i < n_low else 100.0
        rows.append({"date": date, **{c: base + j for j, c in enumerate(FEATURE_ORDER)}})
    return pd.DataFrame(rows)


def test_spread_iqr_ratio_matches_manual():
    daily = make_daily()
    labels = np.array([0] * 6 + [1] * 4)
    spread = discovery.cluster_spread(daily, labels, "kmeans")
    row = spread[(spread["cluster"] == 0) & (spread["pollutant"] == "PM2.5")].iloc[0]
    values = daily.loc[labels == 0, "PM2.5"].to_numpy()
    q1, q3 = np.quantile(values, [0.25, 0.75])
    assert row["iqr"] == pytest.approx(q3 - q1)
    assert row["iqr_over_median"] == pytest.approx((q3 - q1) / np.median(values))
    assert row["n_days"] == 6


def test_overlap_zero_when_clusters_separated():
    daily = make_daily()
    labels = np.array([0] * 6 + [1] * 4)
    spread = discovery.cluster_spread(daily, labels, "kmeans")
    overlap = discovery.overlap_evidence(spread, "kmeans")
    # low cluster ~10-15, high cluster ~100-105: IQRs cannot overlap
    assert (overlap["iqr_overlap_width"] == 0).all()
    assert overlap["iqr_overlap_low"].isna().all()


def test_overlap_positive_when_clusters_share_range():
    daily = make_daily()
    # interleaved assignment -> the two clusters share the same value range
    daily.loc[:, "PM2.5"] = np.arange(len(daily)) + 1.0
    labels = np.array([i % 2 for i in range(len(daily))])
    spread = discovery.cluster_spread(daily, labels, "kmeans")
    overlap = discovery.overlap_evidence(spread, "kmeans")
    pm = overlap[overlap["pollutant"] == "PM2.5"].iloc[0]
    assert pm["iqr_overlap_width"] > 0
    assert 0 < pm["overlap_fraction_of_pooled_iqr"] <= 1


def test_disagreement_case_uses_largest_off_diagonal_and_earliest_dates():
    daily = make_daily(n_low=6, n_high=4)
    km = np.array([0] * 6 + [1] * 4)
    ward = np.array([0] * 4 + [1] * 2 + [1] * 4)  # 2 days: km0 but ward1
    case = discovery.disagreement_case(daily, km, ward, top_n=5)
    assert case["status"] == "ok"
    assert case["cell"]["kmeans_cluster"] == 0
    assert case["cell"]["ward_cluster"] == 1
    assert case["cell"]["n_days"] == 2
    assert case["counts"]["km_only_days"] == 4   # 6 days in km0 minus 2 shared
    assert case["counts"]["ward_only_days"] == 4  # 6 days in ward1 minus 2 shared
    example_dates = [r["date"] for r in case["examples"]]
    assert example_dates == sorted(example_dates)  # earliest-first, fixed rule
    assert example_dates == ["2013-03-05", "2013-03-06"]


def test_disagreement_case_identical_partitions_with_permuted_numbers():
    daily = make_daily()
    labels = np.array([0] * 6 + [1] * 4)
    permuted = np.where(labels == 0, 7, 3)  # same partition, different numbers
    case = discovery.disagreement_case(daily, labels, permuted)
    assert case["status"] == "no_off_diagonal_cell"
    assert case["examples"] == []


def test_season_coverage_denominators():
    dates = pd.date_range("2013-03-01", periods=10).strftime("%Y-%m-%d")
    coverage = pd.DataFrame({"date": dates, "eligible": [True] * 7 + [False] * 3})
    context = pd.DataFrame({"date": dates, "season": ["spring"] * 10})
    out = discovery.season_coverage(coverage, context)
    row = out.iloc[0]
    assert row["calendar_days"] == 10
    assert row["eligible_days"] == 7
    assert row["coverage_fraction"] == pytest.approx(0.7)


def test_year_composition_rows_sum_to_one():
    daily = make_daily()
    context = pd.DataFrame({"date": daily["date"], "year": [2013] * 10})
    labels = np.array([0] * 6 + [1] * 4)
    out = discovery.year_composition(context, labels, daily, "kmeans")
    row = out.iloc[0]
    assert row["eligible_days_in_year"] == 10
    assert row["pct_cluster_0"] + row["pct_cluster_1"] == pytest.approx(1.0)


def test_sensitivity_labels_respect_frozen_k(cfg):
    daily = make_daily(n_low=20, n_high=20)
    labels = discovery.sensitivity_labels(daily, {"kmeans": 2, "ward": 3}, cfg)
    assert len(np.unique(labels["kmeans"])) == 2
    assert len(np.unique(labels["ward"])) == 3
    assert all(len(v) == len(daily) for v in labels.values())
