"""Boundary tests for the prepare stage (docs/05 + tests/README.md).

Small constructed frames verify the rules; the real-source integration check
lives in test_real_source.py and only runs when the raw CSV is present.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from air_quality_clustering import data as dq  # noqa: E402
from air_quality_clustering.config import load_config  # noqa: E402
from air_quality_clustering.features import FeatureError, build_feature_matrix  # noqa: E402

POLLUTANTS = dq.POLLUTANTS
START = date(2013, 3, 1)
END = date(2013, 3, 2)


def make_hourly(day_values: dict[str, dict[str, list[float]]]) -> pd.DataFrame:
    """Build a two-day hourly frame.

    day_values maps ISO date -> pollutant -> list of 24 hourly values
    (np.nan means missing). Weather columns are filled with valid constants.
    """
    rows = []
    for day_iso, pollutants in day_values.items():
        day = date.fromisoformat(day_iso)
        for hour in range(24):
            row = {
                "timestamp": pd.Timestamp(day.year, day.month, day.day, hour),
                "station": "Aotizhongxin",
            }
            for column in POLLUTANTS:
                row[column] = pollutants[column][hour]
            for column in dq.WEATHER_MEAN_COLUMNS:
                row[column] = 10.0
            row["RAIN"] = 0.0
            row["wd"] = "NE"
            rows.append(row)
    return pd.DataFrame(rows)


def full_day(value: float = 50.0) -> dict[str, list[float]]:
    return {column: [value] * 24 for column in POLLUTANTS}


def test_17_hours_one_column_excludes_day():
    day1 = full_day()
    day1["CO"][17:] = [np.nan] * 7  # only 17 valid hours
    hourly = make_hourly({"2013-03-01": day1, "2013-03-02": full_day()})
    coverage = dq.daily_coverage(hourly, START, END, minimum_valid_hours=18)
    assert coverage.loc[0, "eligible"] == False  # noqa: E712
    assert "CO:17h" in coverage.loc[0, "exclusion_reason"]
    assert coverage.loc[1, "eligible"] == True  # noqa: E712
    daily = dq.daily_pollutants(hourly, coverage)
    assert list(daily["date"]) == ["2013-03-02"]


def test_exactly_18_hours_each_column_is_eligible():
    day1 = full_day()
    for column in POLLUTANTS:
        # drop a different set of 6 hours per column: hours need not coincide
        drop = list(range(18, 24)) if column != "O3" else list(range(0, 6))
        for hour in drop:
            day1[column][hour] = np.nan
    hourly = make_hourly({"2013-03-01": day1, "2013-03-02": full_day()})
    coverage = dq.daily_coverage(hourly, START, END, minimum_valid_hours=18)
    assert coverage["eligible"].all()
    daily = dq.daily_pollutants(hourly, coverage)
    assert len(daily) == 2
    # mean over each column's own valid hours (all 50 here)
    assert np.allclose(daily[POLLUTANTS].to_numpy(), 50.0)


def test_daily_mean_uses_own_valid_hours():
    day1 = full_day()
    day1["PM2.5"] = [10.0] * 18 + [np.nan] * 6
    day1["PM10"] = [20.0] * 12 + [40.0] * 6 + [np.nan] * 6  # 18 valid, mean 26.666...
    hourly = make_hourly({"2013-03-01": day1, "2013-03-02": full_day()})
    coverage = dq.daily_coverage(hourly, START, END, minimum_valid_hours=18)
    daily = dq.daily_pollutants(hourly, coverage)
    row = daily.loc[daily["date"] == "2013-03-01"].iloc[0]
    assert row["PM2.5"] == pytest.approx(10.0)
    assert row["PM10"] == pytest.approx((20.0 * 12 + 40.0 * 6) / 18)


def test_zero_kept_negative_invalid_na_missing(tmp_path):
    day1 = full_day()
    day1["SO2"][0] = 0.0       # zero must be kept
    day1["SO2"][1] = -5.0      # negative -> invalid
    day1["NO2"][0] = np.nan    # missing
    hourly = make_hourly({"2013-03-01": day1, "2013-03-02": full_day()})
    # emulate the load_hourly invalidation rule on the constructed frame
    for column in POLLUTANTS:
        hourly.loc[hourly[column] < 0, column] = np.nan
    coverage = dq.daily_coverage(hourly, START, END, minimum_valid_hours=18)
    assert coverage.loc[0, "n_valid_SO2"] == 23  # zero counted, negative not
    assert coverage.loc[0, "eligible"] == True  # noqa: E712
    daily = dq.daily_pollutants(hourly, coverage)
    # 23 valid hours: one zero + 22 values of 50; the zero participates in the mean
    assert daily.loc[0, "SO2"] == pytest.approx((0.0 + 50.0 * 22) / 23)


def test_conflicting_hour_raises(tmp_path):
    csv_path = tmp_path / "conflict.csv"
    rows = []
    for hour in range(24):
        rows.append({
            "No": hour + 1, "year": 2013, "month": 3, "day": 1, "hour": hour,
            **{c: 50.0 for c in POLLUTANTS},
            "TEMP": 10.0, "PRES": 1000.0, "DEWP": 0.0, "RAIN": 0.0,
            "wd": "NE", "WSPM": 2.0, "station": "Aotizhongxin",
        })
    frame = pd.DataFrame(rows)
    duplicate = frame.iloc[[0]].copy()
    duplicate["PM2.5"] = 999.0  # same hour, conflicting value
    pd.concat([frame, duplicate], ignore_index=True).to_csv(csv_path, index=False)
    with pytest.raises(dq.DataQualityError, match="Conflicting"):
        dq.load_hourly(csv_path, "Aotizhongxin")


def test_identical_duplicate_removed(tmp_path):
    csv_path = tmp_path / "dup.csv"
    rows = []
    for hour in range(24):
        rows.append({
            "No": hour + 1, "year": 2013, "month": 3, "day": 1, "hour": hour,
            **{c: 50.0 for c in POLLUTANTS},
            "TEMP": 10.0, "PRES": 1000.0, "DEWP": 0.0, "RAIN": 0.0,
            "wd": "NE", "WSPM": 2.0, "station": "Aotizhongxin",
        })
    frame = pd.DataFrame(rows)
    pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_csv(csv_path, index=False)
    hourly, audit = dq.load_hourly(csv_path, "Aotizhongxin")
    assert len(hourly) == 24
    assert audit.duplicate_identical_rows_removed == 1


def test_unrecognized_token_raises(tmp_path):
    csv_path = tmp_path / "bad_token.csv"
    rows = []
    for hour in range(24):
        rows.append({
            "No": hour + 1, "year": 2013, "month": 3, "day": 1, "hour": hour,
            **{c: "50" for c in POLLUTANTS},
            "TEMP": "10", "PRES": "1000", "DEWP": "0", "RAIN": "0",
            "wd": "NE", "WSPM": "2", "station": "Aotizhongxin",
        })
    rows[0]["PM2.5"] = "oops"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with pytest.raises(dq.DataQualityError, match="Unrecognized"):
        dq.load_hourly(csv_path, "Aotizhongxin")


def test_feature_matrix_sorted_six_columns_only():
    day1 = full_day(30.0)
    day2 = full_day(60.0)
    hourly = make_hourly({"2013-03-02": day2, "2013-03-01": day1})  # reversed order
    coverage = dq.daily_coverage(hourly, START, END, minimum_valid_hours=18)
    daily = dq.daily_pollutants(hourly, coverage)
    daily["season"] = "winter"  # background column must be ignored
    daily["TEMP_mean"] = 1.0
    matrix = build_feature_matrix(daily, "log1p_standard_scaler")
    assert list(matrix.dates) == ["2013-03-01", "2013-03-02"]
    assert matrix.feature_names == POLLUTANTS
    assert matrix.X.shape == (2, 6)
    assert np.isfinite(matrix.X).all()
    # log1p applied: standardized log values, not raw
    expected = (np.log1p([30.0, 60.0]) - np.log1p([30.0, 60.0]).mean()) / np.log1p([30.0, 60.0]).std()
    assert np.allclose(matrix.X[:, 0], expected)


def test_feature_matrix_rejects_wrong_order_and_nan():
    daily = pd.DataFrame({
        "date": ["2013-03-01"],
        **{c: [50.0] for c in POLLUTANTS},
    })
    with pytest.raises(FeatureError):
        build_feature_matrix(daily, "log1p_standard_scaler", feature_order=list(reversed(POLLUTANTS)))
    bad = daily.copy()
    bad.loc[0, "CO"] = np.nan
    with pytest.raises(FeatureError):
        build_feature_matrix(bad, "log1p_standard_scaler")


def test_context_does_not_filter_and_rain_needs_24h():
    day1 = full_day()
    hourly = make_hourly({"2013-03-01": day1, "2013-03-02": full_day()})
    hourly.loc[0, "RAIN"] = np.nan          # day 1 loses one rain hour
    context = dq.daily_context(hourly, START, END, minimum_valid_hours=18)
    assert len(context) == 2                # all calendar days retained
    assert np.isnan(context.loc[0, "RAIN_sum"])      # 23/24 -> NA
    assert context.loc[1, "RAIN_sum"] == pytest.approx(0.0)
    # weather gaps never remove days from the background table
    assert list(context["date"]) == ["2013-03-01", "2013-03-02"]


def test_context_temp_mean_with_23_hours():
    day1 = full_day()
    hourly = make_hourly({"2013-03-01": day1, "2013-03-02": full_day()})
    hourly.loc[24, "TEMP"] = np.nan
    context = dq.daily_context(hourly, START, END, minimum_valid_hours=18)
    assert context.loc[1, "n_valid_TEMP"] == 23
    assert context.loc[1, "TEMP_mean"] == pytest.approx(10.0)


def test_season_convention():
    assert dq._season(3) == "spring"
    assert dq._season(7) == "summer"
    assert dq._season(10) == "autumn"
    assert dq._season(1) == "winter"
    assert dq._season(12) == "winter"


def test_config_rejects_supervised_task(tmp_path):
    config_path = ROOT / "configs" / "project.yaml"
    cfg = load_config(config_path)
    assert cfg.features == POLLUTANTS
    assert cfg.min_cluster_size(1333) == 14
    bad = tmp_path / "bad.yaml"
    text = config_path.read_text(encoding="utf-8").replace(
        "task: unsupervised_clustering", "task: regression"
    )
    bad.write_text(text, encoding="utf-8")
    with pytest.raises(Exception):
        load_config(bad)
