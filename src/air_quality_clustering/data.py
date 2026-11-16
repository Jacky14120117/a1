"""Hourly quality audit, daily aggregation and coverage tables.

Implements docs/05_data_dictionary.md exactly:
- NA / empty / non-finite values are missing; negative concentrations are
  invalid and logged; zeros are kept.
- Local Asia/Shanghai calendar days, no UTC shift before grouping.
- A day is eligible only when every one of the six pollutants has at least
  ``minimum_valid_hours`` valid hours; each daily value is the arithmetic
  mean of that pollutant's own valid hours (hours need not coincide).
- No imputation, no zero-filling, no trimming of real high values.
- Weather is aggregated independently and never filters eligible days.

Implementation notes (precision pass): numeric parsing is vectorized; daily
grouping uses the normalized timestamp directly, so no full-frame copies are
made and the raw hourly frame is never mutated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

POLLUTANTS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
WEATHER_MEAN_COLUMNS = ["TEMP", "PRES", "DEWP", "WSPM"]
EXPECTED_COLUMNS = [
    "No", "year", "month", "day", "hour", *POLLUTANTS,
    "TEMP", "PRES", "DEWP", "RAIN", "wd", "WSPM", "station",
]
_MISSING_TOKENS = ["", "NA", "NaN", "nan", "N/A", "null", "NULL", "None"]


class DataQualityError(ValueError):
    """Raised for source problems that must be inspected, not silently fixed."""


@dataclass
class HourlyAudit:
    """Counts collected while parsing; written into the run manifest."""

    rows_read: int = 0
    rows_written: int = 0
    duplicate_identical_rows_removed: int = 0
    missing_counts: dict[str, int] = field(default_factory=dict)
    negative_counts: dict[str, int] = field(default_factory=dict)
    invalid_tokens: list[dict[str, Any]] = field(default_factory=list)
    first_timestamp: str = ""
    last_timestamp: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows_read": self.rows_read,
            "rows_written": self.rows_written,
            "duplicate_identical_rows_removed": self.duplicate_identical_rows_removed,
            "missing_counts": self.missing_counts,
            "negative_counts": self.negative_counts,
            "invalid_tokens": self.invalid_tokens,
            "first_timestamp": self.first_timestamp,
            "last_timestamp": self.last_timestamp,
        }


def _parse_numeric_column(series: pd.Series, column: str, audit: HourlyAudit) -> pd.Series:
    """Vectorized string -> float conversion with strict token auditing.

    Known missing tokens and non-finite values become NaN and are counted;
    any other unparseable token raises instead of being silently coerced.
    """
    text = series.astype("string").str.strip()
    missing_mask = text.isna() | text.isin(_MISSING_TOKENS)
    numeric = pd.to_numeric(text.mask(missing_mask), errors="coerce")
    bad = ~missing_mask & numeric.isna()
    if bad.any():
        tokens = [str(t) for t in text[bad].unique()]
        for token in tokens:
            audit.invalid_tokens.append({"column": column, "token": token})
        raise DataQualityError(
            f"Unrecognized numeric token(s) {tokens!r} in column {column}; inspect the source"
        )
    values = numeric.to_numpy(dtype="float64", copy=True)
    nonfinite = ~np.isfinite(values)
    audit.missing_counts[column] = int(missing_mask.sum()) + int(nonfinite.sum())
    values[nonfinite] = np.nan
    return pd.Series(values, index=series.index, dtype="float64")


def load_hourly(csv_path: str | Path, station: str) -> tuple[pd.DataFrame, HourlyAudit]:
    """Read the raw station CSV into a cleaned hourly frame (raw stays read-only)."""
    csv_path = Path(csv_path)
    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    audit = HourlyAudit(rows_read=len(frame))
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise DataQualityError(f"Unexpected schema: {list(frame.columns)}")
    if not (frame["station"] == station).all():
        raise DataQualityError(f"Found records from stations other than {station}")

    for column in ("year", "month", "day", "hour"):
        frame[column] = _parse_numeric_column(frame[column], column, audit)
        if frame[column].isna().any():
            raise DataQualityError(f"Missing values in time field {column}")
        frame[column] = frame[column].astype("int64")

    for column in POLLUTANTS + WEATHER_MEAN_COLUMNS + ["RAIN"]:
        frame[column] = _parse_numeric_column(frame[column], column, audit)

    negative: dict[str, int] = {}
    for column in POLLUTANTS:
        mask = frame[column] < 0
        negative[column] = int(mask.sum())
        if mask.any():
            frame.loc[mask, column] = np.nan
    audit.negative_counts = negative

    stamps = pd.to_datetime(
        dict(year=frame["year"], month=frame["month"], day=frame["day"], hour=frame["hour"])
    )
    frame.insert(0, "timestamp", stamps)
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)

    duplicated = frame.duplicated(subset=["timestamp"], keep=False)
    if duplicated.any():
        identical_mask = pd.Series(False, index=frame.index)
        for stamp, group in frame.loc[duplicated].groupby("timestamp"):
            content = group.drop(columns=["timestamp"])
            if content.drop_duplicates().shape[0] == 1:
                identical_mask.loc[group.index[1:]] = True
            else:
                raise DataQualityError(
                    f"Conflicting values at hourly key {stamp}; inspect the source, "
                    "no automatic averaging is performed"
                )
        audit.duplicate_identical_rows_removed = int(identical_mask.sum())
        frame = frame.loc[~identical_mask].reset_index(drop=True)

    audit.rows_written = len(frame)
    audit.first_timestamp = frame["timestamp"].iloc[0].isoformat()
    audit.last_timestamp = frame["timestamp"].iloc[-1].isoformat()
    return frame, audit


def validate_hourly_grid(
    hourly: pd.DataFrame, start: date, end: date, expected_rows: int
) -> None:
    """Assert the complete, unique hourly grid over the configured range."""
    expected_start = datetime(start.year, start.month, start.day, 0)
    expected_end = datetime(end.year, end.month, end.day, 23)
    if hourly["timestamp"].iloc[0] != expected_start or hourly["timestamp"].iloc[-1] != expected_end:
        raise DataQualityError("Hourly range differs from the configured start/end")
    if hourly["timestamp"].duplicated().any():
        raise DataQualityError("Duplicate hourly keys remain after cleaning")
    expected_hours = int((expected_end - expected_start).total_seconds() // 3600) + 1
    if len(hourly) != expected_hours or len(hourly) != expected_rows:
        raise DataQualityError(
            f"Hourly grid has {len(hourly)} rows, expected {expected_rows} unique hours"
        )
    deltas = hourly["timestamp"].diff().dropna()
    if not (deltas == pd.Timedelta(hours=1)).all():
        raise DataQualityError("Hourly grid is not contiguous")


def calendar_days(start: date, end: date) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq="D")


def _by_day(hourly: pd.DataFrame) -> pd.core.groupby.DataFrameGroupBy:
    """Group by local calendar day without copying or mutating the frame."""
    return hourly.groupby(hourly["timestamp"].dt.normalize(), sort=True)


def daily_coverage(
    hourly: pd.DataFrame,
    start: date,
    end: date,
    minimum_valid_hours: int,
) -> pd.DataFrame:
    """One row per calendar day: per-pollutant valid counts and eligibility."""
    days = calendar_days(start, end)
    counts = (
        _by_day(hourly)[POLLUTANTS]
        .count()
        .reindex(days, fill_value=0)
        .astype("int64")
    )
    coverage = pd.DataFrame({"date": days.strftime("%Y-%m-%d")})
    for column in POLLUTANTS:
        coverage[f"n_valid_{column}"] = counts[column].to_numpy()
    count_matrix = counts.to_numpy()
    eligible = (count_matrix >= minimum_valid_hours).all(axis=1)
    coverage["eligible"] = eligible
    reasons: list[str] = []
    for row_idx in range(len(days)):
        if eligible[row_idx]:
            reasons.append("")
            continue
        failing = [
            f"{column}:{int(count_matrix[row_idx, j])}h"
            for j, column in enumerate(POLLUTANTS)
            if count_matrix[row_idx, j] < minimum_valid_hours
        ]
        reasons.append("insufficient_valid_hours(" + ",".join(failing) + ")")
    coverage["exclusion_reason"] = reasons
    return coverage


def daily_pollutants(hourly: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    """Daily means over each pollutant's own valid hours, eligible days only.

    Eligibility is fully determined by ``coverage``; no threshold is re-applied.
    """
    means = _by_day(hourly)[POLLUTANTS].mean()
    means.index.name = "date"
    eligible_dates = pd.to_datetime(coverage.loc[coverage["eligible"], "date"])
    daily = means.reindex(eligible_dates).reset_index()
    daily["date"] = pd.to_datetime(daily["date"]).dt.strftime("%Y-%m-%d")
    daily = daily.sort_values("date").reset_index(drop=True)
    if daily[POLLUTANTS].isna().any().any():
        raise DataQualityError("Eligible day with missing daily mean; coverage rule violated")
    return daily[["date", *POLLUTANTS]]


def excluded_days(coverage: pd.DataFrame) -> pd.DataFrame:
    return coverage.loc[~coverage["eligible"]].reset_index(drop=True)


def _season(month: int) -> str:
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    if month in (9, 10, 11):
        return "autumn"
    return "winter"


def daily_context(
    hourly: pd.DataFrame, start: date, end: date, minimum_valid_hours: int
) -> pd.DataFrame:
    """Background table for ALL calendar days; never filters eligible days.

    TEMP/PRES/DEWP/WSPM: mean when >= minimum_valid_hours valid hours, else NA.
    RAIN: sum only when all 24 hours are present and valid, else NA.
    wd: daily mode; ties resolved alphabetically (documented convention).
    """
    days = calendar_days(start, end)
    grouped = _by_day(hourly)

    context = pd.DataFrame({"date": days.strftime("%Y-%m-%d")})
    context["year"] = days.year.to_numpy()
    context["month"] = days.month.to_numpy()
    context["season"] = [_season(m) for m in days.month]

    for column in WEATHER_MEAN_COLUMNS:
        counts = grouped[column].count().reindex(days, fill_value=0)
        means = grouped[column].mean().reindex(days)
        context[f"{column}_mean"] = means.where(counts >= minimum_valid_hours).to_numpy()
        context[f"n_valid_{column}"] = counts.astype("int64").to_numpy()

    rain_counts = grouped["RAIN"].count().reindex(days, fill_value=0)
    rain_sum = grouped["RAIN"].sum(min_count=1).reindex(days)
    context["RAIN_sum"] = rain_sum.where(rain_counts == 24).to_numpy()
    context["n_valid_RAIN"] = rain_counts.astype("int64").to_numpy()

    def _mode(series: pd.Series) -> str:
        values = series.dropna()
        if values.empty:
            return ""
        counts = values.value_counts()
        top = counts.max()
        return sorted(counts.index[counts == top])[0]

    context["wd_mode"] = grouped["wd"].agg(_mode).reindex(days, fill_value="").to_numpy()
    return context


def prepare_tables(
    csv_path: str | Path,
    *,
    station: str,
    start: date,
    end: date,
    minimum_valid_hours: int,
    expected_rows: int,
) -> dict[str, Any]:
    """Full prepare pipeline returning derived tables plus the audit record."""
    hourly, audit = load_hourly(csv_path, station)
    validate_hourly_grid(hourly, start, end, expected_rows)
    coverage = daily_coverage(hourly, start, end, minimum_valid_hours)
    return {
        "hourly_clean": hourly,
        "daily_coverage": coverage,
        "daily_pollutants": daily_pollutants(hourly, coverage),
        "excluded_days": excluded_days(coverage),
        "daily_context": daily_context(hourly, start, end, minimum_valid_hours),
        "audit": audit,
    }
