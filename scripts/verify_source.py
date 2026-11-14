"""Offline source preflight only; does not fit or evaluate clustering models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip"
STATION_FILE = "PRSA_Data_Aotizhongxin_20130301-20170228.csv"
NESTED_ZIP = "PRSA2017_Data_20130301-20170228.zip"
POLLUTANTS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
EXPECTED_COLUMNS = [
    "No", "year", "month", "day", "hour", *POLLUTANTS,
    "TEMP", "PRES", "DEWP", "RAIN", "wd", "WSPM", "station",
]


def selected_bytes(archive_path: Path) -> tuple[bytes, str, list[str]]:
    with zipfile.ZipFile(archive_path) as outer:
        members = outer.namelist()
        direct = [n for n in members if Path(n).name == STATION_FILE]
        if len(direct) == 1:
            return outer.read(direct[0]), direct[0], members
        nested = [n for n in members if Path(n).name == NESTED_ZIP]
        if len(nested) != 1:
            raise ValueError("Expected exactly one original PRSA archive or selected station CSV")
        with zipfile.ZipFile(io.BytesIO(outer.read(nested[0]))) as inner:
            selected = [n for n in inner.namelist() if Path(n).name == STATION_FILE]
            if len(selected) != 1:
                raise ValueError("Selected station missing or ambiguous in original PRSA archive")
            return inner.read(selected[0]), nested[0] + "::" + selected[0], members


def audit(archive_path: Path, minimum_hours: int = 18) -> tuple[dict, bytes]:
    payload, member, members = selected_bytes(archive_path)
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    if reader.fieldnames != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected schema: {reader.fieldnames}")
    rows = list(reader)
    timestamps = []
    missing = Counter()
    negative = Counter()
    valid_by_day = defaultdict(Counter)
    for row in rows:
        if row["station"] != "Aotizhongxin":
            raise ValueError("Unexpected station")
        stamp = datetime(*(int(row[c]) for c in ["year", "month", "day", "hour"]))
        timestamps.append(stamp)
        day = stamp.date().isoformat()
        valid_by_day[day]  # retain even entirely invalid calendar days
        for feature in POLLUTANTS:
            raw = row[feature].strip()
            try:
                value = float(raw)
            except ValueError:
                if raw not in {"", "NA", "NaN", "nan"}:
                    raise ValueError(f"Unrecognized numeric token {raw!r} in {feature}")
                missing[feature] += 1
                continue
            if not math.isfinite(value):
                missing[feature] += 1
            elif value < 0:
                negative[feature] += 1
            else:
                valid_by_day[day][feature] += 1
    unique_stamps = set(timestamps)
    first, last = min(timestamps), max(timestamps)
    expected_hours = int((last - first).total_seconds() / 3600) + 1
    if len(rows) != 35064 or len(unique_stamps) != len(rows) or expected_hours != len(rows):
        raise ValueError("Station hourly grid differs from the expected 35,064 unique hours")
    if first != datetime(2013, 3, 1) or last != datetime(2017, 2, 28, 23):
        raise ValueError("Unexpected source date range")
    day_count = (last.date() - first.date()).days + 1
    eligible = [day for day, counts in valid_by_day.items()
                if all(counts[f] >= minimum_hours for f in POLLUTANTS)]
    eligible_by_year = Counter(day[:4] for day in eligible)
    by_feature = {f: sum(counts[f] >= minimum_hours for counts in valid_by_day.values())
                  for f in POLLUTANTS}
    result = {
        "purpose": "source/schema/missingness feasibility preflight; no clustering performed",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": SOURCE_URL,
        "dataset_doi": "10.24432/C5RK5G",
        "license": "CC BY 4.0; see UCI source page",
        "archive_path": archive_path.resolve().relative_to(ROOT).as_posix(),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "archive_members": members,
        "selected_member": member,
        "selected_csv_sha256": hashlib.sha256(payload).hexdigest(),
        "columns": EXPECTED_COLUMNS,
        "selected_station": "Aotizhongxin",
        "station_rows": len(rows),
        "unique_hourly_timestamps": len(unique_stamps),
        "duplicate_hourly_timestamps": len(rows) - len(unique_stamps),
        "first_local_timestamp": first.isoformat(),
        "last_local_timestamp": last.isoformat(),
        "timezone_interpretation": "Asia/Shanghai local timestamps, no UTC shift before daily grouping",
        "calendar_days": day_count,
        "clustering_features": POLLUTANTS,
        "hourly_missing_counts": {f: missing[f] for f in POLLUTANTS},
        "hourly_negative_counts": {f: negative[f] for f in POLLUTANTS},
        "minimum_valid_hours_per_pollutant_per_day": minimum_hours,
        "eligible_days_per_pollutant": by_feature,
        "eligible_complete_days": len(eligible),
        "excluded_calendar_days": day_count - len(eligible),
        "eligible_fraction": len(eligible) / day_count,
        "eligible_days_by_year": dict(sorted(eligible_by_year.items())),
        "clustering_status": "not_started",
    }
    return result, payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path,
                        default=ROOT / "data/raw/beijing_air_quality_source.zip")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/metadata/source_preflight.json")
    parser.add_argument("--extract-selected", action="store_true")
    args = parser.parse_args()
    result, payload = audit(args.archive)
    if args.extract_selected:
        target = ROOT / "data/raw" / STATION_FILE
        if target.exists() and target.read_bytes() != payload:
            raise FileExistsError("Refusing to replace a different raw source CSV")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        result["selected_csv_path"] = target.relative_to(ROOT).as_posix()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["station_rows", "calendar_days",
          "eligible_complete_days", "excluded_calendar_days", "clustering_status"]}, indent=2))


if __name__ == "__main__":
    main()
