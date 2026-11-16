"""Load and validate configs/project.yaml as the single parameter source.

No experiment parameter may be hard-coded in research modules; everything
comes from the validated configuration object produced here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

FEATURE_ORDER = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
STAGES = ("prepare", "eda", "cluster", "stability", "compare", "sensitivity", "export", "all")


class ConfigError(ValueError):
    """Raised when the project configuration violates the fixed protocol."""


@dataclass(frozen=True)
class ProjectConfig:
    root: Path
    config_path: Path
    config_sha256: str
    raw: dict[str, Any] = field(repr=False)

    # -- project identity -------------------------------------------------
    @property
    def project_id(self) -> str:
        return self.raw["project"]["id"]

    @property
    def task(self) -> str:
        return self.raw["project"]["task"]

    @property
    def seed(self) -> int:
        return int(self.raw["project"]["seed"])

    # -- data -------------------------------------------------------------
    @property
    def data(self) -> dict[str, Any]:
        return self.raw["data"]

    @property
    def features(self) -> list[str]:
        return list(self.data["features"])

    @property
    def station(self) -> str:
        return self.data["station"]

    @property
    def archive_path(self) -> Path:
        return self.root / self.data["archive"]

    @property
    def csv_path(self) -> Path:
        return self.root / self.data["csv"]

    @property
    def start_date(self) -> date:
        return date.fromisoformat(self.data["start"])

    @property
    def end_date(self) -> date:
        return date.fromisoformat(self.data["end"])

    @property
    def minimum_valid_hours(self) -> int:
        return int(self.data["minimum_valid_hours"])

    @property
    def expected_hourly_rows(self) -> int:
        return int(self.data["expected_hourly_rows"])

    @property
    def expected_calendar_days(self) -> int:
        return int(self.data["expected_calendar_days"])

    @property
    def expected_eligible_days(self) -> int:
        return int(self.data["expected_eligible_days"])

    @property
    def excluded_from_clustering(self) -> list[str]:
        return list(self.data["excluded_from_clustering"])

    # -- preprocessing ----------------------------------------------------
    @property
    def preprocessing(self) -> dict[str, Any]:
        return self.raw["preprocessing"]

    @property
    def main_representation(self) -> str:
        return self.preprocessing["main"]

    @property
    def sensitivity_representation(self) -> str:
        return self.preprocessing["primary_sensitivity"]

    # -- algorithms / selection / stability / evaluation ------------------
    @property
    def algorithms(self) -> dict[str, Any]:
        return self.raw["algorithms"]

    @property
    def selection(self) -> dict[str, Any]:
        return self.raw["selection"]

    @property
    def stability(self) -> dict[str, Any]:
        return self.raw["stability"]

    @property
    def evaluation(self) -> dict[str, Any]:
        return self.raw["evaluation"]

    def min_cluster_size(self, n: int) -> int:
        """Protocol rule: max(absolute floor, ceil(fraction * n))."""
        import math

        absolute = int(self.selection["minimum_cluster_size_absolute"])
        fraction = float(self.selection["minimum_cluster_fraction"])
        return max(absolute, math.ceil(fraction * n))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def validate_config(raw: dict[str, Any]) -> None:
    """Enforce the fixed Topic-3 protocol before any computation runs."""
    project = raw.get("project", {})
    _require(project.get("task") == "unsupervised_clustering",
             "project.task must remain unsupervised_clustering (Topic 3)")

    data = raw.get("data", {})
    _require(data.get("features") == FEATURE_ORDER,
             f"data.features must be exactly {FEATURE_ORDER} in this order")
    _require(data.get("daily_aggregation") == "mean",
             "daily_aggregation must be the arithmetic mean of valid hours")
    _require(data.get("missing_policy") == "complete_eligible_days_no_imputation",
             "missing_policy must not impute or fill ineligible days")
    _require(data.get("negative_concentration") == "mark_invalid",
             "negative concentrations must be marked invalid, not clipped")
    _require(data.get("clip_extremes") is False,
             "clip_extremes must stay false; real high values are retained")
    _require(int(data.get("minimum_valid_hours", 0)) >= 1,
             "minimum_valid_hours must be a positive integer")
    _require(data.get("timezone") == "Asia/Shanghai",
             "timestamps are interpreted as Asia/Shanghai local time")

    excluded = set(data.get("excluded_from_clustering", []))
    _require(not excluded.intersection(FEATURE_ORDER),
             "excluded_from_clustering must not list any main feature")
    for background in ("season", "station", "TEMP", "PRES", "DEWP", "RAIN", "wd", "WSPM"):
        _require(background in excluded,
                 f"background column {background!r} must be excluded from clustering")

    preprocessing = raw.get("preprocessing", {})
    _require(preprocessing.get("main") == "log1p_standard_scaler",
             "main representation must be log1p + StandardScaler")
    _require(preprocessing.get("primary_sensitivity") == "raw_standard_scaler",
             "primary sensitivity representation must be raw + StandardScaler")
    _require(preprocessing.get("pca_for_display_only") is True,
             "PCA may only be used for display, never as model input")

    algorithms = raw.get("algorithms", {})
    _require("kmeans" in algorithms and "ward" in algorithms,
             "K-Means and Ward are the mandatory primary methods")
    _require(algorithms.get("ward", {}).get("linkage") == "ward"
             and algorithms.get("ward", {}).get("metric") == "euclidean",
             "Ward must use euclidean linkage")

    selection = raw.get("selection", {})
    _require(selection.get("no_predefined_final_k") is True,
             "the final k must not be predefined")

    evaluation = raw.get("evaluation", {})
    _require(evaluation.get("ground_truth_labels") == "none",
             "no ground-truth labels may be used")
    _require(evaluation.get("predictive_metrics") == [],
             "predictive metrics are not part of this unsupervised study")


def load_config(path: str | Path, root: str | Path | None = None) -> ProjectConfig:
    """Read, hash and validate the YAML configuration.

    ``root`` defaults to the parent of the configs/ directory so relative
    data paths resolve against the project root regardless of the caller's
    working directory.
    """
    config_path = Path(path).resolve()
    payload = config_path.read_bytes()
    raw = yaml.safe_load(payload.decode("utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError("configuration root must be a mapping")
    validate_config(raw)
    if root is None:
        root = config_path.parent.parent
    return ProjectConfig(
        root=Path(root).resolve(),
        config_path=config_path,
        config_sha256=hashlib.sha256(payload).hexdigest(),
        raw=raw,
    )
