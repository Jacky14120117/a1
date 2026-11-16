"""Feature construction: fixed six-column matrix, log1p + StandardScaler.

The main representation is Z_j = (log(1+x_j) - mean_j) / std_j with the
population standard deviation (sklearn StandardScaler convention). The
sensitivity branch uses raw concentrations + StandardScaler. Background
columns (date, season, weather, station, ...) can never enter the matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import FEATURE_ORDER


class FeatureError(ValueError):
    """Raised when the feature matrix violates the fixed protocol."""


@dataclass
class FeatureMatrix:
    dates: np.ndarray            # ISO date strings, ascending, unique
    X: np.ndarray                # n x 6 transformed matrix, feature order fixed
    feature_names: list[str]
    representation: str
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    log_applied: bool

    def scaler_params(self) -> dict[str, Any]:
        return {
            "representation": self.representation,
            "feature_order": self.feature_names,
            "log_applied": self.log_applied,
            "scaler_mean": self.scaler_mean.tolist(),
            "scaler_scale": self.scaler_scale.tolist(),
            "n_samples": int(self.X.shape[0]),
            "standard_deviation_convention": "population (ddof=0), sklearn StandardScaler",
        }


def build_feature_matrix(
    daily: pd.DataFrame,
    representation: str,
    feature_order: list[str] | None = None,
) -> FeatureMatrix:
    """Build X from the eligible-day table; date is kept separate, never a feature.

    ``daily`` must contain a ``date`` column plus the six pollutant columns.
    Rows are sorted by date; any extra column is ignored (and background
    columns are explicitly rejected if requested as features).
    """
    features = list(feature_order) if feature_order is not None else list(FEATURE_ORDER)
    if features != FEATURE_ORDER:
        raise FeatureError(f"feature order must be exactly {FEATURE_ORDER}")
    if representation not in ("log1p_standard_scaler", "raw_standard_scaler"):
        raise FeatureError(f"unknown representation {representation!r}")
    if "date" not in daily.columns:
        raise FeatureError("daily table must carry a date column")
    missing = [c for c in features if c not in daily.columns]
    if missing:
        raise FeatureError(f"missing pollutant columns: {missing}")

    table = daily.sort_values("date").reset_index(drop=True)
    dates = table["date"].astype(str).to_numpy()
    if len(np.unique(dates)) != len(dates):
        raise FeatureError("duplicate dates in daily table")

    raw = table[features].to_numpy(dtype="float64")
    if not np.isfinite(raw).all():
        raise FeatureError("non-finite values in eligible-day matrix")
    if (raw < 0).any():
        raise FeatureError("negative concentrations in eligible-day matrix")

    log_applied = representation == "log1p_standard_scaler"
    transformed = np.log1p(raw) if log_applied else raw.copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(transformed)
    if not np.isfinite(X).all():
        raise FeatureError("non-finite values after transformation")

    return FeatureMatrix(
        dates=dates,
        X=X,
        feature_names=features,
        representation=representation,
        scaler_mean=scaler.mean_.copy(),
        scaler_scale=scaler.scale_.copy(),
        log_applied=log_applied,
    )
