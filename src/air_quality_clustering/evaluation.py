"""Internal metrics and partition agreement with explicit validity rules.

Protocol v1 (docs/06): silhouette / Davies-Bouldin / Calinski-Harabasz are
computed on the same six-dimensional transformed space. Invalid or
non-finite values are stored as None with a reason, never as 0. DBSCAN
noise (-1) is excluded before scoring and n_scored is always reported.
ARI compares partitions without ground truth and is invariant to label
permutation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


@dataclass
class MetricResult:
    silhouette: float | None
    davies_bouldin: float | None
    calinski_harabasz: float | None
    n_scored: int
    metric_reason: str

    def as_dict(self) -> dict:
        return {
            "silhouette": self.silhouette,
            "davies_bouldin": self.davies_bouldin,
            "calinski_harabasz": self.calinski_harabasz,
            "n_scored": self.n_scored,
            "metric_reason": self.metric_reason,
        }


def _safe(fn, X: np.ndarray, labels: np.ndarray) -> float | None:
    try:
        value = float(fn(X, labels))
    except ValueError:
        return None
    return value if np.isfinite(value) else None


def internal_metrics(
    X: np.ndarray, labels: np.ndarray, *, exclude_noise: bool = False
) -> MetricResult:
    """Score a partition; degenerate cases return nulls plus a reason."""
    labels = np.asarray(labels)
    if exclude_noise:
        keep = labels != -1
        X, labels = X[keep], labels[keep]
    n = int(len(labels))
    if n == 0:
        return MetricResult(None, None, None, 0, "all_noise" if exclude_noise else "no_samples")
    unique = np.unique(labels)
    k = int(len(unique))
    if k < 2:
        return MetricResult(None, None, None, n, "single_cluster")
    if k > n - 1:
        return MetricResult(None, None, None, n, "too_many_clusters_for_n")

    values = {
        "silhouette": _safe(lambda a, b: silhouette_score(a, b, metric="euclidean"), X, labels),
        "davies_bouldin": _safe(davies_bouldin_score, X, labels),
        "calinski_harabasz": _safe(calinski_harabasz_score, X, labels),
    }
    failed = [name for name, value in values.items() if value is None]
    reason = "ok" if not failed else "nonfinite_or_undefined:" + ",".join(failed)
    return MetricResult(
        silhouette=values["silhouette"],
        davies_bouldin=values["davies_bouldin"],
        calinski_harabasz=values["calinski_harabasz"],
        n_scored=n,
        metric_reason=reason,
    )


def ari(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Adjusted Rand index; invariant to cluster-number permutation."""
    return float(adjusted_rand_score(np.asarray(labels_a), np.asarray(labels_b)))
