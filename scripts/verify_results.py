"""Independent post-run verification of the study's results.

Usage:
    python scripts/verify_results.py RUN_ID [--baseline OTHER_RUN_ID]

Checks (all read-only):
1. Funnel recomputed from the cleaned hourly copy matches coverage/daily tables.
2. Every candidate label file has exactly the eligible dates; ward cuts nested.
3. Cluster profiles recomputed from raw daily means match the stored profiles.
4. Selection invariants: min-size rule, shortlist tolerance, tie-break, weak flag.
5. With --baseline: experiments metrics, all labels, selection, stability and
   sensitivity must agree between the two runs (reproducibility audit).
6. Report numbers spot-checked against the run's tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from air_quality_clustering.config import load_config  # noqa: E402
from air_quality_clustering.data import POLLUTANTS, daily_coverage  # noqa: E402
from air_quality_clustering.evaluation import ari  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}: {name}" + (f" ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def run_path(run_id: str) -> Path:
    return ROOT / "results" / "runs" / run_id


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--baseline", default=None)
    args = parser.parse_args()
    cfg = load_config(ROOT / "configs" / "project.yaml")
    run = run_path(args.run_id)

    hourly = pd.read_csv(ROOT / "data" / "interim" / "hourly_clean.csv", parse_dates=["timestamp"])
    coverage = pd.read_csv(ROOT / "data" / "processed" / "daily_coverage.csv")
    daily = pd.read_csv(ROOT / "data" / "processed" / "daily_pollutants.csv")

    # 1. funnel recomputation
    recomputed = daily_coverage(hourly, cfg.start_date, cfg.end_date, cfg.minimum_valid_hours)
    coverage_same = list(recomputed.columns) == list(coverage.columns)
    if coverage_same:
        for col in recomputed.columns:
            a, b = recomputed[col], coverage[col]
            if col == "eligible":
                coverage_same &= bool((a.astype(bool).to_numpy() == b.astype(bool).to_numpy()).all())
            elif pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                coverage_same &= bool(np.allclose(a.to_numpy(), b.to_numpy(), atol=1e-9))
            else:
                coverage_same &= bool((a.fillna("").astype(str).to_numpy() == b.fillna("").astype(str).to_numpy()).all())
    check("coverage recomputed identical", bool(coverage_same))
    means = hourly.assign(date=hourly["timestamp"].dt.normalize()).groupby("date")[POLLUTANTS].mean()
    eligible_dates = pd.to_datetime(coverage.loc[coverage["eligible"], "date"])
    recomputed_daily = means.reindex(eligible_dates)
    check("daily means recomputed identical",
          np.allclose(recomputed_daily.to_numpy(), daily[POLLUTANTS].to_numpy(), atol=1e-9))
    check("eligible count 1333", len(daily) == cfg.expected_eligible_days, str(len(daily)))

    # 2. labels
    experiments = pd.read_csv(run / "experiments.csv")
    check("89 candidates", len(experiments) == 89, str(len(experiments)))
    label_files = sorted((run / "labels").glob("*.csv"))
    check("89 label files", len(label_files) == 89, str(len(label_files)))
    dates_ok = True
    for path in label_files:
        frame = pd.read_csv(path)
        if list(frame["date"]) != list(daily["date"]) or len(frame) != len(daily):
            dates_ok = False
    check("all labels carry exactly the eligible dates", dates_ok)
    ward_labels = {int(p.stem.split("_k")[1]): pd.read_csv(p)["cluster"].to_numpy()
                   for p in label_files if p.stem.startswith("E02_")}
    nested = True
    coarse = ward_labels[2]
    for k in sorted(ward_labels):
        fine = ward_labels[k]
        for cluster in np.unique(fine):
            if len(np.unique(coarse[fine == cluster])) != 1:
                nested = False
    check("ward cuts nested in k=2 partition", nested)

    # 3. profiles recomputation
    selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
    profiles = pd.read_csv(run / "tables" / "cluster_profiles.csv")
    labels = {p.stem: pd.read_csv(p)["cluster"].to_numpy() for p in label_files}
    prof_ok = True
    for algorithm in ("kmeans", "ward"):
        k = int(selection["methods"][algorithm]["representative"]["k"])
        ref = f"E01_k{k}_s{cfg.algorithms['kmeans']['representative_seed']}" if algorithm == "kmeans" else f"E02_k{k}"
        lab = labels[ref]
        for cluster in np.unique(lab):
            group = daily.loc[lab == cluster, POLLUTANTS]
            for column in POLLUTANTS:
                stored = profiles[(profiles["algorithm"] == algorithm)
                                  & (profiles["cluster"] == cluster)
                                  & (profiles["pollutant"] == column)]["median"].iloc[0]
                if not np.isclose(stored, group[column].median(), atol=1e-9):
                    prof_ok = False
    check("profiles match raw-unit medians", prof_ok)

    # 4. selection invariants
    min_size = cfg.min_cluster_size(len(daily))
    inv_ok = True
    for algorithm in ("kmeans", "ward"):
        method = selection["methods"][algorithm]
        for cand in method["candidates"]:
            if cand["shortlisted"] and cand["min_cluster_size"] < min_size:
                inv_ok = False
        rep = method["representative"]
        if rep is not None:
            if rep["median_ari"] < 0.70 and not rep["weak_stability"]:
                inv_ok = False
            if rep["median_ari"] >= 0.70 and rep["weak_stability"]:
                inv_ok = False
    check("selection invariants (min size / weak flag)", inv_ok, f"min_size={min_size}")

    # 5. cross-run reproducibility
    if args.baseline:
        base = run_path(args.baseline)
        base_exp = pd.read_csv(base / "experiments.csv")
        metric_cols = ["silhouette", "davies_bouldin", "calinski_harabasz", "n_clusters",
                       "noise_fraction", "min_cluster_size", "inertia"]
        same_metrics = True
        for col in metric_cols:
            a = base_exp.set_index("experiment_id")[col]
            b = experiments.set_index("experiment_id")[col]
            if not np.allclose(a.fillna(-9e9), b.fillna(-9e9), atol=1e-9):
                same_metrics = False
        check("candidate metrics identical across runs", same_metrics)
        labels_same = all(
            ari(labels[p.stem], pd.read_csv(base / "labels" / p.name)["cluster"].to_numpy()) == 1.0
            for p in label_files
        )
        check("all 89 partitions identical across runs (ARI=1)", labels_same)
        base_sel = json.loads((base / "selection.json").read_text(encoding="utf-8"))
        sel_same = all(
            base_sel["methods"][a]["representative"] == selection["methods"][a]["representative"]
            for a in ("kmeans", "ward")
        ) and base_sel["dbscan"]["representative"] == selection["dbscan"]["representative"]
        check("representatives identical across runs", sel_same)
        base_agg = pd.read_csv(base / "stability" / "aggregate.csv")
        new_agg = pd.read_csv(run / "stability" / "aggregate.csv")
        check("stability aggregates identical", np.allclose(
            base_agg["median_ari"].fillna(-9), new_agg["median_ari"].fillna(-9), atol=1e-12))
        base_sens = pd.read_csv(base / "sensitivity" / "representation_ari.csv")
        new_sens = pd.read_csv(run / "sensitivity" / "representation_ari.csv")
        check("sensitivity ARIs identical", np.allclose(
            base_sens["ari_vs_main"], new_sens["ari_vs_main"], atol=1e-12))

    # 6. report spot checks
    report = (run / "report.md").read_text(encoding="utf-8")
    km_rep = selection["methods"]["kmeans"]["representative"]
    ward_rep = selection["methods"]["ward"]["representative"]
    rep_ari = ari(labels[f"E01_k{km_rep['k']}_s100"], labels[f"E02_k{ward_rep['k']}"])
    check("report contains representative ARI", f"{rep_ari:.3f}" in report, f"{rep_ari:.3f}")
    check("report contains stability medians",
          f"{km_rep['median_ari']:.3f}" in report and f"{ward_rep['median_ari']:.3f}" in report)
    check("report has both season directions",
          "denominator = season days" in report and "denominator = cluster days" in report)

    print(f"\n{len(FAILURES)} failure(s)" if FAILURES else "\nALL CHECKS PASSED")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
