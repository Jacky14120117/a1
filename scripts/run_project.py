"""Unified entry point for the AISC5030 Topic 3 study.

Usage (from any working directory):
    python scripts/run_project.py --stage prepare --config configs/project.yaml
    python scripts/run_project.py --stage eda cluster --config configs/project.yaml
    python scripts/run_project.py --stage all --config configs/project.yaml

Implemented stages: prepare | eda | cluster. Later stages (stability,
compare, sensitivity, export) are defined by docs/03_execution_plan.md and
return an explicit "not implemented" error until built.

One invocation = one run directory = one manifest. Within a run, downstream
stages reuse in-memory tables; across runs, upstream outputs are verified by
SHA256 before use. No stage fabricates results.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from air_quality_clustering import provenance  # noqa: E402
from air_quality_clustering.clustering import (  # noqa: E402
    dbscan_eps_grid,
    run_dbscan_candidates,
    run_kmeans_candidates,
    run_ward_candidates,
)
from air_quality_clustering.config import STAGES, load_config  # noqa: E402
from air_quality_clustering.data import prepare_tables  # noqa: E402
from air_quality_clustering.evaluation import ari  # noqa: E402
from air_quality_clustering.features import build_feature_matrix  # noqa: E402
from air_quality_clustering import plots  # noqa: E402

IMPLEMENTED_STAGES = ("prepare", "eda", "cluster")
PROCESSED_TABLES = ("daily_coverage", "daily_pollutants", "excluded_days", "daily_context")


class UpstreamError(RuntimeError):
    """Raised when a required upstream output is missing or has changed."""


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")


def _latest_completed_run(root: Path, stage: str) -> Path | None:
    runs_dir = root / "results" / "runs"
    if not runs_dir.exists():
        return None
    candidates = []
    for run_dir in runs_dir.iterdir():
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        completed = manifest.get("stages_completed") or manifest.get("stage_summary", {}).get("stages_completed", [])
        if stage in completed and manifest.get("status") in ("completed", "completed_with_warnings"):
            candidates.append((run_dir.name, run_dir, manifest))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _verified_table(root: Path, name: str, run_dir: Path | None) -> pd.DataFrame:
    """Read a processed table, checking its hash against the producing run."""
    path = root / "data" / "processed" / f"{name}.csv"
    if not path.exists():
        raise UpstreamError(f"missing upstream table {path}; run --stage prepare first")
    if run_dir is not None:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        expected = manifest.get("output_sha256", {}).get(path.relative_to(root).as_posix())
        if expected and provenance.sha256_file(path) != expected:
            raise UpstreamError(
                f"{name}.csv changed since run {run_dir.name}; rerun prepare before continuing"
            )
    return pd.read_csv(path)


def stage_prepare(cfg, run_dir: Path, state: dict) -> dict:
    tables = prepare_tables(
        cfg.csv_path,
        station=cfg.station,
        start=cfg.start_date,
        end=cfg.end_date,
        minimum_valid_hours=cfg.minimum_valid_hours,
        expected_rows=cfg.expected_hourly_rows,
    )
    state["tables"] = tables
    processed = cfg.root / "data" / "processed"
    outputs = {"hourly_clean": cfg.root / "data" / "interim" / "hourly_clean.csv"}
    _write_csv(tables["hourly_clean"], outputs["hourly_clean"])
    for name in PROCESSED_TABLES:
        outputs[name] = processed / f"{name}.csv"
        _write_csv(tables[name], outputs[name])

    matrix = build_feature_matrix(tables["daily_pollutants"], cfg.main_representation)
    state["matrix"] = matrix
    features_frame = pd.DataFrame(matrix.X, columns=matrix.feature_names)
    features_frame.insert(0, "date", matrix.dates)
    outputs["features_main"] = processed / "features_main.csv"
    _write_csv(features_frame, outputs["features_main"])

    coverage, daily = tables["daily_coverage"], tables["daily_pollutants"]
    checks = {
        "coverage_rows": len(coverage),
        "expected_calendar_days": cfg.expected_calendar_days,
        "eligible_days": len(daily),
        "expected_eligible_days": cfg.expected_eligible_days,
        "excluded_days": len(tables["excluded_days"]),
        "main_matrix_shape": list(matrix.X.shape),
        "dates_unique_sorted": bool(
            (matrix.dates == sorted(matrix.dates)).all()
            and len(set(matrix.dates)) == len(matrix.dates)
        ),
    }
    warnings: list[str] = []
    if len(coverage) != cfg.expected_calendar_days:
        warnings.append("coverage row count differs from preflight expectation")
    if len(daily) != cfg.expected_eligible_days:
        warnings.append(
            f"eligible days {len(daily)} differ from preflight {cfg.expected_eligible_days}; "
            "investigate source version before proceeding"
        )
    provenance.write_json(run_dir / "artifacts" / "scaler_main.json", matrix.scaler_params())
    summary = {
        "hourly_audit": tables["audit"].as_dict(),
        "acceptance_checks": checks,
        "scaler_params": matrix.scaler_params(),
        "eligible_days_by_year": (
            daily.assign(year=daily["date"].str[:4]).groupby("year").size().to_dict()
        ),
    }
    provenance.write_json(run_dir / "tables" / "prepare_summary.json", summary)
    return {
        "outputs": outputs,
        "summary": summary,
        "warnings": warnings,
        "checks": checks,
    }


def stage_eda(cfg, run_dir: Path, state: dict) -> dict:
    tables = state.get("tables")
    if tables is None:
        upstream = _latest_completed_run(cfg.root, "prepare")
        tables = {name: _verified_table(cfg.root, name, upstream) for name in PROCESSED_TABLES}
        state["tables"] = tables
    coverage, daily = tables["daily_coverage"], tables["daily_pollutants"]

    figures_dir = run_dir / "figures"
    figures = {
        "F01_monthly_coverage": plots.fig_f01_monthly_coverage(coverage, figures_dir / "F01_monthly_coverage.png"),
        "F02_distributions": plots.fig_f02_distributions(daily, figures_dir / "F02_distributions.png"),
        "F03_correlation": plots.fig_f03_correlation(daily, figures_dir / "F03_correlation.png"),
        "F04_time_series": plots.fig_f04_time_series(daily, coverage, figures_dir / "F04_time_series.png"),
    }

    tables_dir = run_dir / "tables"
    quantiles = daily[POLLUTANTS_ORDER].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    quantiles.to_csv(tables_dir / "eda_daily_quantiles.csv", encoding="utf-8")
    spearman = daily[POLLUTANTS_ORDER].corr(method="spearman")
    spearman.to_csv(tables_dir / "eda_spearman.csv", encoding="utf-8")
    missing_summary = {
        "calendar_days": len(coverage),
        "eligible_days": int(coverage["eligible"].sum()),
        "excluded_days": int((~coverage["eligible"]).sum()),
        "hourly_missing_counts": state.get("tables", {}).get("audit").as_dict()["missing_counts"]
        if state.get("tables", {}).get("audit") is not None else None,
    }
    provenance.write_json(tables_dir / "eda_summary.json", missing_summary)

    outputs = {**{k: Path(v) for k, v in figures.items()},
               "eda_daily_quantiles": tables_dir / "eda_daily_quantiles.csv",
               "eda_spearman": tables_dir / "eda_spearman.csv",
               "eda_summary": tables_dir / "eda_summary.json"}
    summary = {"figures": [p.name for p in figures.values()], "missing_summary": missing_summary}
    return {"outputs": outputs, "summary": summary, "warnings": [],
            "checks": {"figures_written": len(figures)}}


def stage_cluster(cfg, run_dir: Path, state: dict) -> dict:
    matrix = state.get("matrix")
    if matrix is None:
        upstream = _latest_completed_run(cfg.root, "prepare")
        daily = _verified_table(cfg.root, "daily_pollutants", upstream)
        matrix = build_feature_matrix(daily, cfg.main_representation)
        state["matrix"] = matrix
    X = matrix.X

    rows: list[dict] = []
    labels: dict[str, np.ndarray] = {}
    km_rows, km_labels = run_kmeans_candidates(X, cfg)
    ward_rows, ward_labels, Z = run_ward_candidates(X, cfg)
    grid = dbscan_eps_grid(X, cfg)
    db_rows, db_labels, grid = run_dbscan_candidates(X, cfg, grid)
    rows.extend(km_rows + ward_rows + db_rows)
    labels.update(km_labels)
    labels.update(ward_labels)
    labels.update(db_labels)

    experiments = pd.DataFrame(rows)
    experiments.insert(0, "run_id", run_dir.name)
    experiments_path = run_dir / "experiments.csv"
    _write_csv(experiments, experiments_path)

    labels_dir = run_dir / "labels"
    label_outputs = {}
    for experiment_id, lab in labels.items():
        frame = pd.DataFrame({"date": matrix.dates, "cluster": lab})
        path = labels_dir / f"{experiment_id}.csv"
        _write_csv(frame, path)
        label_outputs[f"label_{experiment_id}"] = path

    # Seed agreement: 45 pairwise ARIs per k (initialization stability only).
    seed_aris: dict[int, list[float]] = {}
    km_settings = cfg.algorithms["kmeans"]
    for k in km_settings["k"]:
        seeds = km_settings["seeds"]
        pair_values = [
            ari(km_labels[f"E01_k{k}_s{a}"], km_labels[f"E01_k{k}_s{b}"])
            for a, b in itertools.combinations(seeds, 2)
        ]
        seed_aris[int(k)] = pair_values
    seed_df = pd.DataFrame(
        [{"k": k, "pair_count": len(v), "median_ari": float(np.median(v)),
          "q1_ari": float(np.quantile(v, 0.25)), "q3_ari": float(np.quantile(v, 0.75)),
          "min_ari": float(np.min(v))} for k, v in sorted(seed_aris.items())]
    )
    _write_csv(seed_df, run_dir / "tables" / "kmeans_seed_ari.csv")

    # Same-k K-Means(seed=100) vs Ward agreement (algorithm difference at equal k).
    same_k = pd.DataFrame([
        {"k": int(k), "ari_kmeans_seed100_vs_ward": ari(km_labels[f"E01_k{k}_s100"], ward_labels[f"E02_k{k}"])}
        for k in km_settings["k"]
    ])
    _write_csv(same_k, run_dir / "tables" / "same_k_ari.csv")

    np.save(run_dir / "artifacts" / "ward_linkage.npy", Z)
    provenance.write_json(
        run_dir / "artifacts" / "dbscan_eps_grid.json",
        {str(m): {"eps_by_quantile": {str(q): eps for q, eps in g["eps_by_quantile"].items()}}
         for m, g in grid.items()},
    )

    figures_dir = run_dir / "figures"
    figures = {
        "F05_kmeans_metrics": plots.fig_f05_kmeans_metrics(experiments, seed_aris, figures_dir / "F05_kmeans_metrics.png"),
        "F06_dendrogram": plots.fig_f06_dendrogram(Z, figures_dir / "F06_dendrogram.png"),
        "F07_dbscan": plots.fig_f07_dbscan(grid, experiments, figures_dir / "F07_dbscan.png"),
    }

    outputs = {"experiments": experiments_path,
               "kmeans_seed_ari": run_dir / "tables" / "kmeans_seed_ari.csv",
               "same_k_ari": run_dir / "tables" / "same_k_ari.csv",
               "ward_linkage": run_dir / "artifacts" / "ward_linkage.npy",
               "dbscan_eps_grid": run_dir / "artifacts" / "dbscan_eps_grid.json",
               **label_outputs,
               **{k: Path(v) for k, v in figures.items()}}
    status_counts = experiments.groupby(["algorithm", "status"]).size().to_dict()
    summary = {
        "candidate_rows": len(experiments),
        "expected_candidate_rows": 89,
        "status_counts": {f"{a}:{s}": int(n) for (a, s), n in status_counts.items()},
        "label_files": len(label_outputs),
        "figures": [p.name for p in figures.values()],
    }
    warnings = []
    if len(experiments) != 89:
        warnings.append(f"expected 89 base candidates, got {len(experiments)}")
    checks = {"candidate_rows": len(experiments), "label_files": len(label_outputs),
              "same_k_ari_rows": len(same_k)}
    return {"outputs": outputs, "summary": summary, "warnings": warnings, "checks": checks}


POLLUTANTS_ORDER = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
STAGE_FUNCTIONS = {"prepare": stage_prepare, "eda": stage_eda, "cluster": stage_cluster}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", nargs="+", choices=STAGES, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "project.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config, root=ROOT)
    stages = list(IMPLEMENTED_STAGES) if "all" in args.stage else args.stage
    unimplemented = [s for s in stages if s not in IMPLEMENTED_STAGES]
    if unimplemented:
        print(
            f"Stage(s) not implemented yet: {', '.join(unimplemented)}. "
            "See docs/03_execution_plan.md for the implementation order.",
            file=sys.stderr,
        )
        return 2

    run_id = provenance.new_run_id(cfg.config_sha256)
    run_dir = provenance.create_run_dir(cfg.root, run_id)
    print(f"run_id: {run_id}")
    state: dict = {}
    completed: list[str] = []
    all_outputs: dict[str, Path] = {}
    summaries: dict = {}
    all_warnings: list[str] = []
    try:
        for stage in stages:
            result = STAGE_FUNCTIONS[stage](cfg, run_dir, state)
            completed.append(stage)
            all_outputs.update(result["outputs"])
            summaries[stage] = result["summary"]
            all_warnings.extend(result["warnings"])
            print(f"[{stage}] completed")
            for key, value in result["checks"].items():
                print(f"  {key}: {value}")
            for warning in result["warnings"]:
                print(f"  WARNING: {warning}")
    except Exception as exc:
        failed_manifest = provenance.build_manifest(
            run_id=run_id, stage=",".join(stages), status="failed",
            config_path=cfg.config_path, config_sha256=cfg.config_sha256, root=cfg.root,
            input_hashes={}, output_hashes=provenance.file_hashes(all_outputs.values(), cfg.root),
            stage_summary=summaries,
            warnings=all_warnings, error=f"{type(exc).__name__}: {exc}",
        )
        failed_manifest["stages_completed"] = completed
        provenance.write_json(run_dir / "manifest.json", failed_manifest)
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    input_paths = [cfg.archive_path, cfg.csv_path]
    manifest = provenance.build_manifest(
        run_id=run_id, stage=",".join(stages),
        status="completed" if not all_warnings else "completed_with_warnings",
        config_path=cfg.config_path, config_sha256=cfg.config_sha256, root=cfg.root,
        input_hashes=provenance.file_hashes(input_paths, cfg.root),
        output_hashes=provenance.file_hashes(all_outputs.values(), cfg.root),
        stage_summary=summaries,
        warnings=all_warnings,
    )
    manifest["stages_completed"] = completed
    provenance.write_json(run_dir / "manifest.json", manifest)
    (run_dir / "config_snapshot.yaml").write_text(
        cfg.config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    print(f"results: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
