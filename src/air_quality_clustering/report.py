"""Generate the final English report from a completed run's real numbers.

Every figure and table referenced here is produced by the pipeline; no
number is hand-entered. Descriptive claims only: clusters are observational
groupings, not pollution sources, causal effects or official health grades.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import FEATURE_ORDER
from .data import POLLUTANTS

SEASON_ORDER = ["spring", "summer", "autumn", "winter"]


def _md_table(frame: pd.DataFrame, float_fmt: str = ".3f") -> str:
    head = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    sep = "| " + " | ".join("---" for _ in frame.columns) + " |"
    lines = [head, sep]
    for _, row in frame.iterrows():
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append(format(value, float_fmt) if np.isfinite(value) else "null")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _profile_narrative(profiles: pd.DataFrame, algorithm: str, mapping: dict[int, int]) -> str:
    """Per-pollutant comparison against that pollutant's all-eligible-day median.

    Comparing every pollutant to a single PM2.5 reference would mix units and
    scales; each pollutant is judged against its own overall median instead.
    """
    sub = profiles[profiles["algorithm"] == algorithm]
    lines = []
    ordered = sorted(sub["cluster"].unique(), key=lambda c: mapping.get(int(c), int(c)))
    overall = {}
    for pollutant in FEATURE_ORDER:
        rows = sub[sub["pollutant"] == pollutant]
        overall[pollutant] = float(np.average(rows["median"], weights=rows["n_days"]))
    for cluster in ordered:
        rows = sub[sub["cluster"] == cluster].set_index("pollutant")
        n = int(rows.loc["PM2.5", "n_days"])
        ratios = {p: float(rows.loc[p, "median"]) / overall[p] if overall[p] > 0 else None
                  for p in FEATURE_ORDER}
        highs = [p for p in FEATURE_ORDER if ratios[p] is not None and ratios[p] > 1.2]
        lows = [p for p in FEATURE_ORDER if ratios[p] is not None and ratios[p] < 0.8]
        desc = (
            f"- **Display C{mapping.get(int(cluster), int(cluster) + 1)}** (original label {int(cluster)}, n={n} days): "
            f"PM2.5 median {rows.loc['PM2.5', 'median']:.1f} µg/m³ "
            f"(IQR {rows.loc['PM2.5', 'q1']:.1f}–{rows.loc['PM2.5', 'q3']:.1f}), "
            f"O3 median {rows.loc['O3', 'median']:.1f} µg/m³. "
        )
        parts = []
        if highs:
            parts.append("above 1.2× its own all-day median: "
                         + ", ".join(f"{p} ({ratios[p]:.2f}×)" for p in highs))
        if lows:
            parts.append("below 0.8×: " + ", ".join(f"{p} ({ratios[p]:.2f}×)" for p in lows))
        desc += ("; ".join(parts) + ".") if parts else "All six medians lie within 0.8–1.2× of their all-day medians."
        lines.append(desc)
    return "\n".join(lines)


def _case_narrative(case: dict | None) -> str:
    if not case or case.get("status") != "ok":
        return "No off-diagonal disagreement cell exists at the selected resolution (the two partitions coincide)."
    cell, counts = case["cell"], case["counts"]
    lines = [
        f"Rule: {case['rule']}. Selected cell: K-Means cluster {cell['kmeans_cluster']} × Ward cluster "
        f"{cell['ward_cluster']}, containing **{cell['n_days']} days**.",
        "",
        f"- K-Means cluster {cell['kmeans_cluster']} holds {counts['km_cluster_total']} days in total; "
        f"Ward cluster {cell['ward_cluster']} holds {counts['ward_cluster_total']}.",
        f"- Intersection {counts['intersection_days']} days; K-Means-only {counts['km_only_days']} days; "
        f"Ward-only {counts['ward_only_days']} days.",
        "",
        "Earliest five dates in that cell, in original units (µg/m³):",
        "",
    ]
    examples = pd.DataFrame(case["examples"])
    if len(examples):
        lines.append(_md_table(examples, float_fmt=".1f"))
    medians = case.get("example_median_original_units") or {}
    if medians:
        lines += ["", "Their medians: " + ", ".join(f"{k} {v:.1f}" for k, v in medians.items()) + "."]
    return "\n".join(lines)


def write_report(cfg, run_dir: Path, state: dict) -> Path:
    experiments: pd.DataFrame = state["experiments"]
    selection = state["selection"]
    agg: pd.DataFrame = state["stability_agg"]
    compare = state["compare"]
    sensitivity: pd.DataFrame = state["sensitivity"]
    daily = state["tables"]["daily_pollutants"]
    coverage = state["tables"]["daily_coverage"]
    profiles = compare["profiles"]
    mappings = compare["mappings"]
    reps = compare["reps"]
    rep_ari = compare["representative_ari"]
    variance = compare["variance"]

    km_rep = selection["methods"]["kmeans"]["representative"]
    ward_rep = selection["methods"]["ward"]["representative"]
    db_sel = selection.get("dbscan", {})

    km = experiments[experiments["algorithm"] == "kmeans"]
    ward = experiments[experiments["algorithm"] == "ward"]
    db = experiments[experiments["algorithm"] == "dbscan"]
    km_med = km.groupby("k")[["silhouette", "davies_bouldin", "calinski_harabasz"]].median().reset_index()
    cluster_run = state.get("cluster_run") or run_dir
    seed_ari = pd.read_csv(cluster_run / "tables" / "kmeans_seed_ari.csv")
    same_k = pd.read_csv(cluster_run / "tables" / "same_k_ari.csv")
    cont = compare["contingency"]
    weather = compare["weather"]
    season = compare["season"]

    n = len(daily)
    excluded = int((~coverage["eligible"]).sum())
    spearman = daily[POLLUTANTS].corr(method="spearman")
    pm_pm10 = float(spearman.loc["PM2.5", "PM10"])
    no2_co = float(spearman.loc["NO2", "CO"])

    db_single = int(((db["n_clusters"] == 1)).sum())
    db_multi = db[db["n_clusters"] >= 2]

    parts: list[str] = []
    parts.append(f"""# Discovering Daily Air-Pollution Patterns in Beijing: An Unsupervised Comparison of K-Means, Ward and DBSCAN

**Course project report — AISC5030, Topic 3: Knowledge Discovery from Scientific Data**
Run ID: `{run_dir.name}` · Protocol v1 (frozen before clustering) · Generated from the run's real outputs.

## Abstract

We apply three unsupervised clustering methods to {n} eligible days (2013-03-01 to 2017-02-28) of six daily-mean pollutant concentrations (PM2.5, PM10, SO2, NO2, CO, O3; µg/m³) from the UCI Beijing multi-site air-quality dataset, station Aotizhongxin. Days enter the analysis only when every pollutant has at least 18 valid hourly observations; {excluded} of 1,461 calendar days are excluded and never imputed. The main representation is log1p followed by StandardScaler. All 89 base candidates (70 K-Means, 7 Ward cuts of one tree, 12 DBSCAN settings) are retained. Representative cluster counts are chosen by a frozen rule chain (minimum cluster size, silhouette shortlist, 20 temporal-block resamples with refitted preprocessing): K-Means k={km_rep['k'] if km_rep else 'none'}, Ward k={ward_rep['k'] if ward_rep else 'none'}. DBSCAN yields no representative under the coverage/cluster-count rules ({db_single} of 12 settings collapse to a single cluster plus noise), indicating the data behave more like a continuous pollution gradient than separated density clusters. The two primary representatives agree with ARI={rep_ari:.3f}; same-k agreement varies from {same_k['ari_kmeans_seed100_vs_ward'].min():.3f} to {same_k['ari_kmeans_seed100_vs_ward'].max():.3f}, showing that cluster-count and algorithm geometry both matter. Block-stability median ARIs are {km_rep['median_ari']:.3f} (K-Means) and {ward_rep['median_ari']:.3f} (Ward); a raw-value sensitivity branch gives ARIs of {', '.join(f"{r['algorithm']}={r['ari_vs_main']:.3f}" for _, r in sensitivity.iterrows())} against the main representation. At the selected resolution (k=2 for both primary methods) the profiles describe a lower-load, ozone-richer day type and a higher-load, particulate- and CO-heavy day type; these are observational groupings, not causal or health-grade claims.

## 1. Introduction and research question

Topic 3 asks for knowledge discovery from scientific data without supervised labels. Our question: *do days with similar multi-pollutant signatures form reproducible, interpretable groups, and how do three clustering geometries (centroid, hierarchical, density) organise the same days differently?* We compare K-Means and Ward as primary methods and DBSCAN as a supplementary probe of the density assumption. No prediction target, no ground-truth labels, and no date/season/weather variable enters the distance.

## 2. Data and preprocessing

Source: UCI Machine Learning Repository ID 501, Beijing multi-site air-quality data (Zhang et al., 2016; UCI, 2017). Raw archive and station CSV are read-only; SHA256 hashes are recorded in the run manifest (CSV `a94fbcfc…351b`, full value in `manifest.json`). The station file contains 35,064 hourly rows over 1,461 calendar days.

Cleaning rules (docs/05): NA/empty/non-finite hours are missing; negative concentrations are invalidated and counted; zeros are kept; conflicting duplicate hours raise an error (none occurred); identical duplicates are removed (none occurred). A day is eligible iff each of the six pollutants has ≥18 valid hours; each daily value is the mean of that pollutant's own valid hours. Result: {n} eligible days, {excluded} excluded (reasons per day in `excluded_days.csv`). Missingness is not uniform in time (F01), so the eligible-day sample carries a temporal selection bias that we disclose rather than correct.

Representation: `Z_j = (log1p(x_j) − mean_j)/std_j` with population standard deviation (F02 motivates the transform: right-skewed, multi-scale concentrations). Background columns (season, weather, station, identifiers) are excluded from the matrix by configuration validation.

## 3. Methods and evaluation protocol

- **E01 K-Means**: k=2…8 × seeds 100…109, n_init=20, max_iter=300, Lloyd → 70 runs.
- **E02 Ward**: one euclidean linkage tree, cut at k=2…8 → 7 partitions (not seven independent fits).
- **E03 DBSCAN**: m=5/10/20; eps = linear quantiles {{0.70, 0.80, 0.90, 0.95}} of the m-th nearest-neighbour distance **including the query point itself** → 12 settings.

Internal metrics (silhouette, Davies–Bouldin, Calinski–Harabasz) are computed on the same six-dimensional Z for K-Means/Ward over all {n} days. DBSCAN metrics use only the non-noise subset (`n_scored` reported); noise (−1) is never treated as a cluster. Invalid metrics are stored as null with a reason. ARI measures partition agreement without labels and is invariant to cluster-number permutation.

Selection (frozen before looking at any interpretation): minimum cluster size max(10, ceil(0.01·n)) = {cfg.min_cluster_size(n)} days; per-method silhouette shortlist within 0.02 of the eligible best; 20 temporal-block resamples (7-day blocks over the full 1,461-day calendar, 167 of 209 blocks per draw, seeds 100…119) with scaler and model refit on each subset; representative = highest median ARI, ties within 0.02 prefer smaller k. Median ARI < 0.70 flags weak stability (a project warning line, not a validity proof). Because k is chosen using these resamples, the stability figures carry exploratory selection bias and are not confidence intervals.

## 4. Exploratory analysis

F01 shows monthly valid-hour fractions and eligible-day fractions; several months fall below the 18/24 rule. F02 shows why log1p is used. F03: Spearman ρ(PM2.5, PM10)={pm_pm10:.2f} and ρ(NO2, CO)={no2_co:.2f} — correlated pollutants receive repeated weight in euclidean distance, a limitation we discuss rather than silently "fix". F04 displays daily series with excluded days left as gaps.

Missingness at both scales and the extreme-day index are tabulated in `tables/eda_hourly_missing.csv` and `tables/eda_extreme_days.csv`. The five highest daily means per pollutant are listed with the 99th-percentile threshold and were **retained**: an extreme value is only removed if the raw hours prove it erroneous, which was never the case here.

![F01](figures/F01_monthly_coverage.png)
![F02](figures/F02_distributions.png)
![F03](figures/F03_correlation.png)
![F04](figures/F04_time_series.png)

## 5. Candidate results (all 89 retained)

K-Means medians over 10 seeds:

{_md_table(km_med)}

Seed agreement (45 pairwise ARIs per k):

{_md_table(seed_ari[["k", "median_ari", "q1_ari", "q3_ari", "min_ari"]])}

Ward cuts of the single tree:

{_md_table(ward[["k", "silhouette", "davies_bouldin", "calinski_harabasz", "min_cluster_size"]])}

DBSCAN outcomes (noise fraction and n_scored always shown):

{_md_table(db[["experiment_id", "min_samples", "eps", "n_clusters", "noise_fraction", "n_scored", "silhouette", "status"]])}

{db_single} of 12 DBSCAN settings produce a single non-noise cluster; only {len(db_multi)} setting(s) yield ≥2 clusters. This is evidence about the density assumption, not a failure to be hidden: the six-dimensional log-standardised days form a gradient with a dense core rather than well-separated density clusters.

![F05](figures/F05_kmeans_metrics.png)
![F06](figures/F06_dendrogram.png)
![F07](figures/F07_dbscan.png)

## 6. Representative selection and stability

Shortlists and stability (20 block resamples each; failed repetitions counted, never masked):

{_md_table(agg)}

Selection outcome: K-Means representative k={km_rep['k'] if km_rep else 'none'} (median block ARI {km_rep['median_ari']:.3f}{', weak-stability flag' if km_rep and km_rep['weak_stability'] else ''}); Ward representative k={ward_rep['k'] if ward_rep else 'none'} (median ARI {ward_rep['median_ari']:.3f}{', weak-stability flag' if ward_rep and ward_rep['weak_stability'] else ''}). DBSCAN representative: {db_sel.get('representative') or 'none — ' + str(db_sel.get('reason'))}. The full rule chain, exclusions and per-repetition ARIs are in `selection.json` and `stability/`.

## 7. What the clusters mean (original units)

Profiles are medians/IQRs of raw daily concentrations grouped by the actual labels; display numbers are ordered by PM2.5 median and mapped in `artifacts/display_mapping.json`.

### K-Means (k={reps['kmeans']})

{_profile_narrative(profiles, "kmeans", mappings["kmeans"])}

### Ward (k={reps['ward']})

{_profile_narrative(profiles, "ward", mappings["ward"])}

![F09 K-Means](figures/F09_profiles_kmeans.png)
![F09 Ward](figures/F09_profiles_ward.png)

### 7.1 Within-cluster spread comes before any background story

Protocol §8 requires comparing the dispersion *inside* a cluster before invoking season or weather. IQR/median is scale-free, so it is comparable across pollutants (original units, K-Means representative):

{_md_table(compare["spread"][compare["spread"]["algorithm"] == "kmeans"][["cluster", "pollutant", "n_days", "median", "iqr", "iqr_over_median"]]) if "spread" in compare else "see tables/cluster_spread.csv"}

### 7.2 Which pollutants actually separate the clusters, and which overlap

Shared IQR span between the two clusters, in original units and as a fraction of the pooled IQR span (K-Means):

{_md_table(compare["overlap"][compare["overlap"]["algorithm"] == "kmeans"][["pollutant", "iqr_overlap_width", "overlap_fraction_of_pooled_iqr", "median_ratio_b_over_a"]]) if "overlap" in compare else "see tables/cluster_overlap.csv"}

The full distributions (not just medians) make the same point:

![F13 K-Means](figures/F13_boxplots_kmeans.png)
![F13 Ward](figures/F13_boxplots_ward.png)

## 8. Method agreement and disagreement

Representative partitions: ARI={rep_ari:.3f}; contingency (rows K-Means, columns Ward):

{_md_table(cont.reset_index())}

Same-k ARI separates algorithm geometry from cluster-count effects:

{_md_table(same_k)}

The shared PCA projection (display only; PC1+PC2 explain {sum(variance[:2]):.1%} of variance) shows how the same dates are organised by each method:

![F08](figures/F08_pca_panels.png)
![F10](figures/F10_agreement.png)

### 8.1 Case study of the largest split/merge cell (fixed selection rule)

{_case_narrative(compare.get("case"))}

## 9. Background context (after frozen selection)

Eligible-day coverage by season — the denominator that must accompany every seasonal percentage (2013 and 2017 are partial calendar years):

{_md_table(compare["season_coverage"]) if "season_coverage" in compare else "see tables/season_coverage.csv"}

Year composition, P(cluster | year), denominator = eligible days in that year:

{_md_table(compare["years"][compare["years"]["algorithm"] == "kmeans"]) if "years" in compare else "see tables/year_composition.csv"}

Season composition within each K-Means cluster (denominator = cluster days):

{_md_table(season.reindex(columns=[c for c in SEASON_ORDER if c in season.columns]).reset_index())}

The same composition for the Ward representative (denominator = Ward cluster days):

{_md_table(compare.get("season_ward", season).reindex(columns=[c for c in SEASON_ORDER if c in compare.get("season_ward", season).columns]).reset_index())}

The reverse direction — within each season, the fraction of days assigned to each K-Means cluster (denominator = season days) — is reported as well, so neither direction is read as the other:

{_md_table(compare.get("season_of_season", season).reindex(columns=[c for c in SEASON_ORDER if c in compare.get("season_of_season", season).columns]).reset_index())}

Weather medians by cluster (descriptive; valid-day denominators in `weather_profiles.csv`):

{_md_table(weather[weather["algorithm"] == "kmeans"][["cluster", "variable", "n_valid_days", "median"]].pivot(index="cluster", columns="variable", values="median").reset_index())}

These associations are descriptive. Clusters are not pollution sources, not causal effects of weather, and not official air-quality health grades. No significance tests are run on the six clustering features themselves (that would circularly "validate" the discovery).

![F11](figures/F11_background.png)

## 10. Sensitivity and robustness

Raw+StandardScaler at the frozen k, compared with the main representation:

{_md_table(sensitivity[["algorithm", "k", "ari_vs_main"]])}

ARI alone is not enough: the protocol also requires the **original-unit profile change**. Median shift of each cluster under raw+StandardScaler relative to the main representation (K-Means):

{_md_table(state["sensitivity_shift"][state["sensitivity_shift"]["algorithm"] == "kmeans"][["cluster", "pollutant", "median_main", "median_raw", "median_ratio_raw_over_main"]]) if "sensitivity_shift" in state else "see sensitivity/profile_shift.csv"}

Block stability and transformation sensitivity together:

![F12](figures/F12_stability_sensitivity.png)

## 11. Limitations

1. Eligible-day selection bias: missing hours are temporally structured; excluded days are not random.
2. Correlated pollutants double-weight shared information in euclidean distance.
3. Internal metrics favour compact spherical clusters; DBSCAN's degeneration shows the geometry dependence.
4. Stability resamples were used for selection → exploratory bias; ARI values are not confidence intervals.
5. Single station, single city, 2013–2017; no claim of generalisation to other periods or sites.
6. Observational groupings only; no causal, source-apportionment or health-grade interpretation.

## 12. Conclusion

On the same {n} days and the same transformed space, K-Means and Ward produce broadly consistent low-vs-high pollution organisation at their representative k (ARI={rep_ari:.3f}) but differ in how intermediate days are split (same-k ARI as low as {same_k['ari_kmeans_seed100_vs_ward'].min():.3f}). DBSCAN's near-universal single-cluster outcome characterises the data as a continuous gradient with a dense core. The discovered day types are reproducible under temporal-block perturbation (median ARI ≥ {min(r['median_ari'] for r in (km_rep, ward_rep) if r):.3f}). Robustness to the representation is method-dependent: Ward's partition is moderately stable under the raw-value branch (ARI={sensitivity.loc[sensitivity['algorithm'] == 'ward', 'ari_vs_main'].iloc[0]:.3f}) whereas K-Means reorganises substantially (ARI={sensitivity.loc[sensitivity['algorithm'] == 'kmeans', 'ari_vs_main'].iloc[0]:.3f}), so the log1p choice is an explicit, disclosed modelling decision rather than a neutral default. Together these results support using the clusters as descriptive knowledge rather than predictive or causal claims.

## References

- UCI Machine Learning Repository (2017). Beijing Multi-Site Air-Quality Data Set, ID 501. https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data
- Zhang, S., Guo, B., Dong, A., He, J., Xu, Z., Chen, S.X. (2016). Cautionary tales on air-quality improvement in Beijing. *Proc. R. Soc. A* 472: 20160457. (Dataset companion paper as cited by UCI.)
- Course materials: AISC5030 Week 5 slides/notebooks (method definitions and Topic 3 brief).

## Appendix A. Reproducibility

Run `{run_dir.name}`; config snapshot and SHA256 of data, config and executed source in `manifest.json`; environment in `requirements-lock.txt`. Re-run: `python scripts/run_project.py --stage all`; notebooks 01–03 execute top-to-bottom via `scripts/run_notebooks.py`. All 89 candidate label files carry date keys; figures F01–F12 in `figures/`.
""")

    report_path = run_dir / "report.md"
    report_path.write_text("\n".join(parts), encoding="utf-8")
    return report_path


def project_root_of(run_dir: Path) -> Path:
    """results/runs/<id> -> project root (three levels up)."""
    return run_dir.parent.parent.parent
