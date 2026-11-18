# Discovering Daily Air-Pollution Patterns in Beijing: An Unsupervised Comparison of K-Means, Ward and DBSCAN

**Course project report — AISC5030, Topic 3: Knowledge Discovery from Scientific Data**
Run ID: `20261116T193443Z-75ddd759` · Protocol v1 (frozen before clustering) · Generated from the run's real outputs.

## Abstract

We apply three unsupervised clustering methods to 1333 eligible days (2013-03-01 to 2017-02-28) of six daily-mean pollutant concentrations (PM2.5, PM10, SO2, NO2, CO, O3; µg/m³) from the UCI Beijing multi-site air-quality dataset, station Aotizhongxin. Days enter the analysis only when every pollutant has at least 18 valid hourly observations; 128 of 1,461 calendar days are excluded and never imputed. The main representation is log1p followed by StandardScaler. All 89 base candidates (70 K-Means, 7 Ward cuts of one tree, 12 DBSCAN settings) are retained. Representative cluster counts are chosen by a frozen rule chain (minimum cluster size, silhouette shortlist, 20 temporal-block resamples with refitted preprocessing): K-Means k=2, Ward k=2. DBSCAN yields no representative under the coverage/cluster-count rules (11 of 12 settings collapse to a single cluster plus noise), indicating the data behave more like a continuous pollution gradient than separated density clusters. The two primary representatives agree with ARI=0.892; same-k agreement varies from 0.359 to 0.892, showing that cluster-count and algorithm geometry both matter. Block-stability median ARIs are 0.970 (K-Means) and 0.707 (Ward); a raw-value sensitivity branch gives ARIs of kmeans=0.392, ward=0.719 against the main representation. Cluster profiles in original units describe low-load, particulate-heavy and ozone-associated day types; these are observational groupings, not causal or health-grade claims.

## 1. Introduction and research question

Topic 3 asks for knowledge discovery from scientific data without supervised labels. Our question: *do days with similar multi-pollutant signatures form reproducible, interpretable groups, and how do three clustering geometries (centroid, hierarchical, density) organise the same days differently?* We compare K-Means and Ward as primary methods and DBSCAN as a supplementary probe of the density assumption. No prediction target, no ground-truth labels, and no date/season/weather variable enters the distance.

## 2. Data and preprocessing

Source: UCI Machine Learning Repository ID 501, Beijing multi-site air-quality data (Zhang et al., 2016; UCI, 2017). Raw archive and station CSV are read-only; SHA256 hashes are recorded in the run manifest (CSV `a94fbcfc…351b`, full value in `manifest.json`). The station file contains 35,064 hourly rows over 1,461 calendar days.

Cleaning rules (docs/05): NA/empty/non-finite hours are missing; negative concentrations are invalidated and counted; zeros are kept; conflicting duplicate hours raise an error (none occurred); identical duplicates are removed (none occurred). A day is eligible iff each of the six pollutants has ≥18 valid hours; each daily value is the mean of that pollutant's own valid hours. Result: 1333 eligible days, 128 excluded (reasons per day in `excluded_days.csv`). Missingness is not uniform in time (F01), so the eligible-day sample carries a temporal selection bias that we disclose rather than correct.

Representation: `Z_j = (log1p(x_j) − mean_j)/std_j` with population standard deviation (F02 motivates the transform: right-skewed, multi-scale concentrations). Background columns (season, weather, station, identifiers) are excluded from the matrix by configuration validation.

## 3. Methods and evaluation protocol

- **E01 K-Means**: k=2…8 × seeds 100…109, n_init=20, max_iter=300, Lloyd → 70 runs.
- **E02 Ward**: one euclidean linkage tree, cut at k=2…8 → 7 partitions (not seven independent fits).
- **E03 DBSCAN**: m=5/10/20; eps = linear quantiles {0.70, 0.80, 0.90, 0.95} of the m-th nearest-neighbour distance **including the query point itself** → 12 settings.

Internal metrics (silhouette, Davies–Bouldin, Calinski–Harabasz) are computed on the same six-dimensional Z for K-Means/Ward over all 1333 days. DBSCAN metrics use only the non-noise subset (`n_scored` reported); noise (−1) is never treated as a cluster. Invalid metrics are stored as null with a reason. ARI measures partition agreement without labels and is invariant to cluster-number permutation.

Selection (frozen before looking at any interpretation): minimum cluster size max(10, ceil(0.01·n)) = 14 days; per-method silhouette shortlist within 0.02 of the eligible best; 20 temporal-block resamples (7-day blocks over the full 1,461-day calendar, 167 of 209 blocks per draw, seeds 100…119) with scaler and model refit on each subset; representative = highest median ARI, ties within 0.02 prefer smaller k. Median ARI < 0.70 flags weak stability (a project warning line, not a validity proof). Because k is chosen using these resamples, the stability figures carry exploratory selection bias and are not confidence intervals.

## 4. Exploratory analysis

F01 shows monthly valid-hour fractions and eligible-day fractions; several months fall below the 18/24 rule. F02 shows why log1p is used. F03: Spearman ρ(PM2.5, PM10)=0.91 and ρ(NO2, CO)=0.74 — correlated pollutants receive repeated weight in euclidean distance, a limitation we discuss rather than silently "fix". F04 displays daily series with excluded days left as gaps.

![F01](figures/F01_monthly_coverage.png)
![F02](figures/F02_distributions.png)
![F03](figures/F03_correlation.png)
![F04](figures/F04_time_series.png)

## 5. Candidate results (all 89 retained)

K-Means medians over 10 seeds:

| k | silhouette | davies_bouldin | calinski_harabasz |
| --- | --- | --- | --- |
| 2.000 | 0.361 | 1.047 | 1051.979 |
| 3.000 | 0.294 | 1.149 | 948.616 |
| 4.000 | 0.242 | 1.322 | 786.398 |
| 5.000 | 0.240 | 1.344 | 714.355 |
| 6.000 | 0.228 | 1.378 | 646.783 |
| 7.000 | 0.213 | 1.359 | 597.143 |
| 8.000 | 0.198 | 1.423 | 553.195 |

Seed agreement (45 pairwise ARIs per k):

| k | median_ari | q1_ari | q3_ari | min_ari |
| --- | --- | --- | --- | --- |
| 2.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 3.000 | 1.000 | 0.998 | 1.000 | 0.998 |
| 4.000 | 0.998 | 0.995 | 0.998 | 0.991 |
| 5.000 | 0.967 | 0.939 | 0.986 | 0.926 |
| 6.000 | 0.962 | 0.947 | 0.974 | 0.908 |
| 7.000 | 0.966 | 0.956 | 0.973 | 0.940 |
| 8.000 | 0.843 | 0.798 | 0.886 | 0.524 |

Ward cuts of the single tree:

| k | silhouette | davies_bouldin | calinski_harabasz | min_cluster_size |
| --- | --- | --- | --- | --- |
| 2.000 | 0.358 | 1.050 | 1031.880 | 510.000 |
| 3.000 | 0.246 | 1.223 | 764.265 | 206.000 |
| 4.000 | 0.198 | 1.351 | 670.456 | 206.000 |
| 5.000 | 0.190 | 1.546 | 597.890 | 206.000 |
| 6.000 | 0.187 | 1.544 | 529.126 | 105.000 |
| 7.000 | 0.171 | 1.442 | 485.765 | 100.000 |
| 8.000 | 0.164 | 1.403 | 456.460 | 75.000 |

DBSCAN outcomes (noise fraction and n_scored always shown):

| experiment_id | min_samples | eps | n_clusters | noise_fraction | n_scored | silhouette | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E03_m5_q70 | 5.000 | 0.736 | 3 | 0.142 | 1144 | 0.100 | ok |
| E03_m5_q80 | 5.000 | 0.801 | 1 | 0.088 | 1216 | null | ok |
| E03_m5_q90 | 5.000 | 0.920 | 1 | 0.043 | 1276 | null | ok |
| E03_m5_q95 | 5.000 | 1.023 | 1 | 0.017 | 1311 | null | ok |
| E03_m10_q70 | 10.000 | 0.873 | 1 | 0.091 | 1212 | null | ok |
| E03_m10_q80 | 10.000 | 0.939 | 1 | 0.059 | 1254 | null | ok |
| E03_m10_q90 | 10.000 | 1.064 | 1 | 0.025 | 1300 | null | ok |
| E03_m10_q95 | 10.000 | 1.203 | 1 | 0.010 | 1320 | null | ok |
| E03_m20_q70 | 20.000 | 1.039 | 1 | 0.065 | 1247 | null | ok |
| E03_m20_q80 | 20.000 | 1.114 | 1 | 0.037 | 1284 | null | ok |
| E03_m20_q90 | 20.000 | 1.265 | 1 | 0.014 | 1314 | null | ok |
| E03_m20_q95 | 20.000 | 1.426 | 1 | 0.004 | 1328 | null | ok |

11 of 12 DBSCAN settings produce a single non-noise cluster; only 1 setting(s) yield ≥2 clusters. This is evidence about the density assumption, not a failure to be hidden: the six-dimensional log-standardised days form a gradient with a dense core rather than well-separated density clusters.

![F05](figures/F05_kmeans_metrics.png)
![F06](figures/F06_dendrogram.png)
![F07](figures/F07_dbscan.png)

## 6. Representative selection and stability

Shortlists and stability (20 block resamples each; failed repetitions counted, never masked):

| algorithm | k | valid_repetitions | failed_repetitions | median_ari | q1_ari | q3_ari |
| --- | --- | --- | --- | --- | --- | --- |
| kmeans | 2 | 20 | 0 | 0.970 | 0.922 | 0.990 |
| ward | 2 | 20 | 0 | 0.707 | 0.554 | 0.799 |

Selection outcome: K-Means representative k=2 (median block ARI 0.970); Ward representative k=2 (median ARI 0.707). DBSCAN representative: none — no_setting_meets_two_clusters_min_size_coverage_and_valid_metrics. The full rule chain, exclusions and per-repetition ARIs are in `selection.json` and `stability/`.

## 7. What the clusters mean (original units)

Profiles are medians/IQRs of raw daily concentrations grouped by the actual labels; display numbers are ordered by PM2.5 median and mapped in `artifacts/display_mapping.json`.

### K-Means (k=2)

- **Display C1** (original label 0, n=812 days): PM2.5 median 39.0 µg/m³ (IQR 21.8–62.1), O3 median 66.0 µg/m³. Relatively elevated: CO.
- **Display C2** (original label 1, n=521 days): PM2.5 median 119.5 µg/m³ (IQR 84.6–171.6), O3 median 22.8 µg/m³. Relatively elevated: PM2.5, PM10, CO.

### Ward (k=2)

- **Display C1** (original label 0, n=823 days): PM2.5 median 39.8 µg/m³ (IQR 21.9–64.6), O3 median 66.6 µg/m³. Relatively elevated: CO.
- **Display C2** (original label 1, n=510 days): PM2.5 median 118.3 µg/m³ (IQR 82.7–172.5), O3 median 21.5 µg/m³. Relatively elevated: PM2.5, PM10, CO.

![F09 K-Means](figures/F09_profiles_kmeans.png)
![F09 Ward](figures/F09_profiles_ward.png)

## 8. Method agreement and disagreement

Representative partitions: ARI=0.892; contingency (rows K-Means, columns Ward):

| kmeans | 0 | 1 |
| --- | --- | --- |
| 0 | 799 | 13 |
| 1 | 24 | 497 |

Same-k ARI separates algorithm geometry from cluster-count effects:

| k | ari_kmeans_seed100_vs_ward |
| --- | --- |
| 2.000 | 0.892 |
| 3.000 | 0.391 |
| 4.000 | 0.634 |
| 5.000 | 0.589 |
| 6.000 | 0.522 |
| 7.000 | 0.416 |
| 8.000 | 0.359 |

The shared PCA projection (display only; PC1+PC2 explain 82.0% of variance) shows how the same dates are organised by each method:

![F08](figures/F08_pca_panels.png)
![F10](figures/F10_agreement.png)

## 9. Background context (after frozen selection)

Season composition within each K-Means cluster (denominator = cluster days):

| cluster | spring | summer | autumn | winter |
| --- | --- | --- | --- | --- |
| 0.000 | 0.234 | 0.373 | 0.224 | 0.169 |
| 1.000 | 0.286 | 0.063 | 0.290 | 0.361 |

Weather medians by cluster (descriptive; valid-day denominators in `weather_profiles.csv`):

| cluster | DEWP_mean | PRES_mean | RAIN_sum | TEMP_mean | WSPM_mean |
| --- | --- | --- | --- | --- | --- |
| 0.000 | 8.700 | 1007.959 | 0.000 | 20.502 | 1.725 |
| 1.000 | -0.938 | 1014.329 | 0.000 | 6.737 | 1.283 |

These associations are descriptive. Clusters are not pollution sources, not causal effects of weather, and not official air-quality health grades. No significance tests are run on the six clustering features themselves (that would circularly "validate" the discovery).

![F11](figures/F11_background.png)

## 10. Sensitivity and robustness

Raw+StandardScaler at the frozen k, compared with the main representation:

| algorithm | k | ari_vs_main |
| --- | --- | --- |
| kmeans | 2 | 0.392 |
| ward | 2 | 0.719 |

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

On the same 1333 days and the same transformed space, K-Means and Ward produce broadly consistent low-vs-high pollution organisation at their representative k (ARI=0.892) but differ in how intermediate days are split (same-k ARI as low as 0.359). DBSCAN's near-universal single-cluster outcome characterises the data as a continuous gradient with a dense core. The discovered day types are reproducible under temporal-block perturbation (median ARI ≥ 0.707) and moderately robust to the log transform, supporting their use as descriptive knowledge rather than predictive or causal claims.

## References

- UCI Machine Learning Repository (2017). Beijing Multi-Site Air-Quality Data Set, ID 501. https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data
- Zhang, S., Guo, B., Dong, A., He, J., Xu, Z., Chen, S.X. (2016). Cautionary tales on air-quality improvement in Beijing. *Proc. R. Soc. A* 472: 20160457. (Dataset companion paper as cited by UCI.)
- Course materials: AISC5030 Week 5 slides/notebooks (method definitions and Topic 3 brief).

## Appendix A. Reproducibility

Run `20261116T193443Z-75ddd759`; config snapshot and SHA256 of data, config and executed source in `manifest.json`; environment in `requirements-lock.txt`. Re-run: `python scripts/run_project.py --stage all`; notebooks 01–03 execute top-to-bottom via `scripts/run_notebooks.py`. All 89 candidate label files carry date keys; figures F01–F12 in `figures/`.
