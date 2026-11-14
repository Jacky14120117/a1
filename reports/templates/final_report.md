# Discovering Daily Air Pollution Patterns in Beijing

Template status: planning only. Replace the result prompts with measured outputs from a recorded run. This file does not contain experimental conclusions.

## Abstract

State the environmental question, unsupervised methods, actual eligible sample count, key measured differences between clusterings, stability and the scope of the conclusions. Complete this last. Do not predict an expected number of pollution types.

## 1. Scientific question and Topic 3 scope

Explain why joint pollutant patterns are worth exploring and what clustering can establish. Identify the unit as an eligible day at Aotizhongxin. State that no supervised target or predefined pollution classes were used.

## 2. Data provenance and limitations

Cite Chen (2017), UCI DOI 10.24432/C5RK5G. Distinguish the official 12-site hourly dataset from the selected station and the daily analysis subset. Record dates, source version, license, the six input variables and their units. Discuss missingness, one-site scope and historical coverage.

Insert the measured data-flow table: raw hours -> calendar days -> eligible days -> analyzed rows. The preflight count is 1,333 eligible days; verify this against the actual run before reporting it as the final sample.

## 3. Preparation and exploratory analysis

Specify the per-pollutant >=18-hour rule, daily mean, invalid-value handling, no primary imputation or clipping, log1p and standardization. Include coverage, distribution and correlation figures. Explain that meteorological/calendar fields were excluded from clustering.

## 4. Methods and evaluation

Describe K-Means, Ward and supplementary DBSCAN assumptions. Give actual settings and the protocol version. Explain internal metrics, noise coverage, initialization stability, block subsampling and representative-setting selection. PCA is a visualization, not a clustering result.

## 5. Results

### 5.1 Candidate settings and representatives

Insert the real candidate summary and selected settings, including rejected or invalid settings. Give silhouette/DB/CH with the evaluation population and uncertainty summaries. Do not treat internal scores as ground truth.

### 5.2 Cluster profiles

Insert raw-unit medians, IQRs, counts and profile figures. Use data-supported descriptive cluster names. Keep the original label-to-display-name mapping.

### 5.3 Differences between methods

Insert representative ARI/contingency and matched-k comparison. Describe at least two measured differences in splitting, merging, membership, outliers or cluster profiles, with evidence.

### 5.4 Stability and sensitivity

Insert initialization ARI, all block-subsampling summaries and the raw-standardized sensitivity result. Disclose invalid replicates. Explain that these diagnostics were used exploratorily and are not independent future-data validation.

## 6. Knowledge discovery and discussion

For each substantive observation, state its numeric evidence, agreement across methods and limitation. Add seasonal proportions with clear denominators and meteorological summaries with available n. Discuss missingness, correlated features, scale dependence and possible continuous rather than discrete structure.

## 7. Conclusions and limitations

Answer the scientific questions using measured results. No pollution-source, health-standard or causal claims follow from clustering alone. State what additional data or independent validation would be needed. Future work may extend to another fixed station or independent period.

## 8. Reproducibility

Give the actual run_id, code/data/config hashes, Python/dependency versions, commands and result-file mapping. Report what was rerun successfully.

## References

Use verified primary sources from `references/sources.md`, including the dataset, original scientific context and method documentation. Do not cite the course's regression case as evidence for this project's cluster results.
