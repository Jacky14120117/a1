"""Build notebooks 01/02 with nbformat (valid structure guaranteed).

The notebooks only call src modules and read authoritative run outputs;
they contain no duplicated research logic. Re-run this script to regenerate
clean (output-free) sources.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

LOCATOR = '''import json
import sys
from pathlib import Path

import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT / 'src'))

def latest_run(stage):
    runs = []
    for d in (ROOT / 'results' / 'runs').iterdir():
        mp = d / 'manifest.json'
        if not mp.is_file():
            continue
        m = json.loads(mp.read_text(encoding='utf-8'))
        done = m.get('stages_completed') or m.get('stage_summary', {}).get('stages_completed', [])
        if stage in done and m.get('status') in ('completed', 'completed_with_warnings'):
            runs.append((d.name, d, m))
    if not runs:
        raise RuntimeError(f'no completed run contains stage {stage}; run scripts/run_project.py first')
    return max(runs, key=lambda t: t[0])
'''


def build_01() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell(
            "# 01 · Data audit & exploratory analysis (T04)\n\n"
            "AISC5030 Topic 3 — daily air pollution patterns at the Aotizhongxin station.\n\n"
            "This notebook **calls the research modules in `src/air_quality_clustering`** and reads "
            "authoritative run outputs; it does not re-implement cleaning or plotting logic. "
            "Execute top-to-bottom after `--stage prepare eda` has produced a run.\n\n"
            "Protocol references: `docs/05_data_dictionary.md`, `docs/06_experiment_protocol.md`."
        ),
        nbf.v4.new_code_cell(
            LOCATOR + "\n"
            "from air_quality_clustering.config import load_config\n"
            "cfg = load_config(ROOT / 'configs' / 'project.yaml')\n"
            "run_name, RUN, manifest = latest_run('eda')\n"
            "print('run_id:', run_name)\n"
            "print('data version (CSV sha256):', manifest['input_sha256'].get('data/raw/PRSA_Data_Aotizhongxin_20130301-20170228.csv'))\n"
            "print('stages completed:', manifest.get('stages_completed') or manifest.get('stage_summary', {}).get('stages_completed', []))"
        ),
        nbf.v4.new_markdown_cell(
            "## 1. Source and cleaning funnel\n\n"
            "Raw file is read-only; the audit below comes from the actual prepare run."
        ),
        nbf.v4.new_code_cell(
            "summary = json.loads((RUN / 'tables' / 'prepare_summary.json').read_text(encoding='utf-8'))\n"
            "audit = summary['hourly_audit']\n"
            "print('hourly rows read/written:', audit['rows_read'], audit['rows_written'])\n"
            "print('missing hours per pollutant:', audit['missing_counts'])\n"
            "print('negative concentrations (invalidated):', audit['negative_counts'])\n"
            "print('acceptance checks:', summary['acceptance_checks'])\n"
            "print('eligible days by year:', summary['eligible_days_by_year'])"
        ),
        nbf.v4.new_code_cell(
            "coverage = pd.read_csv(ROOT / 'data' / 'processed' / 'daily_coverage.csv')\n"
            "daily = pd.read_csv(ROOT / 'data' / 'processed' / 'daily_pollutants.csv')\n"
            "excluded = pd.read_csv(ROOT / 'data' / 'processed' / 'excluded_days.csv')\n"
            "print('calendar days:', len(coverage), '| eligible:', int(coverage.eligible.sum()), '| excluded:', len(excluded))\n"
            "excluded.head(10)"
        ),
        nbf.v4.new_markdown_cell(
            "## 2. F01 — monthly coverage\n\n"
            "Question: is missingness concentrated in time? The dashed line is the 18/24-hour "
            "quality rule (a project rule, not a regulatory standard)."
        ),
        nbf.v4.new_code_cell("display(Image(filename=str(RUN / 'figures' / 'F01_monthly_coverage.png')))"),
        nbf.v4.new_markdown_cell(
            "## 3. F02 — raw vs log1p distributions\n\n"
            "Question: why transform? Daily concentrations are right-skewed and span orders of "
            "magnitude (CO included, in µg/m³); log1p compresses the tail before standardization."
        ),
        nbf.v4.new_code_cell(
            "display(Image(filename=str(RUN / 'figures' / 'F02_distributions.png')))\n"
            "daily[['PM2.5','PM10','SO2','NO2','CO','O3']].describe(percentiles=[.05,.25,.5,.75,.95]).T.round(2)"
        ),
        nbf.v4.new_markdown_cell(
            "## 4. F03 — Spearman correlation\n\n"
            "Question: do correlated pollutants double-count information in Euclidean distance? "
            "PM2.5/PM10 and NO2/CO correlations are discussed in the report as a limitation of "
            "the distance, not corrected silently."
        ),
        nbf.v4.new_code_cell(
            "display(Image(filename=str(RUN / 'figures' / 'F03_correlation.png')))\n"
            "daily[['PM2.5','PM10','SO2','NO2','CO','O3']].corr(method='spearman').round(3)"
        ),
        nbf.v4.new_markdown_cell(
            "## 5. F04 — daily time series with gaps\n\n"
            "Excluded days remain gaps; no interpolation connects them."
        ),
        nbf.v4.new_code_cell("display(Image(filename=str(RUN / 'figures' / 'F04_time_series.png')))"),
        nbf.v4.new_markdown_cell(
            "## 6. Observations to test with clustering (not conclusions)\n\n"
            "- Do high-particulate days and high-O3 days form different combinations?\n"
            "- Are winter/summer days separable by concentrations alone (season is NOT an input)?\n"
            "- Is the structure cluster-like or a continuous gradient? DBSCAN will probe this.\n\n"
            "No cluster names or k are fixed here; selection follows the frozen protocol in T08."
        ),
    ]
    nb.cells = cells
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return nb


def build_02() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell(
            "# 02 · Clustering candidate experiments (T05–T07)\n\n"
            "All 89 base candidates on the same 1,333 eligible days and the same six-dimensional "
            "log1p+StandardScaler space:\n\n"
            "- **E01 K-Means**: k=2..8 × seeds 100..109 → 70 runs\n"
            "- **E02 Ward**: ONE euclidean linkage tree cut at k=2..8 → 7 cuts\n"
            "- **E03 DBSCAN**: m=5/10/20 × eps quantiles 0.70/0.80/0.90/0.95 of the self-inclusive "
            "m-th NN distance → 12 settings\n\n"
            "This notebook reads the authoritative run outputs; core logic lives in "
            "`src/air_quality_clustering`. **No representative k is chosen here** — selection "
            "with stability is T08."
        ),
        nbf.v4.new_code_cell(
            LOCATOR + "\n"
            "run_name, RUN, manifest = latest_run('cluster')\n"
            "print('run_id:', run_name)\n"
            "experiments = pd.read_csv(RUN / 'experiments.csv')\n"
            "print('candidate rows:', len(experiments))\n"
            "experiments.groupby(['algorithm', 'status']).size()"
        ),
        nbf.v4.new_markdown_cell(
            "## 1. K-Means (E01): metrics vs k and seed stability\n\n"
            "Inertia decreases with k by construction — it is an elbow diagnostic, not a selection "
            "score. Silhouette/DB medians summarize 10 seeds; pairwise ARI (45 pairs per k) measures "
            "initialization sensitivity only."
        ),
        nbf.v4.new_code_cell(
            "km = experiments[experiments.algorithm == 'kmeans']\n"
            "km.groupby('k')[['silhouette', 'davies_bouldin', 'calinski_harabasz']].median().round(4)"
        ),
        nbf.v4.new_code_cell(
            "display(Image(filename=str(RUN / 'figures' / 'F05_kmeans_metrics.png')))\n"
            "pd.read_csv(RUN / 'tables' / 'kmeans_seed_ari.csv').round(4)"
        ),
        nbf.v4.new_markdown_cell(
            "## 2. Ward (E02): one tree, seven cuts\n\n"
            "The dendrogram below is drawn from the SAME linkage matrix that produced the labels "
            "(`artifacts/ward_linkage.npy`). Truncation is display-only."
        ),
        nbf.v4.new_code_cell(
            "experiments[experiments.algorithm == 'ward'][['k', 'silhouette', 'davies_bouldin', 'calinski_harabasz', 'min_cluster_size']].round(4)"
        ),
        nbf.v4.new_code_cell("display(Image(filename=str(RUN / 'figures' / 'F06_dendrogram.png')))"),
        nbf.v4.new_markdown_cell(
            "## 3. DBSCAN (E03): density assumption probe\n\n"
            "Noise (-1) is never a pollution cluster. Internal metrics are computed on the non-noise "
            "subset only (`n_scored` differs from 1,333) and must not be ranked against full-sample "
            "K-Means/Ward scores. Single-cluster outcomes keep null metrics with a reason."
        ),
        nbf.v4.new_code_cell(
            "db = experiments[experiments.algorithm == 'dbscan']\n"
            "db[['experiment_id', 'min_samples', 'eps', 'n_clusters', 'noise_fraction', 'n_scored', 'silhouette', 'metric_reason', 'status']].round(4)"
        ),
        nbf.v4.new_code_cell("display(Image(filename=str(RUN / 'figures' / 'F07_dbscan.png')))"),
        nbf.v4.new_markdown_cell(
            "## 4. Same-k agreement: algorithm difference vs cluster-count difference\n\n"
            "ARI between K-Means(seed=100) and Ward at equal k. Low values at some k mean the two "
            "geometries organize the same days differently — a finding to interpret in T09, not an error."
        ),
        nbf.v4.new_code_cell("pd.read_csv(RUN / 'tables' / 'same_k_ari.csv').round(4)"),
        nbf.v4.new_markdown_cell(
            "## 5. What is NOT decided here\n\n"
            "- No representative k (T08: min-size rule → silhouette shortlist → 20 block-resample ARI).\n"
            "- No cluster names, no season/weather interpretation (only after the selection is frozen).\n"
            "- DBSCAN degenerate settings are retained as evidence about the density assumption."
        ),
    ]
    nb.cells = cells
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return nb


def build_03() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell(
            "# 03 · Interpretation, stability & sensitivity (T08–T10)\n\n"
            "Reads the authoritative run outputs after `--stage stability compare sensitivity`. "
            "Selection was frozen BEFORE any season/weather view; this notebook only interprets. "
            "Core logic lives in `src/air_quality_clustering/stability.py` and `discovery.py`."
        ),
        nbf.v4.new_code_cell(
            LOCATOR + "\n"
            "run_name, RUN, manifest = latest_run('sensitivity')\n"
            "print('run_id:', run_name)\n"
            "selection = json.loads((RUN / 'selection.json').read_text(encoding='utf-8'))\n"
            "for a in ('kmeans', 'ward'):\n"
            "    r = selection['methods'][a]['representative']\n"
            "    print(a, 'representative:', r)\n"
            "print('dbscan:', selection['dbscan']['representative'], selection['dbscan']['reason'])"
        ),
        nbf.v4.new_markdown_cell(
            "## 1. Temporal-block stability (20 refit resamples per shortlisted k)\n\n"
            "Failed repetitions are counted, not masked. These ARIs carry exploratory selection "
            "bias because k was chosen with them; they are not confidence intervals."
        ),
        nbf.v4.new_code_cell(
            "pd.read_csv(RUN / 'stability' / 'aggregate.csv').round(4)"
        ),
        nbf.v4.new_code_cell("display(Image(filename=str(RUN / 'figures' / 'F12_stability_sensitivity.png')))"),
        nbf.v4.new_markdown_cell(
            "## 2. Cluster profiles in ORIGINAL units\n\n"
            "Medians/IQRs of raw daily concentrations by actual label; display order by PM2.5 "
            "median (mapping in artifacts/display_mapping.json). No back-transformed log centroids."
        ),
        nbf.v4.new_code_cell(
            "profiles = pd.read_csv(RUN / 'tables' / 'cluster_profiles.csv')\n"
            "profiles[profiles.algorithm == 'kmeans'].pivot(index='cluster', columns='pollutant', values='median').round(1)"
        ),
        nbf.v4.new_code_cell("display(Image(filename=str(RUN / 'figures' / 'F09_profiles_kmeans.png')))"),
        nbf.v4.new_code_cell("display(Image(filename=str(RUN / 'figures' / 'F09_profiles_ward.png')))"),
        nbf.v4.new_markdown_cell(
            "### 2.1 Within-cluster spread first (protocol §8)\n\n"
            "IQR/median is scale-free, so heterogeneity is comparable across pollutants. "
            "A cluster is only a tidy 'type' if its own spread is small relative to its level."
        ),
        nbf.v4.new_code_cell(
            "spread = pd.read_csv(RUN / 'tables' / 'cluster_spread.csv')\n"
            "spread[spread.algorithm == 'kmeans'][['cluster','pollutant','n_days','median','iqr','iqr_over_median']].round(3)"
        ),
        nbf.v4.new_markdown_cell(
            "### 2.2 Which pollutants separate the clusters, which overlap\n\n"
            "Shared IQR span in original units and as a fraction of the pooled IQR span."
        ),
        nbf.v4.new_code_cell(
            "ov = pd.read_csv(RUN / 'tables' / 'cluster_overlap.csv')\n"
            "ov[ov.algorithm == 'kmeans'][['pollutant','iqr_overlap_width','overlap_fraction_of_pooled_iqr','median_ratio_b_over_a']].round(3)"
        ),
        nbf.v4.new_code_cell("display(Image(filename=str(RUN / 'figures' / 'F13_boxplots_kmeans.png')))"),
        nbf.v4.new_code_cell("display(Image(filename=str(RUN / 'figures' / 'F13_boxplots_ward.png')))"),
        nbf.v4.new_markdown_cell(
            "## 3. Method agreement\n\n"
            "Contingency + representative ARI; same-k ARI separates geometry from cluster count."
        ),
        nbf.v4.new_code_cell(
            "print(pd.read_csv(RUN / 'tables' / 'contingency.csv', index_col=0))\n"
            "pd.read_csv(latest_run('cluster')[1] / 'tables' / 'same_k_ari.csv').round(4)"
        ),
        nbf.v4.new_code_cell("display(Image(filename=str(RUN / 'figures' / 'F08_pca_panels.png')))"),
        nbf.v4.new_code_cell("display(Image(filename=str(RUN / 'figures' / 'F10_agreement.png')))"),
        nbf.v4.new_markdown_cell(
            "### 3.1 Largest split/merge cell, fixed rule\n\n"
            "Largest off-diagonal cell under optimal matching (cluster numbers are arbitrary), "
            "then the earliest five dates — never picked by visual impression."
        ),
        nbf.v4.new_code_cell(
            "case = json.loads((RUN / 'tables' / 'disagreement_case.json').read_text(encoding='utf-8'))\n"
            "print('cell:', case['cell']); print('counts:', case['counts'])\n"
            "pd.DataFrame(case['examples']).round(1)"
        ),
        nbf.v4.new_markdown_cell(
            "## 4. Background AFTER freezing (descriptive only)\n\n"
            "Every proportion states its denominator. No causal claims; clusters are not sources "
            "or health grades."
        ),
        nbf.v4.new_code_cell(
            "print('Eligible-day coverage by season (denominator for all seasonal percentages):')\n"
            "print(pd.read_csv(RUN / 'tables' / 'season_coverage.csv').round(3))\n"
            "pd.read_csv(RUN / 'tables' / 'season_pct_of_cluster_kmeans.csv', index_col=0).round(3)"
        ),
        nbf.v4.new_code_cell("display(Image(filename=str(RUN / 'figures' / 'F11_background.png')))"),
        nbf.v4.new_markdown_cell(
            "## 5. Transformation sensitivity\n\n"
            "raw+StandardScaler at frozen k vs the main log1p representation."
        ),
        nbf.v4.new_code_cell(
            "print(pd.read_csv(RUN / 'sensitivity' / 'representation_ari.csv').round(4))\n"
            "shift = pd.read_csv(RUN / 'sensitivity' / 'profile_shift.csv')\n"
            "shift[shift.algorithm == 'kmeans'][['cluster','pollutant','median_main','median_raw','median_ratio_raw_over_main']].round(2)"
        ),
        nbf.v4.new_markdown_cell(
            "## 6. Report\n\n"
            "The full English report generated from this run's numbers: `report.md` in the run "
            "directory (stage export)."
        ),
    ]
    nb.cells = cells
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return nb


def main() -> None:
    target_dir = ROOT / "notebooks"
    target_dir.mkdir(parents=True, exist_ok=True)
    for name, builder in (
        ("01_data_audit_and_eda.ipynb", build_01),
        ("02_clustering_experiments.ipynb", build_02),
        ("03_interpretation_and_robustness.ipynb", build_03),
    ):
        nb = builder()
        nbf.validate(nb)
        path = target_dir / name
        nbf.write(nb, path)
        print(f"built {path}")


if __name__ == "__main__":
    main()
