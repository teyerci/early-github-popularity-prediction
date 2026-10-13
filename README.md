
[![DOI](https://zenodo.org/badge/1280089833.svg)](https://doi.org/10.5281/zenodo.20856257)

# GitHub Popularity Prediction Pipeline

This repository contains the scripts needed to reproduce the data pipeline and baseline machine-learning results.

## Reproducibility Settings

The reported pipeline uses the following settings.

- Repository cohort: GitHub repositories created in 2024, sampled by `01_gh_repo_list_create.py` as the top-starred repositories for each creation day (55 per day for each of the 366 days, 20,130 in total), ranked by star count when the list was collected. Inclusion therefore depends on later popularity: the cohort covers repositories that became relatively popular, not all repositories created in 2024 (the smallest `stars_now` in the committed list is 56). Re-running step 01 today returns a different list; use the committed `repos_2024_20k.csv.gz` to reproduce the reported cohort.
- Cohort attrition: of the 20,130 listed repositories, 19,611 have snapshot features (the rest were skipped as too large, unavailable or failed to clone), and step 13 then drops the 277 whose 1-year window is not fully covered by GH Archive data, leaving 19,334 for modelling.
- Effective-start rule: `effective_start_date = max(git_actual_start_date, github_created_at)`.
- `git_actual_start_date` is derived from the cloned git history by scanning commits oldest-first. The first qualifying start point is the earliest day with at least 3 commits, or the earliest commit that touches at least 3 files. If neither condition is found, the script falls back to the first commit in the repository history.
- Snapshot windows: `1m`, `3m`, and `6m`, computed as 30, 90, and 180 days after the effective start date.
- GH Archive feature windows: `1m`, `3m`, and `6m`.
- Target window: `1year`, computed as 365 days after the effective start date.
- Large-repository filter: skip repositories above `MAX_REPO_SIZE_MB=500` MB.
- API size safety margin: `API_SIZE_SAFETY_MARGIN_MB=50`, so the default effective API skip threshold is 450 MB before cloning.
- Local clone observation limit: `LOCAL_CLONE_LIMIT_MB=500` MB; already-cloned or unexpectedly large repositories are processed when possible and then removed.
- Train/test split: 80/20 split with `random_state=42`.
- Target stratification: `pd.qcut` on `log_WatchEvent_1year` with 4 bins; the same split is used for every configuration.
- Main target (`TARGET_MODE=future`, the default): `log_WatchEvent_future_{window}`, the log1p of WatchEvents in `[start + window, start + 365 days)`. Each window's Static, Snapshot, GH Archive and Combined models predict only the stars that arrive after their feature cutoff, so the target does not overlap the features.
- Cumulative target (`TARGET_MODE=cumulative`): `log_WatchEvent_1year`, the log1p of WatchEvents in `[start, start + 365 days)`. This target includes the 1m/3m/6m feature windows and is kept only for comparison with earlier results.
- Static features: `is_fork` and `git_start_minus_created_days`. The `description_length`, `topic_count` and `has_license` columns come from the step 01 API response, which was collected after the feature and target windows, so step 13 excludes them by default. Set `INCLUDE_COLLECTION_TIME_METADATA=1` to include them.
- GH Archive date range: 2024-01-01 to 2026-03-14 (the daily files used for the committed outputs).
- Coverage filter: step 11 writes `window_complete_{window}` flags, and step 13 drops repositories whose 365-day window is not fully covered by the daily GH Archive files (late effective starts, or missing days), instead of counting the uncovered days as zero events.
- Baseline model depths: Random Forest `max_depth` values of 6, 8, and 12; the selected main model uses Combined 3m features with `max_depth=8` (test R² 0.402 on `log_WatchEvent_future_3m`).

Dependency versions used for data collection (steps 01–10):

```text
Python 3.13.13
pandas 3.0.2
numpy 2.4.4
scikit-learn 1.8.0
requests 2.33.1
tqdm 4.68.3
```

Dependency versions used to regenerate the committed step 11–13 outputs (future-growth target), pinned in `requirements.txt`; these install on Python 3.10–3.13:

```text
Python 3.10.12
pandas 2.3.3
numpy 2.2.6
scikit-learn 1.7.2
tqdm 4.70.1
```

## Setup

```bash
python3 -m venv .venv3.13
source .venv3.13/bin/activate
pip install -r requirements.txt
```

Set a GitHub API token for scripts that call the GitHub API:

```bash
export GITHUB_TOKEN="your_token_here"
```

## Pipeline Scripts

Run the scripts in this order. The numbering follows the original working project; scripts 02 and 04–07 were exploratory steps (feature fetching, earlier event-count, EDA and modelling scripts) that the final pipeline does not use, so they are not included.

### Reproducing from the committed data

The committed `.csv.gz` / `.json.gz` files are the frozen inputs and outputs of the reported run. pandas reads gzip files directly, so pass the `.csv.gz` names in the environment variables below instead of decompressing them. Steps 01, 08 and 03 query live services and will not reproduce the same cohort or snapshots; to reproduce the models, start from the committed files:

- Step 12 → 13 need only the committed files: `repos_2024_20k.csv.gz`, `repos_snapshot_features_20k_effective_start_1m3m6m.csv.gz` and `repos_2024_20k_gh_archive_windows_effective_start_1m3m6m.csv.gz`.
- Step 11 additionally needs the `out/` daily count files (about 14 GB, not committed), which step 03 rebuilds from GH Archive.

### 1. Create Repository List

```bash
python 01_gh_repo_list_create.py
```

Output:

```text
repos_2024_20k.csv
```

### 2. Clone Repositories and Extract Snapshot Features

```bash
export REPOS_CSV="repos_2024_20k.csv"
export SNAPSHOT_WINDOWS="1m,3m,6m"
export SNAPSHOTS_OUTPUT_CSV="repos_snapshots_20k_effective_start_1m3m6m.csv"
export SNAPSHOT_FEATURES_OUTPUT_CSV="repos_snapshot_features_20k_effective_start_1m3m6m.csv"
python 08_clone_repos.py
```

Outputs:

```text
repos_snapshots_20k_effective_start_1m3m6m.csv
repos_snapshot_features_20k_effective_start_1m3m6m.csv
clone_processed_20k.json
clone_skip_20k.json
```

`08_clone_repos.py` calls `09_analyze_snapshots.py` and `10_extract_snapshot_features.py`.

### 3. Download and Count GH Archive Events

```bash
export REPOS_CSV="repos_2024_20k.csv"
export GH_ARCHIVE_START_DATE="2024-01-01"
export GH_ARCHIVE_END_DATE="2026-03-14"
export GH_DAILY_COUNTS_DIR="out"
python 03_download_gh_archive_daily_counts.py
```

Output pattern:

```text
out/gh_archive_counts_YYYY-MM-DD.csv.gz
out/gh_archive_missing_hours.csv
```

A daily file is written only when every hour was downloaded or reported missing by GH Archive (HTTP 404); hours that GH Archive does not have are listed in `gh_archive_missing_hours.csv`. If an hour still fails after retries, the day is not written and the script exits with an error, so rerunning it retries only the missing days.

### 4. Aggregate GH Archive Windows

```bash
export REPOS_CSV="repos_2024_20k.csv"
export SNAPSHOT_CSV="repos_snapshots_20k_effective_start_1m3m6m.csv"
export SNAPSHOT_FEATURES_CSV="repos_snapshot_features_20k_effective_start_1m3m6m.csv"
export GH_FEATURE_WINDOWS="1m,3m,6m"
export DAILY_COUNTS_GLOB="out/gh_archive_counts_*.csv.gz"
export GH_ARCHIVE_WINDOWS_OUTPUT_CSV="repos_2024_20k_gh_archive_windows_effective_start_1m3m6m.csv"
python 11_gh_archive_window_counts_sample.py
```

Output:

```text
repos_2024_20k_gh_archive_windows_effective_start_1m3m6m.csv
```

### 5. Combine Features

```bash
export BASE_CSV="repos_2024_20k.csv"
export SNAPSHOT_FEATURES_CSV="repos_snapshot_features_20k_effective_start_1m3m6m.csv"
export GH_WINDOWS_CSV="repos_2024_20k_gh_archive_windows_effective_start_1m3m6m.csv"
export COMBINED_FEATURES_OUTPUT_CSV="repos_2024_20k_combined_features_effective_start_1m3m6m.csv"
python 12_combine_snapshot_and_gh_features.py
```

Output:

```text
repos_2024_20k_combined_features_effective_start_1m3m6m.csv
```

### 6. Train Baseline Models

```bash
export COMBINED_FEATURES_CSV="repos_2024_20k_combined_features_effective_start_1m3m6m.csv"
export MODEL_RESULTS_CSV="repos_2024_20k_combined_model_results_effective_start_1m3m6m.csv"
export RF_DEPTH_RESULTS_CSV="repos_2024_20k_rf_depth_comparison_effective_start_1m3m6m.csv"
export TARGET_MODE="future"  # or "cumulative" for the overlapping 1-year target
python 13_gh_repo_ml_combined_features.py
```

Outputs:

```text
repos_2024_20k_combined_model_results_effective_start_1m3m6m.csv
repos_2024_20k_rf_depth_comparison_effective_start_1m3m6m.csv
```

## Notes

- `GITHUB_TOKEN` is read from the environment and should not be committed.
- `out/` contains derived GH Archive daily count files and is ignored by git.
- `cloned_repos/` is temporary and is ignored by git.
- The scripts use `effective_start_date = max(git_actual_start_date, github_created_at)` for snapshot and GH Archive windows.
