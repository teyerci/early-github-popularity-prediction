# GitHub Popularity Prediction Pipeline

This repository contains the scripts needed to reproduce the data pipeline and baseline machine-learning results.

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

Run the scripts in this order.

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
export GH_ARCHIVE_END_DATE="2025-12-31"
export GH_DAILY_COUNTS_DIR="out"
python 03_download_gh_archive_daily_counts.py
```

Output pattern:

```text
out/gh_archive_counts_YYYY-MM-DD.csv.gz
```

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

