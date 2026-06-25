import glob
import os
import re
from collections import defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm

SAMPLE_CSV = os.getenv("REPOS_CSV", "repos_sample_980.csv")
SNAPSHOT_CSV = os.getenv("SNAPSHOT_CSV", "repos_snapshots.csv")
SNAPSHOT_FEATURES_CSV = os.getenv("SNAPSHOT_FEATURES_CSV", "repos_snapshot_features.csv")
DAILY_COUNTS_GLOB = os.getenv("DAILY_COUNTS_GLOB", "out/gh_archive_counts_*.csv.gz")
OUTPUT_CSV = os.getenv("GH_ARCHIVE_WINDOWS_OUTPUT_CSV", "repos_sample_980_gh_archive_windows.csv")

AVAILABLE_WINDOWS = {
    "1m": 30,
    "3m": 90,
    "6m": 180,
}

selected_window_labels = [
    w.strip()
    for w in os.getenv("GH_FEATURE_WINDOWS", "1m,3m,6m").split(",")
    if w.strip()
]
unknown_windows = sorted(set(selected_window_labels) - set(AVAILABLE_WINDOWS))
if unknown_windows:
    raise ValueError(f"Unknown GH feature windows: {unknown_windows}. Valid values: {sorted(AVAILABLE_WINDOWS)}")

WINDOWS = {label: AVAILABLE_WINDOWS[label] for label in selected_window_labels}

TARGET_WINDOWS = {
    "1year": 365,
}

GROWTH_EVENTS = [
    "WatchEvent",
    "ForkEvent",
    "PushEvent",
    "IssuesEvent",
    "PullRequestEvent",
]

GROWTH_PAIRS = []
if "1m" in WINDOWS and "3m" in WINDOWS:
    GROWTH_PAIRS.append(("1m", "3m"))
if "3m" in WINDOWS and "6m" in WINDOWS:
    GROWTH_PAIRS.append(("3m", "6m"))


def load_start_dates():
    if os.path.exists(SNAPSHOT_CSV):
        df = pd.read_csv(SNAPSHOT_CSV, usecols=["repo", "start_date"])
    elif os.path.exists(SNAPSHOT_FEATURES_CSV):
        df = pd.read_csv(SNAPSHOT_FEATURES_CSV, usecols=["repo", "start_date"])
    else:
        raise FileNotFoundError(
            f"Expected {SNAPSHOT_CSV} or {SNAPSHOT_FEATURES_CSV} with repo/start_date columns."
        )

    df = df.rename(columns={"repo": "name"})
    df["start_date"] = pd.to_datetime(df["start_date"], utc=True)
    return df


def load_repo_windows():
    sample = pd.read_csv(SAMPLE_CSV, usecols=["repo_id", "name"])
    starts = load_start_dates()

    repos = sample.merge(starts, on="name", how="inner")
    if repos.empty:
        raise ValueError("No sampled repositories matched the snapshot start-date file.")

    # GH Archive inputs here are daily aggregate files, so windows are date-granular.
    repos["start_day"] = repos["start_date"].dt.floor("D")
    for label, days in {**WINDOWS, **TARGET_WINDOWS}.items():
        repos[f"end_day_{label}"] = (repos["start_date"] + pd.Timedelta(days=days)).dt.floor("D")

    return repos


def parse_daily_file_date(path):
    match = re.search(r"gh_archive_counts_(\d{4}-\d{2}-\d{2})\.csv\.gz$", os.path.basename(path))
    if not match:
        return None
    return pd.to_datetime(match.group(1), utc=True)


def event_columns(df):
    return [c for c in df.columns if c not in {"repo_id", "name", "archive_day"}]


def add_timing_and_burst_features(output, window_df, all_event_cols, label):
    """
    Add date-aware GH Archive features for a single window.

    Daily count files let us capture whether attention was early, late, bursty,
    or absent. Missing first-watch values are encoded as -1 so the ML script can
    keep using numeric features without a separate imputation rule.
    """
    repo_ids = output[["repo_id", "start_date"]].copy()
    window_days = WINDOWS[label]

    if window_df.empty:
        output[f"gh_has_any_events_{label}"] = 0
        output[f"gh_has_WatchEvent_{label}"] = 0
        output[f"gh_any_event_active_days_{label}"] = 0
        output[f"gh_any_event_total_{label}"] = 0
        output[f"gh_WatchEvent_active_days_{label}"] = 0
        output[f"gh_WatchEvent_max_daily_{label}"] = 0
        output[f"gh_WatchEvent_days_to_first_{label}"] = -1
        output[f"gh_WatchEvent_started_late_{label}"] = 0
        return output

    daily = window_df[["repo_id", "archive_day"] + all_event_cols].copy()
    daily[all_event_cols] = daily[all_event_cols].fillna(0)
    daily["any_event_total_day"] = daily[all_event_cols].sum(axis=1)
    daily["has_any_event_day"] = daily["any_event_total_day"] > 0

    any_summary = daily.groupby("repo_id").agg(
        gh_any_event_active_days=("has_any_event_day", "sum"),
        gh_any_event_total=("any_event_total_day", "sum"),
    ).reset_index()

    if "WatchEvent" in daily.columns:
        daily["has_watch_day"] = daily["WatchEvent"] > 0
        watch_daily = daily[daily["has_watch_day"]].copy()
        watch_summary = daily.groupby("repo_id").agg(
            gh_WatchEvent_active_days=("has_watch_day", "sum"),
            gh_WatchEvent_max_daily=("WatchEvent", "max"),
        ).reset_index()

        if watch_daily.empty:
            first_watch = pd.DataFrame({"repo_id": repo_ids["repo_id"]})
            first_watch["first_watch_day"] = pd.NaT
        else:
            first_watch = (
                watch_daily.groupby("repo_id")["archive_day"]
                .min()
                .reset_index(name="first_watch_day")
            )
    else:
        watch_summary = pd.DataFrame({"repo_id": repo_ids["repo_id"]})
        watch_summary["gh_WatchEvent_active_days"] = 0
        watch_summary["gh_WatchEvent_max_daily"] = 0
        first_watch = pd.DataFrame({"repo_id": repo_ids["repo_id"]})
        first_watch["first_watch_day"] = pd.NaT

    features = repo_ids.merge(any_summary, on="repo_id", how="left")
    features = features.merge(watch_summary, on="repo_id", how="left")
    features = features.merge(first_watch, on="repo_id", how="left")

    active_days_col = f"gh_any_event_active_days_{label}"
    total_col = f"gh_any_event_total_{label}"
    watch_active_days_col = f"gh_WatchEvent_active_days_{label}"
    watch_max_daily_col = f"gh_WatchEvent_max_daily_{label}"
    days_to_first_col = f"gh_WatchEvent_days_to_first_{label}"
    started_late_col = f"gh_WatchEvent_started_late_{label}"

    features = features.rename(
        columns={
            "gh_any_event_active_days": active_days_col,
            "gh_any_event_total": total_col,
            "gh_WatchEvent_active_days": watch_active_days_col,
            "gh_WatchEvent_max_daily": watch_max_daily_col,
        }
    )

    features[days_to_first_col] = (
        features["first_watch_day"] - features["start_date"].dt.floor("D")
    ).dt.days
    features[days_to_first_col] = features[days_to_first_col].fillna(-1).astype(int)
    features[started_late_col] = (
        features[days_to_first_col] >= int(window_days * 2 / 3)
    ).astype(int)
    features.loc[features[days_to_first_col] < 0, started_late_col] = 0

    output = output.merge(
        features[
            [
                "repo_id",
                active_days_col,
                total_col,
                watch_active_days_col,
                watch_max_daily_col,
                days_to_first_col,
                started_late_col,
            ]
        ],
        on="repo_id",
        how="left",
    )

    for col in [
        active_days_col,
        total_col,
        watch_active_days_col,
        watch_max_daily_col,
        days_to_first_col,
        started_late_col,
    ]:
        output[col] = output[col].fillna(-1 if col == days_to_first_col else 0)

    output[f"gh_has_any_events_{label}"] = (output[total_col] > 0).astype(int)
    output[f"gh_has_WatchEvent_{label}"] = (
        output[f"gh_WatchEvent_{label}"] > 0
        if f"gh_WatchEvent_{label}" in output.columns
        else 0
    )
    output[f"gh_has_WatchEvent_{label}"] = output[f"gh_has_WatchEvent_{label}"].astype(int)

    return output


def add_growth_features(output):
    for from_label, to_label in GROWTH_PAIRS:
        if from_label not in WINDOWS or to_label not in WINDOWS:
            continue

        from_days = WINDOWS[from_label]
        to_days = WINDOWS[to_label]
        incremental_days = max(to_days - from_days, 1)

        for event in GROWTH_EVENTS:
            from_col = f"gh_{event}_{from_label}"
            to_col = f"gh_{event}_{to_label}"
            if from_col not in output.columns or to_col not in output.columns:
                continue

            growth_col = f"gh_{event}_growth_{from_label}_to_{to_label}"
            ratio_col = f"gh_{event}_ratio_{from_label}_to_{to_label}"
            incremental_daily_col = f"gh_{event}_incremental_daily_{from_label}_to_{to_label}"

            output[growth_col] = output[to_col] - output[from_col]
            output[ratio_col] = output[to_col] / output[from_col].clip(lower=1)
            output[incremental_daily_col] = output[growth_col] / incremental_days

    for event in GROWTH_EVENTS:
        col_1m = f"gh_{event}_1m"
        col_3m = f"gh_{event}_3m"
        col_6m = f"gh_{event}_6m"
        if all(col in output.columns for col in [col_1m, col_3m, col_6m]):
            output[f"gh_{event}_acceleration_1m_3m_6m"] = (
                (output[col_6m] - output[col_3m]) -
                (output[col_3m] - output[col_1m])
            )

    return output


def add_target_features(output):
    watch_col = "gh_WatchEvent_1year"
    if watch_col in output.columns:
        output["WatchEvent_1year"] = output[watch_col].fillna(0).astype(int)
        output["log_WatchEvent_1year"] = np.log1p(output["WatchEvent_1year"])

    return output


def aggregate_window_counts():
    repos = load_repo_windows()
    daily_files = sorted(glob.glob(DAILY_COUNTS_GLOB))
    if not daily_files:
        raise FileNotFoundError(f"No daily GH Archive count files matched {DAILY_COUNTS_GLOB!r}.")

    chunks = defaultdict(list)
    all_event_cols = set()

    aggregation_windows = {**WINDOWS, **TARGET_WINDOWS}

    print(f"Repos CSV             : {SAMPLE_CSV}")
    print(f"Snapshot CSV          : {SNAPSHOT_CSV}")
    print(f"Snapshot features CSV : {SNAPSHOT_FEATURES_CSV}")
    print(f"Daily counts          : {DAILY_COUNTS_GLOB}")
    print(f"Output CSV            : {OUTPUT_CSV}")
    print(f"Matched repos         : {len(repos):,}")
    print(f"Daily files to scan   : {len(daily_files):,}")
    print(f"Feature windows       : {list(WINDOWS)}")
    print(f"Target windows        : {list(TARGET_WINDOWS)}")

    for file_path in tqdm(daily_files, desc="Scanning daily GH Archive files"):
        file_day = parse_daily_file_date(file_path)
        if file_day is None:
            continue

        daily = pd.read_csv(file_path)
        if daily.empty or "repo_id" not in daily.columns:
            continue

        if "name" in daily.columns:
            daily = daily.drop(columns=["name"])

        cols = event_columns(daily)
        all_event_cols.update(cols)

        for label in aggregation_windows:
            valid_repo_ids = repos.loc[
                (file_day >= repos["start_day"]) &
                (file_day < repos[f"end_day_{label}"]),
                ["repo_id"],
            ]

            if valid_repo_ids.empty:
                continue

            matched = daily.merge(valid_repo_ids, on="repo_id", how="inner")
            if not matched.empty:
                matched["archive_day"] = file_day
                chunks[label].append(matched)

    output = repos[["repo_id", "name", "start_date"]].copy()
    all_event_cols = sorted(all_event_cols)

    for label in aggregation_windows:
        if chunks[label]:
            window_df = pd.concat(chunks[label], ignore_index=True)
            for col in all_event_cols:
                if col not in window_df.columns:
                    window_df[col] = 0
            window_df[all_event_cols] = window_df[all_event_cols].fillna(0)
            counts = window_df.groupby("repo_id")[all_event_cols].sum().reset_index()
        else:
            counts = pd.DataFrame({"repo_id": output["repo_id"]})

        rename_map = {col: f"gh_{col}_{label}" for col in all_event_cols}
        counts = counts.rename(columns=rename_map)
        output = output.merge(counts, on="repo_id", how="left")

        gh_cols = [rename_map[col] for col in all_event_cols]
        for col in gh_cols:
            if col not in output.columns:
                output[col] = 0
        output[gh_cols] = output[gh_cols].fillna(0).astype(int)

        if label in WINDOWS:
            output = add_timing_and_burst_features(
                output,
                window_df if chunks[label] else pd.DataFrame(),
                all_event_cols,
                label,
            )

    output = add_growth_features(output)
    output = add_target_features(output)

    output.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(output)} repos to {OUTPUT_CSV}")


if __name__ == "__main__":
    aggregate_window_counts()
