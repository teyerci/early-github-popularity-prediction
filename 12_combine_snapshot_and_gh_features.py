import os

import pandas as pd

BASE_CSV = os.getenv("BASE_CSV", os.getenv("REPOS_CSV", "repos_sample_980.csv"))
SNAPSHOT_FEATURES_CSV = os.getenv("SNAPSHOT_FEATURES_CSV", "repos_snapshot_features.csv")
GH_WINDOWS_CSV = os.getenv("GH_WINDOWS_CSV", "repos_sample_980_gh_archive_windows.csv")
OUTPUT_CSV = os.getenv("COMBINED_FEATURES_OUTPUT_CSV", "repos_sample_980_combined_features.csv")

TARGET_COLUMNS = {
    "WatchEvent_1year",
    "log_WatchEvent_1year",
}

# These columns are older first-month counts from the creation-date pipeline.
# The combined experiment uses gh_*_1m/3m/6m counts aligned to snapshot start_date.
BASE_CREATION_WINDOW_EVENT_COLUMNS = {
    "CreateEvent",
    "DeleteEvent",
    "ForkEvent",
    "IssueCommentEvent",
    "IssuesEvent",
    "PullRequestEvent",
    "PushEvent",
    "ReleaseEvent",
    "WatchEvent",
}


def make_base_static_features(base):
    drop_cols = [c for c in BASE_CREATION_WINDOW_EVENT_COLUMNS if c in base.columns]
    return base.drop(columns=drop_cols)


def add_data_quality_flags(combined):
    feature_columns = {}
    gh_count_cols = [
        c for c in combined.columns
        if c.startswith("gh_")
        and any(c.endswith(f"_{label}") for label in ["1m", "3m", "6m"])
        and not any(
            marker in c
            for marker in [
                "_growth_",
                "_ratio_",
                "_incremental_daily_",
                "_acceleration_",
                "_active_days_",
                "_total_",
                "_max_daily_",
                "_days_to_first_",
                "_started_late_",
                "gh_has_",
            ]
        )
    ]

    for label in ["1m", "3m", "6m"]:
        window_gh_count_cols = [c for c in gh_count_cols if c.endswith(f"_{label}")]
        if not window_gh_count_cols:
            continue

        commit_col = f"{label}_commit_count"
        file_col = f"{label}_file_count"
        if commit_col not in combined.columns and file_col not in combined.columns:
            continue

        gh_total = combined[window_gh_count_cols].fillna(0).sum(axis=1)
        commit_count = (
            combined[commit_col].fillna(0)
            if commit_col in combined.columns
            else pd.Series(0, index=combined.index)
        )
        file_count = (
            combined[file_col].fillna(0)
            if file_col in combined.columns
            else pd.Series(0, index=combined.index)
        )
        active_days_col = f"{label}_active_days"
        active_days = (
            combined[active_days_col].fillna(0)
            if active_days_col in combined.columns
            else pd.Series(0, index=combined.index)
        )

        git_activity = pd.Series(0, index=combined.index)
        if commit_col in combined.columns:
            git_activity = git_activity + commit_count
        if file_col in combined.columns:
            git_activity = git_activity + file_count

        watch_col = f"gh_WatchEvent_{label}"
        watch_count = (
            combined[watch_col].fillna(0)
            if watch_col in combined.columns
            else pd.Series(0, index=combined.index)
        )

        fork_col = f"gh_ForkEvent_{label}"
        fork_count = (
            combined[fork_col].fillna(0)
            if fork_col in combined.columns
            else pd.Series(0, index=combined.index)
        )

        push_col = f"gh_PushEvent_{label}"
        push_count = (
            combined[push_col].fillna(0)
            if push_col in combined.columns
            else pd.Series(0, index=combined.index)
        )

        git_active = git_activity > 0
        many_commits = commit_count >= 50
        many_files = file_count >= 50
        sustained_git_activity = (commit_count >= 20) | (active_days >= 10) | (file_count >= 50)
        weak_archive_activity = gh_total <= 2
        weak_watch_activity = watch_count <= 2

        feature_columns[f"gh_git_activity_score_{label}"] = git_activity
        feature_columns[f"gh_archive_to_git_activity_ratio_{label}"] = (
            gh_total / git_activity.clip(lower=1)
        )
        feature_columns[f"gh_watch_to_git_activity_ratio_{label}"] = (
            watch_count / git_activity.clip(lower=1)
        )
        feature_columns[f"gh_watch_share_of_archive_{label}"] = (
            watch_count / gh_total.clip(lower=1)
        )
        feature_columns[f"gh_fork_watch_balance_{label}"] = (
            fork_count / watch_count.clip(lower=1)
        )
        feature_columns[f"gh_push_watch_balance_{label}"] = (
            push_count / watch_count.clip(lower=1)
        )

        feature_columns[f"gh_zero_archive_but_active_git_{label}"] = (
            (gh_total == 0) & git_active
        ).astype(int)
        feature_columns[f"gh_git_active_but_no_watch_{label}"] = (
            (watch_count == 0) & git_active
        ).astype(int)
        feature_columns[f"gh_archive_coverage_suspect_{label}"] = (
            sustained_git_activity & weak_archive_activity
        ).astype(int)
        feature_columns[f"gh_low_attention_high_activity_{label}"] = (
            sustained_git_activity & weak_watch_activity
        ).astype(int)
        feature_columns[f"gh_zero_watch_but_many_commits_{label}"] = (
            (watch_count == 0) & many_commits
        ).astype(int)
        feature_columns[f"gh_zero_watch_but_many_files_{label}"] = (
            (watch_count == 0) & many_files
        ).astype(int)

    if feature_columns:
        features = pd.DataFrame(feature_columns, index=combined.index)
        combined = pd.concat([combined, features], axis=1).copy()

    return combined


def combine_features():
    base = pd.read_csv(BASE_CSV)
    snapshots = pd.read_csv(SNAPSHOT_FEATURES_CSV)
    gh_windows = pd.read_csv(GH_WINDOWS_CSV)

    base_static = make_base_static_features(base)

    combined = base_static.merge(
        snapshots,
        left_on="name",
        right_on="repo",
        how="inner",
        suffixes=("", "_snapshot"),
    )

    gh_join = gh_windows.drop(
        columns=["start_date", "WatchEvent_1year", "log_WatchEvent_1year"],
        errors="ignore",
    )
    combined = combined.merge(
        gh_join,
        on=["repo_id", "name"],
        how="left",
    )

    for target_col in TARGET_COLUMNS:
        if target_col in base_static.columns:
            continue
        if target_col in gh_windows.columns:
            combined[target_col] = combined["repo_id"].map(
                gh_windows.set_index("repo_id")[target_col]
            )

    gh_cols = [c for c in combined.columns if c.startswith("gh_")]
    combined[gh_cols] = combined[gh_cols].fillna(0)
    combined = add_data_quality_flags(combined)

    for col in combined.select_dtypes(include="bool").columns:
        combined[col] = combined[col].astype(int)

    combined.to_csv(OUTPUT_CSV, index=False)

    snapshot_cols = [c for c in combined.columns if c.startswith(("1m_", "3m_", "6m_"))]
    gh_cols = [c for c in combined.columns if c.startswith("gh_")]
    data_quality_cols = [
        c for c in gh_cols
        if any(
            marker in c
            for marker in [
                "coverage_suspect",
                "active_but_no",
                "low_attention_high_activity",
                "zero_watch_but_many",
                "archive_to_git_activity_ratio",
                "watch_to_git_activity_ratio",
                "watch_share_of_archive",
                "git_activity_score",
                "fork_watch_balance",
                "push_watch_balance",
            ]
        )
    ]
    target_cols = [c for c in TARGET_COLUMNS if c in combined.columns]

    print(f"Saved {len(combined)} repos to {OUTPUT_CSV}")
    print(f"Base/static + target columns: {len(base_static.columns)}")
    print(f"Snapshot feature columns: {len(snapshot_cols)}")
    print(f"GH Archive window columns: {len(gh_cols)}")
    print(f"Data quality / coverage columns: {len(data_quality_cols)}")
    print(f"Target columns present: {target_cols}")


if __name__ == "__main__":
    combine_features()
