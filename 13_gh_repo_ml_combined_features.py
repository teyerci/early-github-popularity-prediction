import warnings

warnings.filterwarnings("ignore")

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

INPUT_CSV = os.getenv("COMBINED_FEATURES_CSV", "repos_sample_980_combined_features.csv")
RESULTS_CSV = os.getenv("MODEL_RESULTS_CSV", "repos_sample_980_combined_model_results.csv")
RF_DEPTH_RESULTS_CSV = os.getenv("RF_DEPTH_RESULTS_CSV", "repos_sample_980_rf_depth_comparison.csv")
SEED = 42
# "future": each window's models predict log1p of WatchEvents in [start + window, start + 1 year),
# so the target does not overlap the features. "cumulative": the original
# log_WatchEvent_1year target over [start, start + 1 year), kept for comparison.
TARGET_MODE = os.getenv("TARGET_MODE", "future")
CUMULATIVE_TARGET = "log_WatchEvent_1year"
COVERAGE_COLUMN = "window_complete_1year"
RF_MAX_DEPTHS = [6, 8, 12]
N_TARGET_STRATA = 4
WINDOW_LABELS = ["1m", "3m", "6m"]

IGNORE_COLUMNS = {
    "repo_id",
    "name",
    "repo",
    "start_date",
    "stars_now",
    "forks_now",
    "WatchEvent_1year",
    "log_WatchEvent_1year",
}
NON_FEATURE_PREFIXES = (
    "WatchEvent_future_",
    "log_WatchEvent_future_",
    "window_complete_",
)


def clean_for_modeling(df):
    df = df.copy()

    for col in df.select_dtypes(include="bool").columns:
        df[col] = df[col].astype(int)

    # Fill missing feature values only; a missing target must not become a zero-star example.
    numeric_cols = [
        c for c in df.select_dtypes(include=np.number).columns
        if c not in IGNORE_COLUMNS and not c.startswith(NON_FEATURE_PREFIXES)
    ]
    df[numeric_cols] = df[numeric_cols].fillna(0)
    return df


def target_for(config_name):
    if TARGET_MODE == "cumulative":
        return CUMULATIVE_TARGET
    if TARGET_MODE != "future":
        raise ValueError(f"Unknown TARGET_MODE {TARGET_MODE!r}. Use 'future' or 'cumulative'.")
    label = config_name.split()[-1]
    return f"log_WatchEvent_future_{label}"


def select_complete_rows(df):
    if COVERAGE_COLUMN not in df.columns:
        raise ValueError(
            f"{INPUT_CSV} has no {COVERAGE_COLUMN!r} column; rerun steps 11 and 12 "
            "so incomplete GH Archive windows can be excluded."
        )
    complete = df[COVERAGE_COLUMN] == 1
    print(f"Dropped {int((~complete).sum()):,} repos whose 1-year window is not fully covered by GH Archive files.")
    return df[complete]


def feature_configs(df):
    snapshot_by_window = {
        label: [c for c in df.columns if c.startswith(f"{label}_")]
        for label in WINDOW_LABELS
    }
    gh_by_window = {
        label: [
            c for c in df.columns
            if c.startswith("gh_") and (
                c.endswith(f"_{label}") or
                f"_to_{label}" in c or
                (label == "6m" and c.endswith("_1m_3m_6m"))
            )
        ]
        for label in WINDOW_LABELS
    }

    non_static = set()
    for label in WINDOW_LABELS:
        non_static.update(snapshot_by_window[label])
        non_static.update(gh_by_window[label])

    non_static.update(c for c in df.columns if c.startswith("gh_") and c.endswith("_1year"))

    numeric_cols = set(df.select_dtypes(include=np.number).columns)
    static = sorted(
        c for c in numeric_cols
        if c not in IGNORE_COLUMNS and c not in non_static and not c.startswith(NON_FEATURE_PREFIXES)
    )

    configs = {}
    for label in WINDOW_LABELS:
        snapshot_features = snapshot_by_window[label]
        gh_features = gh_by_window[label]

        # Static baseline per window, so every config in a window shares its target.
        if TARGET_MODE == "future" or label == WINDOW_LABELS[0]:
            static_name = f"Static {label}" if TARGET_MODE == "future" else "Static"
            configs[static_name] = static

        if snapshot_features:
            configs[f"Snapshot {label}"] = snapshot_features
        if gh_features:
            configs[f"GH Archive {label}"] = gh_features
        if snapshot_features or gh_features:
            configs[f"Combined {label}"] = static + snapshot_features + gh_features

    return configs


def make_target_strata(df):
    strata = pd.qcut(
        df[CUMULATIVE_TARGET],
        q=N_TARGET_STRATA,
        labels=False,
        duplicates="drop",
    )

    if strata.nunique(dropna=True) < 2:
        return None

    return strata


def make_models():
    models = {}
    for max_depth in RF_MAX_DEPTHS:
        models[f"Random Forest max_depth={max_depth}"] = RandomForestRegressor(
            n_estimators=300,
            max_depth=max_depth,
            random_state=SEED,
            n_jobs=-1,
        )

    models["Gradient Boosting"] = GradientBoostingRegressor(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        random_state=SEED,
    )
    return models


def evaluate():
    df = clean_for_modeling(pd.read_csv(INPUT_CSV))
    df = select_complete_rows(df)

    configs = feature_configs(df)
    targets = sorted({target_for(name) for name in configs})
    missing = [t for t in [CUMULATIVE_TARGET, *targets] if t not in df.columns]
    if missing:
        raise ValueError(f"{INPUT_CSV} is missing target columns {missing}; rerun steps 11 and 12.")
    n_missing_target = int(df[targets].isna().any(axis=1).sum())
    if n_missing_target:
        raise ValueError(f"{n_missing_target:,} rows have a missing target; check the step 11/12 merge.")

    # One split for every config, stratified on the cumulative target, so results are comparable.
    target_strata = make_target_strata(df)

    print(f"Input data: {INPUT_CSV}")
    print(f"Rows: {len(df):,}")
    print(f"Target mode: {TARGET_MODE} ({', '.join(targets)})")
    print("Feature configs:")
    for config_name, features in configs.items():
        print(f"  {config_name:<14}: {len(features):,} features")
    print()

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=0.20,
        random_state=SEED,
        stratify=target_strata,
    )
    rows = []

    for config_name, features in configs.items():
        if not features:
            continue

        target = target_for(config_name)
        y_train = df.loc[train_idx, target]
        y_test = df.loc[test_idx, target]
        X_train = df.loc[train_idx, features]
        X_test = df.loc[test_idx, features]

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        for model_name, model in make_models().items():
            model.fit(X_train_sc, y_train)
            pred_train = model.predict(X_train_sc)
            pred_test = model.predict(X_test_sc)
            train_r2 = r2_score(y_train, pred_train)
            test_r2 = r2_score(y_test, pred_test)

            rows.append(
                {
                    "Config": config_name,
                    "Target": target,
                    "Model": model_name,
                    "n_features": len(features),
                    "Train R2": train_r2,
                    "Test R2": test_r2,
                    "Train-Test R2 Gap": train_r2 - test_r2,
                    "Test MAE": mean_absolute_error(y_test, pred_test),
                    "Test RMSE": np.sqrt(mean_squared_error(y_test, pred_test)),
                }
            )

    results = pd.DataFrame(rows).sort_values(
        ["Target", "Test R2", "Test RMSE"],
        ascending=[True, False, True],
    )
    results.to_csv(RESULTS_CSV, index=False)

    rf_depth_results = results[results["Model"].str.startswith("Random Forest")].copy()
    rf_depth_results.to_csv(RF_DEPTH_RESULTS_CSV, index=False)

    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nInput data: {INPUT_CSV}")
    print(f"Rows: {len(df):,}")
    print(f"Saved results to {RESULTS_CSV}")
    print(f"Saved Random Forest depth comparison to {RF_DEPTH_RESULTS_CSV}")

    if target_strata is not None:
        split_balance = pd.DataFrame({
            "target_stratum": target_strata,
            "split": "unused",
        })
        split_balance.loc[train_idx, "split"] = "train"
        split_balance.loc[test_idx, "split"] = "test"
        split_balance = (
            split_balance.groupby(["target_stratum", "split"])
            .size()
            .unstack(fill_value=0)
        )
        print(f"\nUsed target stratification with {target_strata.nunique()} qcut bins:")
        print(split_balance.to_string())


if __name__ == "__main__":
    evaluate()
