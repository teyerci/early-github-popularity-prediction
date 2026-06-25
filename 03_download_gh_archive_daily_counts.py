"""Download GH Archive hourly files and build daily repo/event count files.

This script creates the intermediate files consumed by
11_gh_archive_window_counts_sample.py:

    out/gh_archive_counts_YYYY-MM-DD.csv.gz

Each output row is one repository/day and each event type is a count column.
Only repositories listed in REPOS_CSV are retained, which keeps the daily files
small enough for later window aggregation.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


REPOS_CSV = os.getenv("REPOS_CSV", "repos_2024_20k.csv")
OUTPUT_DIR = Path(os.getenv("GH_DAILY_COUNTS_DIR", "out"))
START_DATE = os.getenv("GH_ARCHIVE_START_DATE", "2024-01-01")
END_DATE = os.getenv("GH_ARCHIVE_END_DATE", "2025-12-31")
REQUEST_TIMEOUT = int(os.getenv("GH_ARCHIVE_TIMEOUT_SECONDS", "60"))
MAX_RETRIES = int(os.getenv("GH_ARCHIVE_MAX_RETRIES", "3"))
SLEEP_SECONDS = float(os.getenv("GH_ARCHIVE_SLEEP_SECONDS", "0.2"))


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def iter_days(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def load_repo_lookup() -> tuple[set[int], dict[int, str]]:
    repos = pd.read_csv(REPOS_CSV, usecols=["repo_id", "name"])
    repos = repos.dropna(subset=["repo_id"])
    repos["repo_id"] = repos["repo_id"].astype(int)
    repo_ids = set(repos["repo_id"])
    names = dict(zip(repos["repo_id"], repos["name"]))
    return repo_ids, names


def fetch_hour(day: date, hour: int) -> bytes | None:
    url = f"https://data.gharchive.org/{day.isoformat()}-{hour}.json.gz"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            if response.status_code == 404:
                return None
            if response.status_code == 200:
                return response.content
            print(f"HTTP {response.status_code} for {url} (attempt {attempt}/{MAX_RETRIES})")
        except requests.RequestException as exc:
            print(f"Request failed for {url} (attempt {attempt}/{MAX_RETRIES}): {exc}")
        time.sleep(SLEEP_SECONDS * attempt)
    return None


def count_events_for_hour(content: bytes, repo_ids: set[int]) -> dict[tuple[int, str], int]:
    counts: dict[tuple[int, str], int] = {}
    with gzip.GzipFile(fileobj=io.BytesIO(content)) as gz:
        for raw_line in gz:
            try:
                event = json.loads(raw_line)
                repo = event.get("repo") or {}
                repo_id = repo.get("id")
                event_type = event.get("type")
            except (json.JSONDecodeError, AttributeError):
                continue

            if repo_id in repo_ids and event_type:
                key = (int(repo_id), str(event_type))
                counts[key] = counts.get(key, 0) + 1
    return counts


def merge_counts(total: dict[tuple[int, str], int], part: dict[tuple[int, str], int]) -> None:
    for key, value in part.items():
        total[key] = total.get(key, 0) + value


def write_daily_counts(day: date, counts: dict[tuple[int, str], int], names: dict[int, str]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"gh_archive_counts_{day.isoformat()}.csv.gz"

    if not counts:
        pd.DataFrame(columns=["repo_id", "name"]).to_csv(out_path, index=False, compression="gzip")
        return

    records = [
        {"repo_id": repo_id, "event_type": event_type, "count": count}
        for (repo_id, event_type), count in counts.items()
    ]
    long_df = pd.DataFrame(records)
    wide = (
        long_df.pivot_table(
            index="repo_id",
            columns="event_type",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )
    wide.columns.name = None
    wide.insert(1, "name", wide["repo_id"].map(names))
    event_cols = [c for c in wide.columns if c not in {"repo_id", "name"}]
    wide[event_cols] = wide[event_cols].astype(int)
    wide.to_csv(out_path, index=False, compression="gzip")


def main() -> None:
    repo_ids, names = load_repo_lookup()
    start = parse_date(START_DATE)
    end = parse_date(END_DATE)

    print(f"Repos CSV       : {REPOS_CSV}")
    print(f"Tracked repos   : {len(repo_ids):,}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Date range      : {start} to {end}")

    for day in tqdm(list(iter_days(start, end)), desc="GH Archive days"):
        out_path = OUTPUT_DIR / f"gh_archive_counts_{day.isoformat()}.csv.gz"
        if out_path.exists():
            continue

        daily_counts: dict[tuple[int, str], int] = {}
        for hour in range(24):
            content = fetch_hour(day, hour)
            if content is None:
                continue
            merge_counts(daily_counts, count_events_for_hour(content, repo_ids))
            time.sleep(SLEEP_SECONDS)

        write_daily_counts(day, daily_counts, names)


if __name__ == "__main__":
    main()
