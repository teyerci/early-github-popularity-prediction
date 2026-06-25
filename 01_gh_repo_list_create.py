import math
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# Select roughly 20,000 repositories created in 2024.
# The script samples a balanced number of top-starred repositories per creation day.
TARGET_REPOS = 20_000
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2024, 12, 31)

OUTPUT_CSV = "repos.csv"
SECONDARY_OUTPUT_CSV = "repos_2024_20k.csv"
CHECKPOINT_CSV = "repos_2024_20k_checkpoint.csv"

PER_PAGE_MAX = 100
MAX_SEARCH_RESULTS_PER_QUERY = 1_000

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()


def date_range(start_date, end_date):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def build_session():
    session = requests.Session()
    session.headers.update({
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    if GITHUB_TOKEN:
        session.headers.update({"Authorization": f"token {GITHUB_TOKEN}"})
    else:
        print("Warning: GITHUB_TOKEN is not set. Unauthenticated GitHub Search API limits are much lower.")
    return session


def wait_for_rate_limit(response):
    remaining = response.headers.get("X-RateLimit-Remaining")
    reset = response.headers.get("X-RateLimit-Reset")

    is_rate_limited = response.status_code in (403, 429) and (
        remaining == "0" or "rate limit" in response.text.lower()
    )
    if not is_rate_limited:
        return False

    if reset and reset.isdigit():
        reset_at = datetime.fromtimestamp(int(reset), timezone.utc)
        sleep_seconds = max((reset_at - datetime.now(timezone.utc)).total_seconds() + 5, 5)
    else:
        sleep_seconds = int(response.headers.get("Retry-After", 60))

    print(f"Rate limit reached. Sleeping {sleep_seconds:.0f}s...")
    time.sleep(sleep_seconds)
    return True


def repo_to_row(repo, collection_day):
    return {
        "repo_id": repo["id"],
        "name": repo["full_name"],
        "created_at": repo["created_at"],
        "stars_now": repo["stargazers_count"],
        "forks_now": repo["forks_count"],
        "language": repo.get("language") or "",
        "description_length": len(repo.get("description") or ""),
        "topic_count": len(repo.get("topics", [])),
        "has_license": int(bool(repo.get("license"))),
        "is_fork": bool(repo["fork"]),
        "collection_day": collection_day,
    }


def fetch_repos_for_day(session, day_str, per_day_limit, request_sleep_seconds):
    rows = []
    per_page = min(PER_PAGE_MAX, per_day_limit)
    max_pages = min(
        math.ceil(per_day_limit / per_page),
        math.ceil(MAX_SEARCH_RESULTS_PER_QUERY / per_page),
    )

    for page in range(1, max_pages + 1):
        params = {
            "q": f"created:{day_str}..{day_str}",
            "sort": "stars",
            "order": "desc",
            "per_page": per_page,
            "page": page,
        }

        while True:
            response = session.get("https://api.github.com/search/repositories", params=params, timeout=30)
            if wait_for_rate_limit(response):
                continue
            response.raise_for_status()
            break

        items = response.json().get("items", [])
        if not items:
            break

        for repo in items:
            rows.append(repo_to_row(repo, day_str))
            if len(rows) >= per_day_limit:
                break

        if len(rows) >= per_day_limit:
            break

        time.sleep(request_sleep_seconds)

    return rows


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_CSV):
        return pd.DataFrame()
    checkpoint = pd.read_csv(CHECKPOINT_CSV)
    print(f"Loaded checkpoint with {len(checkpoint):,} rows from {CHECKPOINT_CSV}")
    return checkpoint


def save_checkpoint(rows):
    pd.DataFrame(rows).to_csv(CHECKPOINT_CSV, index=False)


def main():
    days = list(date_range(START_DATE, END_DATE))
    per_day_limit = math.ceil(TARGET_REPOS / len(days))
    request_sleep_seconds = 2.5 if GITHUB_TOKEN else 7.0

    print(f"Target repositories : {TARGET_REPOS:,}")
    print(f"Date range          : {START_DATE.date()} to {END_DATE.date()} ({len(days)} days)")
    print(f"Daily limit         : {per_day_limit} repositories/day")
    print(f"Expected total      : about {per_day_limit * len(days):,} before deduplication")
    print(f"Output              : {OUTPUT_CSV} and {SECONDARY_OUTPUT_CSV}")

    checkpoint = load_checkpoint()
    all_rows = checkpoint.to_dict("records") if not checkpoint.empty else []
    completed_days = set(checkpoint["collection_day"].astype(str)) if not checkpoint.empty else set()

    session = build_session()

    for day in tqdm(days, desc="Fetching daily repository lists"):
        day_str = day.strftime("%Y-%m-%d")
        if day_str in completed_days:
            continue

        try:
            rows = fetch_repos_for_day(session, day_str, per_day_limit, request_sleep_seconds)
        except requests.HTTPError as exc:
            print(f"\nError on {day_str}: {exc}")
            print("Progress has been saved. Re-run the script to resume.")
            save_checkpoint(all_rows)
            raise

        all_rows.extend(rows)
        completed_days.add(day_str)
        save_checkpoint(all_rows)
        time.sleep(request_sleep_seconds)

    df = pd.DataFrame(all_rows)
    before_dedup = len(df)
    df = df.drop_duplicates(subset="repo_id", keep="first").reset_index(drop=True)

    print(f"\nFetched rows before dedup : {before_dedup:,}")
    print(f"Unique repositories       : {len(df):,}")
    print(f"Watchlist: this is a top-starred-per-day sample, not a random sample.")

    df.to_csv(OUTPUT_CSV, index=False)
    df.to_csv(SECONDARY_OUTPUT_CSV, index=False)
    print(f"Saved {OUTPUT_CSV}")
    print(f"Saved {SECONDARY_OUTPUT_CSV}")


if __name__ == "__main__":
    main()
