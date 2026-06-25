import os
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

CLONE_DIR  = "cloned_repos"
SAMPLE_CSV = "repos_sample_1000.csv"
OUTPUT_CSV = "repos_snapshots.csv"

# A day qualifies as "real start" if it meets either condition
MIN_COMMITS_PER_DAY = 3   # >= 3 commits on same day
MIN_FILES_PER_COMMIT = 3  # a single commit touches >= 3 files
GIT_TIMEOUT_SECONDS = int(os.getenv("GIT_TIMEOUT_SECONDS", "300"))

WINDOW_DAY_MAP = {
    "1m": 30,
    "3m": 90,
    "6m": 180,
}


def selected_windows():
    labels = [w.strip() for w in os.getenv("SNAPSHOT_WINDOWS", "1m,3m,6m").split(",") if w.strip()]
    unknown = [w for w in labels if w not in WINDOW_DAY_MAP]
    if unknown:
        raise ValueError(f"Unknown snapshot windows: {unknown}. Valid values: {sorted(WINDOW_DAY_MAP)}")
    return [(label, WINDOW_DAY_MAP[label]) for label in labels]


# ── git helpers ──────────────────────────────────────────────────────────────

def git(args, cwd, timeout=GIT_TIMEOUT_SECONDS):
    try:
        r = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        cmd = "git " + " ".join(args)
        raise RuntimeError(f"{cmd} timed out after {timeout} seconds") from exc

    return r.stdout.strip() if r.returncode == 0 else None


def parse_git_timestamp(value):
    """Parse git's strict ISO timestamp and normalize it to UTC."""
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def parse_optional_timestamp(value):
    value = scalar_value(value)
    if value is None or pd.isna(value):
        return None
    try:
        return parse_git_timestamp(str(value))
    except ValueError:
        return None


def scalar_value(value):
    """Return the first non-null scalar from pandas objects or the value itself."""
    if isinstance(value, pd.Series):
        non_null = value.dropna()
        return non_null.iloc[0] if not non_null.empty else None
    if isinstance(value, (list, tuple)):
        return next((item for item in value if item is not None), None)
    return value


def repo_name_and_created_at(repo_info):
    """
    Accept either a plain "owner/repo" string or a pandas row/dict with
    name/repo and created_at. Keeping both forms lets this module work both
    standalone and when called by 08_clone_repos.py.
    """
    if isinstance(repo_info, str):
        return repo_info, None

    repo_name = scalar_value(repo_info.get("name", None))
    if repo_name is None:
        repo_name = scalar_value(repo_info.get("repo", None))
    created_at = parse_optional_timestamp(repo_info.get("created_at", None))
    return repo_name, created_at


def parse_commits(repo_path):
    """Return list of {hash, timestamp, date, file_count} from all refs."""
    raw = git(
        ["log", "--all", "--format=COMMIT %H %cI", "--shortstat"],
        repo_path,
    )
    if not raw:
        return []

    commits, current, file_count = [], None, 0
    for line in raw.splitlines():
        if line.startswith("COMMIT "):
            if current is not None:
                current["file_count"] = file_count
                commits.append(current)
            _, chash, timestamp = line.split(maxsplit=2)
            commit_dt = parse_git_timestamp(timestamp)
            current, file_count = {
                "hash": chash,
                "timestamp": commit_dt,
                "date": commit_dt.date().isoformat(),
            }, 0
        else:
            match = re.search(r"(\d+)\s+files?\s+changed", line)
            if match:
                file_count = int(match.group(1))

    if current is not None:
        current["file_count"] = file_count
        commits.append(current)

    return commits


# ── start-date detection ─────────────────────────────────────────────────────

def find_start(commits):
    """
    Walk commits oldest-first. Return (date, hash, condition) for the first
    day that meets either activity threshold. Falls back to the very first
    commit if no threshold day is found.
    """
    if not commits:
        return None, None, None

    by_date = defaultdict(list)
    for c in commits:
        by_date[c["date"]].append(c)

    for date in sorted(by_date):
        day = sorted(by_date[date], key=lambda c: c["timestamp"])
        if len(day) >= MIN_COMMITS_PER_DAY:
            # use the earliest commit that day
            return day[0]["timestamp"].isoformat(), day[0]["hash"], "commits_per_day"
        heavy = [c for c in day if c["file_count"] >= MIN_FILES_PER_COMMIT]
        if heavy:
            return heavy[0]["timestamp"].isoformat(), heavy[0]["hash"], "files_per_commit"

    # fallback: first commit ever
    first = min(commits, key=lambda c: c["timestamp"])
    return first["timestamp"].isoformat(), first["hash"], "fallback"


# ── snapshot helpers ──────────────────────────────────────────────────────────

def last_commit_before(commits, start_str, days):
    """Last commit (hash, date) that falls on or before start + days."""
    cutoff = parse_git_timestamp(start_str) + timedelta(days=days)
    eligible = [c for c in commits if c["timestamp"] <= cutoff]
    if not eligible:
        return None, None

    latest = max(eligible, key=lambda c: c["timestamp"])
    return latest["timestamp"].isoformat(), latest["hash"]


def count_files_at(repo_path, commit_hash):
    """Number of tracked files at a given commit."""
    out = git(["ls-tree", "-r", "--name-only", commit_hash], repo_path)
    return len(out.splitlines()) if out else None


def count_commits_between(commits, start_str, cutoff_hash):
    """How many commits fall within [start, cutoff_hash timestamp]."""
    cutoff = None
    for c in commits:
        if c["hash"] == cutoff_hash:
            cutoff = c["timestamp"]
            break
    if cutoff is None:
        return None
    start = parse_git_timestamp(start_str)
    return sum(1 for c in commits if start <= c["timestamp"] <= cutoff)


def effective_start(git_start_str, github_created_at):
    git_start = parse_git_timestamp(git_start_str)
    if github_created_at is None:
        return git_start.isoformat(), "git_start_date", None

    chosen = max(git_start, github_created_at)
    source = "github_created_at" if github_created_at > git_start else "git_start_date"
    delta_days = (git_start - github_created_at).total_seconds() / 86400
    return chosen.isoformat(), source, round(delta_days, 3)


# ── per-repo analysis ────────────────────────────────────────────────────────

def analyze_repo(repo_info):
    try:
        repo_name, github_created_at = repo_name_and_created_at(repo_info)
        if not repo_name or "/" not in str(repo_name):
            return None, f"invalid repo name: {repo_name!r}"
        repo_name = str(repo_name)
        owner, repo = repo_name.split("/", 1)
        repo_path = os.path.join(CLONE_DIR, owner, repo)

        if not os.path.isdir(repo_path):
            return None, "not cloned"

        commits = parse_commits(repo_path)
        if not commits:
            return None, "no commits"

        git_start_date, git_start_hash, condition = find_start(commits)
        if not git_start_date:
            return None, "could not determine start"

        start_date, start_source, start_created_delta_days = effective_start(
            git_start_date,
            github_created_at,
        )

        snapshots = {}
        for label, days in selected_windows():
            date, chash = last_commit_before(commits, start_date, days)
            snapshots[label] = {
                "date":    date,
                "commit":  chash,
                "files":   count_files_at(repo_path, chash) if chash else None,
                "commits": count_commits_between(commits, start_date, chash) if chash else None,
            }

        result = {
            "repo":                      repo_name,
            "start_date":                start_date,
            "start_date_source":         start_source,
            "github_created_at":         github_created_at.isoformat() if github_created_at else None,
            "git_start_date":            git_start_date,
            "git_start_commit":          git_start_hash,
            "git_start_condition":       condition,
            "git_start_minus_created_days": start_created_delta_days,
            "start_commit":              git_start_hash,
            "start_condition":           condition,
            "total_commits":             len(commits),
        }
        for label, snapshot in snapshots.items():
            result[f"snap_{label}_date"] = snapshot["date"]
            result[f"snap_{label}_commit"] = snapshot["commit"]
            result[f"snap_{label}_files"] = snapshot["files"]
            result[f"snap_{label}_commits"] = snapshot["commits"]

        return result, None
    except RuntimeError as exc:
        return None, str(exc)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(SAMPLE_CSV)
    results, errors = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing"):
        record, err = analyze_repo(row)
        if record:
            results.append(record)
        else:
            errors.append((row.get("name", row.get("repo")), err))

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved {len(results)} repos → {OUTPUT_CSV}")
    print(f"Skipped: {len(errors)}")
    if errors:
        for name, err in errors[:10]:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
