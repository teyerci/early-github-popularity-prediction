import json
import os
import shutil
import subprocess
import time
import importlib.util

import pandas as pd
import requests

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


CLONE_DIR = "cloned_repos"
CSV_FILE = os.getenv("REPOS_CSV", "repos_2024_20k.csv")
SKIP_FILE = os.getenv("CLONE_SKIP_FILE", "clone_skip_20k.json")
PROCESSED_FILE = os.getenv("CLONE_PROCESSED_FILE", "clone_processed_20k.json")
SNAPSHOTS_OUTPUT_CSV = os.getenv("SNAPSHOTS_OUTPUT_CSV", "repos_snapshots_20k.csv")
FEATURES_OUTPUT_CSV = os.getenv("SNAPSHOT_FEATURES_OUTPUT_CSV", "repos_snapshot_features_20k.csv")
REMOVE_CLONE_AFTER_PROCESS = os.getenv("REMOVE_CLONE_AFTER_PROCESS", "1") != "0"
RETRY_PROCESSING_FAILURES = os.getenv("RETRY_PROCESSING_FAILURES", "1") != "0"

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()
MAX_SIZE_MB = int(os.getenv("MAX_REPO_SIZE_MB", "500"))
API_SIZE_SAFETY_MARGIN_MB = int(os.getenv("API_SIZE_SAFETY_MARGIN_MB", "50"))
EFFECTIVE_API_MAX_SIZE_MB = max(MAX_SIZE_MB - API_SIZE_SAFETY_MARGIN_MB, 1)
LOCAL_CLONE_LIMIT_MB = int(os.getenv("LOCAL_CLONE_LIMIT_MB", str(MAX_SIZE_MB)))
CLONE_NO_CHECKOUT = os.getenv("CLONE_NO_CHECKOUT", "1") != "0"
MAX_RETRIES = 3
RETRY_SLEEP = 60
SNAPSHOT_WINDOWS = os.getenv("SNAPSHOT_WINDOWS", "1m,3m,6m")

RATE_LIMIT_SIGNALS = [
    "rate limit",
    "429",
    "too many requests",
    "connection reset",
    "timed out",
    "temporarily unavailable",
    "server error",
    "503",
]


def progress_write(message):
    if hasattr(tqdm, "write"):
        tqdm.write(message)
    else:
        print(message)


def load_skip_list():
    if os.path.exists(SKIP_FILE):
        with open(SKIP_FILE) as f:
            return json.load(f)
    return {}


def load_json_dict(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_skip_list(skip_list):
    with open(SKIP_FILE, "w") as f:
        json.dump(skip_list, f, indent=2, sort_keys=True)


def save_json_dict(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def should_retry_skip_reason(reason):
    return RETRY_PROCESSING_FAILURES and str(reason).startswith("processing failed:")


def load_script_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def append_record_csv(record, path):
    df = pd.DataFrame([record])
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def repos_already_in_csv(path, repo_col):
    if not os.path.exists(path):
        return set()
    try:
        return set(pd.read_csv(path, usecols=[repo_col])[repo_col].dropna().astype(str))
    except Exception:
        return set()


def remove_clone_dir(target_dir, repo_name):
    clone_root = os.path.abspath(CLONE_DIR)
    target_abs = os.path.abspath(target_dir)

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
        progress_write(f"\nRemoved cloned repo folder for {repo_name}")

    parent = os.path.dirname(target_abs)
    while parent.startswith(clone_root + os.sep) and parent != clone_root:
        try:
            os.rmdir(parent)
        except OSError:
            break
        parent = os.path.dirname(parent)


def build_session():
    session = requests.Session()
    session.headers.update({
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    if GITHUB_TOKEN:
        session.headers.update({"Authorization": f"token {GITHUB_TOKEN}"})
    else:
        print("Warning: GITHUB_TOKEN is not set. API size checks may hit low unauthenticated limits.")
    return session


def wait_for_api_rate_limit(response):
    remaining = response.headers.get("X-RateLimit-Remaining")
    reset = response.headers.get("X-RateLimit-Reset")
    retry_after = response.headers.get("Retry-After")

    is_rate_limited = response.status_code in (403, 429) and (
        remaining == "0" or "rate limit" in response.text.lower()
    )
    if not is_rate_limited:
        return False

    if retry_after and retry_after.isdigit():
        sleep_seconds = int(retry_after)
    elif reset and reset.isdigit():
        sleep_seconds = max(int(reset) - int(time.time()) + 5, 5)
    else:
        sleep_seconds = RETRY_SLEEP

    progress_write(f"\nGitHub API rate limit reached. Sleeping {sleep_seconds}s...")
    time.sleep(sleep_seconds)
    return True


def api_repo_size_mb(session, repo_name):
    """
    Return (size_mb, error).

    GitHub's repository API returns `size` in KB. We require this check before
    cloning so large repositories can be skipped without downloading them.
    """
    owner, repo = repo_name.split("/", 1)
    url = f"https://api.github.com/repos/{owner}/{repo}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(url, timeout=20)
        except requests.RequestException as exc:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP * attempt)
                continue
            return None, f"API request failed: {exc}"

        if wait_for_api_rate_limit(response):
            continue

        if response.status_code == 200:
            size_kb = response.json().get("size")
            if size_kb is None:
                return None, "API response did not include repository size"
            return size_kb / 1024, None

        if response.status_code in (401, 403):
            return None, f"API auth/permission error: HTTP {response.status_code}"

        if response.status_code in (404, 409, 451):
            return None, f"repository unavailable: HTTP {response.status_code}"

        if response.status_code >= 500 and attempt < MAX_RETRIES:
            time.sleep(RETRY_SLEEP * attempt)
            continue

        return None, f"API size check failed: HTTP {response.status_code}"

    return None, "API size check failed after retries"


def is_valid_clone(path):
    result = subprocess.run(
        ["git", "rev-parse", "--git-dir"],
        cwd=path,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def dir_size_mb(path):
    result = subprocess.run(["du", "-sk", path], capture_output=True, text=True)
    if result.returncode != 0:
        return 0
    return int(result.stdout.split()[0]) / 1024


def is_rate_limited(stderr):
    err = stderr.lower()
    return any(signal in err for signal in RATE_LIMIT_SIGNALS)


def clone_with_retry(url, target_dir, repo_name):
    for attempt in range(1, MAX_RETRIES + 1):
        cmd = ["git", "clone", "--quiet"]
        if CLONE_NO_CHECKOUT:
            cmd.append("--no-checkout")
        cmd.extend([url, target_dir])

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        while process.poll() is None:
            if os.path.exists(target_dir) and dir_size_mb(target_dir) > LOCAL_CLONE_LIMIT_MB:
                process.kill()
                _, stderr = process.communicate()
                remove_clone_dir(target_dir, repo_name)
                return False, (
                    f"clone exceeded local disk limit: > {LOCAL_CLONE_LIMIT_MB} MB. "
                    f"stderr: {(stderr or '').strip()[:200]}"
                )
            time.sleep(2)

        _, stderr = process.communicate()
        if process.returncode == 0:
            return True, None

        err = (stderr or "").strip()
        if is_rate_limited(err) and attempt < MAX_RETRIES:
            wait = RETRY_SLEEP * attempt
            progress_write(f"\nGit clone rate/connection issue on {repo_name}; sleeping {wait}s...")
            time.sleep(wait)
            remove_clone_dir(target_dir, repo_name)
            continue

        return False, err or "git clone failed"

    return False, f"failed after {MAX_RETRIES} retries"


def process_cloned_repo(repo_row, snapshot_module, feature_module):
    try:
        snapshot_record, err = snapshot_module.analyze_repo(repo_row)
        if not snapshot_record:
            return None, None, err or "snapshot analysis failed"

        feature_record = feature_module.analyze_repo(pd.Series(snapshot_record))
        if not feature_record:
            return snapshot_record, None, "snapshot feature extraction failed"

        return snapshot_record, feature_record, None
    except Exception as exc:
        return None, None, f"processing exception: {type(exc).__name__}: {exc}"


def main():
    os.makedirs(CLONE_DIR, exist_ok=True)

    df = pd.read_csv(CSV_FILE)
    if "name" not in df.columns:
        raise ValueError(f"{CSV_FILE} must contain a 'name' column.")

    skip_list = load_skip_list()
    processed = load_json_dict(PROCESSED_FILE)

    for repo_name in repos_already_in_csv(FEATURES_OUTPUT_CSV, "repo"):
        processed.setdefault(repo_name, {"status": "processed_from_existing_features_csv"})

    session = build_session()
    snapshot_module = load_script_module("snapshot_analyzer", "09_analyze_snapshots.py")
    feature_module = load_script_module("snapshot_feature_extractor", "10_extract_snapshot_features.py")
    snapshot_module.CLONE_DIR = CLONE_DIR
    feature_module.CLONE_DIR = CLONE_DIR

    counts = {
        "cloned": 0,
        "used_existing_clone": 0,
        "processed": 0,
        "already_processed": 0,
        "already_skipped": 0,
        "skipped_size": 0,
        "skipped_api": 0,
        "failed_clone": 0,
        "failed_processing": 0,
        "oversize_after_clone_seen": 0,
        "removed_after_processing": 0,
    }
    failures = []

    print(f"Repos in CSV          : {len(df):,}")
    print(f"CSV file              : {CSV_FILE}")
    print(f"Clone directory       : {CLONE_DIR}")
    print(f"Skip log              : {SKIP_FILE}")
    print(f"Processed log         : {PROCESSED_FILE}")
    print(f"Snapshots output      : {SNAPSHOTS_OUTPUT_CSV}")
    print(f"Features output       : {FEATURES_OUTPUT_CSV}")
    print(f"Maximum API size      : {MAX_SIZE_MB} MB")
    print(f"API safety margin     : {API_SIZE_SAFETY_MARGIN_MB} MB")
    print(f"Effective API max     : {EFFECTIVE_API_MAX_SIZE_MB} MB")
    print(f"Local clone limit     : {LOCAL_CLONE_LIMIT_MB} MB")
    print(f"Clone no checkout     : {CLONE_NO_CHECKOUT}")
    print(f"Snapshot windows      : {SNAPSHOT_WINDOWS}")
    print(f"Git analysis timeout  : {os.getenv('GIT_TIMEOUT_SECONDS', '300')} seconds")
    print("Policy                : fetch GitHub API size before every clone")
    print(f"Remove after process  : {REMOVE_CLONE_AFTER_PROCESS}")
    print(f"Retry processing skips: {RETRY_PROCESSING_FAILURES}")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Cloning repositories"):
        repo_name = row["name"]
        owner, repo = repo_name.split("/", 1)
        target_dir = os.path.join(CLONE_DIR, owner, repo)

        if repo_name in processed:
            counts["already_processed"] += 1
            if REMOVE_CLONE_AFTER_PROCESS and os.path.exists(target_dir):
                remove_clone_dir(target_dir, repo_name)
            continue

        if repo_name in skip_list:
            if should_retry_skip_reason(skip_list[repo_name]):
                progress_write(f"\nRetrying previous processing failure for {repo_name}")
                del skip_list[repo_name]
                save_skip_list(skip_list)
            else:
                counts["already_skipped"] += 1
                continue

        cloned_this_run = False
        use_existing_clone = False
        size_mb = None

        if os.path.exists(target_dir):
            if is_valid_clone(target_dir):
                counts["used_existing_clone"] += 1
                actual_mb = dir_size_mb(target_dir)
                if actual_mb > MAX_SIZE_MB:
                    progress_write(
                        f"\nExisting clone for {repo_name} is {actual_mb:.1f} MB "
                        f"(>{MAX_SIZE_MB} MB); processing it before removal."
                    )
                    counts["oversize_after_clone_seen"] += 1
                size_mb = actual_mb
                use_existing_clone = True
            else:
                progress_write(f"\nIncomplete clone found for {repo_name}; removing and re-cloning...")
                remove_clone_dir(target_dir, repo_name)

        if not use_existing_clone:
            size_mb, size_error = api_repo_size_mb(session, repo_name)
            if size_error:
                reason = f"size check failed before clone: {size_error}"
                progress_write(f"\nSkipping {repo_name} — {reason}")
                skip_list[repo_name] = reason
                save_skip_list(skip_list)
                counts["skipped_api"] += 1
                continue

            if size_mb > EFFECTIVE_API_MAX_SIZE_MB:
                reason = (
                    f"too large/near limit by API: {size_mb:.1f} MB "
                    f"> effective limit {EFFECTIVE_API_MAX_SIZE_MB} MB "
                    f"(configured max {MAX_SIZE_MB} MB, safety margin {API_SIZE_SAFETY_MARGIN_MB} MB)"
                )
                progress_write(f"\nSkipping {repo_name} — {reason}")
                skip_list[repo_name] = reason
                save_skip_list(skip_list)
                counts["skipped_size"] += 1
                continue

            os.makedirs(os.path.dirname(target_dir), exist_ok=True)
            url = f"https://github.com/{repo_name}.git"
            success, err = clone_with_retry(url, target_dir, repo_name)

            if not success:
                reason = f"clone failed: {err[:200]}"
                failures.append((repo_name, err))
                skip_list[repo_name] = reason
                save_skip_list(skip_list)
                counts["failed_clone"] += 1
                continue

            cloned_this_run = True
            counts["cloned"] += 1

        # Defensive post-clone observation because GitHub API size is approximate.
        # Once the repo is already cloned, process it and then remove it to recover space.
        actual_mb = dir_size_mb(target_dir)
        if actual_mb > MAX_SIZE_MB:
            progress_write(
                f"\nClone for {repo_name} is {actual_mb:.1f} MB "
                f"(>{MAX_SIZE_MB} MB); processing it before removal."
            )
            counts["oversize_after_clone_seen"] += 1

        snapshot_record, feature_record, process_error = process_cloned_repo(
            row, snapshot_module, feature_module
        )
        if process_error:
            reason = f"processing failed: {process_error}"
            progress_write(f"\nSkipping {repo_name} — {reason}")
            skip_list[repo_name] = reason
            save_skip_list(skip_list)
            counts["failed_processing"] += 1
            if REMOVE_CLONE_AFTER_PROCESS:
                remove_clone_dir(target_dir, repo_name)
                counts["removed_after_processing"] += 1
            continue

        append_record_csv(snapshot_record, SNAPSHOTS_OUTPUT_CSV)
        append_record_csv(feature_record, FEATURES_OUTPUT_CSV)
        processed[repo_name] = {
            "status": "processed",
            "size_mb": round(float(size_mb), 2),
            "cloned_this_run": cloned_this_run,
        }
        save_json_dict(PROCESSED_FILE, processed)
        counts["processed"] += 1

        if REMOVE_CLONE_AFTER_PROCESS:
            remove_clone_dir(target_dir, repo_name)
            counts["removed_after_processing"] += 1

    print("\nDone:")
    for key, value in counts.items():
        print(f"  {key:<24}: {value:,}")

    if failures:
        print("\nFirst clone failures:")
        for name, err in failures[:20]:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
