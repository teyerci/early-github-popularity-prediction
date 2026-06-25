import os
import re
import subprocess
from datetime import datetime

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

CLONE_DIR    = "cloned_repos"
SNAPSHOT_CSV = "repos_snapshots.csv"
OUTPUT_CSV   = "repos_snapshot_features.csv"

WINDOW_LABELS = {"1m", "3m", "6m"}


def scalar_value(value):
    """Return the first non-null scalar from pandas objects or the value itself."""
    if isinstance(value, pd.Series):
        non_null = value.dropna()
        return non_null.iloc[0] if not non_null.empty else None
    if isinstance(value, (list, tuple)):
        return next((item for item in value if item is not None), None)
    return value


def selected_windows(row=None):
    configured = [w.strip() for w in os.getenv("SNAPSHOT_WINDOWS", "1m,3m,6m").split(",") if w.strip()]
    unknown = [w for w in configured if w not in WINDOW_LABELS]
    if unknown:
        raise ValueError(f"Unknown snapshot windows: {unknown}. Valid values: {sorted(WINDOW_LABELS)}")

    if row is None:
        return configured

    return [label for label in configured if f"snap_{label}_commit" in row.index]

# git's empty tree SHA — diffing against this gives total lines in a commit
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"

LINTER_BASENAMES = {
    '.flake8', 'ruff.toml', '.ruff.toml', '.pylintrc', 'pylintrc',
    '.eslintrc', '.eslintrc.js', '.eslintrc.json', '.eslintrc.yml', '.eslintrc.yaml',
    '.prettierrc', '.prettierrc.js', '.prettierrc.json', '.prettierrc.yml',
    'tslint.json', '.rubocop.yml', '.editorconfig', '.stylelintrc',
    '.golangci.yml', '.golangci.yaml', 'golangci.yml',
}

CHANGELOG_BASENAMES = {
    'changelog.md', 'changelog.txt', 'changelog.rst',
    'history.md', 'changes.md', 'changelog',
}

README_BASENAMES = {'readme.md', 'readme.rst', 'readme.txt', 'readme'}

LANG_EXTS = {
    '.py': 'Python',   '.js': 'JavaScript', '.ts': 'TypeScript',
    '.jsx': 'JavaScript', '.tsx': 'TypeScript', '.go': 'Go',
    '.rs': 'Rust',     '.java': 'Java',      '.c': 'C',
    '.cpp': 'C++',     '.cc': 'C++',         '.h': 'C/C++',
    '.rb': 'Ruby',     '.php': 'PHP',        '.cs': 'C#',
    '.swift': 'Swift', '.kt': 'Kotlin',      '.r': 'R',
    '.jl': 'Julia',    '.scala': 'Scala',    '.sh': 'Shell',
}

TEST_RE = re.compile(
    r'(^|/)tests?/'
    r'|_test\.(py|go|rs|java|c|cpp)$'
    r'|\.test\.(js|ts|jsx|tsx)$'
    r'|\.spec\.(js|ts|jsx|tsx)$'
    r'|(^|/)specs?/',
    re.IGNORECASE,
)

BADGE_RE     = re.compile(r'!\[.*?\]\(.*?\)|!\[.*?\]\[.*?\]')
CODE_RE      = re.compile(r'```')
INSTALL_RE   = re.compile(r'pip install|npm install|yarn add|go get|cargo add|gem install|brew install', re.IGNORECASE)
DEMO_RE      = re.compile(r'demo|screenshot|\.gif|live preview|playground|try it', re.IGNORECASE)
LINK_RE      = re.compile(r'https?://')


# ── git helpers ───────────────────────────────────────────────────────────────

def git(args, cwd, timeout=30):
    try:
        r = subprocess.run(
            ['git'] + args, cwd=cwd,
            capture_output=True, text=True, timeout=timeout,
            encoding='utf-8', errors='replace',   # tolerate non-UTF-8 bytes in git output
        )
        return r.stdout if r.returncode == 0 else None
    except subprocess.TimeoutExpired:
        return None


def get_file_tree(repo_path, commit):
    """Return list of (path, size_bytes) for every blob at commit."""
    out = git(['ls-tree', '-r', '-l', commit], repo_path)
    if not out:
        return []
    result = []
    for line in out.splitlines():
        parts = line.split(None, 4)
        if len(parts) == 5 and parts[1] == 'blob':
            size = int(parts[3]) if parts[3].isdigit() else 0
            result.append((parts[4], size))
    return result


def get_log_entries(repo_path, commit):
    """Return list of {date, email, is_merge, msg_len} for all commits up to commit."""
    out = git(
        ['log', commit, '--format=%ad|%ae|%P|%s', '--date=short'],
        repo_path, timeout=60,
    )
    if not out:
        return []
    entries = []
    for line in out.splitlines():
        parts = line.split('|', 3)
        if len(parts) < 4:
            continue
        date_str, email, parents, subject = parts
        entries.append({
            'date':     date_str.strip(),
            'email':    email.strip().lower(),
            'is_merge': ' ' in parents.strip(),
            'msg_len':  len(subject.strip()),
        })
    return entries


def get_file_content(repo_path, commit, path):
    return git(['show', f'{commit}:{path}'], repo_path, timeout=10)


# ── feature extractors ────────────────────────────────────────────────────────

def extract_structure(files):
    paths  = [p for p, _ in files]
    sizes  = [s for _, s in files]
    bases  = {os.path.basename(p).lower() for p in paths}
    dirs   = {os.path.dirname(p) for p in paths if os.path.dirname(p)}
    tops   = {p.split('/')[0] for p in paths if '/' in p}
    exts   = [os.path.splitext(p)[1].lower() for p in paths]
    langs  = {LANG_EXTS[e] for e in exts if e in LANG_EXTS}
    tests  = [p for p in paths if TEST_RE.search(p)]

    return {
        'file_count':           len(files),
        'dir_count':            len(dirs),
        'total_size_bytes':     sum(sizes),
        'avg_file_size_bytes':  round(sum(sizes) / len(sizes), 1) if sizes else 0,
        'max_file_size_bytes':  max(sizes) if sizes else 0,
        'lang_count':           len(langs),
        'test_file_count':      len(tests),
        'test_ratio':           round(len(tests) / len(files), 4) if files else 0,
        'has_src_or_lib':       int('src' in tops or 'lib' in tops),
        'has_docs_dir':         int('docs' in tops or 'doc' in tops),
        'has_changelog':        int(bool(bases & CHANGELOG_BASENAMES)),
        'has_contributing':     int(any(b.startswith('contributing') for b in bases)),
        'has_pyproject':        int('pyproject.toml' in bases),
        'has_package_json':     int('package.json' in bases),
        'has_makefile':         int('makefile' in bases),
        'has_docker_compose':   int(any('docker-compose' in p for p in paths)),
        'has_linter_config':    int(bool(bases & LINTER_BASENAMES)),
        'has_github_actions':   int(any('.github/workflows' in p for p in paths)),
    }


def extract_activity(log_entries, start_date_str):
    empty = {
        'unique_authors': 0, 'commit_count': 0, 'merge_ratio': 0.0,
        'avg_commit_msg_len': 0.0, 'commits_per_week': 0.0,
        'avg_days_between_commits': None, 'active_days': 0,
    }
    if not log_entries:
        return empty

    try:
        start_date = pd.to_datetime(start_date_str, utc=True).date()
    except Exception:
        start_date = None

    if start_date is not None:
        filtered = []
        for entry in log_entries:
            try:
                entry_date = datetime.strptime(entry['date'], '%Y-%m-%d').date()
            except ValueError:
                continue
            if entry_date >= start_date:
                filtered.append(entry)
        log_entries = filtered

    if not log_entries:
        return empty

    total   = len(log_entries)
    merges  = sum(1 for e in log_entries if e['is_merge'])
    authors = {e['email'] for e in log_entries}
    avg_msg = sum(e['msg_len'] for e in log_entries) / total

    dates = []
    for e in log_entries:
        try:
            dates.append(datetime.strptime(e['date'], '%Y-%m-%d').date())
        except ValueError:
            pass

    if dates:
        earliest, latest = min(dates), max(dates)
        elapsed_days = max((latest - earliest).days, 1)
    else:
        elapsed_days = 1

    commits_per_week = total / (elapsed_days / 7) if elapsed_days >= 7 else float(total)

    sorted_dates = sorted(dates)
    if len(sorted_dates) > 1:
        gaps = [(sorted_dates[i + 1] - sorted_dates[i]).days
                for i in range(len(sorted_dates) - 1)]
        avg_gap = round(sum(gaps) / len(gaps), 1)
    else:
        avg_gap = None

    return {
        'unique_authors':           len(authors),
        'commit_count':             total,
        'merge_ratio':              round(merges / total, 4),
        'avg_commit_msg_len':       round(avg_msg, 1),
        'commits_per_week':         round(commits_per_week, 2),
        'avg_days_between_commits': avg_gap,
        'active_days':              len(set(dates)),
    }


def extract_readme(content):
    if not content:
        return {
            'readme_word_count': 0,    'readme_has_badges': 0,
            'readme_has_install': 0,   'readme_has_code_blocks': 0,
            'readme_has_demo': 0,      'readme_has_links': 0,
        }
    return {
        'readme_word_count':      len(content.split()),
        'readme_has_badges':      int(bool(BADGE_RE.search(content))),
        'readme_has_install':     int(bool(INSTALL_RE.search(content))),
        'readme_has_code_blocks': int(bool(CODE_RE.search(content))),
        'readme_has_demo':        int(bool(DEMO_RE.search(content))),
        'readme_has_links':       int(bool(LINK_RE.search(content))),
    }


def extract_code_quality(repo_path, commit, files):
    # Total lines: diff entire commit against the empty tree
    total_lines = None
    diff_out = git(['diff', '--shortstat', EMPTY_TREE, commit], repo_path, timeout=30)
    if diff_out:
        m = re.search(r'(\d+) insertion', diff_out)
        if m:
            total_lines = int(m.group(1))

    # Files containing TODO/FIXME/HACK/XXX
    todo_out = git(
        ['grep', '-i', '-l', '-E', 'TODO|FIXME|HACK|XXX', commit],
        repo_path, timeout=20,
    )
    todo_file_count = len(todo_out.splitlines()) if todo_out else 0

    # Python type hints (files with return annotations or typed args)
    py_files = [p for p, _ in files if p.endswith('.py')]
    has_type_hints = 0
    if py_files:
        hints_out = git(
            ['grep', '-i', '-l', '-E', r'def .+->.+:|:\s+[A-Z][a-zA-Z]+', commit, '--', '*.py'],
            repo_path, timeout=20,
        )
        has_type_hints = int(bool(hints_out and hints_out.strip()))

    return {
        'total_lines':     total_lines,
        'todo_file_count': todo_file_count,
        'has_type_hints':  has_type_hints,
    }


# ── snapshot orchestration ────────────────────────────────────────────────────

def analyze_snapshot(repo_path, commit, start_date_str, label):
    """Run all extractors for one snapshot; return dict with label-prefixed keys."""
    if not commit:
        return {}

    files   = get_file_tree(repo_path, commit)
    log     = get_log_entries(repo_path, commit)

    readme_path    = next((p for p, _ in files if os.path.basename(p).lower() in README_BASENAMES), None)
    readme_content = get_file_content(repo_path, commit, readme_path) if readme_path else None

    combined = {
        **extract_structure(files),
        **extract_activity(log, start_date_str),
        **extract_readme(readme_content),
        **extract_code_quality(repo_path, commit, files),
    }

    prefix = f'{label}_'
    return {f'{prefix}{k}': v for k, v in combined.items()}


def analyze_repo(row):
    repo_name = scalar_value(row['repo'])
    if not repo_name or "/" not in str(repo_name):
        raise ValueError(f"invalid repo name: {repo_name!r}")
    repo_name = str(repo_name)

    owner, repo = repo_name.split('/', 1)
    repo_path   = os.path.join(CLONE_DIR, owner, repo)

    if not os.path.isdir(repo_path):
        return None

    start_date = row['start_date']
    result = {
        'repo': repo_name,
        'start_date': start_date,
    }
    for col in [
        'start_date_source',
        'github_created_at',
        'git_start_date',
        'git_start_condition',
        'git_start_minus_created_days',
    ]:
        if col in row.index:
            result[col] = row[col]

    for label in selected_windows(row):
        col = f'snap_{label}_commit'
        commit = row[col] if col in row.index and pd.notna(row[col]) else None
        result.update(analyze_snapshot(repo_path, commit, start_date, label))

    return result


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    snaps = pd.read_csv(SNAPSHOT_CSV)
    print(f"Repos to analyze: {len(snaps)}")

    results, errors = [], []

    for _, row in tqdm(snaps.iterrows(), total=len(snaps), desc='Extracting features'):
        try:
            rec = analyze_repo(row)
            if rec:
                results.append(rec)
            else:
                errors.append((row['repo'], 'not cloned'))
        except Exception as e:
            errors.append((row['repo'], str(e)))

    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f'\nSaved {len(results)} repos → {OUTPUT_CSV}')
    print(f'Errors: {len(errors)}')
    if errors:
        for name, err in errors[:10]:
            print(f'  {name}: {err}')


if __name__ == '__main__':
    main()
