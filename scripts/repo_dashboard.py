from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPO_DEFAULT = Path(__file__).resolve().parents[1]
IMPORTANT_DIRS = [
    ("agent", "LLM agents, prompts, and orchestration logic"),
    ("baseline", "Baseline implementations and comparators"),
    ("common", "Shared utilities and helpers"),
    ("config", "Runtime config loader and defaults"),
    ("configs", "YAML selector / experiment configs"),
    ("datasets", "Dataset abstractions and loaders"),
    ("docs", "Research notes, summaries, and plans"),
    ("eval_metrics", "Metric implementations and evaluation logic"),
    ("experiments", "Run artifacts, traces, and outputs"),
    ("results", "Curated outputs and aggregation tables"),
    ("runtime", "Training engine, benchmark runner, server"),
    ("scripts", "CLI entry points and helpers"),
    ("segment_selection", "Evidence selection and scoring"),
    ("selector", "Performance selector / ablation tools"),
    ("tests", "Regression and behavior tests"),
]
IMPORTANT_ROOT_MARKDOWN = [
    "README.md",
    "RESULTS_AND_INTERPRETATION.md",
    "work_brief_argos_curation_fixes.md",
]


@dataclass
class FileItem:
    path: Path
    updated: datetime
    title: str
    summary: str


@dataclass
class RunItem:
    path: Path
    updated: datetime
    group: str
    dataset: str
    chunk_size: str
    status: str
    status_tone: str
    notes: str
    meta: dict


@dataclass
class DirItem:
    name: str
    rel: str
    file_count: int
    updated: datetime | None
    modified_files: int
    note: str
    highlight: str


# -------------------------
# Utilities
# -------------------------

def run_git(repo: Path, args: list[str]) -> str:
    try:
        res = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return res.stdout
    except Exception:
        return ""


def format_dt(dt: datetime | None) -> str:
    if dt is None:
        return "—"
    return dt.astimezone().strftime("%Y-%m-%d %H:%M")


def human_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def normalize_search(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def file_mtime(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def count_files(path: Path) -> int:
    total = 0
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d != ".git" and d != "__pycache__"]
        total += len(files)
    return total


def latest_mtime(path: Path) -> datetime | None:
    latest: float | None = None
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d != ".git" and d != "__pycache__"]
        for name in files:
            full = Path(root) / name
            try:
                mtime = full.stat().st_mtime
            except OSError:
                continue
            if latest is None or mtime > latest:
                latest = mtime
    if latest is None:
        return None
    return datetime.fromtimestamp(latest, tz=timezone.utc)


def read_text(path: Path, limit: int | None = None) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    if limit is not None:
        return text[:limit]
    return text


def extract_title_and_summary(path: Path) -> tuple[str, str]:
    text = read_text(path, limit=12_000)
    lines = [line.rstrip() for line in text.splitlines()]
    title = path.stem.replace("-", " ").replace("_", " ").title()
    summary_lines: list[str] = []
    in_first_section = False

    for line in lines:
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            if title == path.stem.replace("-", " ").replace("_", " ").title():
                title = line.lstrip("# ").strip() or title
            if in_first_section:
                break
            in_first_section = True
            continue
        if in_first_section:
            clean = re.sub(r"^[-*]\s+", "", line).strip()
            clean = re.sub(r"^\d+[.)]\s+", "", clean)
            if clean:
                summary_lines.append(clean)
            if len(summary_lines) >= 2:
                break

    if not summary_lines:
        for line in lines:
            clean = line.strip()
            if clean and not clean.startswith("#"):
                summary_lines.append(re.sub(r"^[-*]\s+", "", clean))
            if len(summary_lines) >= 2:
                break

    summary = " ".join(summary_lines).strip()
    if len(summary) > 190:
        summary = summary[:187].rstrip() + "…"
    return title, summary or "No summary captured from the file body."


def git_status_items(repo: Path) -> list[tuple[str, str]]:
    raw = run_git(repo, ["status", "--short"]).splitlines()
    items: list[tuple[str, str]] = []
    for line in raw:
        if len(line) < 4:
            continue
        code = line[:2]
        path = line[3:].strip()
        items.append((code, path))
    return items


def top_level_prefix(path: str) -> str:
    return Path(path).parts[0] if Path(path).parts else path


def collect_dir_items(repo: Path, git_changes: list[tuple[str, str]]) -> list[DirItem]:
    modified_counts = Counter(top_level_prefix(path) for _, path in git_changes)
    items: list[DirItem] = []
    for name, note in IMPORTANT_DIRS:
        path = repo / name
        if not path.exists():
            continue
        file_count = count_files(path)
        updated = latest_mtime(path)
        modified = modified_counts.get(name, 0)
        highlight = "changed" if modified else ("large" if file_count >= 200 else "stable")
        items.append(
            DirItem(
                name=name,
                rel=name,
                file_count=file_count,
                updated=updated,
                modified_files=modified,
                note=note,
                highlight=highlight,
            )
        )
    return items


def collect_recent_files(paths: Iterable[Path], limit: int = 6) -> list[FileItem]:
    items: list[FileItem] = []
    for path in paths:
        if not path.exists():
            continue
        title, summary = extract_title_and_summary(path)
        items.append(
            FileItem(
                path=path,
                updated=file_mtime(path),
                title=title,
                summary=summary,
            )
        )
    items.sort(key=lambda item: item.updated, reverse=True)
    return items[:limit]


def detect_run_status(run_dir: Path) -> tuple[str, str, str]:
    best_rule_path = run_dir / "best_rule_path.txt"
    log_candidates = []
    for name in ["output.log", *sorted(p.name for p in run_dir.glob("driver_stdout_attempt_*.log"))]:
        p = run_dir / name
        if p.exists():
            log_candidates.append(p)

    merged = "\n".join(read_text(p, limit=40_000) for p in log_candidates)
    compact = merged.lower()

    if "traceback" in compact or "calledprocesserror" in compact:
        return "blocked", "danger", "Execution halted by an exception or failed subprocess."
    if best_rule_path.exists():
        return "done", "success", "Final rule artifact exists."
    if "all_done" in compact or "[done]" in compact:
        return "done", "success", "Wrapper signaled completion."
    if "initialized llm" in compact or "start to train rules" in compact:
        return "running", "accent", "Active run with progress logs present."
    if log_candidates:
        return "queued", "muted", "Run directory exists, but no clear completion marker yet."
    return "missing", "muted", "No runtime logs found."


def collect_run_items(repo: Path, limit: int = 6) -> list[RunItem]:
    run_dirs: list[Path] = []
    for meta in repo.glob("experiments/**/metadata.json"):
        run_dirs.append(meta.parent)
    items: list[RunItem] = []
    for run_dir in run_dirs:
        meta_path = run_dir / "metadata.json"
        meta = {}
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        status, tone, notes = detect_run_status(run_dir)
        updated = latest_mtime(run_dir) or file_mtime(meta_path)
        dataset = str(meta.get("dataset") or Path(str(meta.get("dataset_path", ""))).stem or run_dir.parent.parent.name)
        chunk_size = str(meta.get("chunk_size") or run_dir.parent.name.replace("chunk_", ""))
        group = run_dir.parent.parent.parent.name if run_dir.parent.parent.parent != repo else run_dir.parent.parent.name
        items.append(
            RunItem(
                path=run_dir,
                updated=updated,
                group=group,
                dataset=dataset,
                chunk_size=chunk_size,
                status=status,
                status_tone=tone,
                notes=notes,
                meta=meta,
            )
        )
    items.sort(key=lambda item: item.updated, reverse=True)
    return items[:limit]


def collect_git_summary(repo: Path) -> dict:
    branch = run_git(repo, ["status", "--short", "--branch"]).splitlines()
    branch_line = branch[0] if branch else ""
    status_items = git_status_items(repo)
    dirty = len(status_items)
    modified_dirs = Counter(top_level_prefix(path) for _, path in status_items)
    recent_commits = run_git(repo, ["log", "-5", "--date=short", "--pretty=format:%h|%ad|%s"]).splitlines()
    return {
        "branch_line": branch_line,
        "dirty": dirty,
        "modified_dirs": modified_dirs,
        "recent_commits": recent_commits,
        "status_items": status_items,
    }


def summarize_modifications(status_items: list[tuple[str, str]]) -> list[dict[str, str]]:
    rows = []
    for code, path in status_items[:10]:
        kind = "untracked" if code == "??" else "modified"
        rows.append(
            {
                "code": code,
                "kind": kind,
                "path": path,
                "scope": top_level_prefix(path),
            }
        )
    return rows


def render_badge(text: str, tone: str = "neutral") -> str:
    return f'<span class="badge badge-{tone}">{html.escape(text)}</span>'


def render_dashboard(repo: Path) -> str:
    git = collect_git_summary(repo)
    status_items = git["status_items"]
    dir_items = collect_dir_items(repo, status_items)

    doc_candidates = [
        *(repo / "docs").glob("*.md"),
        *(repo.glob("*.md")),
    ]
    # avoid README duplication and keep the most recent, high-signal notes first
    doc_candidates = [p for p in doc_candidates if p.name not in {"README.md"}]
    recent_docs = collect_recent_files(doc_candidates, limit=8)
    recent_runs = collect_run_items(repo, limit=6)
    modifications = summarize_modifications(status_items)
    modified_dir_counts = git["modified_dirs"]
    recent_commit_lines = git["recent_commits"]

    updated_overall = max(
        [item.updated for item in dir_items if item.updated] + [item.updated for item in recent_docs] + [item.updated for item in recent_runs],
        default=datetime.now(timezone.utc),
    )

    total_files = sum(item.file_count for item in dir_items)
    dirs_changed = sum(1 for item in dir_items if item.modified_files)
    active_runs = sum(1 for item in recent_runs if item.status == "running")

    summary_cards = [
        ("Branch", git["branch_line"].replace("##", "").strip() or "unknown", "quiet"),
        ("Dirty files", str(git["dirty"]), "warn" if git["dirty"] else "ok"),
        ("Modified areas", str(dirs_changed), "accent" if dirs_changed else "quiet"),
        ("Active runs", str(active_runs), "accent" if active_runs else "quiet"),
    ]

    def card_label(item: DirItem) -> str:
        tone = "accent" if item.modified_files else ("ok" if item.highlight == "stable" else "quiet")
        mods = f" · {item.modified_files} changed" if item.modified_files else ""
        return f"<article class='dir-card filterable' data-search='{html.escape(normalize_search(f'{item.name} {item.note} {item.file_count} {item.modified_files}'))}'>"
        
    html_parts: list[str] = []
    html_parts.append("<!doctype html>")
    html_parts.append("<html lang='ko'>")
    html_parts.append("<head>")
    html_parts.append("<meta charset='utf-8'>")
    html_parts.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    html_parts.append("<title>Argos Repo Dashboard</title>")
    html_parts.append(
        "<link href='https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap' rel='stylesheet'>"
    )
    html_parts.append("<style>")
    html_parts.append(
        """
        :root {
          color-scheme: dark;
          --bg: #08090a;
          --panel: #0f1011;
          --surface: #191a1b;
          --surface-2: #212225;
          --line: rgba(255,255,255,.08);
          --line-soft: rgba(255,255,255,.05);
          --text: #f7f8f8;
          --text-2: #d0d6e0;
          --muted: #8a8f98;
          --quiet: #62666d;
          --accent: #7170ff;
          --accent-2: #828fff;
          --success: #27a644;
          --warning: #f5a524;
          --danger: #ef6b6b;
          --mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace;
          --sans: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
        }
        * { box-sizing: border-box; }
        html, body { min-height: 100%; }
        body {
          margin: 0;
          background:
            radial-gradient(circle at top left, rgba(113,112,255,.12), transparent 30%),
            radial-gradient(circle at 80% 0%, rgba(39,166,68,.08), transparent 18%),
            var(--bg);
          color: var(--text);
          font-family: var(--sans);
          font-feature-settings: 'cv01' 1, 'ss03' 1;
          -webkit-font-smoothing: antialiased;
          text-rendering: geometricPrecision;
        }
        a { color: inherit; text-decoration: none; }
        .shell {
          max-width: 1500px;
          margin: 0 auto;
          padding: 28px 24px 40px;
        }
        .hero {
          display: grid;
          gap: 18px;
          grid-template-columns: 1.35fr .65fr;
          align-items: stretch;
          margin-bottom: 18px;
        }
        .panel {
          background: rgba(255,255,255,.03);
          border: 1px solid var(--line);
          box-shadow: 0 0 0 1px rgba(0,0,0,.15), 0 16px 40px rgba(0,0,0,.25);
          border-radius: 18px;
        }
        .hero-main {
          padding: 28px;
          position: relative;
          overflow: hidden;
        }
        .hero-main::after {
          content: '';
          position: absolute;
          inset: -2px;
          background: linear-gradient(120deg, transparent, rgba(113,112,255,.06), transparent 60%);
          pointer-events: none;
        }
        .eyebrow {
          display: inline-flex;
          gap: 8px;
          align-items: center;
          color: var(--muted);
          font-size: 12px;
          letter-spacing: .08em;
          text-transform: uppercase;
          margin-bottom: 14px;
        }
        .title {
          margin: 0;
          font-size: clamp(34px, 5vw, 64px);
          line-height: .94;
          letter-spacing: -0.05em;
          font-weight: 600;
          max-width: 12ch;
        }
        .subtitle {
          margin: 14px 0 0;
          max-width: 70ch;
          color: var(--text-2);
          font-size: 15px;
          line-height: 1.7;
        }
        .hero-side {
          display: grid;
          gap: 12px;
        }
        .summary-grid {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 12px;
        }
        .summary-card, .mini-panel, .dir-card, .doc-card, .run-card, .commit-card {
          background: rgba(255,255,255,.025);
          border: 1px solid var(--line-soft);
          border-radius: 16px;
        }
        .summary-card {
          padding: 14px 14px 16px;
          min-height: 108px;
        }
        .summary-label {
          color: var(--muted);
          font-size: 12px;
          margin-bottom: 10px;
          text-transform: uppercase;
          letter-spacing: .08em;
        }
        .summary-value {
          font-size: 22px;
          line-height: 1.1;
          font-weight: 600;
          letter-spacing: -0.04em;
        }
        .summary-meta {
          margin-top: 8px;
          color: var(--quiet);
          font-size: 12px;
        }
        .mini-panel {
          padding: 14px;
        }
        .mini-title {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 12px;
          margin-bottom: 12px;
        }
        .mini-title h2, .section-head h2 {
          margin: 0;
          font-size: 15px;
          font-weight: 600;
          letter-spacing: -0.02em;
        }
        .mini-title span, .section-head span {
          color: var(--muted);
          font-size: 12px;
        }
        .searchbar {
          display: flex;
          align-items: center;
          gap: 10px;
          padding: 14px 16px;
          margin: 18px 0;
          background: rgba(255,255,255,.022);
          border: 1px solid var(--line);
          border-radius: 16px;
        }
        .searchbar input {
          width: 100%;
          background: transparent;
          border: 0;
          outline: none;
          color: var(--text);
          font: inherit;
          font-size: 15px;
        }
        .searchbar input::placeholder { color: var(--quiet); }
        .chip-row {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
          margin-top: 10px;
        }
        .badge {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          padding: 6px 10px;
          border-radius: 999px;
          border: 1px solid var(--line-soft);
          color: var(--text-2);
          font-size: 12px;
          line-height: 1;
        }
        .badge-accent { background: rgba(113,112,255,.11); color: #d9dbff; border-color: rgba(113,112,255,.35); }
        .badge-ok { background: rgba(39,166,68,.12); color: #d7f7df; border-color: rgba(39,166,68,.35); }
        .badge-warn { background: rgba(245,165,36,.12); color: #ffe7ba; border-color: rgba(245,165,36,.35); }
        .badge-danger { background: rgba(239,107,107,.12); color: #ffd5d5; border-color: rgba(239,107,107,.35); }
        .badge-quiet { background: rgba(255,255,255,.02); color: var(--text-2); }
        .section {
          margin-top: 18px;
          padding: 18px;
        }
        .section-head {
          display: flex;
          align-items: baseline;
          justify-content: space-between;
          gap: 12px;
          margin-bottom: 14px;
        }
        .section-head .section-sub {
          color: var(--muted);
          font-size: 12px;
        }
        .grid-structure {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 12px;
        }
        .dir-card {
          padding: 14px;
          min-height: 150px;
          position: relative;
          overflow: hidden;
        }
        .dir-card::before {
          content: '';
          position: absolute;
          inset: 0 auto auto 0;
          width: 100%;
          height: 3px;
          background: linear-gradient(90deg, var(--accent), rgba(113,112,255,.15));
          opacity: .35;
        }
        .dir-top {
          display: flex;
          justify-content: space-between;
          gap: 12px;
          align-items: flex-start;
        }
        .dir-name {
          font-size: 18px;
          font-weight: 600;
          letter-spacing: -0.03em;
          margin: 0;
        }
        .dir-note {
          margin-top: 6px;
          color: var(--text-2);
          font-size: 13px;
          line-height: 1.55;
        }
        .dir-meta {
          display: flex;
          gap: 8px;
          flex-wrap: wrap;
          margin-top: 12px;
        }
        .dir-stat {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          padding: 6px 10px;
          border-radius: 999px;
          background: rgba(255,255,255,.02);
          border: 1px solid var(--line-soft);
          color: var(--text-2);
          font-size: 12px;
        }
        .progress {
          margin-top: 12px;
          height: 7px;
          border-radius: 999px;
          background: rgba(255,255,255,.05);
          overflow: hidden;
        }
        .progress > span {
          display: block;
          height: 100%;
          border-radius: inherit;
          background: linear-gradient(90deg, var(--accent), var(--accent-2));
        }
        .updates-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 12px;
        }
        .stack {
          display: grid;
          gap: 10px;
        }
        .doc-card, .run-card, .commit-card {
          padding: 14px;
        }
        .doc-head, .run-head, .commit-head {
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          gap: 12px;
        }
        .doc-title, .run-title {
          margin: 0;
          font-size: 15px;
          font-weight: 600;
          line-height: 1.35;
        }
        .doc-path, .run-path, .commit-path {
          margin-top: 4px;
          color: var(--muted);
          font-size: 12px;
          word-break: break-all;
        }
        .doc-summary, .run-summary, .commit-summary {
          margin-top: 10px;
          color: var(--text-2);
          font-size: 13px;
          line-height: 1.6;
        }
        .mono { font-family: var(--mono); }
        .run-kpis {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
          margin-top: 10px;
        }
        .run-grid {
          display: grid;
          gap: 12px;
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .footer {
          margin-top: 20px;
          color: var(--muted);
          font-size: 12px;
          display: flex;
          justify-content: space-between;
          gap: 12px;
          flex-wrap: wrap;
          padding: 0 4px;
        }
        @media (max-width: 1180px) {
          .hero, .updates-grid, .run-grid, .grid-structure { grid-template-columns: 1fr; }
        }
        @media (max-width: 720px) {
          .shell { padding: 16px; }
          .hero-main, .section { padding: 16px; }
          .summary-grid { grid-template-columns: 1fr; }
          .title { max-width: unset; }
        }
        """
    )
    html_parts.append("</style>")
    html_parts.append("</head>")
    html_parts.append("<body>")
    html_parts.append("<div class='shell'>")
    html_parts.append("<div class='hero'>")
    html_parts.append("<section class='panel hero-main'>")
    html_parts.append("<div class='eyebrow'><span class='badge badge-accent'>LIVE SNAPSHOT</span><span>Argos repository dashboard</span></div>")
    html_parts.append("<h1 class='title'>Code structure, updates, and experiment pulse — at a glance.</h1>")
    html_parts.append(
        "<p class='subtitle'>This dashboard is generated from the current repo state. It surfaces the architecture map, recently changed files, fresh notes, and the latest experiment runs so you can keep orientation while other work is still running in the background.</p>"
    )
    html_parts.append(
        f"<div class='chip-row'>"
        f"{render_badge('Repo: argos', 'quiet')}"
        f"{render_badge(f'Total tracked area: {human_count(total_files)} files', 'quiet')}"
        f"{render_badge(f'Last refresh: {format_dt(updated_overall)}', 'quiet')}"
        f"{render_badge('Linear-style dark UI', 'accent')}"
        f"</div>"
    )
    html_parts.append("</section>")
    html_parts.append("<aside class='hero-side'>")
    html_parts.append("<div class='panel mini-panel'>")
    html_parts.append("<div class='mini-title'><h2>Repository pulse</h2><span>current state</span></div>")
    html_parts.append("<div class='summary-grid'>")
    for label, value, tone in summary_cards:
        html_parts.append(
            f"<div class='summary-card'>"
            f"<div class='summary-label'>{html.escape(label)}</div>"
            f"<div class='summary-value'>{html.escape(value)}</div>"
            f"<div class='summary-meta'>{render_badge('status', tone)}</div>"
            f"</div>"
        )
    html_parts.append("</div>")
    html_parts.append("</div>")
    html_parts.append("<div class='panel mini-panel'>")
    html_parts.append("<div class='mini-title'><h2>Git change radar</h2><span>top scopes</span></div>")
    if modified_dir_counts:
        radar = ", ".join(f"{name}×{count}" for name, count in modified_dir_counts.most_common(6))
        html_parts.append(f"<div class='subtitle' style='margin:0;color:var(--text-2)'>{html.escape(radar)}</div>")
    else:
        html_parts.append("<div class='subtitle' style='margin:0;color:var(--text-2)'>No current git changes detected.</div>")
    html_parts.append("</div>")
    html_parts.append("</aside>")
    html_parts.append("</div>")

    html_parts.append("<div class='searchbar panel'>")
    html_parts.append("<span class='badge badge-quiet'>/</span>")
    html_parts.append("<input id='filter' type='text' placeholder='Filter structure, updates, runs, or file paths…'>")
    html_parts.append("<span class='badge badge-quiet'>Cmd/Ctrl + K vibe</span>")
    html_parts.append("</div>")

    # Structure section
    html_parts.append("<section class='panel section'>")
    html_parts.append("<div class='section-head'><h2>Structure map</h2><span class='section-sub'>Where the repo lives, how big each area is, and which parts are moving.</span></div>")
    html_parts.append("<div class='grid-structure'>")
    for item in dir_items:
        width = min(100, max(8, int(item.file_count ** 0.5 * 4)))
        tone = "accent" if item.modified_files else ("ok" if item.highlight == "stable" else "quiet")
        search = normalize_search(f"{item.name} {item.note} {item.file_count} {item.modified_files} {item.rel}")
        html_parts.append(
            f"<article class='dir-card filterable' data-search='{html.escape(search)}'>"
            f"<div class='dir-top'><div><h3 class='dir-name'>{html.escape(item.name)}</h3><div class='dir-note'>{html.escape(item.note)}</div></div>{render_badge(item.highlight, tone)}</div>"
            f"<div class='dir-meta'>"
            f"<span class='dir-stat mono'>{human_count(item.file_count)} files</span>"
            f"<span class='dir-stat mono'>{format_dt(item.updated)}</span>"
            f"<span class='dir-stat mono'>{item.modified_files} touched</span>"
            f"</div>"
            f"<div class='progress' aria-hidden='true'><span style='width:{width}%' ></span></div>"
            f"</article>"
        )
    html_parts.append("</div>")
    html_parts.append("</section>")

    # Updates section
    html_parts.append("<section class='panel section'>")
    html_parts.append("<div class='section-head'><h2>Recent updates</h2><span class='section-sub'>Markdown notes, repo docs, and current code churn.</span></div>")
    html_parts.append("<div class='updates-grid'>")

    html_parts.append("<div class='stack'>")
    for doc in recent_docs:
        search = normalize_search(f"{doc.title} {doc.path.name} {doc.path} {doc.summary}")
        html_parts.append(
            f"<article class='doc-card filterable' data-search='{html.escape(search)}'>"
            f"<div class='doc-head'>"
            f"<div><h3 class='doc-title'>{html.escape(doc.title)}</h3><div class='doc-path mono'>{html.escape(str(doc.path.relative_to(repo)))}</div></div>"
            f"{render_badge(format_dt(doc.updated), 'quiet')}"
            f"</div>"
            f"<div class='doc-summary'>{html.escape(doc.summary)}</div>"
            f"</article>"
        )
    html_parts.append("</div>")

    html_parts.append("<div class='stack'>")
    for item in modifications:
        search = normalize_search(f"{item['path']} {item['kind']} {item['scope']}")
        tone = "accent" if item["code"] != "??" else "warn"
        html_parts.append(
            f"<article class='commit-card filterable' data-search='{html.escape(search)}'>"
            f"<div class='commit-head'>"
            f"<div><h3 class='doc-title'>{html.escape(item['path'])}</h3><div class='commit-path mono'>{html.escape(item['scope'])}</div></div>"
            f"{render_badge(item['code'], tone)}"
            f"</div>"
            f"<div class='commit-summary'>{html.escape('Tracked as ' + item['kind'] + ' change in the current branch state.')}</div>"
            f"</article>"
        )
    if not modifications:
        html_parts.append(
            "<article class='commit-card'><div class='commit-summary'>No current uncommitted changes in git status.</div></article>"
        )
    html_parts.append("</div>")
    html_parts.append("</div>")
    html_parts.append("</section>")

    # Runs section
    html_parts.append("<section class='panel section'>")
    html_parts.append("<div class='section-head'><h2>Experiment pulse</h2><span class='section-sub'>Most recent run directories and the state they are in.</span></div>")
    html_parts.append("<div class='run-grid'>")
    for run in recent_runs:
        meta = run.meta
        split_stats = meta.get("split_stats", {})
        test = split_stats.get("test", {}) if isinstance(split_stats, dict) else {}
        flags = split_stats.get("flags", {}) if isinstance(split_stats, dict) else {}
        test_points = test.get("anomaly_point_count", None)
        test_events = test.get("anomaly_event_count", None)
        flag_badges = []
        for key, value in flags.items():
            flag_badges.append(render_badge(f"{key}: {str(value).lower() if isinstance(value, bool) else value}", "warn" if value else "quiet"))
        if not flag_badges:
            flag_badges.append(render_badge('no flags', 'quiet'))
        search = normalize_search(
            f"{run.group} {run.dataset} {run.chunk_size} {run.status} {run.notes} {run.path} {json.dumps(meta, ensure_ascii=False)}"
        )
        html_parts.append(
            f"<article class='run-card filterable' data-search='{html.escape(search)}'>"
            f"<div class='run-head'>"
            f"<div><h3 class='run-title'>{html.escape(run.dataset)} · chunk {html.escape(run.chunk_size)}</h3>"
            f"<div class='run-path mono'>{html.escape(str(run.path.relative_to(repo)))}</div></div>"
            f"{render_badge(run.status, run.status_tone)}"
            f"</div>"
            f"<div class='run-summary'>"
            f"<div>Mode: <span class='mono'>{html.escape(str(meta.get('mode', '—')))}</span> · Provider: <span class='mono'>{html.escape(str(meta.get('llm_provider', '—')))}</span></div>"
            f"<div>Selector: <span class='mono'>{html.escape(str(meta.get('segment_selection_mode', '—')))}</span> · Seed: <span class='mono'>{html.escape(str(meta.get('seed', '—')))}</span></div>"
            f"<div>Test split: <span class='mono'>{html.escape(str(test_points if test_points is not None else '—'))}</span> anomaly points · <span class='mono'>{html.escape(str(test_events if test_events is not None else '—'))}</span> anomaly events</div>"
            f"</div>"
            f"<div class='run-kpis'>"
            f"{''.join(flag_badges)}"
            f"{render_badge(format_dt(run.updated), 'quiet')}"
            f"{render_badge(run.notes, 'quiet')}"
            f"</div>"
            f"</article>"
        )
    html_parts.append("</div>")
    html_parts.append("</section>")

    # Recent commits section
    html_parts.append("<section class='panel section'>")
    html_parts.append("<div class='section-head'><h2>Recent commits</h2><span class='section-sub'>Latest history from git log.</span></div>")
    html_parts.append("<div class='stack'>")
    if recent_commit_lines:
        for line in recent_commit_lines:
            try:
                short, date, subject = line.split("|", 2)
            except ValueError:
                short, date, subject = "—", "—", line
            html_parts.append(
                "<article class='commit-card'>"
                f"<div class='commit-head'><div><h3 class='doc-title mono'>{html.escape(short)}</h3><div class='commit-path mono'>{html.escape(date)}</div></div>{render_badge('commit', 'quiet')}</div>"
                f"<div class='commit-summary'>{html.escape(subject)}</div>"
                "</article>"
            )
    else:
        html_parts.append("<article class='commit-card'><div class='commit-summary'>No commit history available from git log.</div></article>")
    html_parts.append("</div>")
    html_parts.append("</section>")

    html_parts.append(
        "<div class='footer'>"
        f"<div>Generated from <span class='mono'>{html.escape(str(repo))}</span></div>"
        f"<div>Refresh time: <span class='mono'>{html.escape(format_dt(updated_overall))}</span></div>"
        "</div>"
    )
    html_parts.append("</div>")

    html_parts.append(
        "<script>\n"
        "const input = document.getElementById('filter');\n"
        "const cards = [...document.querySelectorAll('.filterable')];\n"
        "input.addEventListener('input', () => {\n"
        "  const q = input.value.trim().toLowerCase();\n"
        "  cards.forEach(card => {\n"
        "    const hay = card.dataset.search || '';\n"
        "    card.style.display = !q || hay.includes(q) ? '' : 'none';\n"
        "  });\n"
        "});\n"
        "document.addEventListener('keydown', (e) => {\n"
        "  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {\n"
        "    e.preventDefault();\n"
        "    input.focus();\n"
        "  }\n"
        "  if (e.key === '/' && document.activeElement !== input) {\n"
        "    e.preventDefault();\n"
        "    input.focus();\n"
        "  }\n"
        "});\n"
        "</script>"
    )
    html_parts.append("</body></html>")
    return "\n".join(html_parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a visual dashboard for the Argos repository.")
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT, help="Repo root")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_DEFAULT / "docs" / "repo-dashboard.html",
        help="Output HTML path",
    )
    args = parser.parse_args()

    repo = args.repo.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    html_text = render_dashboard(repo)
    output.write_text(html_text, encoding="utf-8")
    print(str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
