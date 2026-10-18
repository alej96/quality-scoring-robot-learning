#!/usr/bin/env python3
"""Score a curated list of public LeRobot datasets and write docs/benchmarks.md.

Run after changing a check or threshold and diff ``docs/benchmarks.md`` against the
previous commit to see which real datasets' scores moved, and by how much. One bad
dataset (unreachable, wrong format) doesn't stop the rest: it's reported as an error
row instead.

    python scripts/benchmark.py
    python scripts/benchmark.py --repo some/other-dataset --repo and/another
    python scripts/benchmark.py --max-episodes 50 --out /tmp/bench.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from dataclasses import dataclass
from pathlib import Path

from demoqc import __version__
from demoqc.checks import Config, DatasetContext, evaluate
from demoqc.dataset import load_dataset
from demoqc.dedupe import find_duplicates, flag_duplicates
from demoqc.report import build_report

# Real community teleoperation datasets (not the huge OpenX/DROID re-releases, which are
# orders of magnitude larger and not what this tool is calibrated for or aimed at), plus
# the two reference datasets used throughout development.
DEFAULT_REPOS = [
    "lerobot/svla_so101_pickplace",
    "lerobot/pusht",
    "masato-ka/so100_cutlery_handling_simple",
    "aaronsu11/so100_lego",
    "shylee/so100_cube",
    "jccj/so100_block_in_cup",
]


@dataclass
class BenchmarkRow:
    repo_id: str
    error: str | None = None
    episodes: int = 0
    frames: int = 0
    mean_score: float = 0.0
    median_score: float = 0.0
    trimmable_idle_s: float = 0.0
    duplicate_groups: int = 0
    flags_by_check: dict[str, int] | None = None


def score_repo(repo_id: str, max_episodes: int | None, revision: str | None = None) -> BenchmarkRow:
    """Score one dataset, catching anything that goes wrong (network, format, a bug in a
    check) so a single broken or unreachable dataset doesn't stop the rest of the run."""
    try:
        ds = load_dataset(repo_id, revision=revision, max_episodes=max_episodes)
        cfg = Config()
        results = evaluate(ds, cfg)
        ctx = DatasetContext.build(ds, cfg)
        groups = find_duplicates(ds, results, ctx, cfg)
        flag_duplicates(results, groups)
        report = build_report(ds, results, min_score=70.0, duplicate_groups=groups)
        s = report["summary"]
        return BenchmarkRow(
            repo_id=repo_id,
            episodes=report["dataset"]["episodes"],
            frames=report["dataset"]["frames"],
            mean_score=s["mean_score"],
            median_score=s["median_score"],
            trimmable_idle_s=s["trimmable_idle_s"],
            duplicate_groups=len(groups),
            flags_by_check=s["flags_by_check"],
        )
    except Exception as exc:
        return BenchmarkRow(repo_id=repo_id, error=f"{type(exc).__name__}: {exc}")


def format_benchmark_markdown(
    rows: list[BenchmarkRow], generated_at: dt.datetime, max_episodes: int | None
) -> str:
    cap = f" (capped at {max_episodes} episodes/dataset)" if max_episodes else ""
    lines = [
        "# demoqc community benchmark",
        "",
        f"Generated {generated_at.strftime('%Y-%m-%d')} with demoqc v{__version__}{cap}. "
        "Regenerate with `python scripts/benchmark.py`; this also doubles as a regression "
        "check for the heuristics — a diff here after a threshold change shows exactly "
        "which real datasets moved, and by how much.",
        "",
        "| Dataset | Episodes | Frames | Mean score | Median score | Trimmable idle | "
        "Dup. groups | Top flags |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r.error:
            lines.append(f"| {r.repo_id} | — | — | — | — | — | — | error: {r.error} |")
            continue
        flags = ", ".join(f"{k} x{v}" for k, v in (r.flags_by_check or {}).items()) or "none"
        lines.append(
            f"| {r.repo_id} | {r.episodes} | {r.frames} | {r.mean_score:.1f} | "
            f"{r.median_score:.1f} | {r.trimmable_idle_s:.1f}s | {r.duplicate_groups} | {flags} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--repo",
        dest="repos",
        action="append",
        metavar="REPO_ID",
        help="dataset to score (repeatable); default: a curated list of real community "
        "teleoperation datasets",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=200,
        metavar="N",
        help="cap episodes scored per dataset, so one huge dataset can't blow the "
        "benchmark's runtime (default: 200; pass 0 to disable the cap)",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("docs/benchmarks.md"), help="where to write the report"
    )
    args = parser.parse_args(argv)

    repos = args.repos or DEFAULT_REPOS
    max_episodes = args.max_episodes or None
    rows = [score_repo(repo, max_episodes) for repo in repos]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        format_benchmark_markdown(rows, dt.datetime.now(dt.timezone.utc), max_episodes)
    )
    print(f"wrote {args.out}")

    failed = [r for r in rows if r.error]
    for r in failed:
        print(f"benchmark: {r.repo_id}: {r.error}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
