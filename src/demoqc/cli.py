"""Command-line interface: ``demoqc score <local-path-or-hub-repo-id>``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from demoqc import __version__
from demoqc.checks import Config, evaluate
from demoqc.dataset import load_dataset
from demoqc.report import build_report, format_summary, write_csv, write_json


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="demoqc", description=__doc__)
    p.add_argument("--version", action="version", version=f"demoqc {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    score = sub.add_parser(
        "score",
        help="score every episode of a LeRobot dataset",
        description="Score every episode of a LeRobot dataset (local dir or Hub repo id). "
        "Only metadata and tabular data are downloaded, never videos.",
    )
    score.add_argument("source", help="local dataset directory or Hub repo id, e.g. lerobot/pusht")
    score.add_argument("--revision", help="Hub branch, tag or commit")
    score.add_argument("--json", type=Path, help="write the full report as JSON")
    score.add_argument("--csv", type=Path, help="write one row per episode as CSV")
    score.add_argument(
        "--min-score",
        type=float,
        default=70.0,
        help="episodes below this go on the reject list (default: 70)",
    )
    score.add_argument("--top", type=int, default=10, help="lowest-scoring episodes to print")
    score.add_argument(
        "--fail-under",
        type=float,
        help="exit with status 1 if the mean score is below this (for CI)",
    )
    score.add_argument("--quiet", action="store_true", help="don't print the summary")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "score":
        return _score(args)
    return 2


def _score(args: argparse.Namespace) -> int:
    try:
        ds = load_dataset(args.source, revision=args.revision)
    except (FileNotFoundError, ValueError) as exc:
        print(f"demoqc: {exc}", file=sys.stderr)
        return 2
    results = evaluate(ds, Config())
    report = build_report(ds, results, min_score=args.min_score)
    if args.json:
        write_json(report, args.json)
    if args.csv:
        write_csv(report, args.csv)
    if not args.quiet:
        print(format_summary(report, top=args.top))
    if args.fail_under is not None and report["summary"]["mean_score"] < args.fail_under:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
