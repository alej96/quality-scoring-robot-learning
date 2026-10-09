"""Command-line interface: ``demoqc score <local-path-or-hub-repo-id>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from demoqc import __version__
from demoqc.checks import Config, DatasetContext, evaluate
from demoqc.dataset import load_dataset
from demoqc.dedupe import find_duplicates, flag_duplicates, format_duplicates
from demoqc.export import export_dataset, keep_from_report
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

    dedupe = sub.add_parser(
        "dedupe",
        help="find exact and near-duplicate episodes",
        description="Find exact (byte-identical) and near-duplicate (same trajectory, "
        "resampled) episodes in a LeRobot dataset.",
    )
    dedupe.add_argument("source", help="local dataset directory or Hub repo id")
    dedupe.add_argument("--revision", help="Hub branch, tag or commit")
    dedupe.add_argument("--json", type=Path, help="write the duplicate groups as JSON")

    export = sub.add_parser(
        "export",
        help="write a cleaned copy of a dataset without the rejected episodes",
        description="Write a LeRobot v3.0 dataset containing only the episodes a `demoqc score "
        "--json` report keeps. Episodes are renumbered and meta/info.json, meta/episodes and "
        "meta/stats.json are rewritten. Only tabular data and metadata are written; videos are "
        "neither read nor copied.",
    )
    export.add_argument("source", help="local dataset directory or Hub repo id")
    export.add_argument("--revision", help="Hub branch, tag or commit")
    export.add_argument(
        "--keep-from",
        type=Path,
        required=True,
        metavar="REPORT.json",
        help="report from `demoqc score --json` for this dataset; its summary.keep list is kept",
    )
    export.add_argument(
        "-o", "--output", type=Path, required=True, help="directory to write (must be empty)"
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "score":
        return _score(args)
    if args.command == "dedupe":
        return _dedupe(args)
    if args.command == "export":
        return _export(args)
    return 2


def _load(args: argparse.Namespace):
    try:
        return load_dataset(args.source, revision=args.revision)
    except (FileNotFoundError, ValueError) as exc:
        print(f"demoqc: {exc}", file=sys.stderr)
        return None


def _score(args: argparse.Namespace) -> int:
    ds = _load(args)
    if ds is None:
        return 2
    cfg = Config()
    results = evaluate(ds, cfg)
    ctx = DatasetContext.build(ds, cfg)
    duplicate_groups = find_duplicates(ds, results, ctx, cfg)
    flag_duplicates(results, duplicate_groups)
    report = build_report(ds, results, min_score=args.min_score, duplicate_groups=duplicate_groups)
    if args.json:
        write_json(report, args.json)
    if args.csv:
        write_csv(report, args.csv)
    if not args.quiet:
        print(format_summary(report, top=args.top))
    if args.fail_under is not None and report["summary"]["mean_score"] < args.fail_under:
        return 1
    return 0


def _dedupe(args: argparse.Namespace) -> int:
    ds = _load(args)
    if ds is None:
        return 2
    cfg = Config()
    results = evaluate(ds, cfg)
    ctx = DatasetContext.build(ds, cfg)
    groups = find_duplicates(ds, results, ctx, cfg)
    print(format_duplicates(groups))
    if args.json:
        args.json.write_text(
            json.dumps(
                [
                    {"episodes": g.episodes, "kind": g.kind, "max_distance": g.max_distance}
                    for g in groups
                ],
                indent=2,
            )
            + "\n"
        )
    return 0


def _export(args: argparse.Namespace) -> int:
    ds = _load(args)
    if ds is None:
        return 2
    try:
        keep = keep_from_report(args.keep_from, ds)
        result = export_dataset(ds.root, args.output, keep)
    except (ValueError, FileExistsError) as exc:
        print(f"demoqc: {exc}", file=sys.stderr)
        return 2
    print(
        f"kept {len(result.kept)} of {len(result.kept) + len(result.dropped)} episodes "
        f"({result.frames} frames) -> {result.out}"
    )
    if result.dropped:
        print("dropped episodes: " + ", ".join(map(str, result.dropped)))
    for w in result.warnings:
        print(f"warning: {w}", file=sys.stderr)
    if result.has_videos:
        print(
            "videos were not copied. Their episode time spans are unchanged, so copy the "
            "source videos/ directory into the output to use it for training."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
