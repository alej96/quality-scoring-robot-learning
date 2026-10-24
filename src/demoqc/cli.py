"""Command-line interface: ``demoqc score <local-path-or-hub-repo-id>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from demoqc import __version__
from demoqc.checks import CHECK_NAMES, Config, DatasetContext, evaluate
from demoqc.dataset import load_dataset
from demoqc.dedupe import find_duplicates, flag_duplicates, format_duplicates
from demoqc.export import (
    export_dataset,
    keep_from_report,
    keep_from_report_dict,
    trim_from_report,
    trim_from_report_dict,
)
from demoqc.report import build_report, format_markdown, format_summary, write_csv, write_json


def _episode_spec(spec: str) -> list[int]:
    """Parse '0-9,20,35-40' into a sorted list of episode indices."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            start, end = int(start), int(end)
            if start > end:
                raise ValueError(f"invalid range '{part}': start > end")
            out.update(range(start, end + 1))
        else:
            out.add(int(part))
    if not out:
        raise ValueError("no episodes in spec")
    return sorted(out)


def _positive_int(spec: str) -> int:
    n = int(spec)
    if n < 1:
        raise ValueError("must be >= 1")
    return n


def _add_episode_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--episodes",
        type=_episode_spec,
        metavar="SPEC",
        help="only load these episodes, e.g. '0-99' or '5,10,20-30' (comma-separated indices "
        "and/or ranges); from the Hub, downloads only the data file(s) that contain them",
    )
    parser.add_argument(
        "--max-episodes",
        type=_positive_int,
        metavar="N",
        help="only load the first N episodes (combinable with --episodes); "
        "for trying the tool on a huge dataset without loading all of it",
    )


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
    _add_episode_args(score)
    score.add_argument(
        "--config",
        type=Path,
        metavar="FILE.toml",
        help="TOML file overriding Config thresholds/penalties (see README's config table)",
    )
    score.add_argument("--json", type=Path, help="write the full report as JSON")
    score.add_argument("--csv", type=Path, help="write one row per episode as CSV")
    score.add_argument(
        "--md", type=Path, help="write a Markdown summary, for a dataset card or a Hub/GitHub PR"
    )
    score.add_argument(
        "--min-score",
        type=float,
        default=70.0,
        help="episodes below this go on the reject list (default: 70)",
    )
    score.add_argument(
        "--disable-check",
        action="append",
        choices=CHECK_NAMES,
        metavar="NAME",
        help="skip a check entirely (repeatable), e.g. a dataset with no gripper but a "
        "noisy sync check from an unsynced third camera clock",
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
    _add_episode_args(dedupe)
    dedupe.add_argument(
        "--config",
        type=Path,
        metavar="FILE.toml",
        help="TOML file overriding Config thresholds/penalties (see README's config table)",
    )
    dedupe.add_argument("--json", type=Path, help="write the duplicate groups as JSON")

    export = sub.add_parser(
        "export",
        help="write a cleaned copy of a dataset without the rejected episodes",
        description="Write a LeRobot v3.0 or v2.x dataset containing only the episodes a "
        "`demoqc score --json` report keeps. Episodes are renumbered and the metadata/stats "
        "are rewritten. With --apply-trim, each kept episode's idle start/end frames are also "
        "cut. Only tabular data and metadata are written by default; videos are neither read "
        "nor copied (for v2.x, each kept episode's video files must be renamed to match its "
        "new index, and a trimmed episode's video must also be cut to match; see the printed "
        "warning). --copy-videos copies (and, for v2.x, renames) the source's video files for "
        "a local dataset, without reading, decoding or re-encoding them; --symlink-videos does "
        "the same with symlinks instead, to avoid duplicating video data on disk, at the cost "
        "of the export depending on the source directory staying where it is. --episodes/"
        "--max-episodes restrict which episodes are considered at all (as for score/dedupe); "
        "with --keep-from, the report must have been made from the same subset.",
    )
    export.add_argument("source", help="local dataset directory or Hub repo id")
    export.add_argument("--revision", help="Hub branch, tag or commit")
    _add_episode_args(export)
    keep_source = export.add_mutually_exclusive_group(required=True)
    keep_source.add_argument(
        "--keep-from",
        type=Path,
        metavar="REPORT.json",
        help="report from `demoqc score --json` for this dataset; its summary.keep list is kept",
    )
    keep_source.add_argument(
        "--min-score",
        type=float,
        metavar="N",
        help="score the dataset here (default thresholds) and keep episodes scoring >= N, "
        "instead of requiring a separate `demoqc score --json` run",
    )
    export.add_argument(
        "-o", "--output", type=Path, required=True, help="directory to write (must be empty)"
    )
    export.add_argument(
        "--apply-trim",
        action="store_true",
        help="cut each kept episode's idle start/end frames (per the report's trim "
        "suggestion), shifting timestamps and narrowing video spans to match",
    )
    video_mode = export.add_mutually_exclusive_group()
    video_mode.add_argument(
        "--copy-videos",
        action="store_true",
        help="copy (and, for v2.x, rename) the source's video files into the export; "
        "local sources only, since videos are never downloaded from the Hub",
    )
    video_mode.add_argument(
        "--symlink-videos",
        action="store_true",
        help="symlink (instead of copy) the source's video files into the export, saving disk "
        "space; local sources only, and the export then depends on the source staying in place",
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
        return load_dataset(
            args.source,
            revision=args.revision,
            episodes=getattr(args, "episodes", None),
            max_episodes=getattr(args, "max_episodes", None),
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"demoqc: {exc}", file=sys.stderr)
        return None


def _config(args: argparse.Namespace) -> Config | None:
    if not args.config:
        return Config()
    try:
        return Config.from_file(args.config)
    except (OSError, ValueError) as exc:
        print(f"demoqc: {exc}", file=sys.stderr)
        return None


def _score(args: argparse.Namespace) -> int:
    ds = _load(args)
    if ds is None:
        return 2
    cfg = _config(args)
    if cfg is None:
        return 2
    disabled = frozenset(args.disable_check or [])
    results = evaluate(ds, cfg, disabled)
    duplicate_groups = []
    if "duplicate" not in disabled:
        ctx = DatasetContext.build(ds, cfg)
        duplicate_groups = find_duplicates(ds, results, ctx, cfg)
        flag_duplicates(results, duplicate_groups)
    report = build_report(ds, results, min_score=args.min_score, duplicate_groups=duplicate_groups)
    if args.json:
        write_json(report, args.json)
    if args.csv:
        write_csv(report, args.csv)
    if args.md:
        args.md.write_text(format_markdown(report, top=args.top) + "\n")
    if not args.quiet:
        print(format_summary(report, top=args.top))
    if args.fail_under is not None and report["summary"]["mean_score"] < args.fail_under:
        return 1
    return 0


def _dedupe(args: argparse.Namespace) -> int:
    ds = _load(args)
    if ds is None:
        return 2
    cfg = _config(args)
    if cfg is None:
        return 2
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
    if (args.copy_videos or args.symlink_videos) and ds.repo_id is not None:
        flag = "--copy-videos" if args.copy_videos else "--symlink-videos"
        print(
            f"demoqc: {flag} requires a local dataset directory; videos are never "
            "downloaded from the Hub",
            file=sys.stderr,
        )
        return 2
    try:
        if args.keep_from is not None:
            keep = keep_from_report(args.keep_from, ds)
            trim = trim_from_report(args.keep_from, ds) if args.apply_trim else None
        else:
            cfg = Config()
            results = evaluate(ds, cfg)
            ctx = DatasetContext.build(ds, cfg)
            duplicate_groups = find_duplicates(ds, results, ctx, cfg)
            flag_duplicates(results, duplicate_groups)
            report = build_report(
                ds, results, min_score=args.min_score, duplicate_groups=duplicate_groups
            )
            keep = keep_from_report_dict(report)
            trim = trim_from_report_dict(report) if args.apply_trim else None
        result = export_dataset(
            ds.root,
            args.output,
            keep,
            trim=trim,
            copy_videos=args.copy_videos,
            symlink_videos=args.symlink_videos,
        )
    except (ValueError, FileExistsError) as exc:
        print(f"demoqc: {exc}", file=sys.stderr)
        return 2
    print(
        f"kept {len(result.kept)} of {len(result.kept) + len(result.dropped)} episodes "
        f"({result.frames} frames) -> {result.out}"
    )
    if result.dropped:
        print("dropped episodes: " + ", ".join(map(str, result.dropped)))
    if result.trimmed_frames:
        print(
            f"trimmed {result.trimmed_frames} idle frames ({result.trimmed_frames / ds.fps:.1f}s)"
        )
    for w in result.warnings:
        print(f"warning: {w}", file=sys.stderr)
    if (
        result.has_videos
        and result.version.startswith("v3")
        and not (result.videos_copied or result.videos_symlinked)
    ):
        print(
            "videos were not copied. Their episode time spans are unchanged, so copy the "
            "source videos/ directory into the output to use it for training, or re-run "
            "with --copy-videos or --symlink-videos."
        )
    if result.videos_copied:
        print(f"videos copied to {result.out / 'videos'}")
    if result.videos_symlinked:
        print(
            f"videos symlinked to {result.out / 'videos'} "
            "(the export now depends on the source directory staying where it is)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
