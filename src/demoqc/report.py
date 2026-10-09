"""Turning check results into machine-readable reports and a terminal summary."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from demoqc import __version__
from demoqc.checks import EpisodeResult
from demoqc.dataset import Dataset
from demoqc.dedupe import DuplicateGroup

SCHEMA_VERSION = 1


def build_report(
    ds: Dataset,
    results: list[EpisodeResult],
    min_score: float,
    duplicate_groups: list[DuplicateGroup] | None = None,
) -> dict:
    scores = np.array([r.score for r in results]) if results else np.array([0.0])
    flag_counts = Counter(f.check for r in results for f in r.flags if f.penalty > 0)
    duplicate_groups = duplicate_groups or []
    return {
        "schema_version": SCHEMA_VERSION,
        "demoqc_version": __version__,
        "dataset": {
            "source": ds.repo_id or str(ds.root),
            "codebase_version": ds.codebase_version,
            "robot_type": ds.robot_type,
            "fps": ds.fps,
            "episodes": len(ds.episodes),
            "frames": ds.total_frames,
        },
        "summary": {
            "mean_score": round(float(scores.mean()), 2),
            "median_score": round(float(np.median(scores)), 2),
            "min_score_threshold": min_score,
            "episodes_below_threshold": int((scores < min_score).sum()),
            "keep": [r.episode_index for r in results if r.score >= min_score],
            "trimmable_idle_s": round(
                sum(
                    r.metrics.get("lead_idle_s", 0) + r.metrics.get("trail_idle_s", 0)
                    for r in results
                    if r.trim
                ),
                2,
            ),
            "flags_by_check": dict(flag_counts.most_common()),
            "duplicate_groups": [
                {"episodes": g.episodes, "kind": g.kind, "max_distance": g.max_distance}
                for g in duplicate_groups
            ],
        },
        "episodes": [_episode_dict(r) for r in results],
    }


def _episode_dict(r: EpisodeResult) -> dict:
    return {
        "episode_index": r.episode_index,
        "score": round(r.score, 2),
        "length": r.length,
        "duration_s": round(r.duration_s, 3),
        "task": r.task,
        "trim": list(r.trim) if r.trim else None,
        "flags": [
            {
                "check": f.check,
                "severity": f.severity,
                "message": f.message,
                "penalty": round(f.penalty, 2),
            }
            for f in r.flags
        ],
        "metrics": {k: _clean(v) for k, v in sorted(r.metrics.items())},
    }


def _clean(v):
    v = float(v)
    return None if math.isnan(v) or math.isinf(v) else round(v, 4)


def write_json(report: dict, path: Path) -> None:
    path.write_text(json.dumps(report, indent=2) + "\n")


def write_csv(report: dict, path: Path) -> None:
    metric_keys = sorted({k for e in report["episodes"] for k in e["metrics"]})
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "episode_index",
                "score",
                "length",
                "duration_s",
                "trim_start",
                "trim_end",
                "flags",
                *metric_keys,
            ]
        )
        for e in report["episodes"]:
            trim = e["trim"] or ["", ""]
            flags = "; ".join(f"{f['check']}: {f['message']}" for f in e["flags"])
            writer.writerow(
                [
                    e["episode_index"],
                    e["score"],
                    e["length"],
                    e["duration_s"],
                    *trim,
                    flags,
                    *(e["metrics"].get(k, "") for k in metric_keys),
                ]
            )


def format_summary(report: dict, top: int) -> str:
    d, s = report["dataset"], report["summary"]
    lines = [
        f"{d['source']}  (LeRobot {d['codebase_version']}, {d['robot_type'] or 'unknown robot'}, "
        f"{d['fps']:g} fps)",
        f"{d['episodes']} episodes, {d['frames']} frames",
        f"score: mean {s['mean_score']:.1f}, median {s['median_score']:.1f}; "
        f"{s['episodes_below_threshold']} below {s['min_score_threshold']:g}",
        f"trimmable idle: {s['trimmable_idle_s']:.1f}s",
    ]
    if s["flags_by_check"]:
        lines.append("flags: " + ", ".join(f"{k} x{v}" for k, v in s["flags_by_check"].items()))
    groups = s.get("duplicate_groups") or []
    if groups:
        exact = sum(1 for g in groups if g["kind"] == "exact")
        near = len(groups) - exact
        dup_episodes = sum(len(g["episodes"]) - 1 for g in groups)
        lines.append(
            f"duplicates: {exact} exact group(s), {near} near group(s), "
            f"{dup_episodes} episode(s) to drop"
        )
    worst = sorted(report["episodes"], key=lambda e: e["score"])[:top]
    worst = [e for e in worst if e["flags"]]
    if worst:
        lines += ["", f"lowest-scoring episodes (top {len(worst)}):"]
        for e in worst:
            lines.append(
                f"  ep {e['episode_index']:>5}  score {e['score']:5.1f}  {e['duration_s']:6.1f}s"
            )
            for f in e["flags"]:
                lines.append(f"      [{f['severity']}] {f['check']}: {f['message']}")
    return "\n".join(lines)
