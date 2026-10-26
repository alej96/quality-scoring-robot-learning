"""Loading LeRobot-format datasets (local directories or Hugging Face Hub repos).

Only tabular data and metadata are read; videos are never downloaded. The loader is
format-agnostic across LeRobot v2.x (one parquet per episode, ``meta/episodes.jsonl``)
and v3.0 (many episodes per parquet, ``meta/episodes/*.parquet``) because it groups
frames by ``episode_index`` instead of relying on file layout.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HUB_PATTERNS = ["meta/*", "meta/**", "data/*", "data/**"]
_FRAME_COLUMNS = ["episode_index", "frame_index", "timestamp", "action", "observation.state"]


@dataclass
class Episode:
    index: int
    timestamps: np.ndarray
    frame_index: np.ndarray
    action: np.ndarray | None
    state: np.ndarray | None
    task: str | None = None
    meta_length: int | None = None
    # camera key -> (from_timestamp, to_timestamp) in seconds, v3.0 only
    video_spans: dict[str, tuple[float, float]] = field(default_factory=dict)
    # camera key -> (pixel mean, pixel std) over the whole episode, from LeRobot's own
    # per-episode video stats (meta/episodes on v3.0, meta/episodes_stats.jsonl on v2.x)
    camera_stats: dict[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.timestamps)


@dataclass
class Dataset:
    root: Path
    fps: float
    codebase_version: str
    robot_type: str | None
    episodes: list[Episode]
    repo_id: str | None = None
    action_names: list[str] | None = None  # per-dim feature names, e.g. "gripper.pos"
    state_names: list[str] | None = None  # observation.state per-dim feature names
    camera_names: list[str] = field(default_factory=list)  # video-dtype feature names
    video_path_template: str | None = None  # info.json's "video_path", v2.x layout
    chunks_size: int = 1000  # info.json's "chunks_size", for v2.x episode_chunk math

    @property
    def total_frames(self) -> int:
        return sum(e.length for e in self.episodes)


def load_dataset(
    source: str | Path,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    episodes: Iterable[int] | None = None,
    max_episodes: int | None = None,
) -> Dataset:
    """Load a dataset from a local LeRobot directory or a Hub repo id like ``lerobot/pusht``.

    ``episodes`` and/or ``max_episodes`` restrict which episodes are loaded, useful for a
    huge dataset (e.g. DROID-scale) where loading everything isn't practical. From the Hub,
    only the data file(s) that actually contain the requested episodes are downloaded
    (metadata is always small and downloaded in full); locally, data files that contain none
    of the requested episodes are never opened. The narrowing is best-effort: if a dataset's
    layout can't be mapped from metadata alone, it falls back to reading everything, still
    correct but without the savings.
    """
    path = Path(source).expanduser()
    repo_id = None
    narrowing = episodes is not None or max_episodes is not None
    if not path.exists():
        repo_id = str(source)
        path = _download(repo_id, revision, cache_dir, ["meta/*", "meta/**"] if narrowing else None)

    info_path = path / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"{info_path} not found; is this a LeRobot dataset?")
    info = json.loads(info_path.read_text())

    camera_names = sorted(
        name
        for name, feat in info.get("features", {}).items()
        if isinstance(feat, dict) and feat.get("dtype") == "video"
    )
    episode_meta = _read_episode_meta(path, camera_names)
    want = _resolve_episodes(episode_meta, episodes, max_episodes)

    if repo_id is not None and narrowing:
        names = _needed_data_filenames(path, info, want) if want is not None else None
        patterns = None
        if names is not None:
            patterns = [f"data/*/{n}" for n in sorted(names)] + ["meta/*", "meta/**"]
        path = _download(repo_id, revision, cache_dir, patterns)

    frames = _read_frames(path, want, info)
    tasks = _read_tasks(path)

    episodes_out = []
    for ep_idx, group in frames.groupby("episode_index", sort=True):
        group = group.sort_values("frame_index", kind="stable")
        meta = episode_meta.get(int(ep_idx), {})
        task = meta.get("task")
        if task is None and "task_index" in group and len(tasks):
            task = tasks.get(int(group["task_index"].iloc[0]))
        episodes_out.append(
            Episode(
                index=int(ep_idx),
                timestamps=group["timestamp"].to_numpy(dtype=np.float64),
                frame_index=group["frame_index"].to_numpy(dtype=np.int64),
                action=_stack(group, "action"),
                state=_stack(group, "observation.state"),
                task=task,
                meta_length=meta.get("length"),
                video_spans=meta.get("video_spans", {}),
                camera_stats=meta.get("camera_stats", {}),
            )
        )

    return Dataset(
        root=path,
        fps=float(info["fps"]),
        codebase_version=str(info.get("codebase_version", "unknown")),
        robot_type=info.get("robot_type"),
        episodes=episodes_out,
        repo_id=repo_id,
        action_names=info.get("features", {}).get("action", {}).get("names"),
        state_names=info.get("features", {}).get("observation.state", {}).get("names"),
        camera_names=camera_names,
        video_path_template=info.get("video_path"),
        chunks_size=int(info.get("chunks_size", 1000)),
    )


def _resolve_episodes(
    episode_meta: dict[int, dict], episodes: Iterable[int] | None, max_episodes: int | None
) -> frozenset[int] | None:
    """The concrete set of episode indices to load, or ``None`` for "all of them"."""
    if episodes is None and max_episodes is None:
        return None
    existing = sorted(episode_meta)
    if episodes is not None:
        want = sorted({int(e) for e in episodes})
        unknown = sorted(set(want) - set(existing))
        if unknown:
            raise ValueError(f"episode(s) not in the dataset: {unknown}")
    else:
        want = existing
    if max_episodes is not None:
        want = want[:max_episodes]
    if not want:
        raise ValueError("no episodes selected")
    return frozenset(want)


def _needed_data_filenames(root: Path, info: dict, want: frozenset[int]) -> set[str] | None:
    """Basenames of the data files that contain any of ``want``'s episodes.

    Returns ``None`` when metadata alone can't narrow it down (an unrecognised layout), so
    the caller falls back to reading/downloading everything.
    """
    version = str(info.get("codebase_version", ""))
    if not version.startswith("v3"):
        # v2.x: one episode per file, named episode_<index>.parquet regardless of chunk dir
        return {f"episode_{idx:06d}.parquet" for idx in want}

    template = info.get("data_path")
    episode_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not template or not episode_files:
        return None
    cols = ["episode_index", "data/chunk_index", "data/file_index"]
    frames = []
    for f in episode_files:
        if not set(cols).issubset(pq.read_schema(f).names):
            return None
        frames.append(pq.read_table(f, columns=cols).to_pandas())
    df = pd.concat(frames, ignore_index=True)
    hit = df[df["episode_index"].isin(want)]
    pairs = hit[["data/chunk_index", "data/file_index"]].drop_duplicates()
    return {
        Path(template.format(chunk_index=int(c), file_index=int(fi))).name
        for c, fi in pairs.itertuples(index=False)
    }


def _download(
    repo_id: str, revision: str | None, cache_dir: str | Path | None, patterns: list[str] | None
) -> Path:
    from huggingface_hub import snapshot_download

    local = snapshot_download(
        repo_id,
        repo_type="dataset",
        revision=revision,
        allow_patterns=patterns or HUB_PATTERNS,
        cache_dir=cache_dir,
    )
    return Path(local)


def _read_frames(root: Path, want: frozenset[int] | None, info: dict) -> pd.DataFrame:
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {root / 'data'}")
    if want is not None:
        names = _needed_data_filenames(root, info, want)
        if names is not None:
            narrowed = [f for f in files if f.name in names]
            if narrowed:
                files = narrowed
    tables = []
    for f in files:
        names = set(pq.read_schema(f).names)
        tables.append(
            pq.read_table(f, columns=[c for c in _FRAME_COLUMNS + ["task_index"] if c in names])
        )
    table = pa.concat_tables(tables, promote_options="default")
    df = table.to_pandas()
    for required in ("episode_index", "frame_index", "timestamp"):
        if required not in df:
            raise ValueError(f"dataset frames are missing the '{required}' column")
    if want is not None:
        df = df[df["episode_index"].isin(want)]
    return df


def _stack(group: pd.DataFrame, column: str) -> np.ndarray | None:
    if column not in group:
        return None
    values = group[column].to_numpy()
    if len(values) == 0:
        return np.zeros((0, 0))
    arr = np.stack(values).astype(np.float64)
    return arr.reshape(len(values), -1)


def _flatten_numbers(value: object, out: list[float]) -> None:
    if value is None:
        return
    if isinstance(value, list | tuple | np.ndarray):
        for v in value:
            _flatten_numbers(v, out)
        return
    with contextlib.suppress(TypeError, ValueError):
        out.append(float(value))


def _image_stat_scalar(value: object) -> float | None:
    """Collapse a nested per-channel video stat (e.g. ``[[[0.58]], [[0.61]], [[0.55]]]``)
    to a single scalar by averaging across channels and any other nesting. Parquet round-
    trips these as ragged (non-rectangular) arrays, so this walks them recursively instead
    of relying on ``np.asarray`` to infer a uniform shape."""
    out: list[float] = []
    _flatten_numbers(value, out)
    if not out:
        return None
    arr = np.array(out, dtype=np.float64)
    if not np.isfinite(arr).all():
        return None
    return float(arr.mean())


def _read_episode_meta(root: Path, camera_names: list[str] = ()) -> dict[int, dict]:
    """Per-episode metadata: task, declared length, (v3.0) video time spans, and each
    camera's whole-episode pixel mean/std from LeRobot's own per-episode video stats."""
    meta: dict[int, dict] = {}
    v3_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if v3_files:
        df = pd.concat([pd.read_parquet(f) for f in v3_files], ignore_index=True)
        cameras = sorted(
            {
                c.split("/")[1]
                for c in df.columns
                if c.startswith("videos/") and c.endswith("/from_timestamp")
            }
        )
        stat_cams = [
            cam
            for cam in camera_names
            if f"stats/{cam}/mean" in df.columns and f"stats/{cam}/std" in df.columns
        ]
        for row in df.to_dict("records"):
            spans = {
                cam: (
                    float(row[f"videos/{cam}/from_timestamp"]),
                    float(row[f"videos/{cam}/to_timestamp"]),
                )
                for cam in cameras
                if pd.notna(row.get(f"videos/{cam}/from_timestamp"))
            }
            camera_stats = {}
            for cam in stat_cams:
                mean = _image_stat_scalar(row.get(f"stats/{cam}/mean"))
                std = _image_stat_scalar(row.get(f"stats/{cam}/std"))
                if mean is not None and std is not None:
                    camera_stats[cam] = (mean, std)
            meta[int(row["episode_index"])] = {
                "length": int(row["length"]) if "length" in row else None,
                "task": _first_task(row.get("tasks")),
                "video_spans": spans,
                "camera_stats": camera_stats,
            }
        return meta

    episode_stats = _read_v2_episode_stats(root, camera_names)
    jsonl = root / "meta" / "episodes.jsonl"
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                ep_idx = int(row["episode_index"])
                meta[ep_idx] = {
                    "length": row.get("length"),
                    "task": _first_task(row.get("tasks")),
                    "camera_stats": episode_stats.get(ep_idx, {}),
                }
    return meta


def _read_v2_episode_stats(
    root: Path, camera_names: list[str]
) -> dict[int, dict[str, tuple[float, float]]]:
    """v2.x per-episode video stats from ``meta/episodes_stats.jsonl``: camera -> (pixel
    mean, pixel std) over the whole episode."""
    out: dict[int, dict[str, tuple[float, float]]] = {}
    jsonl = root / "meta" / "episodes_stats.jsonl"
    if not camera_names or not jsonl.exists():
        return out
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        stats = row.get("stats") or {}
        cams = {}
        for cam in camera_names:
            s = stats.get(cam)
            if not s:
                continue
            mean = _image_stat_scalar(s.get("mean"))
            std = _image_stat_scalar(s.get("std"))
            if mean is not None and std is not None:
                cams[cam] = (mean, std)
        out[int(row["episode_index"])] = cams
    return out


def _read_tasks(root: Path) -> dict[int, str]:
    parquet = root / "meta" / "tasks.parquet"
    if parquet.exists():
        df = pd.read_parquet(parquet)
        # v3.0 stores the task string as the index and task_index as a column
        return {int(i): str(t) for t, i in zip(df.index, df["task_index"], strict=True)}
    jsonl = root / "meta" / "tasks.jsonl"
    if jsonl.exists():
        rows = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
        return {int(r["task_index"]): str(r["task"]) for r in rows}
    return {}


def _first_task(tasks) -> str | None:
    if tasks is None:
        return None
    if isinstance(tasks, str):
        return tasks
    tasks = list(tasks)
    return str(tasks[0]) if tasks else None
