"""Loading LeRobot-format datasets (local directories or Hugging Face Hub repos).

Only tabular data and metadata are read; videos are never downloaded. The loader is
format-agnostic across LeRobot v2.x (one parquet per episode, ``meta/episodes.jsonl``)
and v3.0 (many episodes per parquet, ``meta/episodes/*.parquet``) because it groups
frames by ``episode_index`` instead of relying on file layout.
"""

from __future__ import annotations

import json
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

    @property
    def total_frames(self) -> int:
        return sum(e.length for e in self.episodes)


def load_dataset(
    source: str | Path, revision: str | None = None, cache_dir: str | Path | None = None
) -> Dataset:
    """Load a dataset from a local LeRobot directory or a Hub repo id like ``lerobot/pusht``."""
    path = Path(source).expanduser()
    repo_id = None
    if not path.exists():
        repo_id = str(source)
        path = _download(repo_id, revision=revision, cache_dir=cache_dir)

    info_path = path / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"{info_path} not found; is this a LeRobot dataset?")
    info = json.loads(info_path.read_text())

    frames = _read_frames(path)
    episode_meta = _read_episode_meta(path)
    tasks = _read_tasks(path)

    episodes = []
    for ep_idx, group in frames.groupby("episode_index", sort=True):
        group = group.sort_values("frame_index", kind="stable")
        meta = episode_meta.get(int(ep_idx), {})
        task = meta.get("task")
        if task is None and "task_index" in group and len(tasks):
            task = tasks.get(int(group["task_index"].iloc[0]))
        episodes.append(
            Episode(
                index=int(ep_idx),
                timestamps=group["timestamp"].to_numpy(dtype=np.float64),
                frame_index=group["frame_index"].to_numpy(dtype=np.int64),
                action=_stack(group, "action"),
                state=_stack(group, "observation.state"),
                task=task,
                meta_length=meta.get("length"),
                video_spans=meta.get("video_spans", {}),
            )
        )

    return Dataset(
        root=path,
        fps=float(info["fps"]),
        codebase_version=str(info.get("codebase_version", "unknown")),
        robot_type=info.get("robot_type"),
        episodes=episodes,
        repo_id=repo_id,
    )


def _download(repo_id: str, revision: str | None, cache_dir: str | Path | None) -> Path:
    from huggingface_hub import snapshot_download

    local = snapshot_download(
        repo_id,
        repo_type="dataset",
        revision=revision,
        allow_patterns=HUB_PATTERNS,
        cache_dir=cache_dir,
    )
    return Path(local)


def _read_frames(root: Path) -> pd.DataFrame:
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {root / 'data'}")
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
    return df


def _stack(group: pd.DataFrame, column: str) -> np.ndarray | None:
    if column not in group:
        return None
    values = group[column].to_numpy()
    if len(values) == 0:
        return np.zeros((0, 0))
    arr = np.stack(values).astype(np.float64)
    return arr.reshape(len(values), -1)


def _read_episode_meta(root: Path) -> dict[int, dict]:
    """Per-episode metadata: task, declared length and (v3.0) video time spans."""
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
        for row in df.to_dict("records"):
            spans = {
                cam: (
                    float(row[f"videos/{cam}/from_timestamp"]),
                    float(row[f"videos/{cam}/to_timestamp"]),
                )
                for cam in cameras
                if pd.notna(row.get(f"videos/{cam}/from_timestamp"))
            }
            meta[int(row["episode_index"])] = {
                "length": int(row["length"]) if "length" in row else None,
                "task": _first_task(row.get("tasks")),
                "video_spans": spans,
            }
        return meta

    jsonl = root / "meta" / "episodes.jsonl"
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                meta[int(row["episode_index"])] = {
                    "length": row.get("length"),
                    "task": _first_task(row.get("tasks")),
                }
    return meta


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
