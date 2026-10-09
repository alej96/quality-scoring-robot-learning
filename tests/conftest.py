"""Synthetic LeRobot datasets with controllable defects."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FPS = 30
DIMS = 6


@dataclass
class EpisodeSpec:
    seconds: float = 8.0
    lead_idle_s: float = 0.0
    trail_idle_s: float = 0.0
    spikes_at: list[int] = field(default_factory=list)  # frame indices with a 1-frame glitch
    drop_frames: list[int] = field(default_factory=list)  # frames removed (timestamp gaps)
    stale_state_from: tuple[int, int] | None = None  # [start, end) frames state is frozen
    video_span_s: float | None = None  # override video duration in episode metadata
    meta_length_delta: int = 0
    state_lag_frames: int = 1  # observation.state tracks action with this many frames' delay
    state_decorrelated: bool = False  # state is unrelated motion, not a delayed copy of action
    seed: int = 0


def make_episode(spec: EpisodeSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (timestamps, action, state) for one episode."""
    rng = np.random.default_rng(spec.seed)
    lead = int(spec.lead_idle_s * FPS)
    trail = int(spec.trail_idle_s * FPS)
    n_move = int(spec.seconds * FPS) - lead - trail
    t = np.linspace(0, 1, n_move)
    phases = rng.uniform(0, 2 * np.pi, DIMS)
    amps = rng.uniform(20, 60, DIMS)
    # smooth reach-and-return motion: zero velocity at both ends
    profile = (1 - np.cos(2 * np.pi * t))[:, None] / 2
    moving = amps * profile * np.sin(phases + 3 * t[:, None]) + 10 * np.arange(DIMS)
    start, end = moving[0], moving[-1]
    action = np.concatenate(
        [np.repeat(start[None], lead, 0), moving, np.repeat(end[None], trail, 0)]
    )
    action += rng.normal(0, 0.02, action.shape)  # encoder noise
    if spec.state_decorrelated:
        # a scrambled stream: same per-frame distribution, no temporal correspondence
        state = rng.permutation(action, axis=0)
    elif spec.state_lag_frames > 0:
        lag = spec.state_lag_frames
        state = np.vstack([np.repeat(action[:1], lag, 0), action[:-lag]])
    else:
        state = action.copy()

    for i in spec.spikes_at:
        action[i] += 40.0
    if spec.stale_state_from:
        s, e = spec.stale_state_from
        state[s:e] = state[s - 1]

    n = len(action)
    timestamps = np.arange(n) / FPS
    keep = np.setdiff1d(np.arange(n), spec.drop_frames)
    return timestamps[keep], action[keep], state[keep]


def _stats(x: np.ndarray) -> dict[str, list]:
    """LeRobot-style stats of an (n, d) matrix: per-dim min/max/mean/std and the row count."""
    x = np.asarray(x, dtype=np.float64).reshape(len(x), -1)
    return {
        "min": x.min(0).tolist(),
        "max": x.max(0).tolist(),
        "mean": x.mean(0).tolist(),
        "std": x.std(0).tolist(),
        "count": [len(x)],
    }


def _image_stats(rng: np.random.Generator, count: int) -> dict[str, list]:
    """Per-episode video stats are (channels, 1, 1) nested lists; the values are arbitrary."""
    mean = rng.uniform(0.3, 0.7, 3)
    return {
        "min": [[[0.0]]] * 3,
        "max": [[[float(m + 0.2)]] for m in mean],
        "mean": [[[float(m)]] for m in mean],
        "std": [[[float(rng.uniform(0.05, 0.2))]] for _ in mean],
        "count": [count],
    }


def write_dataset(
    root: Path, specs: list[EpisodeSpec], version: str = "v3.0", data_files: int = 1
) -> Path:
    """``data_files`` > 1 spreads the episodes over that many v3.0 data files."""
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    frames, episodes = [], []
    index = 0
    rng = np.random.default_rng(123)
    for ep, spec in enumerate(specs):
        ts, action, state = make_episode(spec)
        n = len(ts)
        frames.append(
            pd.DataFrame(
                {
                    "action": list(action.astype(np.float32)),
                    "observation.state": list(state.astype(np.float32)),
                    "timestamp": ts.astype(np.float32),
                    "frame_index": np.arange(n),
                    "episode_index": ep,
                    "index": np.arange(index, index + n),
                    "task_index": 0,
                }
            )
        )
        span = spec.video_span_s if spec.video_span_s is not None else n / FPS
        stats = {
            "action": _stats(action),
            "observation.state": _stats(state),
            "index": _stats(np.arange(index, index + n)),
            "episode_index": _stats(np.full(n, ep)),
            "observation.images.top": _image_stats(rng, n),
        }
        episodes.append(
            {
                "episode_index": ep,
                "data/chunk_index": 0,
                "data/file_index": ep * data_files // len(specs),
                "videos/observation.images.top/chunk_index": 0,
                "videos/observation.images.top/file_index": 0,
                **{f"stats/{f}/{k}": v for f, st in stats.items() for k, v in st.items()},
                "tasks": ["pick the cube"],
                "length": n + spec.meta_length_delta,
                "dataset_from_index": index,
                "dataset_to_index": index + n,
                "videos/observation.images.top/from_timestamp": 100.0,
                "videos/observation.images.top/to_timestamp": 100.0 + span,
            }
        )
        index += n

    info = {
        "codebase_version": version,
        "robot_type": "so101_follower",
        "fps": FPS,
        "total_episodes": len(specs),
        "total_frames": index,
    }
    info["splits"] = {"train": f"0:{len(specs)}"}
    scalar = {"shape": [1], "names": None}
    info["features"] = {
        "action": {"dtype": "float32", "shape": [DIMS], "names": None},
        "observation.state": {"dtype": "float32", "shape": [DIMS], "names": None},
        "observation.images.top": {"dtype": "video", "shape": [480, 640, 3], "names": None},
        "timestamp": {"dtype": "float32", **scalar},
        "frame_index": {"dtype": "int64", **scalar},
        "episode_index": {"dtype": "int64", **scalar},
        "index": {"dtype": "int64", **scalar},
        "task_index": {"dtype": "int64", **scalar},
    }
    info["total_tasks"] = 1
    info["chunks_size"] = 1000
    info["data_path"] = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    (root / "meta" / "info.json").write_text(json.dumps(info))
    all_frames = pd.concat(frames)
    dataset_stats = {
        "action": _stats(np.stack(all_frames["action"])),
        "observation.state": _stats(np.stack(all_frames["observation.state"])),
        "index": _stats(all_frames["index"].to_numpy()),
        "episode_index": _stats(all_frames["episode_index"].to_numpy()),
        "observation.images.top": _image_stats(rng, index),
    }
    (root / "meta" / "stats.json").write_text(json.dumps(dataset_stats))

    if version.startswith("v3"):
        for k in range(data_files):
            part = [f for ep, f in enumerate(frames) if ep * data_files // len(specs) == k]
            pd.concat(part, ignore_index=True).to_parquet(
                root / "data" / "chunk-000" / f"file-{k:03d}.parquet", index=False
            )
        (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
        pd.DataFrame(episodes).to_parquet(
            root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        )
        pd.DataFrame({"task_index": [0]}, index=["pick the cube"]).to_parquet(
            root / "meta" / "tasks.parquet"
        )
    else:
        for ep, df in enumerate(frames):
            df.to_parquet(root / "data" / "chunk-000" / f"episode_{ep:06d}.parquet")
        with (root / "meta" / "episodes.jsonl").open("w") as fh:
            for e in episodes:
                fh.write(json.dumps({k: e[k] for k in ("episode_index", "tasks", "length")}) + "\n")
        (root / "meta" / "tasks.jsonl").write_text(
            json.dumps({"task_index": 0, "task": "pick the cube"}) + "\n"
        )
    return root


def clean_specs(n: int = 12) -> list[EpisodeSpec]:
    return [EpisodeSpec(seconds=8.0 + 0.3 * (i % 4), seed=i) for i in range(n)]


@pytest.fixture
def make_dataset(tmp_path):
    def _make(specs: list[EpisodeSpec], version: str = "v3.0", data_files: int = 1) -> Path:
        return write_dataset(tmp_path / f"ds-{version}", specs, version, data_files)

    return _make
