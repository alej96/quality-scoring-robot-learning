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


def write_dataset(root: Path, specs: list[EpisodeSpec], version: str = "v3.0") -> Path:
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    frames, episodes = [], []
    index = 0
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
        episodes.append(
            {
                "episode_index": ep,
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
    (root / "meta" / "info.json").write_text(json.dumps(info))

    if version.startswith("v3"):
        pd.concat(frames).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")
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
    def _make(specs: list[EpisodeSpec], version: str = "v3.0") -> Path:
        return write_dataset(tmp_path / f"ds-{version}", specs, version)

    return _make
