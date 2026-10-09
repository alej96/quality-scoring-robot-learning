"""Exact and near-duplicate episode detection.

Community uploads sometimes contain byte-identical episodes (a push re-run that
appended data instead of overwriting it) or near-identical ones (an operator repeats
the same demonstration back-to-back). Either wastes training compute and biases a
policy toward whatever got duplicated, so these are grouped for `demoqc dedupe` and
the `duplicates` section of `demoqc score` to flag.

Exact duplicates are found by hashing each episode's raw ``action``/``observation.state``
arrays. Near-duplicates resample each episode's trimmed, dataset-scaled action
trajectory to a fixed length and compare pairwise L2 distance against the dataset's own
nearest-neighbour distance distribution, so the threshold adapts to how similar episodes
of the same task normally are instead of using an absolute cutoff.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from demoqc.checks import ERROR, WARN, Config, DatasetContext, EpisodeResult, Flag
from demoqc.dataset import Dataset, Episode


@dataclass
class DuplicateGroup:
    episodes: list[int]
    kind: str  # "exact" or "near"
    max_distance: float  # largest pairwise normalized L2 distance within the group; 0 for exact


def _content_hash(ep: Episode) -> str:
    h = hashlib.sha1()
    for arr in (ep.action, ep.state):
        if arr is not None:
            h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def _trimmed_action(ep: Episode, result: EpisodeResult) -> np.ndarray | None:
    if ep.action is None:
        return None
    start, end = result.trim if result.trim else (0, ep.length)
    seg = ep.action[start:end]
    return seg if len(seg) >= 2 else None


def _resample(traj: np.ndarray, n: int) -> np.ndarray:
    """Linearly resample a (T, D) trajectory to n points along its normalized time axis."""
    x_old = np.linspace(0.0, 1.0, len(traj))
    x_new = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(x_new, x_old, traj[:, d]) for d in range(traj.shape[1])], axis=1)


class _UnionFind:
    def __init__(self, items: list[int]) -> None:
        self._parent = {i: i for i in items}

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb

    def groups(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for i in self._parent:
            out.setdefault(self.find(i), []).append(i)
        return out


def find_duplicates(
    ds: Dataset, results: list[EpisodeResult], ctx: DatasetContext, cfg: Config
) -> list[DuplicateGroup]:
    by_hash: dict[str, list[int]] = {}
    for ep in ds.episodes:
        by_hash.setdefault(_content_hash(ep), []).append(ep.index)
    exact_sets = {tuple(sorted(v)) for v in by_hash.values() if len(v) > 1}
    groups = [DuplicateGroup(list(members), "exact", 0.0) for members in exact_sets]
    already_exact = {idx for members in exact_sets for idx in members}

    if ctx.action_scale is None:
        return groups

    result_by_idx = {r.episode_index: r for r in results}
    indices: list[int] = []
    vectors: list[np.ndarray] = []
    for ep in ds.episodes:
        if ep.index in already_exact:
            continue  # already reported as an exact group; keep the near-duplicate matrix smaller
        seg = _trimmed_action(ep, result_by_idx[ep.index])
        if seg is None:
            continue
        scaled = (seg / ctx.action_scale)[:, np.isfinite(ctx.action_scale)]
        if scaled.shape[1] == 0:
            continue
        indices.append(ep.index)
        vectors.append(_resample(scaled, cfg.dedupe_resample_len).ravel())

    if len(indices) < 5:
        return groups

    mat = np.stack(vectors)
    diffs = mat[:, None, :] - mat[None, :, :]
    dist = np.sqrt(np.mean(diffs**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    nn = dist.min(axis=1)
    median = float(np.median(nn))
    mad = float(np.median(np.abs(nn - median))) * 1.4826
    if mad < 1e-9:
        mad = max(median * 0.1, 1e-6)
    threshold = max(0.0, median - cfg.dedupe_z * mad)

    uf = _UnionFind(indices)
    n = len(indices)
    for i in range(n):
        for j in range(i + 1, n):
            if dist[i, j] <= threshold:
                uf.union(indices[i], indices[j])

    pos = {idx: i for i, idx in enumerate(indices)}
    for members in uf.groups().values():
        if len(members) < 2:
            continue
        rows = [pos[m] for m in members]
        sub = dist[np.ix_(rows, rows)]
        max_d = float(sub[np.triu_indices(len(rows), k=1)].max())
        groups.append(DuplicateGroup(sorted(members), "near", round(max_d, 4)))

    return groups


def flag_duplicates(results: list[EpisodeResult], groups: list[DuplicateGroup]) -> None:
    """Flag every episode in a duplicate group except the lowest-indexed (kept) one."""
    by_idx = {r.episode_index: r for r in results}
    for g in groups:
        keep = min(g.episodes)
        for idx in g.episodes:
            if idx == keep:
                continue
            r = by_idx[idx]
            if g.kind == "exact":
                r.flags.append(Flag("duplicate", ERROR, f"exact duplicate of episode {keep}", 50.0))
            else:
                r.flags.append(
                    Flag(
                        "duplicate",
                        WARN,
                        f"near-duplicate of episode {keep} (distance {g.max_distance:.3f})",
                        25.0,
                    )
                )


def format_duplicates(groups: list[DuplicateGroup]) -> str:
    if not groups:
        return "no duplicate episodes found"
    exact = [g for g in groups if g.kind == "exact"]
    near = [g for g in groups if g.kind == "near"]
    lines = []
    if exact:
        lines.append(f"exact duplicates: {len(exact)} group(s)")
        for g in sorted(exact, key=lambda g: g.episodes[0]):
            lines.append(f"  episodes {g.episodes}")
    if near:
        lines.append(f"near duplicates: {len(near)} group(s)")
        for g in sorted(near, key=lambda g: g.episodes[0]):
            lines.append(f"  episodes {g.episodes}  max distance {g.max_distance:.3f}")
    return "\n".join(lines)
