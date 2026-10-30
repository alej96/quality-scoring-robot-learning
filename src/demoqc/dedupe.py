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
    if len(seg) < 2 or not np.isfinite(seg).all():
        # a corrupted (NaN/Inf) episode is already flagged by check_integrity; comparing
        # its trajectory would propagate NaN into np.interp (see _resample) and corrupt
        # that one row of the pairwise distance matrix rather than just excluding it
        return None
    return seg


def _resample(traj: np.ndarray, n: int) -> np.ndarray:
    """Linearly resample a (T, D) trajectory to n points along its normalized time axis."""
    x_old = np.linspace(0.0, 1.0, len(traj))
    x_new = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(x_new, x_old, traj[:, d]) for d in range(traj.shape[1])], axis=1)


def _pairwise_mean_sq_l2(mat: np.ndarray) -> np.ndarray:
    """Pairwise ``sqrt(mean squared difference))`` between rows of ``mat`` (n, D).

    Mathematically ``np.sqrt(np.mean((mat[:, None] - mat[None, :]) ** 2, axis=2))``, but
    never materializes that (n, n, D) intermediate: it is D times larger than the (n, n)
    result this function actually returns, and D (resample length times active action
    dims) is typically in the hundreds, so it is what actually limits how many episodes
    this check can handle in memory -- not the (n, n) distance matrix itself. Uses
    ``||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b``, computed with one BLAS matrix multiply, so
    peak memory is O(n^2 + n*D) instead of O(n^2*D).

    Only used on a single duplicate group's rows (always small): the full-dataset search
    in ``find_duplicates`` uses the row-blocked helpers below instead, which never hold the
    whole (n, n) matrix either.
    """
    sq_norms = np.sum(mat * mat, axis=1)
    gram = mat @ mat.T
    sq_dist = sq_norms[:, None] + sq_norms[None, :] - 2.0 * gram
    np.maximum(sq_dist, 0.0, out=sq_dist)  # clip tiny negative values from float error
    return np.sqrt(sq_dist / mat.shape[1])


# Each row block computes a (block_rows, n) distance slab; capping its element count bounds
# peak memory to a small constant instead of growing with n^2. 2_000_000 float64 elements is
# 16 MB, negligible next to everything else `score`/`dedupe` already holds in memory.
_DEDUPE_MAX_BLOCK_ELEMS = 2_000_000


def _block_rows_for(n: int, max_block_elems: int) -> int:
    return max(1, min(n, max_block_elems // max(n, 1)))


def _row_block_distances(mat: np.ndarray, sq_norms: np.ndarray, start: int, end: int) -> np.ndarray:
    """Distances from rows ``[start, end)`` of ``mat`` to every row of ``mat``, via the same
    Gram-matrix identity ``_pairwise_mean_sq_l2`` uses, without ever forming the full (n, n)
    matrix -- only this (end - start, n) slab."""
    gram = mat[start:end] @ mat.T
    sq_dist = sq_norms[start:end, None] + sq_norms[None, :] - 2.0 * gram
    np.maximum(sq_dist, 0.0, out=sq_dist)
    return np.sqrt(sq_dist / mat.shape[1])


def _nearest_neighbor_distances(
    mat: np.ndarray, max_block_elems: int = _DEDUPE_MAX_BLOCK_ELEMS
) -> np.ndarray:
    """Per-row nearest-neighbour distance (excluding self), one row block at a time so peak
    memory is O(block_rows * n) instead of ``_pairwise_mean_sq_l2``'s O(n^2) -- the dedupe
    check's own follow-up: at a dataset size large enough for this tool's ``--episodes``
    support to matter, even the (n, n) *result* (not just the O(n^2*D) intermediate the
    2026-10-25 fix removed) no longer fits in memory."""
    n = len(mat)
    sq_norms = np.sum(mat * mat, axis=1)
    block_rows = _block_rows_for(n, max_block_elems)
    nn = np.empty(n)
    for start in range(0, n, block_rows):
        end = min(start + block_rows, n)
        dist = _row_block_distances(mat, sq_norms, start, end)
        local = np.arange(end - start)
        dist[local, start + local] = np.inf  # exclude self
        nn[start:end] = dist.min(axis=1)
    return nn


def _pairs_within(
    mat: np.ndarray, threshold: float, max_block_elems: int = _DEDUPE_MAX_BLOCK_ELEMS
) -> list[tuple[int, int]]:
    """``(i, j)`` row-index pairs with ``i < j`` and distance ``<= threshold``, same memory
    bound as ``_nearest_neighbor_distances``. Duplicate pairs are rare, so this is a small
    list even for a huge ``mat``."""
    n = len(mat)
    sq_norms = np.sum(mat * mat, axis=1)
    block_rows = _block_rows_for(n, max_block_elems)
    pairs: list[tuple[int, int]] = []
    for start in range(0, n, block_rows):
        end = min(start + block_rows, n)
        dist = _row_block_distances(mat, sq_norms, start, end)
        for local_i, i in enumerate(range(start, end)):
            hits = np.flatnonzero(dist[local_i, i + 1 :] <= threshold)
            pairs.extend((i, int(j) + i + 1) for j in hits)
    return pairs


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
    nn = _nearest_neighbor_distances(mat)
    median = float(np.median(nn))
    mad = float(np.median(np.abs(nn - median))) * 1.4826
    if mad < 1e-9:
        mad = max(median * 0.1, 1e-6)
    threshold = max(0.0, median - cfg.dedupe_z * mad)

    uf = _UnionFind(indices)
    for i, j in _pairs_within(mat, threshold):
        uf.union(indices[i], indices[j])

    pos = {idx: i for i, idx in enumerate(indices)}
    for members in uf.groups().values():
        if len(members) < 2:
            continue
        rows = [pos[m] for m in members]
        # a duplicate group is always small, so the O(n^2) _pairwise_mean_sq_l2 here is on
        # the group's own rows only, never on the full (potentially huge) `mat`.
        sub = _pairwise_mean_sq_l2(mat[rows])
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
