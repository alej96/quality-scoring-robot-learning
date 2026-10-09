"""Write a cleaned copy of a LeRobot v3.0 dataset: tabular data and metadata only.

Dropping episodes from a v3.0 dataset is more than filtering rows. ``episode_index`` and the
global ``index`` must stay contiguous, ``meta/episodes`` rows need their new indices and
``dataset_from_index``/``dataset_to_index`` ranges, ``meta/info.json`` totals change, and the
normalisation statistics in ``meta/stats.json`` (plus the per-episode ``stats/index`` and
``stats/episode_index`` columns) would otherwise describe episodes that are no longer there.

Data and episode-metadata files keep their relative paths, so the ``data/*`` and ``videos/*``
chunk/file indices stored in ``meta/episodes`` stay valid, and the original ``videos/``
directory can be copied next to the export unchanged (videos are never read or copied here).
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from demoqc.dataset import Dataset

# per-episode stats that describe the (now rewritten) indices rather than the demonstrations
_REINDEXED = ("episode_index", "index")
_QUANTILE_KEY = re.compile(r"q(\d\d)$")
_HANDLED_META = {"info.json", "stats.json"}


@dataclass
class ExportResult:
    out: Path
    kept: list[int]
    dropped: list[int]
    frames: int
    has_videos: bool
    warnings: list[str] = field(default_factory=list)


def keep_from_report(report_path: Path, ds: Dataset) -> list[int]:
    """Episode indices to keep according to a ``demoqc score --json`` report.

    Refuses a report that was clearly made for another dataset (different episode or frame
    count), because applying its indices would silently drop the wrong episodes.
    """
    try:
        report = json.loads(Path(report_path).read_text())
        keep = [int(i) for i in report["summary"]["keep"]]
        episodes, frames = report["dataset"]["episodes"], report["dataset"]["frames"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"{report_path} is not a demoqc score report ({exc!r})") from exc
    if (episodes, frames) != (len(ds.episodes), ds.total_frames):
        raise ValueError(
            f"{report_path} was made for a different dataset: it has {episodes} episodes / "
            f"{frames} frames, this one has {len(ds.episodes)} / {ds.total_frames}"
        )
    return keep


def export_dataset(src: Path, out: Path, keep: Iterable[int]) -> ExportResult:
    """Write the episodes in ``keep`` from the v3.0 dataset at ``src`` to ``out``.

    Kept episodes are renumbered 0..k-1 in their original order.
    """
    src, out = Path(src), Path(out)
    info = json.loads((src / "meta" / "info.json").read_text())
    version = str(info.get("codebase_version", "unknown"))
    if not version.startswith("v3"):
        raise ValueError(
            f"export supports LeRobot v3.0 datasets; this one is {version} "
            "(v2.x export is not implemented yet)"
        )
    if out.resolve() == src.resolve():
        raise ValueError("the output directory must differ from the source dataset")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} already exists and is not empty")

    data_files = sorted((src / "data").rglob("*.parquet"))
    episode_files = sorted((src / "meta" / "episodes").rglob("*.parquet"))
    if not data_files or not episode_files:
        raise ValueError(f"{src} has no data/ or meta/episodes/ parquet files")

    stats_path = src / "meta" / "stats.json"
    old_stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}

    keep_ids = np.unique(np.asarray(list(keep), dtype=np.int64))
    if keep_ids.size == 0:
        raise ValueError("no episodes to keep")
    present = np.unique(
        np.concatenate(
            [pq.read_table(f, columns=["episode_index"])[0].to_numpy() for f in data_files]
        )
    )
    unknown = np.setdiff1d(keep_ids, present)
    if unknown.size:
        raise ValueError(f"episodes not in the dataset: {unknown.tolist()}")

    warnings: list[str] = []
    stat_arrays: dict[str, list[np.ndarray]] = {k: [] for k in old_stats}
    new_episode_rows: list[np.ndarray] = []
    offset = 0
    for f in data_files:
        table = pq.read_table(f)
        mask = np.isin(table["episode_index"].to_numpy(), keep_ids)
        if not mask.any():
            continue
        table = table.filter(pa.array(mask))
        n = table.num_rows
        new_ep = np.searchsorted(keep_ids, table["episode_index"].to_numpy())
        table = _set(table, "episode_index", new_ep)
        table = _set(table, "index", offset + np.arange(n))
        for name in stat_arrays:
            if name in table.column_names:
                stat_arrays[name].append(_as_matrix(table[name]))
        new_episode_rows.append(new_ep)
        _write(table, out / f.relative_to(src))
        offset += n

    new_ep_all = np.concatenate(new_episode_rows)
    if np.any(np.diff(new_ep_all) < 0):
        raise ValueError("episodes are not stored in ascending order across the data files")
    counts = np.bincount(new_ep_all, minlength=len(keep_ids))
    ends = np.cumsum(counts)
    starts = ends - counts

    episode_tables = []
    for f in episode_files:
        table = pq.read_table(f)
        table = table.filter(pa.array(np.isin(table["episode_index"].to_numpy(), keep_ids)))
        if table.num_rows == 0:
            continue
        new_ep = np.searchsorted(keep_ids, table["episode_index"].to_numpy())
        table = _set(table, "episode_index", new_ep)
        table = _set(table, "dataset_from_index", starts[new_ep])
        table = _set(table, "dataset_to_index", ends[new_ep])
        if "length" in table.column_names:
            declared = table["length"].to_numpy()
            if np.any(declared != counts[new_ep]):
                warnings.append(
                    "meta/episodes 'length' disagrees with the data for some episodes "
                    "(run `demoqc score` to see the metadata flags)"
                )
        episode_tables.append((f, table))
    if sum(t.num_rows for _, t in episode_tables) != len(keep_ids):
        raise ValueError("meta/episodes is missing rows for some of the episodes to keep")

    episode_tables = [(f, _restat_indices(t, starts, ends)) for f, t in episode_tables]
    for f, table in episode_tables:
        _write(table, out / f.relative_to(src))

    if old_stats:
        stats = _recompute_stats(
            old_stats, stat_arrays, pa.concat_tables([t for _, t in episode_tables]), warnings
        )
        (out / "meta").mkdir(parents=True, exist_ok=True)
        (out / "meta" / "stats.json").write_text(json.dumps(stats, indent=4) + "\n")
    else:
        warnings.append("no meta/stats.json in the source; none written")

    info["total_episodes"] = len(keep_ids)
    info["total_frames"] = int(ends[-1])
    info["splits"] = {"train": f"0:{len(keep_ids)}"}
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=4) + "\n")

    for p in sorted((src / "meta").rglob("*")):
        rel = p.relative_to(src / "meta")
        if p.is_file() and rel.parts[0] != "episodes" and rel.as_posix() not in _HANDLED_META:
            (out / "meta" / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, out / "meta" / rel)

    dropped = np.setdiff1d(present, keep_ids)
    return ExportResult(
        out=out,
        kept=keep_ids.tolist(),
        dropped=dropped.tolist(),
        frames=int(ends[-1]),
        has_videos=any(f.get("dtype") == "video" for f in info.get("features", {}).values()),
        warnings=sorted(set(warnings)),
    )


def _write(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _set(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    if name not in table.column_names:
        return table
    i = table.schema.get_field_index(name)
    return table.set_column(
        i, table.schema.field(i), pa.array(values, type=table.schema.field(i).type)
    )


def _as_matrix(column: pa.ChunkedArray) -> np.ndarray:
    """A column as float64 of shape (rows, dims); scalar columns have one dim."""
    values = column.to_numpy()
    if values.dtype == object:
        values = np.stack(values)
    return values.astype(np.float64).reshape(len(values), -1)


def _summarise(x: np.ndarray, keys: Iterable[str]) -> dict[str, list]:
    """LeRobot-style stats (per dimension, population std) of an (n, d) matrix."""
    out: dict[str, list] = {}
    for key in keys:
        quantile = _QUANTILE_KEY.fullmatch(key)
        if key == "min":
            out[key] = x.min(axis=0).tolist()
        elif key == "max":
            out[key] = x.max(axis=0).tolist()
        elif key == "mean":
            out[key] = x.mean(axis=0).tolist()
        elif key == "std":
            out[key] = x.std(axis=0).tolist()
        elif key == "count":
            out[key] = [len(x)]
        elif quantile:
            out[key] = np.quantile(x, int(quantile.group(1)) / 100, axis=0).tolist()
    return out


def _restat_indices(table: pa.Table, starts: np.ndarray, ends: np.ndarray) -> pa.Table:
    """Recompute the per-episode stats of ``index`` and ``episode_index`` after re-indexing."""
    new_ep = table["episode_index"].to_numpy()
    for feature in _REINDEXED:
        keys = [c.split("/")[2] for c in table.column_names if c.startswith(f"stats/{feature}/")]
        if not keys:
            continue
        per_episode = []
        for e in new_ep:
            values = (
                np.arange(starts[e], ends[e])
                if feature == "index"
                else np.full(ends[e] - starts[e], e)
            )
            per_episode.append(_summarise(values.astype(np.float64)[:, None], keys))
        for key in keys:
            name = f"stats/{feature}/{key}"
            i = table.schema.get_field_index(name)
            field_ = table.schema.field(i)
            column = pa.array([row[key] for row in per_episode], type=field_.type)
            table = table.set_column(i, field_, column)
    return table


def _recompute_stats(
    old: dict, arrays: dict[str, list[np.ndarray]], episodes: pa.Table, warnings: list[str]
) -> dict:
    """Dataset-level stats for the kept episodes.

    Tabular features are recomputed exactly from the kept frames. Video features (which we
    cannot read) are combined from the per-episode stats in ``meta/episodes``.
    """
    stats = {}
    for feature, old_stats in old.items():
        keys = list(old_stats)
        if arrays.get(feature):
            stats[feature] = _summarise(np.concatenate(arrays[feature]), keys)
            continue
        combined = _combine_episode_stats(episodes, feature, keys)
        if combined is None:
            warnings.append(
                f"stats for '{feature}' could not be recomputed and still describe the "
                "original dataset"
            )
            stats[feature] = old_stats
        else:
            stats[feature] = combined
    return stats


def _combine_episode_stats(episodes: pa.Table, feature: str, keys: list[str]) -> dict | None:
    names = [f"stats/{feature}/{k}" for k in keys]
    if not all(n in episodes.column_names for n in names):
        return None
    try:
        per_episode = {
            k: np.array([np.asarray(v, dtype=np.float64) for v in episodes[n].to_pylist()])
            for k, n in zip(keys, names, strict=True)
        }
    except (TypeError, ValueError):
        return None  # null stats for some episode
    if "count" not in per_episode:
        return None
    count = per_episode["count"].reshape(len(per_episode["count"]), -1)[:, 0]
    weight = count / count.sum()

    def weighted(a: np.ndarray) -> np.ndarray:
        return (weight.reshape(-1, *([1] * (a.ndim - 1))) * a).sum(axis=0)

    out: dict[str, list] = {}
    mean = weighted(per_episode["mean"]) if "mean" in per_episode else None
    for key in keys:
        a = per_episode[key]
        if key == "min":
            out[key] = a.min(axis=0).tolist()
        elif key == "max":
            out[key] = a.max(axis=0).tolist()
        elif key == "mean":
            out[key] = mean.tolist()
        elif key == "std" and mean is not None:
            out[key] = np.sqrt(weighted(a**2 + (per_episode["mean"] - mean) ** 2)).tolist()
        elif key == "count":
            out[key] = [int(count.sum())]
        elif _QUANTILE_KEY.fullmatch(key):
            out[key] = weighted(a).tolist()
        else:
            return None
    return out
