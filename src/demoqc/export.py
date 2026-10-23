"""Write a cleaned copy of a LeRobot dataset: tabular data and metadata only.

Dropping episodes is more than filtering rows. ``episode_index`` and the global ``index`` must
stay contiguous, per-episode metadata needs new indices, ``meta/info.json`` totals change, and
the normalisation statistics in ``meta/stats.json`` (plus any per-episode ``index``/
``episode_index`` stats) would otherwise describe episodes that are no longer there.

v3.0 (``_export_v3``) and v2.x (``_export_v2``) lay out data differently (many episodes per
consolidated parquet/``meta/episodes`` file vs. one parquet per episode and JSONL metadata), so
they're handled separately, sharing only the stats-recomputation helpers below.

For v3.0, data and episode-metadata files keep their relative paths, so the ``data/*`` and
``videos/*`` chunk/file indices stored in ``meta/episodes`` stay valid, and the original
``videos/`` directory can be copied next to the export unchanged. For v2.x, video files are
named after ``episode_index`` directly, so renumbering episodes means each kept episode's video
files must be renamed to match (see the warning ``_export_v2`` returns); copying ``videos/``
unchanged is not enough. ``export_dataset(..., copy_videos=True)`` does this copy/rename
itself for a local source that has a ``videos/`` directory on disk (video bytes are copied
as-is, never read, decoded or re-encoded); the CLI rejects ``--copy-videos`` for a Hub source
up front, since videos are never downloaded by this tool and there would be nothing on disk
to copy.

Optionally, ``export_dataset`` can also apply the ``trim`` suggested by ``demoqc score`` for an
episode: drop the idle frames at its start/end, renumber ``frame_index`` and shift ``timestamp``
so it still starts near 0. For v3.0 this also shrinks the episode's
``videos/*/from_timestamp``/``to_timestamp`` window by the same number of frames, so the (never
read or re-encoded) video stays aligned with the trimmed data. v2.x has no such per-episode time
span to narrow -- its video file covers the whole original episode -- so a trimmed v2.x export
instead leaves a warning that the video itself must also be cut to the same frame range before
training on it.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from demoqc.dataset import Dataset

# per-episode stats that describe the (now rewritten) indices rather than the demonstrations
_REINDEXED = ("episode_index", "index")
_QUANTILE_KEY = re.compile(r"q(\d\d)$")
_HANDLED_META_V3 = {"info.json", "stats.json"}
_HANDLED_META_V2 = {"info.json", "stats.json", "episodes.jsonl", "episodes_stats.jsonl"}
_DEFAULT_DATA_PATH_V2 = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
_DEFAULT_VIDEO_PATH_V2 = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
)


@dataclass
class ExportResult:
    out: Path
    kept: list[int]
    dropped: list[int]
    frames: int
    has_videos: bool
    version: str = "v3.0"
    trimmed_frames: int = 0
    warnings: list[str] = field(default_factory=list)
    videos_copied: bool = False


def _load_report(report_path: Path, ds: Dataset) -> dict:
    """Parse a ``demoqc score --json`` report and check it was made for this dataset.

    Refuses a report that was clearly made for another dataset (different episode or frame
    count), because applying its indices would silently drop or trim the wrong episodes.
    """
    try:
        report = json.loads(Path(report_path).read_text())
        episodes, frames = report["dataset"]["episodes"], report["dataset"]["frames"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"{report_path} is not a demoqc score report ({exc!r})") from exc
    if (episodes, frames) != (len(ds.episodes), ds.total_frames):
        raise ValueError(
            f"{report_path} was made for a different dataset: it has {episodes} episodes / "
            f"{frames} frames, this one has {len(ds.episodes)} / {ds.total_frames}"
        )
    return report


def keep_from_report_dict(report: dict) -> list[int]:
    """Episode indices to keep according to an in-memory ``demoqc score`` report."""
    return [int(i) for i in report["summary"]["keep"]]


def trim_from_report_dict(report: dict) -> dict[int, tuple[int, int]]:
    """Per-episode ``[start, end)`` frame ranges to keep, from an in-memory report.

    Only episodes with a suggested trim (idle time at the start/end) are included.
    """
    return {
        int(e["episode_index"]): (int(e["trim"][0]), int(e["trim"][1]))
        for e in report["episodes"]
        if e.get("trim")
    }


def keep_from_report(report_path: Path, ds: Dataset) -> list[int]:
    """Episode indices to keep according to a ``demoqc score --json`` report file."""
    report = _load_report(report_path, ds)
    try:
        return keep_from_report_dict(report)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{report_path} is not a demoqc score report ({exc!r})") from exc


def trim_from_report(report_path: Path, ds: Dataset) -> dict[int, tuple[int, int]]:
    """Per-episode ``[start, end)`` frame ranges to keep, from a ``demoqc score --json`` report
    file. Only episodes with a suggested trim (idle time at the start/end) are included.
    """
    report = _load_report(report_path, ds)
    try:
        return trim_from_report_dict(report)
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError(f"{report_path} is not a demoqc score report ({exc!r})") from exc


def export_dataset(
    src: Path,
    out: Path,
    keep: Iterable[int],
    trim: dict[int, tuple[int, int]] | None = None,
    copy_videos: bool = False,
) -> ExportResult:
    """Write the episodes in ``keep`` from the LeRobot dataset at ``src`` to ``out``.

    Kept episodes are renumbered 0..k-1 in their original order. ``trim`` optionally maps an
    original episode index to the ``[start, end)`` frame range to keep for it (see
    :func:`trim_from_report`); episodes not in ``trim`` are kept in full. ``copy_videos``
    additionally copies (for v2.x, renaming to match the new episode indices) the source's
    video files next to the export; it requires ``src`` to be a local directory that actually
    has a ``videos/`` directory (never the case for a dataset downloaded from the Hub, since
    videos are never downloaded by this tool).
    """
    src, out = Path(src), Path(out)
    info = json.loads((src / "meta" / "info.json").read_text())
    version = str(info.get("codebase_version", "unknown"))
    fps = float(info["fps"])
    if out.resolve() == src.resolve():
        raise ValueError("the output directory must differ from the source dataset")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} already exists and is not empty")

    keep_ids = np.unique(np.asarray(list(keep), dtype=np.int64))
    if keep_ids.size == 0:
        raise ValueError("no episodes to keep")

    if not version.startswith("v3"):
        return _export_v2(src, out, info, fps, keep_ids, trim, copy_videos)
    return _export_v3(src, out, info, fps, keep_ids, trim, copy_videos)


def _export_v3(
    src: Path,
    out: Path,
    info: dict,
    fps: float,
    keep_ids: np.ndarray,
    trim: dict[int, tuple[int, int]] | None,
    copy_videos: bool = False,
) -> ExportResult:
    data_files = sorted((src / "data").rglob("*.parquet"))
    episode_files = sorted((src / "meta" / "episodes").rglob("*.parquet"))
    if not data_files or not episode_files:
        raise ValueError(f"{src} has no data/ or meta/episodes/ parquet files")

    stats_path = src / "meta" / "stats.json"
    old_stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}

    all_ep = np.concatenate(
        [pq.read_table(f, columns=["episode_index"])[0].to_numpy() for f in data_files]
    )
    present, present_lengths = np.unique(all_ep, return_counts=True)
    unknown = np.setdiff1d(keep_ids, present)
    if unknown.size:
        raise ValueError(f"episodes not in the dataset: {unknown.tolist()}")
    raw_lengths = present_lengths[np.searchsorted(present, keep_ids)]

    trim = dict(trim) if trim else {}
    trim_starts = np.zeros(len(keep_ids), dtype=np.int64)
    trim_ends = raw_lengths.copy()  # default: keep the whole episode
    if trim:
        keep_set = set(keep_ids.tolist())
        bad = []
        for ep, (s, e) in trim.items():
            if ep not in keep_set:
                continue  # trimmed episode is being dropped, not kept
            pos = int(np.searchsorted(keep_ids, ep))
            length = int(raw_lengths[pos])
            if not (0 <= s < e <= length):
                bad.append((ep, s, e, length))
                continue
            trim_starts[pos], trim_ends[pos] = s, e
        if bad:
            raise ValueError(f"invalid trim (start, end, episode length): {bad}")

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
        new_ep = np.searchsorted(keep_ids, table["episode_index"].to_numpy())
        if trim:
            local_frame = table["frame_index"].to_numpy()
            keep_frame = (local_frame >= trim_starts[new_ep]) & (local_frame < trim_ends[new_ep])
            if not keep_frame.all():
                table = table.filter(pa.array(keep_frame))
                new_ep = new_ep[keep_frame]
                local_frame = local_frame[keep_frame]
        n = table.num_rows
        if n == 0:
            continue
        table = _set(table, "episode_index", new_ep)
        table = _set(table, "index", offset + np.arange(n))
        if trim:
            shift = trim_starts[new_ep]
            table = _set(table, "frame_index", local_frame - shift)
            if "timestamp" in table.column_names:
                table = _set(table, "timestamp", table["timestamp"].to_numpy() - shift / fps)
        for name in stat_arrays:
            if name in table.column_names:
                stat_arrays[name].append(_as_matrix(table[name]))
        new_episode_rows.append(new_ep)
        _write(table, out / f.relative_to(src))
        offset += n

    if not new_episode_rows:
        raise ValueError("no frames left after trimming")
    new_ep_all = np.concatenate(new_episode_rows)
    if np.any(np.diff(new_ep_all) < 0):
        raise ValueError("episodes are not stored in ascending order across the data files")
    counts = np.bincount(new_ep_all, minlength=len(keep_ids))
    ends = np.cumsum(counts)
    starts = ends - counts
    trimmed_frames = int(raw_lengths.sum() - counts.sum())

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
            if np.any(declared != raw_lengths[new_ep]):
                warnings.append(
                    "meta/episodes 'length' disagrees with the data for some episodes "
                    "(run `demoqc score` to see the metadata flags)"
                )
            table = _set(table, "length", counts[new_ep])
        if trim:
            table = _shift_video_spans(table, new_ep, trim_starts, raw_lengths - trim_ends, fps)
        episode_tables.append((f, table))
    if sum(t.num_rows for _, t in episode_tables) != len(keep_ids):
        raise ValueError("meta/episodes is missing rows for some of the episodes to keep")

    episode_tables = [(f, _restat_indices(t, starts, ends)) for f, t in episode_tables]
    for f, table in episode_tables:
        _write(table, out / f.relative_to(src))

    if old_stats:
        episodes_table = pa.concat_tables([t for _, t in episode_tables])
        stats = _recompute_stats(
            old_stats,
            stat_arrays,
            warnings,
            lambda feature, keys: _combine_episode_stats(episodes_table, feature, keys),
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
        if p.is_file() and rel.parts[0] != "episodes" and rel.as_posix() not in _HANDLED_META_V3:
            (out / "meta" / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, out / "meta" / rel)

    has_videos = any(f.get("dtype") == "video" for f in info.get("features", {}).values())
    videos_copied = False
    if copy_videos:
        videos_src = src / "videos"
        if videos_src.is_dir():
            shutil.copytree(videos_src, out / "videos")
            videos_copied = True
        elif has_videos:
            warnings.append("--copy-videos requested but the source has no videos/ directory")

    dropped = np.setdiff1d(present, keep_ids)
    return ExportResult(
        out=out,
        kept=keep_ids.tolist(),
        dropped=dropped.tolist(),
        frames=int(ends[-1]),
        has_videos=has_videos,
        version="v3.0",
        trimmed_frames=trimmed_frames,
        warnings=sorted(set(warnings)),
        videos_copied=videos_copied,
    )


def _export_v2(
    src: Path,
    out: Path,
    info: dict,
    fps: float,
    keep_ids: np.ndarray,
    trim: dict[int, tuple[int, int]] | None = None,
    copy_videos: bool = False,
) -> ExportResult:
    """Write the episodes in ``keep_ids`` from a v2.x dataset (one parquet per episode,
    JSONL metadata) to ``out``.

    Unlike v3.0, a v2.x video file is named after its episode's index, so renaming episodes
    invalidates the ``videos/`` directory's filenames; the caller is warned to rename them.
    A v2.x video also has no per-episode time span stored in its metadata to narrow when
    trimming (unlike v3.0's ``videos/*/from_timestamp``/``to_timestamp``), so a trimmed export
    leaves a warning that the video itself must be cut to match instead.
    """
    episode_rows = {
        int(r["episode_index"]): r for r in _read_jsonl(src / "meta" / "episodes.jsonl")
    }
    present = np.array(sorted(episode_rows), dtype=np.int64)
    unknown = np.setdiff1d(keep_ids, present)
    if unknown.size:
        raise ValueError(f"episodes not in the dataset: {unknown.tolist()}")

    data_files: dict[int, Path] = {}
    for f in sorted((src / "data").rglob("episode_*.parquet")):
        m = re.fullmatch(r"episode_(\d+)\.parquet", f.name)
        if m:
            data_files[int(m.group(1))] = f
    missing = [int(e) for e in keep_ids.tolist() if e not in data_files]
    if missing:
        raise ValueError(f"no data file for episode(s): {missing}")

    trim = dict(trim) if trim else {}
    keep_set = set(keep_ids.tolist())
    bad = []
    for ep, (s, e) in trim.items():
        if ep not in keep_set:
            continue  # trimmed episode is being dropped, not kept
        length = pq.read_metadata(data_files[ep]).num_rows
        if not (0 <= s < e <= length):
            bad.append((ep, s, e, length))
    if bad:
        raise ValueError(f"invalid trim (start, end, episode length): {bad}")

    stat_rows_by_ep = {
        int(r["episode_index"]): r for r in _read_jsonl(src / "meta" / "episodes_stats.jsonl")
    }
    video_features = sorted(
        name
        for name, feat in info.get("features", {}).items()
        if isinstance(feat, dict) and feat.get("dtype") == "video"
    )
    chunks_size = int(info.get("chunks_size", 1000))
    data_path = info.get("data_path") or _DEFAULT_DATA_PATH_V2
    video_path = info.get("video_path") or _DEFAULT_VIDEO_PATH_V2

    stats_path = src / "meta" / "stats.json"
    old_stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}

    stat_arrays: dict[str, list[np.ndarray]] = {k: [] for k in old_stats}
    new_episode_rows: list[dict] = []
    new_stat_rows: list[dict] = []
    missing_videos: set[tuple[str, int]] = set()
    offset = 0
    trimmed_frames = 0
    for new_idx, orig_idx in enumerate(keep_ids.tolist()):
        table = pq.read_table(data_files[orig_idx])
        full_n = table.num_rows
        start, end = trim.get(orig_idx, (0, full_n))
        if (start, end) != (0, full_n):
            table = table.slice(start, end - start)
            trimmed_frames += full_n - (end - start)
        n = table.num_rows
        table = _set(table, "episode_index", np.full(n, new_idx, dtype=np.int64))
        table = _set(table, "index", offset + np.arange(n))
        table = _set(table, "frame_index", np.arange(n))
        if start and "timestamp" in table.column_names:
            table = _set(table, "timestamp", table["timestamp"].to_numpy() - start / fps)
        for name in stat_arrays:
            if name in table.column_names:
                stat_arrays[name].append(_as_matrix(table[name]))
        dest = out / data_path.format(episode_chunk=new_idx // chunks_size, episode_index=new_idx)
        _write(table, dest)

        if copy_videos:
            for camera in video_features:
                src_video = src / video_path.format(
                    episode_chunk=orig_idx // chunks_size, video_key=camera, episode_index=orig_idx
                )
                if not src_video.exists():
                    missing_videos.add((camera, orig_idx))
                    continue
                dest_video = out / video_path.format(
                    episode_chunk=new_idx // chunks_size, video_key=camera, episode_index=new_idx
                )
                dest_video.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src_video, dest_video)

        row = dict(episode_rows[orig_idx])
        row["episode_index"] = new_idx
        row["length"] = n
        new_episode_rows.append(row)

        stat_row = stat_rows_by_ep.get(orig_idx)
        if stat_row is not None:
            stats = dict(stat_row.get("stats") or {})
            for feature, values in (
                ("index", offset + np.arange(n)),
                ("episode_index", np.full(n, new_idx)),
            ):
                if feature in stats:
                    stats[feature] = _summarise(
                        values.astype(np.float64)[:, None], list(stats[feature])
                    )
            new_stat_rows.append({"episode_index": new_idx, "stats": stats})

        offset += n

    (out / "meta").mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "meta" / "episodes.jsonl", new_episode_rows)

    warnings: list[str] = []
    if new_stat_rows:
        _write_jsonl(out / "meta" / "episodes_stats.jsonl", new_stat_rows)
        if len(new_stat_rows) != len(keep_ids):
            warnings.append(
                "meta/episodes_stats.jsonl is missing rows for some kept episodes; "
                "meta/stats.json was not fully recomputed from them"
            )

    if old_stats:
        stats = _recompute_stats(
            old_stats,
            stat_arrays,
            warnings,
            lambda feature, keys: _combine_episode_stats_v2(new_stat_rows, feature, keys),
        )
        (out / "meta" / "stats.json").write_text(json.dumps(stats, indent=4) + "\n")
    else:
        warnings.append("no meta/stats.json in the source; none written")

    info = dict(info)
    info["total_episodes"] = len(keep_ids)
    info["total_frames"] = offset
    info["splits"] = {"train": f"0:{len(keep_ids)}"}
    if "total_chunks" in info:
        info["total_chunks"] = max(1, -(-len(keep_ids) // chunks_size))
    if "total_videos" in info and video_features:
        info["total_videos"] = len(keep_ids) * len(video_features)
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=4) + "\n")

    for p in sorted((src / "meta").rglob("*")):
        rel = p.relative_to(src / "meta")
        if p.is_file() and rel.as_posix() not in _HANDLED_META_V2:
            (out / "meta" / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, out / "meta" / rel)

    if video_features and not copy_videos:
        video_warning = (
            "videos were not copied or renamed: a v2.x video file is named after its episode's "
            "index, so copy each kept episode's video (every camera) from "
            "episode_<old index>.mp4 to episode_<new index>.mp4, where <new index> is its "
            "0-based position in `kept`"
        )
        if trimmed_frames:
            video_warning += (
                "; a v2.x video also has no stored time span to narrow (unlike v3.0), so any "
                "trimmed episode's video must also be cut to its `trim` frame range from the "
                "report before training on it, or it will run longer than the trimmed data"
            )
        warnings.append(video_warning)
    elif video_features and copy_videos:
        if missing_videos:
            warnings.append(
                f"--copy-videos: no video file found for (camera, episode) {sorted(missing_videos)}"
            )
        if trimmed_frames:
            warnings.append(
                "videos were copied and renamed to match the new episode indices, but a v2.x "
                "video has no stored per-episode time span to narrow (unlike v3.0), so each "
                "trimmed episode's copied video still covers the whole original span; cut it "
                "to the `trim` frame range from the report before training on it"
            )

    dropped = np.setdiff1d(present, keep_ids)
    return ExportResult(
        out=out,
        kept=keep_ids.tolist(),
        dropped=dropped.tolist(),
        frames=offset,
        has_videos=bool(video_features),
        version="v2.x",
        trimmed_frames=trimmed_frames,
        warnings=sorted(set(warnings)),
        videos_copied=copy_videos and bool(video_features),
    )


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _write(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _shift_video_spans(
    table: pa.Table, new_ep: np.ndarray, lead: np.ndarray, trail: np.ndarray, fps: float
) -> pa.Table:
    """Narrow each camera's ``from_timestamp``/``to_timestamp`` window by ``lead``/``trail``
    frames (indexed by ``new_ep``), so it still points at exactly the kept part of the video."""
    prefixes = {
        c[: -len("/from_timestamp")]
        for c in table.column_names
        if c.startswith("videos/") and c.endswith("/from_timestamp")
    }
    for prefix in prefixes:
        from_col, to_col = f"{prefix}/from_timestamp", f"{prefix}/to_timestamp"
        if to_col not in table.column_names:
            continue
        table = _set(table, from_col, table[from_col].to_numpy() + lead[new_ep] / fps)
        table = _set(table, to_col, table[to_col].to_numpy() - trail[new_ep] / fps)
    return table


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
    old: dict,
    arrays: dict[str, list[np.ndarray]],
    warnings: list[str],
    combine: Callable[[str, list[str]], dict | None],
) -> dict:
    """Dataset-level stats for the kept episodes.

    Tabular features are recomputed exactly from the kept frames (``arrays``). Video features
    (which we cannot read) are combined from each format's own per-episode stats via ``combine``.
    """
    stats = {}
    for feature, old_stats in old.items():
        keys = list(old_stats)
        if arrays.get(feature):
            stats[feature] = _summarise(np.concatenate(arrays[feature]), keys)
            continue
        combined = combine(feature, keys)
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
    """v3.0: per-episode stats as ``stats/{feature}/{key}`` columns of ``meta/episodes``."""
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
    return _combine_from_arrays(per_episode, keys)


def _combine_episode_stats_v2(stat_rows: list[dict], feature: str, keys: list[str]) -> dict | None:
    """v2.x: per-episode stats as nested dicts, one ``meta/episodes_stats.jsonl`` row each."""
    if not stat_rows:
        return None
    try:
        per_episode = {
            k: np.array([np.asarray(r["stats"][feature][k], dtype=np.float64) for r in stat_rows])
            for k in keys
        }
    except (KeyError, TypeError, ValueError):
        return None  # feature/key missing or null stats for some episode
    return _combine_from_arrays(per_episode, keys)


def _combine_from_arrays(per_episode: dict[str, np.ndarray], keys: list[str]) -> dict | None:
    """Count-weighted combination of one feature's per-episode stats into dataset-level stats."""
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
