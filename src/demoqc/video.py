"""Frame-level video checks (opt-in, ``--decode-video``).

The ``camera`` check in ``checks.py`` reads LeRobot's own per-episode pixel mean/std from
metadata, so it can only see a camera that is dead for an *entire* episode. A camera that
disconnects and reconnects mid-recording, or a video file that is truncated or otherwise
fails to decode, averages out across the whole episode and slips past it. Seeing either
needs actual frame sampling, so this stays behind an optional dependency (``av``, PyAV) and
an explicit CLI flag; it is never touched by the default code path, and only for a local
source whose video files are already on disk (videos are never downloaded from the Hub).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Must match export.py's _DEFAULT_VIDEO_PATH_V2; duplicated (rather than imported) so this
# module, and the optional `av` dependency it pulls in, stays independent of the export path.
DEFAULT_VIDEO_PATH_V2 = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
)

# LeRobot v3.0's default ``info.json["video_path"]`` template: unlike v2.x, a v3.0 video file
# is shared by several consecutive episodes (one per camera per chunk/file), so it's addressed
# by chunk/file index rather than by episode index directly.
DEFAULT_VIDEO_PATH_V3 = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"


@dataclass
class FrameSample:
    t: float
    mean: float  # pixel mean, 0-1 scale
    std: float  # pixel std, 0-1 scale
    # small block-averaged thumbnail, for frame-to-frame diffing (longest_frozen_run);
    # None for samples built without a real decoded frame (e.g. in tests of the pure
    # mean/std-based longest_dark_run).
    thumb: np.ndarray | None = None


def _thumbnail(arr: np.ndarray, size: int = 32) -> np.ndarray:
    """Block-averaged ``size`` x ``size`` downsample of a grayscale frame, small and cheap
    enough to hold one per sampled frame and diff frame-to-frame.

    ``size`` needs to be large enough that a small, localised change (a robot arm that
    occupies a corner of a wide shot) still moves at least one block's average
    noticeably -- real validation against a public dataset found that an 8x8 thumbnail
    diluted real motion below the noise floor of a genuinely frozen frame; 32x32 keeps
    each block small enough to still catch it without comparing full-resolution frames.
    """
    h, w = arr.shape
    ys = np.linspace(0, h, size + 1).astype(int)
    xs = np.linspace(0, w, size + 1).astype(int)
    row_sums = np.add.reduceat(arr, ys[:-1], axis=0) / np.diff(ys)[:, None]
    return np.add.reduceat(row_sums, xs[:-1], axis=1) / np.diff(xs)[None, :]


def is_available() -> bool:
    """Whether the optional ``av`` dependency (``pip install demoqc[video]``) is installed."""
    import importlib.util

    return importlib.util.find_spec("av") is not None


def resolve_v2_video_path(
    root: Path, template: str | None, chunks_size: int, episode_index: int, camera: str
) -> Path:
    """Local path LeRobot v2.x stores ``camera``'s video for ``episode_index`` at."""
    tpl = template or DEFAULT_VIDEO_PATH_V2
    return root / tpl.format(
        episode_chunk=episode_index // chunks_size, video_key=camera, episode_index=episode_index
    )


def resolve_v3_video_path(
    root: Path, template: str | None, chunk_index: int, file_index: int, camera: str
) -> Path:
    """Local path LeRobot v3.0 stores ``camera``'s video file at, for the chunk/file index
    an episode's ``meta/episodes`` row names (``videos/{camera}/chunk_index``/``file_index``).
    The file itself may hold several consecutive episodes; the caller narrows to one
    episode's footage with its own ``from_timestamp``/``to_timestamp`` span."""
    tpl = template or DEFAULT_VIDEO_PATH_V3
    return root / tpl.format(chunk_index=chunk_index, file_index=file_index, video_key=camera)


def decode_samples(
    path: Path, max_samples: int = 20, time_range: tuple[float, float] | None = None
) -> list[FrameSample] | None:
    """Up to ``max_samples`` evenly-spaced grayscale frame stats from the video at ``path``.

    ``time_range``, when given, narrows decoding to ``[start, end)`` seconds within the
    file instead of the whole thing -- needed for LeRobot v3.0, where one video file holds
    several consecutive episodes' footage back to back (unlike v2.x, one file per episode).

    Returns ``None`` if the file can't be decoded at all (corrupted container, unsupported
    codec, truncated download) -- a real defect in its own right, not something to let
    propagate and abort the whole ``score`` run.
    """
    import av  # optional dependency; only ever imported once this path actually runs

    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else 1.0
            if time_range is not None:
                start_s, end_s = time_range
                container.seek(max(0, int(start_s * 1_000_000)), backward=True)
                in_range = []
                for i, frame in enumerate(container.decode(stream)):
                    t = float(frame.time) if frame.time is not None else i / fps
                    if t < start_s:
                        continue
                    if t >= end_s:
                        break
                    arr = frame.to_ndarray(format="gray").astype(np.float64) / 255.0
                    in_range.append(
                        FrameSample(t, float(arr.mean()), float(arr.std()), _thumbnail(arr))
                    )
                if len(in_range) <= max_samples:
                    return in_range
                positions = np.linspace(0, len(in_range) - 1, max_samples)
                idx = sorted(set(positions.round(0).astype(int)))
                return [in_range[i] for i in idx]
            total = stream.frames or 0
            step = max(1, total // max_samples) if total else 1
            out = []
            for i, frame in enumerate(container.decode(stream)):
                if i % step != 0:
                    continue
                arr = frame.to_ndarray(format="gray").astype(np.float64) / 255.0
                t = float(frame.time) if frame.time is not None else i / fps
                out.append(FrameSample(t, float(arr.mean()), float(arr.std()), _thumbnail(arr)))
                if len(out) >= max_samples:
                    break
            return out
    except Exception:
        return None


def longest_dark_run(
    samples: list[FrameSample], dark_mean_max: float, dark_std_max: float, min_run_frac: float
) -> tuple[int, int] | None:
    """Index range ``[start, end)`` of the longest contiguous near-black run in ``samples``.

    ``None`` if the longest run doesn't cover at least ``min_run_frac`` of the samples, or
    if the whole clip is dark -- a whole-episode dead camera is the cheaper, metadata-only
    ``camera`` check's job; this is specifically for a *partial* run mid-episode.
    """
    dark = [s.mean <= dark_mean_max and s.std <= dark_std_max for s in samples]
    if not any(dark) or all(dark):
        return None
    best: tuple[int, int] | None = None
    start = None
    for i, d in enumerate([*dark, False]):
        if d and start is None:
            start = i
        elif not d and start is not None:
            if best is None or i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    if best is None or (best[1] - best[0]) / len(samples) < min_run_frac:
        return None
    return best


def longest_frozen_run(
    samples: list[FrameSample], max_abs_diff_max: float, min_run_frac: float
) -> tuple[int, int] | None:
    """Index range ``[start, end)`` of the longest run of consecutive samples that are
    pixel-for-pixel identical (largest per-block thumbnail difference <=
    ``max_abs_diff_max``): a camera or recorder that stalled and kept emitting the same
    frame, as opposed to the near-black feed ``longest_dark_run`` looks for. Unlike that
    function, a run spanning the *whole* clip still counts -- a frozen (but not dark) feed
    for an entire episode has normal-looking pixel mean/std and so isn't caught by the
    metadata-only ``camera`` check.

    Uses the *maximum* per-block difference, not the mean: real robot motion that occupies
    only a small part of the frame (a distant or wide-angle camera) can average out to a
    tiny *mean* thumbnail difference over a couple of seconds even though it clearly moved
    somewhere in frame, which the mean alone can't tell apart from an actually frozen feed.

    ``None`` if fewer than two samples have a thumbnail, no pair is near-identical, or the
    longest such run doesn't cover at least ``min_run_frac`` of the samples.
    """
    if len(samples) < 2 or any(s.thumb is None for s in samples):
        return None
    same = [
        float(np.max(np.abs(samples[i].thumb - samples[i + 1].thumb))) <= max_abs_diff_max
        for i in range(len(samples) - 1)
    ]
    if not any(same):
        return None
    best: tuple[int, int] | None = None
    start = None
    for i, d in enumerate([*same, False]):
        if d and start is None:
            start = i
        elif not d and start is not None:
            if best is None or i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    if best is None:
        return None
    run = (best[0], best[1] + 1)  # best spans adjacent-pair indices; +1 sample to include both ends
    if (run[1] - run[0]) / len(samples) < min_run_frac:
        return None
    return run
