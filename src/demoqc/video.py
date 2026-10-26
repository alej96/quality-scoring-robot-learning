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


@dataclass
class FrameSample:
    t: float
    mean: float  # pixel mean, 0-1 scale
    std: float  # pixel std, 0-1 scale


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


def decode_samples(path: Path, max_samples: int = 20) -> list[FrameSample] | None:
    """Up to ``max_samples`` evenly-spaced grayscale frame stats from the video at ``path``.

    Returns ``None`` if the file can't be decoded at all (corrupted container, unsupported
    codec, truncated download) -- a real defect in its own right, not something to let
    propagate and abort the whole ``score`` run.
    """
    import av  # optional dependency; only ever imported once this path actually runs

    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            total = stream.frames or 0
            step = max(1, total // max_samples) if total else 1
            fps = float(stream.average_rate) if stream.average_rate else 1.0
            out = []
            for i, frame in enumerate(container.decode(stream)):
                if i % step != 0:
                    continue
                arr = frame.to_ndarray(format="gray").astype(np.float64) / 255.0
                t = float(frame.time) if frame.time is not None else i / fps
                out.append(FrameSample(t, float(arr.mean()), float(arr.std())))
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
