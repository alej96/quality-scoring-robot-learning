"""Tests for the opt-in `--decode-video` check (`video_freeze`).

Pure decision logic (`longest_dark_run`, `resolve_v2_video_path`) is tested without the
optional `av` dependency so it always runs in CI. Anything that actually decodes a video
needs `av` installed (`pip install demoqc[video]`) and is skipped otherwise, same as the
`lerobot`-optional test in test_export.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from demoqc import video as video_mod
from demoqc.checks import Config, evaluate
from demoqc.cli import main
from demoqc.dataset import load_dataset
from tests.conftest import EpisodeSpec, clean_specs


def _samples(means: list[float], stds: list[float] | None = None) -> list[video_mod.FrameSample]:
    stds = stds if stds is not None else [0.0] * len(means)
    return [
        video_mod.FrameSample(t=float(i), mean=m, std=s)
        for i, (m, s) in enumerate(zip(means, stds, strict=True))
    ]


# -- longest_dark_run (pure, no av needed) -------------------------------------------------


def test_longest_dark_run_finds_a_partial_mid_episode_run():
    means = [0.5] * 5 + [0.01] * 4 + [0.5] * 5
    run = video_mod.longest_dark_run(_samples(means), 0.03, 0.02, min_run_frac=0.15)
    assert run == (5, 9)


def test_longest_dark_run_ignores_a_run_shorter_than_min_frac():
    means = [0.5] * 9 + [0.01]
    run = video_mod.longest_dark_run(_samples(means), 0.03, 0.02, min_run_frac=0.5)
    assert run is None


def test_longest_dark_run_is_none_for_a_clean_clip():
    run = video_mod.longest_dark_run(_samples([0.5] * 10), 0.03, 0.02, min_run_frac=0.15)
    assert run is None


def test_longest_dark_run_is_none_when_the_whole_clip_is_dark():
    # a camera dead for the whole episode is the (cheaper, metadata-only) `camera` check's
    # job; this one only looks for a *partial* run, so a fully-dark clip isn't its concern
    run = video_mod.longest_dark_run(_samples([0.01] * 10), 0.03, 0.02, min_run_frac=0.15)
    assert run is None


# -- resolve_v2_video_path (pure, no av needed) --------------------------------------------


def test_resolve_v2_video_path_applies_the_chunk_math():
    path = video_mod.resolve_v2_video_path(
        Path("/data"), None, chunks_size=1000, episode_index=1234, camera="cam"
    )
    assert path == Path("/data/videos/chunk-001/cam/episode_001234.mp4")


def test_resolve_v2_video_path_uses_a_custom_template():
    tpl = "vids/{video_key}/{episode_index:06d}-c{episode_chunk}.mp4"
    path = video_mod.resolve_v2_video_path(Path("/d"), tpl, 100, episode_index=250, camera="top")
    assert path == Path("/d/vids/top/000250-c2.mp4")


def test_resolve_v3_video_path_uses_chunk_and_file_index():
    path = video_mod.resolve_v3_video_path(
        Path("/data"), None, chunk_index=2, file_index=7, camera="observation.images.top"
    )
    assert path == Path("/data/videos/observation.images.top/chunk-002/file-007.mp4")


def test_resolve_v3_video_path_uses_a_custom_template():
    tpl = "vids/{video_key}/{chunk_index:02d}-{file_index:02d}.mp4"
    path = video_mod.resolve_v3_video_path(
        Path("/d"), tpl, chunk_index=1, file_index=3, camera="top"
    )
    assert path == Path("/d/vids/top/01-03.mp4")


# -- av-dependent: decode_samples and the full check, on real encoded video ----------------

av = pytest.importorskip("av")


def _write_video(path: Path, frame_values: list[int], fps: int = 10, size: int = 16) -> None:
    """A tiny real .mp4 whose frames are solid colour `frame_values[i]` (0-255, grayscale)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=fps)
    stream.width, stream.height, stream.pix_fmt = size, size, "yuv420p"
    for v in frame_values:
        arr = np.full((size, size, 3), v, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24").reformat(format=stream.pix_fmt)
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def test_decode_samples_reads_a_real_video_and_finds_the_dark_run(tmp_path):
    path = tmp_path / "clip.mp4"
    _write_video(path, [220] * 10 + [2] * 10 + [220] * 10)

    samples = video_mod.decode_samples(path, max_samples=15)

    assert samples
    assert len(samples) <= 15
    run = video_mod.longest_dark_run(
        samples, dark_mean_max=0.1, dark_std_max=0.1, min_run_frac=0.15
    )
    assert run is not None


def test_decode_samples_returns_none_for_an_undecodable_file(tmp_path):
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"this is not a video file")

    assert video_mod.decode_samples(path) is None


def test_decode_samples_time_range_narrows_to_one_segment_of_a_shared_file(tmp_path):
    # a v3.0 video file holds several episodes back to back; a dark run outside the
    # requested [start, end) window must not show up in that window's samples. The second
    # segment's dark run is only *part* of the segment (not all of it), matching
    # longest_dark_run's "partial run" contract -- a wholly-dark segment is the (cheaper,
    # metadata-only) `camera` check's job instead.
    path = tmp_path / "shared.mp4"
    fps = 10
    # [0, 2s): clean. [2, 4s): clean, dark (2.5-3.4s), clean. [4, 6s): clean.
    _write_video(path, [220] * 20 + [220] * 5 + [2] * 10 + [220] * 5 + [220] * 20, fps=fps)

    first = video_mod.decode_samples(path, max_samples=15, time_range=(0.0, 2.0))
    second = video_mod.decode_samples(path, max_samples=15, time_range=(2.0, 4.0))

    assert first and all(s.mean > 0.5 for s in first)
    assert second and any(s.mean < 0.1 for s in second)
    run = video_mod.longest_dark_run(second, dark_mean_max=0.1, dark_std_max=0.1, min_run_frac=0.15)
    assert run is not None


def test_video_freeze_check_fires_on_a_dark_run_and_a_decode_error_only_when_requested(
    make_dataset,
):
    specs = clean_specs(5)
    root = make_dataset(specs, version="v2.1")
    ds = load_dataset(root)
    camera = "observation.images.top"

    def video_path(idx: int) -> Path:
        return video_mod.resolve_v2_video_path(
            ds.root, ds.video_path_template, ds.chunks_size, idx, camera
        )

    _write_video(video_path(0), [220] * 30)  # clean: no defect
    _write_video(video_path(1), [220] * 10 + [2] * 10 + [220] * 10)  # dark mid-episode
    video_path(2).parent.mkdir(parents=True, exist_ok=True)
    video_path(2).write_bytes(b"garbage, not a real video")  # fails to decode
    # episodes 3 and 4 have no video file on disk at all -- must be silently skipped

    def checks(r):
        return {f.check for f in r.flags}

    off = evaluate(load_dataset(root), Config(), decode_video=False)
    assert all("video_freeze" not in checks(r) for r in off), "must be a no-op by default"

    on = evaluate(load_dataset(root), Config(), decode_video=True)
    assert "video_freeze" not in checks(on[0])
    assert "video_freeze" in checks(on[1])
    assert "video_freeze" in checks(on[2])
    assert any(f.check == "video_freeze" and f.severity == "error" for f in on[2].flags)
    assert "video_freeze" not in checks(on[3])
    assert "video_freeze" not in checks(on[4])


def test_video_freeze_check_on_v3_narrows_to_each_episodes_own_segment(make_dataset):
    # Two v3.0 episodes sharing one physical video file (the real layout: LeRobot v3.0
    # concatenates several episodes' footage into one file, addressed by chunk/file index,
    # unlike v2.x's one-file-per-episode). A *partial* dark run placed only in episode 1's
    # segment must flag episode 1 and leave episode 0 (sharing the same file) untouched.
    import pandas as pd

    specs = clean_specs(2)
    root = make_dataset(specs, version="v3.0")
    camera = "observation.images.top"
    episodes_file = next((root / "meta" / "episodes").rglob("*.parquet"))
    df = pd.read_parquet(episodes_file)
    spans = {0: (0.0, 2.0), 1: (2.0, 4.0)}
    for ep, (start, end) in spans.items():
        df.loc[df["episode_index"] == ep, f"videos/{camera}/from_timestamp"] = start
        df.loc[df["episode_index"] == ep, f"videos/{camera}/to_timestamp"] = end
    df.to_parquet(episodes_file)

    ds = load_dataset(root)
    chunk_index, file_index = ds.episodes[0].video_chunk_file[camera]
    video_path = video_mod.resolve_v3_video_path(
        ds.root, ds.video_path_template, chunk_index, file_index, camera
    )
    # ep0 [0, 2s): clean. ep1 [2, 4s): clean, dark (2.5-3.4s), clean -- a partial run that
    # sits entirely in episode 1's own segment.
    _write_video(video_path, [220] * 20 + [220] * 5 + [2] * 10 + [220] * 5, fps=10)

    results = evaluate(load_dataset(root), Config(), decode_video=True)

    assert "video_freeze" not in {f.check for f in results[0].flags}
    assert "video_freeze" in {f.check for f in results[1].flags}


def test_video_freeze_skips_an_episode_the_camera_check_already_flagged(make_dataset):
    specs = clean_specs(5)
    specs[0] = EpisodeSpec(camera_dead=True, seed=0)
    root = make_dataset(specs, version="v2.1")
    ds = load_dataset(root)
    camera = "observation.images.top"
    path = video_mod.resolve_v2_video_path(
        ds.root, ds.video_path_template, ds.chunks_size, 0, camera
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"garbage, not a real video")  # would flag video_freeze if decoded

    results = evaluate(load_dataset(root), Config(), decode_video=True)

    assert "camera" in {f.check for f in results[0].flags}
    assert "video_freeze" not in {f.check for f in results[0].flags}


# -- CLI wiring (--decode-video) ------------------------------------------------------------


def test_decode_video_requires_a_local_source(make_dataset, tmp_path, monkeypatch, capsys):
    src = make_dataset(clean_specs(3), version="v2.1")
    ds = load_dataset(src)
    ds.repo_id = "user/some-hub-dataset"
    monkeypatch.setattr("demoqc.cli.load_dataset", lambda *a, **k: ds)

    code = main(["score", "user/some-hub-dataset", "--decode-video", "--quiet"])

    assert code == 2
    assert "requires a local dataset" in capsys.readouterr().err


def test_decode_video_requires_av_installed(make_dataset, monkeypatch, capsys):
    src = make_dataset(clean_specs(3), version="v2.1")
    monkeypatch.setattr("demoqc.cli.video.is_available", lambda: False)

    code = main(["score", str(src), "--decode-video", "--quiet"])

    assert code == 2
    assert "demoqc[video]" in capsys.readouterr().err


def test_decode_video_runs_on_a_local_v3_dataset_with_no_video_files(make_dataset, capsys):
    # no video files on disk at all: every camera path lookup must miss and skip silently,
    # rather than erroring out the way v3.0 used to before chunk/file-index resolution existed
    src = make_dataset(clean_specs(3), version="v3.0")

    code = main(["score", str(src), "--decode-video", "--quiet"])

    assert code == 0
    assert capsys.readouterr().err == ""
