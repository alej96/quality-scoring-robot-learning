import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from demoqc.cli import main
from demoqc.dataset import load_dataset
from demoqc.export import export_dataset, keep_from_report, trim_from_report
from tests.conftest import FPS, EpisodeSpec, clean_specs, write_dataset

KEEP = [0, 2, 3, 5]
VIDEO_FROM = "videos/observation.images.top/from_timestamp"
VIDEO_TO = "videos/observation.images.top/to_timestamp"


def _frames(root):
    files = sorted((root / "data").rglob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _episodes(root):
    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


@pytest.fixture
def exported(make_dataset, tmp_path):
    # distinct lengths and a 3-file layout, so an offset or ordering bug can't hide
    specs = [EpisodeSpec(seconds=6.0 + i, seed=i) for i in range(6)]
    src = make_dataset(specs, data_files=3)
    result = export_dataset(src, tmp_path / "out", KEEP)
    return src, tmp_path / "out", result


def test_frames_are_filtered_and_reindexed(exported):
    src, out, result = exported
    before, after = _frames(src), _frames(out)

    assert result.kept == KEEP and result.dropped == [1, 4]
    assert after["episode_index"].unique().tolist() == [0, 1, 2, 3]
    assert after["index"].tolist() == list(range(len(after)))
    assert result.frames == len(after) == (before["episode_index"].isin(KEEP)).sum()
    # content of every kept episode is untouched, in the original order
    for new, old in enumerate(KEEP):
        a = np.stack(before[before["episode_index"] == old]["action"])
        b = np.stack(after[after["episode_index"] == new]["action"])
        np.testing.assert_array_equal(a, b)
    # files keep their paths; the one holding only dropped episodes is not written
    assert sorted(p.name for p in (out / "data" / "chunk-000").iterdir()) == [
        "file-000.parquet",
        "file-001.parquet",
        "file-002.parquet",
    ]


def test_file_holding_only_dropped_episodes_is_skipped(make_dataset, tmp_path):
    src = make_dataset(clean_specs(6), data_files=3)  # episodes 0-1 | 2-3 | 4-5
    export_dataset(src, tmp_path / "out", [0, 1, 4, 5])
    names = sorted(p.name for p in (tmp_path / "out" / "data" / "chunk-000").iterdir())
    assert names == ["file-000.parquet", "file-002.parquet"]
    frames = _frames(tmp_path / "out")
    assert frames["index"].tolist() == list(range(len(frames)))
    # the episode metadata still points at the files that exist
    eps = _episodes(tmp_path / "out")
    assert eps["data/file_index"].tolist() == [0, 0, 2, 2]


def test_episode_metadata_matches_the_data(exported):
    src, out, _ = exported
    eps, frames = _episodes(out), _frames(out)

    assert eps["episode_index"].tolist() == [0, 1, 2, 3]
    assert eps["tasks"].map(list).tolist() == [["pick the cube"]] * 4
    for row in eps.itertuples():
        mine = frames[frames["episode_index"] == row.episode_index]
        assert (row.dataset_from_index, row.dataset_to_index) == (
            mine["index"].min(),
            mine["index"].max() + 1,
        )
        assert row.length == len(mine)
    # video spans (unchanged mp4s) are carried over verbatim for the kept episodes
    old = _episodes(src).set_index("episode_index").loc[KEEP]
    col = "videos/observation.images.top/to_timestamp"
    assert eps[col].tolist() == old[col].tolist()


def test_info_and_tasks(exported):
    src, out, result = exported
    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 4
    assert info["total_frames"] == result.frames
    assert info["splits"] == {"train": "0:4"}
    assert info["fps"] == 30 and info["robot_type"] == "so101_follower"
    assert (out / "meta" / "tasks.parquet").read_bytes() == (
        src / "meta" / "tasks.parquet"
    ).read_bytes()
    assert result.has_videos
    assert not (out / "videos").exists()


def test_dataset_stats_describe_only_the_kept_episodes(exported):
    _, out, _ = exported
    stats = json.loads((out / "meta" / "stats.json").read_text())
    frames = _frames(out)

    action = np.stack(frames["action"]).astype(np.float64)
    assert stats["action"]["count"] == [len(frames)]
    np.testing.assert_allclose(stats["action"]["mean"], action.mean(0))
    np.testing.assert_allclose(stats["action"]["std"], action.std(0))
    np.testing.assert_allclose(stats["action"]["min"], action.min(0))
    np.testing.assert_allclose(stats["action"]["max"], action.max(0))
    assert stats["index"]["max"] == [len(frames) - 1]
    assert stats["episode_index"]["max"] == [3]


def test_video_stats_are_combined_from_episode_stats(exported):
    src, out, _ = exported
    stats = json.loads((out / "meta" / "stats.json").read_text())["observation.images.top"]
    old = _episodes(src).set_index("episode_index").loc[KEEP]

    means = np.array([[c[0][0] for c in m] for m in old["stats/observation.images.top/mean"]])
    stds = np.array([[c[0][0] for c in m] for m in old["stats/observation.images.top/std"]])
    counts = old["stats/observation.images.top/count"].map(lambda c: c[0]).to_numpy()
    w = counts / counts.sum()
    mean = (w[:, None] * means).sum(0)
    var = (w[:, None] * (stds**2 + (means - mean) ** 2)).sum(0)

    assert stats["count"] == [counts.sum()]
    np.testing.assert_allclose(np.array(stats["mean"]).reshape(3), mean)
    np.testing.assert_allclose(np.array(stats["std"]).reshape(3), np.sqrt(var))


def test_per_episode_index_stats_are_rewritten(exported):
    _, out, _ = exported
    eps = _episodes(out)
    # the source values referred to the original global indices / episode numbers
    assert [m[0] for m in eps["stats/index/min"]] == eps["dataset_from_index"].tolist()
    assert [m[0] for m in eps["stats/index/max"]] == (eps["dataset_to_index"] - 1).tolist()
    assert [m[0] for m in eps["stats/episode_index/min"]] == [0, 1, 2, 3]
    assert [m[0] for m in eps["stats/episode_index/max"]] == [0, 1, 2, 3]


def test_exported_dataset_loads_and_scores(exported):
    _, out, _ = exported
    ds = load_dataset(out)
    assert [e.index for e in ds.episodes] == [0, 1, 2, 3]
    assert all(e.meta_length == e.length for e in ds.episodes)


def test_keep_everything_reproduces_the_source(make_dataset, tmp_path):
    src = make_dataset(clean_specs(5), data_files=2)
    export_dataset(src, tmp_path / "out", range(5))
    for rel in ("meta/info.json", "meta/stats.json"):
        a = json.loads((src / rel).read_text())
        b = json.loads((tmp_path / "out" / rel).read_text())
        assert a.keys() == b.keys()
    pd.testing.assert_frame_equal(_frames(src), _frames(tmp_path / "out"))
    pd.testing.assert_frame_equal(_episodes(src), _episodes(tmp_path / "out"))


def test_refuses_bad_input(make_dataset, tmp_path):
    src = make_dataset(clean_specs(4))
    with pytest.raises(ValueError, match="not in the dataset"):
        export_dataset(src, tmp_path / "a", [0, 9])
    with pytest.raises(ValueError, match="no episodes"):
        export_dataset(src, tmp_path / "b", [])
    with pytest.raises(ValueError, match="differ"):
        export_dataset(src, src, [0])
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "x").write_text("keep me")
    with pytest.raises(FileExistsError):
        export_dataset(src, tmp_path / "c", [0])
    assert (tmp_path / "c" / "x").read_text() == "keep me"


def test_v2_is_rejected_with_a_clear_message(make_dataset, tmp_path):
    src = make_dataset(clean_specs(3), version="v2.1")
    with pytest.raises(ValueError, match=r"v3\.0.*v2\.1"):
        export_dataset(src, tmp_path / "out", [0])


def test_keep_from_report_rejects_a_report_for_another_dataset(make_dataset, tmp_path):
    a = make_dataset(clean_specs(6))
    report = tmp_path / "r.json"
    assert main(["score", str(a), "--json", str(report), "--quiet"]) == 0
    other = load_dataset(write_dataset(tmp_path / "other", clean_specs(4)))
    with pytest.raises(ValueError, match="different dataset"):
        keep_from_report(report, other)
    junk = tmp_path / "junk.json"
    junk.write_text("{}")
    with pytest.raises(ValueError, match="not a demoqc score report"):
        keep_from_report(junk, other)


def test_export_command_drops_rejected_episodes(make_dataset, tmp_path, capsys):
    specs = clean_specs()
    specs[3] = EpisodeSpec(seconds=8.0, spikes_at=[40, 80, 120, 160, 200], seed=3)
    src = make_dataset(specs)
    report, out = tmp_path / "r.json", tmp_path / "clean"
    assert main(["score", str(src), "--json", str(report), "--quiet", "--min-score", "90"]) == 0
    keep = json.loads(report.read_text())["summary"]["keep"]
    assert 3 not in keep and len(keep) < len(specs)
    capsys.readouterr()

    assert main(["export", str(src), "--keep-from", str(report), "-o", str(out)]) == 0

    said = capsys.readouterr().out
    assert f"kept {len(keep)} of {len(specs)}" in said and "videos were not copied" in said
    assert pq.read_table(out / "data" / "chunk-000" / "file-000.parquet").num_rows == sum(
        len(load_dataset(src).episodes[i].timestamps) for i in keep
    )
    assert json.loads((out / "meta" / "info.json").read_text())["total_episodes"] == len(keep)


def test_export_command_errors_exit_2(make_dataset, tmp_path, capsys):
    src = make_dataset(clean_specs(4))
    report = tmp_path / "r.json"
    main(["score", str(src), "--json", str(report), "--quiet"])
    out = tmp_path / "clean"
    assert main(["export", str(src), "--keep-from", str(report), "-o", str(out)]) == 0
    capsys.readouterr()
    # output already populated
    assert main(["export", str(src), "--keep-from", str(report), "-o", str(out)]) == 2
    assert "not empty" in capsys.readouterr().err
    assert main(["export", str(src), "--keep-from", str(tmp_path / "nope.json"), "-o", "x"]) == 2


def test_exported_dataset_loads_with_lerobot(exported):
    """The real loader accepts the export: indices, episode ranges and video paths all line up."""
    pytest.importorskip("lerobot")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    _, out, result = exported
    meta = LeRobotDatasetMetadata("local/export-test", out)
    for ep in range(meta.total_episodes):  # LeRobot also wants the (not copied) video files
        for key in meta.video_keys:
            path = out / meta.get_video_file_path(ep, key)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

    ds = LeRobotDataset("local/export-test", root=out, download_videos=False)
    assert (ds.num_episodes, len(ds)) == (len(KEEP), result.frames)
    index = np.asarray(ds.hf_dataset["index"])
    episode = np.asarray(ds.hf_dataset["episode_index"])
    for e in ds.meta.episodes:
        rows = (index >= e["dataset_from_index"]) & (index < e["dataset_to_index"])
        assert rows.sum() == e["length"]
        assert (episode[rows] == e["episode_index"]).all()
    assert ds.meta.stats["action"]["count"].tolist() == [result.frames]


def test_apply_trim_cuts_frames_shifts_timestamps_and_narrows_video_span(make_dataset, tmp_path):
    src = make_dataset(clean_specs(4))
    before = _frames(src)
    ep1_before = before[before["episode_index"] == 1]
    ep1_len = len(ep1_before)
    lead, trail = 10, 5
    trim = {1: (lead, ep1_len - trail)}

    result = export_dataset(src, tmp_path / "out", keep=[0, 1, 2, 3], trim=trim)

    assert result.trimmed_frames == lead + trail
    after = _frames(tmp_path / "out")
    assert after["index"].tolist() == list(range(len(after)))
    ep1_after = after[after["episode_index"] == 1]
    assert len(ep1_after) == ep1_len - lead - trail
    assert ep1_after["frame_index"].tolist() == list(range(len(ep1_after)))
    np.testing.assert_array_equal(
        np.stack(ep1_after["action"]), np.stack(ep1_before["action"])[lead : ep1_len - trail]
    )
    np.testing.assert_allclose(
        ep1_after["timestamp"].to_numpy(),
        ep1_before["timestamp"].to_numpy()[lead : ep1_len - trail] - lead / FPS,
        rtol=1e-5,
        atol=1e-6,
    )

    eps, old_eps = _episodes(tmp_path / "out"), _episodes(src)
    row1 = eps[eps["episode_index"] == 1].iloc[0]
    old_row1 = old_eps[old_eps["episode_index"] == 1].iloc[0]
    assert row1["length"] == len(ep1_after)
    assert row1[VIDEO_FROM] == pytest.approx(old_row1[VIDEO_FROM] + lead / FPS)
    assert row1[VIDEO_TO] == pytest.approx(old_row1[VIDEO_TO] - trail / FPS)

    # untrimmed episodes are carried over exactly, including their video spans
    row0 = eps[eps["episode_index"] == 0].iloc[0]
    old_row0 = old_eps[old_eps["episode_index"] == 0].iloc[0]
    assert row0[VIDEO_FROM] == old_row0[VIDEO_FROM] and row0[VIDEO_TO] == old_row0[VIDEO_TO]
    np.testing.assert_array_equal(
        np.stack(after[after["episode_index"] == 0]["action"]),
        np.stack(before[before["episode_index"] == 0]["action"]),
    )


def test_apply_trim_rejects_out_of_range_trims(make_dataset, tmp_path):
    src = make_dataset(clean_specs(3))
    length = len(_frames(src).loc[lambda d: d["episode_index"] == 0])
    with pytest.raises(ValueError, match="invalid trim"):
        export_dataset(src, tmp_path / "a", keep=[0, 1, 2], trim={0: (5, 2)})
    with pytest.raises(ValueError, match="invalid trim"):
        export_dataset(src, tmp_path / "b", keep=[0, 1, 2], trim={0: (0, length + 10)})


def test_apply_trim_for_a_dropped_episode_is_ignored(make_dataset, tmp_path):
    src = make_dataset(clean_specs(4))
    result = export_dataset(src, tmp_path / "out", keep=[0, 2, 3], trim={1: (0, 999999)})
    assert result.kept == [0, 2, 3] and result.trimmed_frames == 0


def test_apply_trim_without_any_trim_is_a_no_op(exported):
    """trim=None (the default) must reproduce the untrimmed export exactly."""
    src, out, result = exported
    assert result.trimmed_frames == 0


def test_export_command_apply_trim(make_dataset, tmp_path, capsys):
    specs = clean_specs(4)
    specs[1] = EpisodeSpec(seconds=8.0, lead_idle_s=2.0, trail_idle_s=1.5, seed=1)
    src = make_dataset(specs)
    report, out = tmp_path / "r.json", tmp_path / "clean"
    assert main(["score", str(src), "--json", str(report), "--quiet"]) == 0
    episodes = json.loads(report.read_text())["episodes"]
    assert 1 in json.loads(report.read_text())["summary"]["keep"]
    start, end = episodes[1]["trim"]
    capsys.readouterr()

    assert (
        main(["export", str(src), "--keep-from", str(report), "--apply-trim", "-o", str(out)]) == 0
    )

    said = capsys.readouterr().out
    assert "trimmed" in said
    after = _frames(out)
    ep1 = after[after["episode_index"] == 1]
    assert len(ep1) == end - start
    assert ep1["frame_index"].tolist() == list(range(end - start))


def test_export_command_without_apply_trim_keeps_episodes_whole(make_dataset, tmp_path):
    specs = clean_specs(4)
    specs[1] = EpisodeSpec(seconds=8.0, lead_idle_s=2.0, trail_idle_s=1.5, seed=1)
    src = make_dataset(specs)
    report, out = tmp_path / "r.json", tmp_path / "clean"
    main(["score", str(src), "--json", str(report), "--quiet"])
    main(["export", str(src), "--keep-from", str(report), "-o", str(out)])

    before_len = len(load_dataset(src).episodes[1].timestamps)
    after_len = len(_frames(out)[_frames(out)["episode_index"] == 1])
    assert after_len == before_len


def test_trim_from_report_rejects_a_report_for_another_dataset(make_dataset, tmp_path):
    a = make_dataset(clean_specs(6))
    report = tmp_path / "r.json"
    assert main(["score", str(a), "--json", str(report), "--quiet"]) == 0
    other = load_dataset(write_dataset(tmp_path / "other", clean_specs(4)))
    with pytest.raises(ValueError, match="different dataset"):
        trim_from_report(report, other)
