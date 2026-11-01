import json

import numpy as np
import pandas as pd
import pytest

from demoqc.dataset import load_dataset
from tests.conftest import FPS, clean_specs


@pytest.mark.parametrize("version", ["v3.0", "v2.1"])
def test_loads_both_layouts(make_dataset, version):
    specs = clean_specs(4)
    ds = load_dataset(make_dataset(specs, version))

    assert ds.fps == FPS
    assert ds.codebase_version == version
    assert [e.index for e in ds.episodes] == [0, 1, 2, 3]
    for ep, spec in zip(ds.episodes, specs, strict=True):
        assert ep.length == int(spec.seconds * FPS)
        assert ep.meta_length == ep.length
        assert ep.task == "pick the cube"
        assert ep.action.shape == (ep.length, 6)
        assert ep.state.shape == (ep.length, 6)
        assert np.all(np.diff(ep.frame_index) == 1)


def test_v3_video_spans(make_dataset):
    ds = load_dataset(make_dataset(clean_specs(2)))
    start, end = ds.episodes[0].video_spans["observation.images.top"]
    assert end - start == pytest.approx(ds.episodes[0].length / FPS)


def test_missing_info_json(tmp_path):
    with pytest.raises(FileNotFoundError, match="info.json"):
        load_dataset(tmp_path)


def test_episodes_filter_v3_skips_other_data_files(make_dataset):
    # 6 episodes over 3 files: file-000 has 0-1, file-001 has 2-3, file-002 has 4-5
    root = make_dataset(clean_specs(6), "v3.0", data_files=3)
    data_dir = root / "data" / "chunk-000"
    files = sorted(data_dir.glob("*.parquet"))
    assert len(files) == 3
    for f in files[1:]:  # corrupt the files episodes 0-1 don't need
        f.write_bytes(b"not a parquet file")

    ds = load_dataset(root, episodes=[0, 1])
    assert [e.index for e in ds.episodes] == [0, 1]


def test_episodes_filter_v2_skips_other_episode_files(make_dataset):
    root = make_dataset(clean_specs(4), "v2.1")
    data_dir = root / "data" / "chunk-000"
    for f in sorted(data_dir.glob("*.parquet")):
        if f.name != "episode_000002.parquet":
            f.write_bytes(b"not a parquet file")

    ds = load_dataset(root, episodes=[2])
    assert [e.index for e in ds.episodes] == [2]


def test_episodes_filter_matches_full_load(make_dataset):
    root = make_dataset(clean_specs(6), "v3.0", data_files=3)
    full = load_dataset(root)
    filtered = load_dataset(root, episodes=[1, 4])
    assert [e.index for e in filtered.episodes] == [1, 4]
    for i in (1, 4):
        full_ep = next(e for e in full.episodes if e.index == i)
        filt_ep = next(e for e in filtered.episodes if e.index == i)
        np.testing.assert_array_equal(full_ep.action, filt_ep.action)
        np.testing.assert_array_equal(full_ep.timestamps, filt_ep.timestamps)


def test_unknown_episode_raises(make_dataset):
    root = make_dataset(clean_specs(3))
    with pytest.raises(ValueError, match=r"\[99\]"):
        load_dataset(root, episodes=[99])


def test_max_episodes(make_dataset):
    root = make_dataset(clean_specs(6))
    ds = load_dataset(root, max_episodes=3)
    assert [e.index for e in ds.episodes] == [0, 1, 2]


def test_episodes_and_max_episodes_combine(make_dataset):
    root = make_dataset(clean_specs(6))
    ds = load_dataset(root, episodes=[4, 1, 5], max_episodes=2)
    assert [e.index for e in ds.episodes] == [1, 4]


def test_duplicate_episode_index_v3_is_excluded_not_merged(make_dataset):
    """A real community dataset (kmg0620/so101_bomb_pick_place_v1) had two batches of
    episodes uploaded under overlapping episode_index values (a second recording session
    that renumbered from 0 instead of continuing the first). Grouping frames by
    episode_index -- the whole point of being format-agnostic across data file layouts --
    silently concatenated the two unrelated recordings into one nonsense "episode": wrong
    length, a pile of non-increasing timestamp steps, and a cascade of bogus flags on every
    other check. There's no way to tell which recording is the real one, so both must be
    dropped instead of merged or arbitrarily picked.
    """
    root = make_dataset(clean_specs(4), "v3.0", data_files=2)  # file-000: ep 0,1; file-001: ep 2,3
    meta_dir = root / "meta" / "episodes" / "chunk-000"
    meta_df = pd.read_parquet(meta_dir / "file-000.parquet")
    dup_meta = meta_df[meta_df["episode_index"] == 1].copy()
    dup_meta["data/file_index"] = 2
    dup_meta.to_parquet(meta_dir / "file-001b.parquet")

    data_df = pd.read_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    ep1 = data_df[data_df["episode_index"] == 1]
    dup_data = ep1.iloc[: len(ep1) // 2].copy()  # a second, unrelated "episode 1" recording
    dup_data.to_parquet(root / "data" / "chunk-000" / "file-002.parquet")

    ds = load_dataset(root)

    assert [e.index for e in ds.episodes] == [0, 2, 3]
    assert ds.warnings == [
        "1 episode index(es) appear more than once in this dataset's own metadata, so "
        "their frames were excluded (there's no way to tell which recording is the real "
        "one): [1]"
    ]


def test_duplicate_episode_index_v2_is_excluded_not_merged(make_dataset):
    root = make_dataset(clean_specs(3), "v2.1")
    jsonl = root / "meta" / "episodes.jsonl"
    lines = [json.loads(line) for line in jsonl.read_text().splitlines()]
    lines.append(lines[1])  # a second episodes.jsonl entry for episode_index 1
    jsonl.write_text("\n".join(json.dumps(line) for line in lines) + "\n")

    ds = load_dataset(root)

    assert [e.index for e in ds.episodes] == [0, 2]
    assert ds.warnings == [
        "1 episode index(es) appear more than once in this dataset's own metadata, so "
        "their frames were excluded (there's no way to tell which recording is the real "
        "one): [1]"
    ]


def test_requesting_a_duplicate_episode_index_raises_a_specific_error(make_dataset):
    root = make_dataset(clean_specs(3), "v2.1")
    jsonl = root / "meta" / "episodes.jsonl"
    lines = [json.loads(line) for line in jsonl.read_text().splitlines()]
    lines.append(lines[1])
    jsonl.write_text("\n".join(json.dumps(line) for line in lines) + "\n")

    with pytest.raises(ValueError, match="appear more than once"):
        load_dataset(root, episodes=[1])
