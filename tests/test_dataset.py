import numpy as np
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
