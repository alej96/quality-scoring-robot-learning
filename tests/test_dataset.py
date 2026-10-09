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
