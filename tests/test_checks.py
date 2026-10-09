import pytest

from demoqc.checks import evaluate
from demoqc.dataset import load_dataset
from tests.conftest import FPS, EpisodeSpec, clean_specs


def _results(make_dataset, specs):
    return evaluate(load_dataset(make_dataset(specs)))


def _checks(result, min_penalty=0.0):
    return {f.check for f in result.flags if f.penalty > min_penalty}


def test_clean_dataset_scores_high(make_dataset):
    results = _results(make_dataset, clean_specs())
    for r in results:
        assert r.score >= 95, (r.episode_index, r.flags)
        assert not [f for f in r.flags if f.severity != "info"], r.flags


def test_idle_trim(make_dataset):
    specs = clean_specs()
    specs[3] = EpisodeSpec(seconds=10, lead_idle_s=2.0, trail_idle_s=1.5, seed=3)
    r = _results(make_dataset, specs)[3]

    assert "idle" in _checks(r)
    assert r.metrics["lead_idle_s"] == pytest.approx(2.0, abs=0.3)
    assert r.metrics["trail_idle_s"] == pytest.approx(1.5, abs=0.3)
    start, end = r.trim
    assert start == pytest.approx(int(1.75 * FPS), abs=10)
    assert end == pytest.approx(int((10 - 1.25) * FPS), abs=10)


def test_dropped_frames(make_dataset):
    specs = clean_specs()
    specs[5] = EpisodeSpec(drop_frames=list(range(100, 106)), seed=5)
    results = _results(make_dataset, specs)

    assert results[5].metrics["dropped_frames"] == 6
    assert "timing" in _checks(results[5])
    assert all("timing" not in _checks(r) for i, r in enumerate(results) if i != 5)


def test_single_frame_spike(make_dataset):
    specs = clean_specs()
    specs[2] = EpisodeSpec(spikes_at=[120], seed=2)
    results = _results(make_dataset, specs)

    assert results[2].metrics["spikes"] == 1
    assert "smoothness" in _checks(results[2])
    assert all(r.metrics["spikes"] == 0 for i, r in enumerate(results) if i != 2)


def test_stale_state(make_dataset):
    specs = clean_specs()
    specs[7] = EpisodeSpec(stale_state_from=(60, 150), seed=7)
    results = _results(make_dataset, specs)

    assert results[7].metrics["stale_state_fraction"] > 0.2
    assert "stale_state" in _checks(results[7])
    assert all("stale_state" not in _checks(r) for i, r in enumerate(results) if i != 7)


def test_video_and_metadata_mismatch(make_dataset):
    specs = clean_specs()
    specs[1] = EpisodeSpec(video_span_s=6.0, seed=1)
    specs[4] = EpisodeSpec(meta_length_delta=-12, seed=4)
    results = _results(make_dataset, specs)

    assert "video_sync" in _checks(results[1])
    assert "metadata" in _checks(results[4])


def test_fixed_duration_recordings_are_not_length_outliers(make_dataset):
    # same wall-clock length, tiny spread: no episode should be called "aborted"
    specs = [EpisodeSpec(seconds=10, trail_idle_s=1.0 + 0.01 * i, seed=i) for i in range(12)]
    results = _results(make_dataset, specs)
    assert all("length" not in _checks(r) for r in results)


def test_short_active_episode_is_flagged(make_dataset):
    specs = clean_specs()
    specs[9] = EpisodeSpec(seconds=3.0, seed=9)
    results = _results(make_dataset, specs)
    assert "length" in _checks(results[9])


def test_sync_lag_outlier(make_dataset):
    # every episode's follower tracks the leader with a 1-frame delay, except one that
    # was recorded with a USB latency spike and lags by 6 frames (0.2s at 30fps)
    specs = clean_specs()
    specs[6] = EpisodeSpec(state_lag_frames=6, seed=6)
    results = _results(make_dataset, specs)

    assert results[6].metrics["sync_lag_frames"] == pytest.approx(6, abs=1)
    assert "sync" in _checks(results[6])
    assert all("sync" not in _checks(r) for i, r in enumerate(results) if i != 6)


def test_sync_low_correlation(make_dataset):
    # a recorder bug scrambles observation.state relative to action for one episode
    specs = clean_specs()
    specs[8] = EpisodeSpec(state_decorrelated=True, seed=8)
    results = _results(make_dataset, specs)

    assert results[8].metrics["sync_peak_corr"] < 0.4
    assert "sync" in _checks(results[8])
    assert all("sync" not in _checks(r) for i, r in enumerate(results) if i != 8)


def test_clean_dataset_has_consistent_sync_lag(make_dataset):
    results = _results(make_dataset, clean_specs())
    for r in results:
        assert r.metrics["sync_lag_frames"] == pytest.approx(1, abs=1)
        assert r.metrics["sync_peak_corr"] > 0.8
