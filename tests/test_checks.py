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


def test_disable_check_suppresses_flags_and_metrics(make_dataset):
    # a dataset with a real sync defect: --disable-check sync must silence it completely,
    # not just hide the flag, so a noisy check can be turned off without polluting the
    # report with metrics for a check the user doesn't trust
    specs = clean_specs()
    specs[6] = EpisodeSpec(state_lag_frames=6, seed=6)
    ds = load_dataset(make_dataset(specs))

    results = evaluate(ds, disabled_checks=frozenset({"sync"}))

    assert all("sync" not in _checks(r) for r in results)
    assert all("sync_lag_frames" not in r.metrics for r in results)


def test_disable_check_leaves_other_checks_running(make_dataset):
    # disabling one check must not affect the (split) metadata/video_sync checks or any
    # other check in the same episode
    specs = clean_specs()
    specs[1] = EpisodeSpec(video_span_s=6.0, seed=1)
    specs[4] = EpisodeSpec(meta_length_delta=-12, seed=4)
    ds = load_dataset(make_dataset(specs))

    results = evaluate(ds, disabled_checks=frozenset({"video_sync"}))

    assert "video_sync" not in _checks(results[1])
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


def test_disable_check_length_has_no_episode_half_but_still_works(make_dataset):
    # "length" is only ever emitted by check_relative (there's no EPISODE_CHECKS entry to
    # disable), so it needs its own disabled_checks plumbing through evaluate()
    specs = clean_specs()
    specs[9] = EpisodeSpec(seconds=3.0, seed=9)
    ds = load_dataset(make_dataset(specs))

    results = evaluate(ds, disabled_checks=frozenset({"length"}))

    assert all("length" not in _checks(r) for r in results)


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


def test_sync_matches_dims_by_name_when_reordered(make_dataset):
    # observation.state's columns are in a different order than action's (e.g. the
    # recorder enumerates sensors differently per stream); matching dims positionally
    # would compare unrelated joints and see no correlation anywhere
    specs = clean_specs()
    path = make_dataset(specs, state_perm=[5, 1, 2, 3, 4, 0])
    results = evaluate(load_dataset(path))

    for r in results:
        assert r.metrics["sync_lag_frames"] == pytest.approx(1, abs=1)
        assert r.metrics["sync_peak_corr"] > 0.8
        assert "sync" not in _checks(r)


def test_sync_survives_extra_unmatched_action_dim(make_dataset):
    # an action-only "force_sensor" column (no observation.state counterpart) used to make
    # action.shape[1] != state.shape[1] and disable the sync check for the whole dataset
    specs = clean_specs()
    specs[6] = EpisodeSpec(state_lag_frames=6, seed=6)
    path = make_dataset(specs, extra_action_dim=True)
    results = evaluate(load_dataset(path))

    assert results[6].metrics["sync_lag_frames"] == pytest.approx(6, abs=1)
    assert "sync" in _checks(results[6])
    assert all("sync" not in _checks(r) for i, r in enumerate(results) if i != 6)


@pytest.mark.parametrize("version", ["v3.0", "v2.1"])
def test_camera_dead(make_dataset, version):
    # one episode's camera stats are near-black for the whole episode: a disconnected
    # camera or a lens cap left on, caught from metadata alone (no video ever opened)
    specs = clean_specs()
    specs[4] = EpisodeSpec(camera_dead=True, seed=4)
    results = evaluate(load_dataset(make_dataset(specs, version=version)))

    assert "camera" in _checks(results[4])
    assert results[4].metrics["camera_dead_count"] == 1
    assert all("camera" not in _checks(r) for i, r in enumerate(results) if i != 4)


def test_gripper_chatter(make_dataset):
    # the operator rapidly double-taps the gripper for half a second: an anxious squeeze,
    # not a deliberate open/close
    specs = clean_specs()
    specs[3] = EpisodeSpec(gripper_chatter_at=(100, 115), seed=3)
    results = _results(make_dataset, specs)

    assert results[3].metrics["gripper_toggles_per_window"] >= 4
    assert "gripper_chatter" in _checks(results[3])
    assert all("gripper_chatter" not in _checks(r) for i, r in enumerate(results) if i != 3)


def test_integrity_flags_nonfinite_action(make_dataset):
    # a sensor glitch or corrupted write leaves a NaN in one frame's action
    specs = clean_specs()
    specs[5] = EpisodeSpec(nan_action_at=[50], seed=5)
    results = _results(make_dataset, specs)

    flags = [f for f in results[5].flags if f.check == "integrity"]
    assert len(flags) == 1
    assert flags[0].severity == "error"
    assert "action" in flags[0].message
    assert all("integrity" not in _checks(r) for i, r in enumerate(results) if i != 5)


def test_integrity_flags_nonfinite_state(make_dataset):
    specs = clean_specs()
    specs[2] = EpisodeSpec(nan_state_at=[30, 31], seed=2)
    results = _results(make_dataset, specs)

    flags = [f for f in results[2].flags if f.check == "integrity"]
    assert len(flags) == 1
    assert "observation.state" in flags[0].message
    assert all("integrity" not in _checks(r) for i, r in enumerate(results) if i != 2)


def test_nan_in_one_episode_does_not_disable_checks_on_others(make_dataset):
    # DatasetContext.build used to compute action_scale with plain np.percentile/.std,
    # which propagate NaN across an entire column: a single bad frame in one episode's
    # gripper dim silently set action_scale[gripper] to NaN for the *whole dataset*,
    # disabling gripper_chatter (which requires a finite scale) on every episode, not
    # just the corrupted one.
    specs = clean_specs()
    specs[0] = EpisodeSpec(nan_gripper_at=[50], seed=0)
    specs[3] = EpisodeSpec(gripper_chatter_at=(100, 115), seed=3)
    results = _results(make_dataset, specs)

    assert "integrity" in _checks(results[0])
    assert "gripper_chatter" in _checks(results[3])
