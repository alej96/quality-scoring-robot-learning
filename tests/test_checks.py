import warnings
from pathlib import Path

import numpy as np
import pytest

from demoqc.checks import (
    Config,
    DatasetContext,
    _angular_action_dims,
    _half_quaternion_action_groups,
    _quaternion_action_groups,
    evaluate,
)
from demoqc.dataset import Dataset, Episode, load_dataset
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


def test_frozen_action_whole_dataset_flags_no_motion(make_dataset):
    # Found on a real Hub dataset (omkarmayekar555/robot-learning-tutorial-data): every
    # action dim was bit-identical for the whole episode (a broken teleop/recorder
    # session). DatasetContext.build's action_scale collapses to NaN for every dim when
    # nothing in the dataset ever varies, which made _speed() return NaN (a RuntimeWarning
    # on its own) for every frame; "NaN < idle_speed" is always False, so check_idle's
    # absolute "no motion detected" fallback silently never fired -- a completely dead
    # recording scored as if it were fully active, with no flag pointing at the real
    # defect (confirmed against the real dataset: mean score 90.0, only an unrelated weak
    # `sync` flag, nothing about the action never moving).
    # same seed for every episode: the dataset-wide action scale (DatasetContext.build)
    # is computed from *all* episodes' frames pooled together, so it only collapses to
    # NaN in every dim -- the exact real-dataset failure mode -- when nothing varies
    # across episodes either, not just within one.
    specs = [EpisodeSpec(seconds=5.0, frozen_action=True, seed=0) for _ in range(5)]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        results = _results(make_dataset, specs)
    for r in results:
        assert "idle" in _checks(r)
        assert any("no motion detected" in f.message for f in r.flags), r.flags


def test_very_short_frozen_episode_flags_no_motion(make_dataset):
    # Found on a real Hub dataset (lerobot/svla_so100_stacking): a 1-frame episode
    # (presumably an aborted recording cut off before any motion happened) scored 90.0
    # with only an unrelated "unusually short" `length` warning, while other episodes in
    # the same dataset with the exact same bit-identical (zero-motion) action but a few
    # more frames correctly scored 20.0 with `idle: no motion detected`. check_idle's old
    # unconditional `ep.length < 3: return` guard -- needed because the smoothed
    # lead/trail idle profile needs at least a couple of real transitions -- also skipped
    # its own absolute no-motion fallback, so the *shortest*, most obviously broken
    # episodes were the only ones it stayed silent on.
    specs = clean_specs()
    specs[1] = EpisodeSpec(seconds=1.5 / FPS, frozen_action=True, seed=1)  # 1 frame
    specs[3] = EpisodeSpec(seconds=2.5 / FPS, frozen_action=True, seed=3)  # 2 frames
    results = _results(make_dataset, specs)

    for i in (1, 3):
        assert "idle" in _checks(results[i]), (i, results[i].flags)
        assert any("no motion detected" in f.message for f in results[i].flags)
    assert all("idle" not in _checks(r) for i, r in enumerate(results) if i not in (1, 3))


def test_structured_action_state_feeds_real_checks(make_dataset):
    # Follow-up to the yaak-ai/L2D exact-duplicate fix (see ROADMAP.md's Log): that fix
    # only stopped the false "exact duplicate" flag by excluding action/state-less
    # episodes from hashing -- every other check still saw ep.action/ep.state as None for
    # any dataset whose meta/info.json has no literal action/observation.state feature,
    # since the real dataset stores it under structured per-key names instead. Now that
    # dataset.py reconstructs a flat action/state array from the numeric structured
    # sub-features (test_dataset.py), an ordinary check like `idle` should fire on a real
    # defect in a structured-feature dataset exactly as it would on a flat one.
    specs = clean_specs()
    specs[2] = EpisodeSpec(seconds=1.5 / FPS, frozen_action=True, seed=2)  # 1 frame, no motion
    ds = make_dataset(specs, structured_action_state=True)
    results = evaluate(load_dataset(ds))

    assert "idle" in _checks(results[2])
    assert any("no motion detected" in f.message for f in results[2].flags)
    assert all("idle" not in _checks(r) for i, r in enumerate(results) if i != 2)


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


def _angular_dataset() -> Dataset:
    """A synthetic dataset with one action dim that's a radian angle wrapped to
    [-pi, pi] (like ``eef_rot_euler_z_rad``), plus a normal, unrelated joint dim.

    Built by hand (not via ``make_dataset``/``write_dataset``) because the shared
    ``make_episode`` fixture generates every dim at the same large, non-angular scale;
    testing the wrap fix needs one dim whose values are genuinely confined to [-pi, pi]
    across the whole dataset.
    """
    n = 90  # 3s at FPS
    t = np.arange(n)

    def episode(idx: int, angle_step: float, jump_at: int | None = None) -> Episode:
        angle = np.mod(angle_step * t + np.pi, 2 * np.pi) - np.pi
        joint = 5.0 * np.sin(2 * np.pi * t / n) + 0.01 * idx
        action = np.stack([angle, joint], axis=1)
        if jump_at is not None:
            action[jump_at:, 1] += 40.0  # a real, unrelated abrupt jump, not wrapping
        return Episode(
            index=idx, timestamps=t / FPS, frame_index=t, action=action, state=action.copy()
        )

    # bounded well short of +-pi: never wraps, same as any ordinary joint
    bounded = [episode(i, angle_step=0.02) for i in range(6)]
    wrap_ep = episode(6, angle_step=0.15)  # sweeps > 2 full turns -> several wrap crossings
    jump_ep = episode(7, angle_step=0.02, jump_at=45)  # a real defect, unrelated to wrapping
    return Dataset(
        root=Path("."),
        fps=float(FPS),
        codebase_version="v3.0",
        robot_type=None,
        episodes=[*bounded, wrap_ep, jump_ep],
    )


def test_smoothness_ignores_wrapped_angle_crossings():
    # Found on a real Hub dataset (RoboCOIN/Agilex_Cobot_Magic_heat_burger): its
    # eef_rot_euler_x/z_rad action dims are Euler angles wrapped to [-pi, pi], and every
    # episode's wrist rotation crosses that +-pi boundary at least once during otherwise
    # perfectly smooth motion. check_smoothness's plain np.diff saw each crossing as a
    # ~2*pi single-frame action -- tens of times the dataset's own step_p99 for that dim
    # -- so "abrupt action jump" fired on nearly every episode of a dataset with nothing
    # actually jerky in it (27/819 episodes in a sampled subset, mean score dragged down
    # to the 60s-70s on several of them).
    results = {r.episode_index: r for r in evaluate(_angular_dataset())}

    wrap_result = results[6]
    assert wrap_result.metrics["spikes"] == 0
    assert wrap_result.metrics["jumps"] == 0
    assert not [f for f in wrap_result.flags if f.check == "smoothness"], wrap_result.flags

    # the fix must not blind the check to a real, unrelated jump on a non-angular dim
    jump_result = results[7]
    assert jump_result.metrics["jumps"] > 0
    assert "smoothness" in _checks(jump_result)


def test_angular_action_dims_requires_reaching_both_wrap_boundaries():
    # dim 0 reaches close to both +pi and -pi (a real wrapped angle); dim 1 stays well
    # within the bound without ever approaching it (an ordinary joint that merely happens
    # to sit under pi); dim 2 exceeds the bound outright (clearly not a wrapped angle).
    allA = np.array(
        [
            [-3.14, 1.5, 4.0],
            [0.0, -1.5, -4.0],
            [3.14, 1.0, 0.0],
        ]
    )
    mask = _angular_action_dims(allA, Config())
    np.testing.assert_array_equal(mask, [True, False, False])


def _gimbal_lock_dataset() -> Dataset:
    """A synthetic dataset with two Euler-angle action dims (``..._x_rad``/``..._z_rad``,
    both wrapped to [-pi, pi]) that pair up under ``_angular_pair_axes``, plus a normal
    joint dim.

    Built by hand for the same reason as ``_angular_dataset``: it needs dims genuinely
    confined to [-pi, pi] and, here, a frame where both axes jump by the *same* amount
    (the gimbal-lock signature -- ``x - z`` barely moves even though each axis's own step
    is huge) versus a frame where only one axis jumps (a real, unrelated defect).
    """
    n = 90  # 3s at FPS
    t = np.arange(n)
    names = ["arm_rot_euler_x_rad", "arm_rot_euler_z_rad", "joint"]

    def episode(idx: int, x: np.ndarray, z: np.ndarray) -> Episode:
        joint = 5.0 * np.sin(2 * np.pi * t / n) + 0.01 * idx
        action = np.stack([x, z, joint], axis=1)
        return Episode(
            index=idx, timestamps=t / FPS, frame_index=t, action=action, state=action.copy()
        )

    def wrap(angle_step: float, phase: float) -> np.ndarray:
        return np.mod(angle_step * t + phase + np.pi, 2 * np.pi) - np.pi

    # bounded well short of +-pi: ordinary slow motion, used to build a normal step_p99
    # baseline (x and z independent, not correlated)
    clean = [episode(i, wrap(0.02, 0.1 * i), wrap(0.017, -0.1 * i)) for i in range(6)]
    # each axis independently sweeps several full turns so the dataset-wide range reaches
    # both +pi and -pi for both dims (qualifying them as angular, like _angular_dataset)
    wrap_provider = episode(6, wrap(0.15, 0.0), wrap(0.13, 1.0))

    gimbal_x = np.full(n, 0.3)
    gimbal_z = np.full(n, -0.2)
    gimbal_x[45:] += 1.2  # both axes jump by the *same* amount at the same frame: a real
    gimbal_z[45:] += 1.2  # rotation would leave (x - z) almost unchanged, this doesn't
    gimbal_ep = episode(7, gimbal_x, gimbal_z)

    jump_x = np.full(n, 0.3)
    jump_z = np.full(n, -0.2)
    jump_x[45:] += 1.2  # only one axis jumps: a real, unrelated defect
    jump_ep = episode(8, jump_x, jump_z)

    return Dataset(
        root=Path("."),
        fps=float(FPS),
        codebase_version="v3.0",
        robot_type=None,
        episodes=[*clean, wrap_provider, gimbal_ep, jump_ep],
        action_names=names,
    )


def test_smoothness_ignores_gimbal_lock_pair_cancellation():
    # Found on a real Hub dataset (RoboCOIN/Agilex_Cobot_Magic_place_towel_flat): its
    # eef_rot_euler_x/z_rad dims (already known to wrap, see the wrap-crossing fix above)
    # also swing by a large, precisely correlated amount whenever the episode's pitch axis
    # passes near its own gimbal-lock singularity, even though the end effector barely
    # moved -- check_smoothness's per-dim wrap-aware diff still saw each axis's own huge
    # swing as an "abrupt action jump", just as before the wrap fix, on nearly every
    # episode of two real datasets (RoboCOIN/Agilex_Cobot_Magic_heat_burger and
    # _place_towel_flat).
    results = {r.episode_index: r for r in evaluate(_gimbal_lock_dataset())}

    gimbal_result = results[7]
    assert gimbal_result.metrics["spikes"] == 0
    assert gimbal_result.metrics["jumps"] == 0
    assert not [f for f in gimbal_result.flags if f.check == "smoothness"], gimbal_result.flags

    # the fix must not blind the check to a real jump on just one of the paired dims
    jump_result = results[8]
    assert jump_result.metrics["jumps"] > 0
    assert "smoothness" in _checks(jump_result)


def _quaternion_dataset() -> Dataset:
    """A synthetic dataset with one action dim group that's a unit quaternion
    (``ee_rot.qw/qx/qy/qz``, a smoothly rotating end-effector orientation) plus a normal,
    unrelated joint dim.

    Built by hand for the same reason as ``_angular_dataset``: it needs dims that are
    genuinely a unit quaternion (sum-of-squares exactly 1 every frame), with one episode
    where the representation's sign flips outright between two frames (dot product -1,
    the real dataset's failure mode) without the underlying orientation actually changing.
    """
    n = 90  # 3s at FPS
    t = np.arange(n)
    names = ["ee_rot.qw", "ee_rot.qx", "ee_rot.qy", "ee_rot.qz", "joint"]
    axis = np.array([0.3, 0.6, 0.7])
    axis = axis / np.linalg.norm(axis)

    def quat(theta: np.ndarray) -> np.ndarray:
        half = theta / 2
        return np.stack(
            [np.cos(half), *(a * np.sin(half) for a in axis)],
            axis=1,
        )

    def episode(idx: int, theta: np.ndarray, jump_at: int | None = None) -> Episode:
        q = quat(theta)
        joint = 5.0 * np.sin(2 * np.pi * t / n) + 0.01 * idx
        if jump_at is not None:
            joint = joint.copy()
            joint[jump_at:] += 40.0  # a real, unrelated abrupt jump, not a sign flip
        action = np.concatenate([q, joint[:, None]], axis=1)
        return Episode(
            index=idx, timestamps=t / FPS, frame_index=t, action=action, state=action.copy()
        )

    # small, continuous rotation step every frame: a normal dataset-wide step_p99 baseline
    clean = [episode(i, 0.01 * (t + i)) for i in range(6)]

    # the same smooth rotation as a clean episode, but the sign is flipped outright
    # starting mid-episode -- the exact same orientation, represented differently
    flip_theta = 0.01 * t
    flip_q = quat(flip_theta)
    flip_q[45:] *= -1.0
    flip_joint = 5.0 * np.sin(2 * np.pi * t / n) + 0.01 * 6
    flip_action = np.concatenate([flip_q, flip_joint[:, None]], axis=1)
    flip_ep = Episode(
        index=6, timestamps=t / FPS, frame_index=t, action=flip_action, state=flip_action.copy()
    )

    jump_ep = episode(7, 0.01 * t, jump_at=45)  # a real defect, unrelated to the quaternion
    return Dataset(
        root=Path("."),
        fps=float(FPS),
        codebase_version="v3.0",
        robot_type=None,
        episodes=[*clean, flip_ep, jump_ep],
        action_names=names,
    )


def test_smoothness_ignores_quaternion_sign_flip():
    # Found on a real Hub dataset (adityx23/icl-dataset, a bimanual arm teleoperated via
    # end-effector pose): two episodes had a frame where the right_ee.qw/qx/qy/qz action
    # dims all flipped sign at once (confirmed directly: the dot product of the two
    # frames' quaternions is exactly -1.0), which check_smoothness's plain per-dim diff
    # read as 4 dims all jumping by up to their full range in one frame -- a "smoothness"
    # flag with no real defect, since a unit quaternion and its negation represent the
    # exact same orientation.
    results = {r.episode_index: r for r in evaluate(_quaternion_dataset())}

    flip_result = results[6]
    assert flip_result.metrics["spikes"] == 0
    assert flip_result.metrics["jumps"] == 0
    assert not [f for f in flip_result.flags if f.check == "smoothness"], flip_result.flags

    # the fix must not blind the check to a real, unrelated jump on a non-quaternion dim
    jump_result = results[7]
    assert jump_result.metrics["jumps"] > 0
    assert "smoothness" in _checks(jump_result)


def test_quaternion_action_groups_requires_unit_norm():
    # 4 dims whose sum-of-squares is always 1 (a real quaternion) are detected; 4 more
    # dims that happen to be named the same way but whose sum-of-squares is nowhere near 1
    # are a naming coincidence, not a real quaternion, and must be rejected.
    theta = np.linspace(0, 2 * np.pi, 12)
    real = np.stack([np.cos(theta / 2), np.sin(theta / 2), np.zeros(12), np.zeros(12)], axis=1)
    fake = 2.0 * real  # sum-of-squares is always 4, not 1
    allA = np.concatenate([real, fake], axis=1)
    names = [
        "real.qw",
        "real.qx",
        "real.qy",
        "real.qz",
        "fake.qw",
        "fake.qx",
        "fake.qy",
        "fake.qz",
    ]
    groups = _quaternion_action_groups(names, allA, Config())
    assert groups == [(0, 1, 2, 3)]


def _half_quaternion_dataset() -> Dataset:
    """A synthetic dataset with a *half* unit quaternion action dim pair
    (``base_orient_quat_w``/``base_orient_quat_z``, a wheeled base's yaw) plus a normal,
    unrelated joint dim.

    Found on a real Hub dataset (``RoboCOIN/Galbot_G1_load_floor_trash_3``, a wheeled-base
    humanoid): ``w = cos(theta/2)`` barely moves for a modest turn (its derivative vanishes
    near ``theta=0``), so its own dataset-wide IQR-based scale is tiny; diffing ``w`` (and
    ``z``) directly and normalising by that tiny scale read an ordinary, smooth, modest base
    turn as an "abrupt action jump" on nearly half the dataset's episodes. Built by hand for
    the same reason as ``_quaternion_dataset``: it needs dims that are genuinely a (half)
    unit quaternion (``w**2 + z**2`` exactly 1 every frame), with many clean episodes
    establishing a baseline of only tiny jitter (the base mostly not turning, as in the real
    dataset) plus one episode with a modest, perfectly smooth turn and one with a real,
    abrupt discontinuity.
    """
    n = 90  # 3s at FPS
    t = np.arange(n)
    n_clean = 60
    names = ["joint", "base_orient_quat_w", "base_orient_quat_z"]

    def quat_wz(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        half = theta / 2
        return np.cos(half), np.sin(half)

    def episode(idx: int, theta: np.ndarray) -> Episode:
        w, z = quat_wz(theta)
        joint = 5.0 * np.sin(2 * np.pi * t / n) + 0.01 * idx
        action = np.stack([joint, w, z], axis=1)
        return Episode(
            index=idx, timestamps=t / FPS, frame_index=t, action=action, state=action.copy()
        )

    def smooth_bump(mag: float, start: int, width: int) -> np.ndarray:
        bump = np.zeros(n)
        ramp = mag * 0.5 * (1 - np.cos(np.pi * np.arange(width) / (width - 1)))
        bump[start : start + width] = ramp
        bump[start + width :] = mag
        return bump

    # tiny dataset-wide jitter: the base holds still (within a fraction of a degree) for
    # virtually every clean episode, same as the real dataset
    clean = [episode(i, 0.01 * np.sin(2 * np.pi * t / 30 + 0.1 * i)) for i in range(n_clean)]

    # one genuine, smooth, modest turn (~6 degrees) -- real motion, not a glitch, and far
    # smaller than the dataset's own observed max on the real robot
    turn_ep = episode(n_clean, smooth_bump(0.1, 25, 25))

    # one real, abrupt discontinuity (unrelated to the smooth-turn representation issue)
    jerky_theta = np.zeros(n)
    jerky_theta[45:] = 1.5
    jerky_ep = episode(n_clean + 1, jerky_theta)

    return Dataset(
        root=Path("."),
        fps=float(FPS),
        codebase_version="v3.0",
        robot_type=None,
        episodes=[*clean, turn_ep, jerky_ep],
        action_names=names,
    )


def test_smoothness_ignores_ordinary_half_quaternion_turn():
    results = {r.episode_index: r for r in evaluate(_half_quaternion_dataset())}

    turn_result = results[60]
    assert turn_result.metrics["spikes"] == 0
    assert turn_result.metrics["jumps"] == 0
    assert not [f for f in turn_result.flags if f.check == "smoothness"], turn_result.flags

    # the fix must not blind the check to a real, unrelated abrupt discontinuity
    jerky_result = results[61]
    assert jerky_result.metrics["jumps"] > 0
    assert "smoothness" in _checks(jerky_result)

    # every clean episode (tiny jitter only) stays silent
    for i in range(60):
        assert results[i].metrics["jumps"] == 0, (i, results[i].metrics)


def test_half_quaternion_action_groups_requires_unit_norm():
    # a w/z pair whose sum-of-squares is always 1 (a real half-quaternion) is detected; a
    # same-named pair whose sum-of-squares is nowhere near 1 is a naming coincidence, not
    # a real one, and must be rejected. A 4-member group (a full quaternion) must not also
    # be picked up here -- that's _quaternion_action_groups' job.
    theta = np.linspace(0, 2 * np.pi, 12)
    real = np.stack([np.cos(theta / 2), np.sin(theta / 2)], axis=1)
    fake = 2.0 * real  # sum-of-squares is always 4, not 1
    full = np.stack([np.cos(theta / 2), np.sin(theta / 2), np.zeros(12), np.zeros(12)], axis=1)
    allA = np.concatenate([real, fake, full], axis=1)
    names = [
        "base.quat_w",
        "base.quat_z",
        "fake.quat_w",
        "fake.quat_z",
        "full.qw",
        "full.qx",
        "full.qy",
        "full.qz",
    ]
    groups = _half_quaternion_action_groups(names, allA, Config())
    assert groups == [(0, 1)]


def _sustained_ramp_dataset() -> Dataset:
    """A synthetic dataset with one action dim that sits nearly still almost the whole
    time (like a dexterous-hand finger joint that only moves to grasp) plus a normal,
    unrelated joint dim.

    Built by hand (not via ``make_dataset``/``write_dataset``) for the same reason as
    ``_angular_dataset``: it needs a dim whose dataset-wide step is tiny almost everywhere,
    with many clean episodes to dilute the handful of large steps below the 99th
    percentile used as the check's own baseline.
    """
    n = 90  # 3s at FPS
    t = np.arange(n)
    n_clean = 120

    def episode(idx: int, still_dim: np.ndarray) -> Episode:
        joint = 5.0 * np.sin(2 * np.pi * t / n) + 0.01 * (idx % 10)
        action = np.stack([joint, still_dim], axis=1)
        return Episode(
            index=idx, timestamps=t / FPS, frame_index=t, action=action, state=action.copy()
        )

    def baseline(idx: int) -> np.ndarray:
        return 0.2 + 0.01 * np.sin(2 * np.pi * t / 13 + idx)  # tiny dataset-wide jitter

    clean = [episode(i, baseline(i)) for i in range(n_clean)]

    # one smooth, monotonic, continuous ramp (a fast but perfectly smooth grasp closing),
    # held at its new position afterwards -- never reverses, never jumps again
    ramp = baseline(n_clean).copy()
    ramp[40:61] = np.linspace(ramp[40], 0.9, 21)
    ramp[61:] = 0.9
    ramp_ep = episode(n_clean, ramp)

    # two genuinely separate single-frame discontinuities, far apart in time
    two_jumps = baseline(n_clean + 1).copy()
    two_jumps[30:] += 0.5
    two_jumps[70:] += 0.5
    jump_ep = episode(n_clean + 1, two_jumps)

    return Dataset(
        root=Path("."),
        fps=float(FPS),
        codebase_version="v3.0",
        robot_type=None,
        episodes=[*clean, ramp_ep, jump_ep],
    )


def test_smoothness_counts_a_sustained_ramp_as_one_jump():
    # Found on a real Hub dataset (zekaiwang/trex_dataset, a dual-arm dexterous-hand
    # dataset): its finger-joint targets sit still almost the whole episode, so
    # step_p99 for those dims is tiny; a single smooth, monotonic grasp-closing motion
    # spanning ~20 frames was reported as "20 abrupt action jump(s)" (big.any(axis=1).sum()
    # counts every frame the ramp spans), even though none of those per-frame steps is
    # actually discontinuous -- it's one continuous fast motion. 19/50 episodes carried
    # this inflated flag, each capped at the check's max 20-point penalty.
    results = {r.episode_index: r for r in evaluate(_sustained_ramp_dataset())}

    ramp_result = results[120]
    assert ramp_result.metrics["spikes"] == 0
    assert ramp_result.metrics["jumps"] == 1
    assert ramp_result.metrics["jumps"] < 20  # pre-fix: one flag per frame the ramp spans

    # the fix must not merge two genuinely separate discontinuities into one
    jump_result = results[121]
    assert jump_result.metrics["jumps"] == 2

    # every clean episode (dim at dataset-wide baseline jitter only) stays silent
    for i in range(120):
        assert results[i].metrics["jumps"] == 0, (i, results[i].metrics)


def _noisy_sustained_ramp_dataset() -> Dataset:
    """Like ``_sustained_ramp_dataset``'s ramp episode, but the ramp's per-frame step
    alternates between just-above and just-below the spike threshold instead of exceeding
    it on every single frame.

    Found on a real Hub dataset (``RoboCOIN/Galbot_G1_load_floor_trash_3``, a wheeled-base
    humanoid): its base orientation is logged as a quaternion z/w pair, and ``w`` sits
    within 0.045 of 1.0 for virtually the whole dataset (the robot rarely rotates its
    base), so its own step_p99 baseline is tiny. A genuine, continuous, smooth base
    rotation then produces a long run of frames whose step ratio exceeds
    ``spike_factor * step_p99`` -- but ordinary frame-to-frame measurement noise, not a
    perfectly uniform step size, drops some of those frames back under the threshold
    mid-turn, so the already-fixed sustained-ramp collapsing (only literally consecutive
    big frames merge into one run) still split one continuous rotation into 3-17 separate
    "jumps" per episode instead of the one real event it is.
    """
    n = 90
    t = np.arange(n)
    n_clean = 120

    def episode(idx: int, still_dim: np.ndarray) -> Episode:
        joint = 5.0 * np.sin(2 * np.pi * t / n) + 0.01 * (idx % 10)
        action = np.stack([joint, still_dim], axis=1)
        return Episode(
            index=idx, timestamps=t / FPS, frame_index=t, action=action, state=action.copy()
        )

    def baseline(idx: int) -> np.ndarray:
        return 0.2 + 0.01 * np.sin(2 * np.pi * t / 13 + idx)

    clean = [episode(i, baseline(i)) for i in range(n_clean)]

    # one continuous, monotonic rotation whose per-frame step alternates big/small (net
    # +0.8 over 20 frames), held afterwards -- never reverses direction
    noisy_ramp = baseline(n_clean).copy()
    deltas = np.tile([0.08, 0.002], 10)
    noisy_ramp[40:60] = noisy_ramp[39] + np.cumsum(deltas)
    noisy_ramp[60:] = noisy_ramp[59]
    ramp_ep = episode(n_clean, noisy_ramp)

    # two genuinely separate single-frame discontinuities, far apart in time -- must stay
    # two jumps, not get bridged into one just because they're both "big"
    two_jumps = baseline(n_clean + 1).copy()
    two_jumps[30:] += 0.5
    two_jumps[70:] += 0.5
    jump_ep = episode(n_clean + 1, two_jumps)

    return Dataset(
        root=Path("."),
        fps=float(FPS),
        codebase_version="v3.0",
        robot_type=None,
        episodes=[*clean, ramp_ep, jump_ep],
    )


def test_smoothness_bridges_short_gaps_in_a_sustained_ramp():
    results = {r.episode_index: r for r in evaluate(_noisy_sustained_ramp_dataset())}

    ramp_result = results[120]
    assert ramp_result.metrics["spikes"] == 0
    assert ramp_result.metrics["jumps"] == 1  # pre-fix: 10, one per contiguous sub-run

    # the fix must not merge two genuinely separate discontinuities into one
    jump_result = results[121]
    assert jump_result.metrics["jumps"] == 2

    # every clean episode (dim at dataset-wide baseline jitter only) stays silent
    for i in range(120):
        assert results[i].metrics["jumps"] == 0, (i, results[i].metrics)


def _noise_dominated_dim_dataset() -> Dataset:
    """A synthetic dataset with five normal, actively-moving joint dims plus one dim that
    sits almost frozen for the whole dataset (like a torso/neck joint the task never
    really uses), with occasional sensor-noise blips.

    Built by hand for the same reason as the other smoothness fixtures: it needs five
    dims with a natural spread of ``step_p99`` (so the new dim-outlier check has a
    meaningful population to compare against) plus a sixth dim whose own ``step_p99`` is
    a clear outlier among them.
    """
    n = 90  # 3s at FPS
    t = np.arange(n)
    n_clean = 40
    rng = np.random.default_rng(0)
    freqs = np.array([1.0, 1.6, 0.6, 2.2, 0.9])
    amps = np.array([5.0, 3.0, 7.0, 2.0, 6.0])
    phases0 = np.array([0.3, 1.1, 2.0, 2.9, 4.0])

    def active_dims(idx: int) -> np.ndarray:
        return amps * np.sin(2 * np.pi * freqs[None, :] * t[:, None] / n + phases0 + 0.01 * idx)

    def noise_dim(idx: int, blip_frame: int | None = None, blip_mag: float = 0.0) -> np.ndarray:
        d = 0.002 * np.sin(2 * np.pi * t / 37 + idx) + rng.normal(0, 0.0003, n)
        if blip_frame is not None:
            d = d.copy()
            d[blip_frame:] += blip_mag  # a one-off sensor blip, held (not an immediate
            # reversal, which check_smoothness already treats as a "spike" glitch)
        return d

    def episode(
        idx: int,
        noise: np.ndarray,
        jump_dim: int | None = None,
        jump_frame: int | None = None,
        jump_mag: float = 0.0,
    ) -> Episode:
        act = active_dims(idx)
        if jump_dim is not None:
            act = act.copy()
            act[jump_frame:, jump_dim] += jump_mag
        action = np.concatenate([act, noise[:, None]], axis=1)
        return Episode(
            index=idx, timestamps=t / FPS, frame_index=t, action=action, state=action.copy()
        )

    blip_episodes = range(0, n_clean, 7)
    clean = []
    for i in range(n_clean):
        if i in blip_episodes:
            noise = noise_dim(i, blip_frame=20 + (i % 50), blip_mag=0.05 if i % 2 == 0 else -0.05)
        else:
            noise = noise_dim(i)
        clean.append(episode(i, noise))

    # a genuine, unrelated jump on one of the five normal dims
    jump_ep = episode(n_clean, noise_dim(n_clean + 900), jump_dim=1, jump_frame=45, jump_mag=10.0)

    return Dataset(
        root=Path("."),
        fps=float(FPS),
        codebase_version="v3.0",
        robot_type=None,
        episodes=[*clean, jump_ep],
    )


def test_smoothness_ignores_a_noise_dominated_dim():
    # Found on a real Hub dataset (RoboCOIN/AI2_Alphabot_2_insert_water_pipe): its torso
    # joints sit still for virtually the whole task (dataset-wide range under 0.08 rad,
    # vs. 0.17+ for every arm/eef/gripper dim), so step_p99 for those dims reflects
    # nothing but sensor/controller noise around a fixed setpoint. That noise, scattered
    # as separate single-frame blips across almost every episode (not one sustained ramp,
    # so the run-counting fix above doesn't help), was flagged as a dataset-wide pile of
    # "abrupt action jump(s)": 19/20 episodes carried a `smoothness` flag, on a dataset
    # whose other checks looked clean (mean score 85.1, 1 episode below 70).
    results = {r.episode_index: r for r in evaluate(_noise_dominated_dim_dataset())}

    for i in range(40):
        assert results[i].metrics["jumps"] == 0, (i, results[i].metrics)
        assert results[i].metrics["spikes"] == 0, (i, results[i].metrics)

    # the fix must not blind the check to a real jump on one of the normal dims
    jump_result = results[40]
    assert jump_result.metrics["jumps"] == 1
    assert "smoothness" in _checks(jump_result)


def _jerkier_than_peers_dataset() -> Dataset:
    """A single-dim dataset where most episodes are a close variation on the same
    smooth reach motion (a tight cluster, by construction), a handful are notably
    smoother still (pulling the cluster's own spread, not its median), one episode is
    only moderately jerkier, and one is genuinely, drastically jerkier.
    """
    n = 240  # 8s at FPS
    t = np.linspace(0, 1, n)
    reach = 50.0 * (1 - np.cos(2 * np.pi * t)) / 2 * np.sin(2 * np.pi * t)

    def make(wiggle_amp: float, wiggle_freq: float, wiggle_phase: float) -> np.ndarray:
        wiggle = wiggle_amp * np.sin(wiggle_freq * 2 * np.pi * t + wiggle_phase)
        return (reach + wiggle)[:, None]

    def episode(idx: int, action: np.ndarray) -> Episode:
        frame_index = np.arange(n)
        return Episode(
            index=idx,
            timestamps=frame_index / FPS,
            frame_index=frame_index,
            action=action,
            state=action.copy(),
        )

    episodes = [episode(i, make(0.3 + 0.02 * i, 5.0, 0.1 * i)) for i in range(14)]
    episodes += [episode(14 + i, make(0.02 + 0.002 * i, 5.0, 0.2 * i)) for i in range(6)]
    episodes.append(episode(20, make(1.1, 5.0, 0.0)))  # moderately jerkier only
    episodes.append(episode(21, make(0.3, 20.0, 0.0)))  # genuinely, drastically jerkier

    return Dataset(
        root=Path("."), fps=FPS, codebase_version="v3.0", robot_type=None, episodes=episodes
    )


def _scale_outlier_dim_dataset() -> Dataset:
    """A synthetic dataset with five normal, actively-moving joint dims plus one dim
    that moves smoothly but through a vanishingly small range (like a torso/neck joint
    whose own typical per-frame step, after normalising by its own tiny range, still
    looks like ordinary motion -- so ``_exclude_noise_dominated_dims``'s step_p99
    comparison across dims doesn't flag it), with an occasional held sensor blip that
    is tiny in absolute terms but huge relative to the dim's own near-zero range.
    """
    n = 90  # 3s at FPS
    t = np.arange(n)
    n_clean = 40
    rng = np.random.default_rng(0)
    freqs = np.array([1.0, 1.6, 0.6, 2.2, 0.9])
    amps = np.array([5.0, 3.0, 7.0, 2.0, 6.0])
    phases0 = np.array([0.3, 1.1, 2.0, 2.9, 4.0])

    def active_dims(idx: int) -> np.ndarray:
        return amps * np.sin(2 * np.pi * freqs[None, :] * t[:, None] / n + phases0 + 0.01 * idx)

    def tiny_dim(idx: int, blip_frame: int | None = None, blip_mag: float = 0.0) -> np.ndarray:
        d = 0.00008 * np.sin(2 * np.pi * 0.3 * t / n + 0.05 * idx) + rng.normal(0, 2e-6, n)
        if blip_frame is not None:
            d = d.copy()
            d[blip_frame : blip_frame + 5] += blip_mag  # held, not an immediate reversal
        return d

    def episode(
        idx: int,
        tiny: np.ndarray,
        jump_dim: int | None = None,
        jump_frame: int | None = None,
        jump_mag: float = 0.0,
    ) -> Episode:
        act = active_dims(idx)
        if jump_dim is not None:
            act = act.copy()
            act[jump_frame:, jump_dim] += jump_mag
        action = np.concatenate([act, tiny[:, None]], axis=1)
        return Episode(
            index=idx, timestamps=t / FPS, frame_index=t, action=action, state=action.copy()
        )

    clean = [episode(i, tiny_dim(i)) for i in range(n_clean)]
    # a real, false-positive-triggering noise blip on the tiny dim: tiny in absolute
    # terms (0.00015) but ~2x the dim's own dataset-wide range
    blip_ep = episode(n_clean, tiny_dim(n_clean, blip_frame=30, blip_mag=0.00015))
    # a genuine, unrelated jump on one of the five normal dims
    jump_ep = episode(n_clean + 1, tiny_dim(n_clean + 1), jump_dim=1, jump_frame=45, jump_mag=10.0)

    return Dataset(
        root=Path("."),
        fps=float(FPS),
        codebase_version="v3.0",
        robot_type=None,
        episodes=[*clean, blip_ep, jump_ep],
    )


def test_smoothness_ignores_a_scale_outlier_dim():
    # Found on a real Hub dataset (RoboCOIN/AI2_Alphabot_2_insert_water_pipe), as a
    # residual of the noise-dominated-dim fix above: its `neck_joint_1/2_rad` and
    # `torso_joint_4_rad` dims have a dataset-wide range under 0.0002 rad (scale outlier
    # z=4.1-5.1 against the dataset's other action dims' own scale), but their own
    # step_p99 -- already normalised by that same tiny range -- lands close enough to a
    # genuinely driven dim's step_p99 (outlier z only 1.1-3.0, short of the 4.0
    # threshold) that `_exclude_noise_dominated_dims` let them through, so they kept
    # contributing scattered single-frame blips to the dataset's `smoothness` flags.
    results = {r.episode_index: r for r in evaluate(_scale_outlier_dim_dataset())}

    for i in range(40):
        assert results[i].metrics["jumps"] == 0, (i, results[i].metrics)
        assert results[i].metrics["spikes"] == 0, (i, results[i].metrics)
    blip_result = results[40]
    assert blip_result.metrics["jumps"] == 0, blip_result.metrics
    assert blip_result.metrics["spikes"] == 0, blip_result.metrics

    # the fix must not blind the check to a real jump on one of the normal dims
    jump_result = results[41]
    assert jump_result.metrics["jumps"] == 1
    assert "smoothness" in _checks(jump_result)


def _unused_side_dataset() -> Dataset:
    """A synthetic bimanual dataset where the "right_*" joints are actively driven and
    every matching "left_*" joint sits frozen (small encoder jitter only) for the whole
    dataset -- a single-arm task on a dual-arm robot, as found on real Hub datasets
    (RoboCOIN/AI2_Alphabot_2_package_fruit, _stamp_document). The frozen dims occasionally
    hold a small sensor-noise blip, exactly like the already-fixed noise-dominated-dim
    case, but here there are 7 of them (not 1), too large a minority for the dataset's own
    median/MAD-based outlier tests to separate from the 7 genuinely driven dims. The 7
    driven dims (not fewer) also give the scale-outlier population enough points of its
    own, once the frozen side is removed from it, for a stable robust z-score.
    """
    n = 90  # 3s at FPS
    t = np.arange(n)
    n_clean = 40
    rng = np.random.default_rng(0)
    n_dims = 7
    freqs = np.array([1.0, 1.6, 0.6, 2.2, 0.9, 1.3, 1.9])
    amps = np.array([5.0, 5.5, 6.0, 5.2, 6.5, 5.8, 6.2])
    phases0 = np.array([0.3, 1.1, 2.0, 2.9, 4.0, 0.7, 3.3])
    names = [f"right_joint_{i}_rad" for i in range(n_dims)] + [
        f"left_joint_{i}_rad" for i in range(n_dims)
    ]

    def active_dims(idx: int) -> np.ndarray:
        return amps * np.sin(2 * np.pi * freqs[None, :] * t[:, None] / n + phases0 + 0.01 * idx)

    def frozen_dims(idx: int, blip_frame: int | None = None, blip_mag: float = 0.0) -> np.ndarray:
        d = 0.0003 * rng.normal(0, 1, (n, n_dims)) + 2.6e-5 * np.sin(
            2 * np.pi * 0.3 * t[:, None] / n + 0.05 * idx
        )
        if blip_frame is not None:
            d = d.copy()
            d[blip_frame:, idx % n_dims] += blip_mag  # a held sensor blip on one frozen dim
        return d

    def episode(
        idx: int,
        frozen: np.ndarray,
        jump_dim: int | None = None,
        jump_frame: int | None = None,
        jump_mag: float = 0.0,
    ) -> Episode:
        act = active_dims(idx)
        if jump_dim is not None:
            act = act.copy()
            act[jump_frame:, jump_dim] += jump_mag
        action = np.concatenate([act, frozen], axis=1)
        return Episode(
            index=idx, timestamps=t / FPS, frame_index=t, action=action, state=action.copy()
        )

    blip_episodes = range(0, n_clean, 7)
    clean = []
    for i in range(n_clean):
        if i in blip_episodes:
            frozen = frozen_dims(
                i, blip_frame=20 + (i % 50), blip_mag=0.02 if i % 2 == 0 else -0.02
            )
        else:
            frozen = frozen_dims(i)
        clean.append(episode(i, frozen))

    # a genuine, unrelated jump on one of the five driven dims
    jump_ep = episode(n_clean, frozen_dims(n_clean + 900), jump_dim=1, jump_frame=45, jump_mag=10.0)

    return Dataset(
        root=Path("."),
        fps=float(FPS),
        codebase_version="v3.0",
        robot_type=None,
        episodes=[*clean, jump_ep],
        action_names=names,
    )


def test_smoothness_ignores_an_unused_bimanual_side():
    # Found on real Hub datasets beyond the existing validation set
    # (RoboCOIN/AI2_Alphabot_2_package_fruit, RoboCOIN/AI2_Alphabot_2_stamp_document):
    # both are single-arm tasks on a dual-arm robot, so the whole "left_*" group of
    # action dims -- 10+ of ~30 dims, too large a minority for the dataset-wide
    # median/MAD tests above to separate from the genuinely driven "right_*" dims --
    # sits frozen for every episode, yet kept contributing scattered noise blips as
    # `smoothness` "abrupt action jump" flags on 10-14/20 episodes of each dataset.
    ctx = DatasetContext.build(_unused_side_dataset(), Config())
    assert np.isnan(ctx.step_p99[7:14]).all(), ctx.step_p99  # every "left_*" dim excluded
    assert not np.isnan(ctx.step_p99[:7]).any(), ctx.step_p99  # "right_*" dims untouched

    results = {r.episode_index: r for r in evaluate(_unused_side_dataset())}
    for i in range(40):
        assert results[i].metrics["jumps"] == 0, (i, results[i].metrics)
        assert results[i].metrics["spikes"] == 0, (i, results[i].metrics)

    # the fix must not blind the check to a real jump on one of the driven dims
    jump_result = results[40]
    assert jump_result.metrics["jumps"] == 1
    assert "smoothness" in _checks(jump_result)


def test_smoothness_jerkier_than_peers_needs_a_material_ldlj_gap():
    # Found by running `demoqc score` against a real Hub dataset beyond the existing
    # validation set (EPITECH-LILLE/eval_training): episode 37, scored against a
    # 20-episode peer group, carried a "much jerkier than peers (smoothness z=-3.6)"
    # flag -- but the same episode, re-scored against the dataset's full 40 episodes
    # instead, got z=-0.68 (not flagged) for an almost identical raw LDLJ value. The
    # z-score had changed, not the episode: most of this dataset's episodes are a close
    # variation on the same motion, so the robust MAD reflects only that tight
    # majority's own tiny spread, even with a few notably smoother episodes pulling the
    # population in the other direction -- a moderate, probably-insignificant LDLJ gap
    # from the median (here, ~0.7 nats) gets inflated into an extreme z-score purely by
    # which episodes happen to be in the peer group.
    results = {r.episode_index: r for r in evaluate(_jerkier_than_peers_dataset())}

    for i in range(20):
        assert "smoothness" not in _checks(results[i]), (i, results[i].flags)

    # episode 20 trips the raw z-score threshold (reproducing the pre-fix false
    # positive exactly) but its LDLJ gap from the median is too small to be material
    borderline = results[20]
    assert borderline.metrics["ldlj_z"] < Config().jerk_z
    assert "smoothness" not in _checks(borderline), borderline.flags

    # a genuinely, drastically jerkier episode must still be caught
    jerky = results[21]
    assert jerky.metrics["ldlj_z"] < Config().jerk_z
    assert any("jerkier than peers" in f.message for f in jerky.flags)


def _gripper_snap_dataset() -> Dataset:
    """A synthetic dataset with five normal, continuously-moving joint dims plus a
    gripper dim that sits at a steady setpoint with small encoder jitter for almost
    every frame and snaps across its whole 0..1 range in a single frame during an
    ordinary grasp -- rare enough, relative to each episode's length, that the
    dataset-wide ``step_p99`` for the gripper dim reflects only the jitter, not the
    snap (mirroring a real bang-bang actuator's duty cycle).
    """
    n = 150  # 5s at FPS
    t = np.arange(n)
    n_clean = 36
    rng = np.random.default_rng(0)
    freqs = np.array([1.0, 1.6, 0.6, 2.2, 0.9])
    amps = np.array([5.0, 3.0, 7.0, 2.0, 6.0])
    phases0 = np.array([0.3, 1.1, 2.0, 2.9, 4.0])

    def active_dims(idx: int) -> np.ndarray:
        return amps * np.sin(2 * np.pi * freqs[None, :] * t[:, None] / n + phases0 + 0.01 * idx)

    def gripper_dim(idx: int, snap_frame: int) -> np.ndarray:
        g = np.zeros(n) + rng.normal(0, 0.01, n)
        g[snap_frame:] = 1.0 + rng.normal(0, 0.01, n - snap_frame)
        return g

    def episode(idx: int, action: np.ndarray) -> Episode:
        return Episode(
            index=idx, timestamps=t / FPS, frame_index=t, action=action, state=action.copy()
        )

    clean = []
    for i in range(n_clean):
        act = active_dims(i)
        grip = gripper_dim(i, snap_frame=60 + (i % 20))
        clean.append(episode(i, np.concatenate([act, grip[:, None]], axis=1)))

    # the episode under test: an ordinary grasp snap, same as every clean episode
    snap_act = active_dims(n_clean + 500)
    snap_grip = gripper_dim(n_clean + 500, snap_frame=75)
    snap_ep = episode(n_clean, np.concatenate([snap_act, snap_grip[:, None]], axis=1))

    # a genuine, unrelated jump on one of the five normal dims must still be caught
    jump_act = active_dims(n_clean + 501)
    jump_act[90:, 1] += 10.0
    jump_grip = gripper_dim(n_clean + 501, snap_frame=75)
    jump_ep = episode(n_clean + 1, np.concatenate([jump_act, jump_grip[:, None]], axis=1))

    return Dataset(
        root=Path("."),
        fps=float(FPS),
        codebase_version="v3.0",
        robot_type=None,
        episodes=[*clean, snap_ep, jump_ep],
        action_names=["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "gripper"],
    )


def test_smoothness_ignores_gripper_snap():
    # Found by running `demoqc score` against a real Hub dataset beyond the existing
    # validation set (allenai/MolmoAct2-BimanualYAM-Dataset): 13/20 episodes carried a
    # spurious "abrupt action jump(s)" smoothness flag, every single flagged frame on a
    # gripper dim (confirmed by inspecting the flagged dim/frame directly), on a dataset
    # whose arm joints -- continuously driven, so their own step_p99 already reflects
    # normal fast motion -- carried zero smoothness flags. A gripper sits static almost
    # the whole episode and snaps across its full range in 1-4 frames during a deliberate
    # grasp/release, so the dataset-wide 99th-percentile step for that dim reflects only
    # its idle-frame jitter, not the snap -- any percentile-based "this step is an
    # outlier" threshold ends up flagging every ordinary grasp.
    results = {r.episode_index: r for r in evaluate(_gripper_snap_dataset())}

    for i in range(36):
        assert results[i].metrics["jumps"] == 0, (i, results[i].metrics)
        assert "smoothness" not in _checks(results[i]), (i, results[i].flags)

    snap_result = results[36]
    assert snap_result.metrics["jumps"] == 0
    assert "smoothness" not in _checks(snap_result), snap_result.flags

    # a genuine, unrelated jump on a normal joint dim must still be caught
    jump_result = results[37]
    assert jump_result.metrics["jumps"] == 1
    assert "smoothness" in _checks(jump_result)


def test_gripper_dims_matches_vr_controller_trigger_squeeze_names():
    # Found on BitRobot/HIW-500-LeRobot (a Unitree G1 humanoid teleoperated via a VR
    # controller): its gripper-actuation action/state dims are named "left_trigger",
    # "left_squeeze", "right_trigger", "right_squeeze" -- a controller's index-trigger and
    # grip-squeeze axes, not literally "gripper". gripper_dims matched only the substring
    # "gripper", so none of these four were ever detected.
    ds = _gripper_snap_dataset()
    ds.action_names = [
        "joint_0",
        "joint_1",
        "joint_2",
        "joint_3",
        "left_trigger",
        "right_squeeze",
    ]
    ctx = DatasetContext.build(ds, Config())
    assert ctx.gripper_dims == [4, 5]


def test_smoothness_ignores_vr_controller_trigger_snap():
    # Same real dataset as above: 19/20 episodes carried spurious "abrupt action jump(s)"
    # smoothness flags, almost entirely on `right_trigger`/`right_squeeze` -- those dims
    # snap between two steady setpoints during a deliberate grasp, exactly like the
    # already-fixed literal-"gripper"-named case, just under VR-controller naming.
    ds = _gripper_snap_dataset()
    ds.action_names = ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "right_trigger"]
    results = {r.episode_index: r for r in evaluate(ds)}

    for i in range(36):
        assert results[i].metrics["jumps"] == 0, (i, results[i].metrics)
        assert "smoothness" not in _checks(results[i]), (i, results[i].flags)

    snap_result = results[36]
    assert snap_result.metrics["jumps"] == 0
    assert "smoothness" not in _checks(snap_result), snap_result.flags

    # a genuine, unrelated jump on a normal joint dim must still be caught
    jump_result = results[37]
    assert jump_result.metrics["jumps"] == 1
    assert "smoothness" in _checks(jump_result)


def test_gripper_dims_matches_dexterous_hand_joint_names():
    # Found on RoboCOIN/Unitree_G1edu_u3_place_bread_1 (a Unitree G1 humanoid with a
    # dexterous multi-finger hand instead of a simple gripper): its grasp-actuation action
    # dims are named "left_hand_joint_1_rad".."left_hand_joint_7_rad" (and the same for
    # "right_"), not literally "gripper". gripper_dims matched only "gripper"/"trigger"/
    # "squeeze", so none of these 14 per-finger dims were ever detected.
    ds = _gripper_snap_dataset()
    ds.action_names = [
        "joint_0",
        "joint_1",
        "joint_2",
        "joint_3",
        "left_hand_joint_4_rad",
        "right_hand_joint_7_rad",
    ]
    ctx = DatasetContext.build(ds, Config())
    assert ctx.gripper_dims == [4, 5]


def test_smoothness_ignores_dexterous_hand_joint_snap():
    # Same real dataset as above: 5/10 episodes carried spurious "abrupt action jump(s)"
    # smoothness flags, every flagged frame on a "*_hand_joint_*" dim -- each finger joint
    # sits at a steady setpoint almost the whole episode and snaps through its own range in
    # 1-2 frames during a deliberate grasp/release, exactly the already-fixed
    # literal-"gripper"-named case, just under a dexterous-hand naming convention.
    ds = _gripper_snap_dataset()
    ds.action_names = [
        "joint_0",
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "left_hand_joint_3_rad",
    ]
    results = {r.episode_index: r for r in evaluate(ds)}

    for i in range(36):
        assert results[i].metrics["jumps"] == 0, (i, results[i].metrics)
        assert "smoothness" not in _checks(results[i]), (i, results[i].flags)

    snap_result = results[36]
    assert snap_result.metrics["jumps"] == 0
    assert "smoothness" not in _checks(snap_result), snap_result.flags

    # a genuine, unrelated jump on a normal joint dim must still be caught
    jump_result = results[37]
    assert jump_result.metrics["jumps"] == 1
    assert "smoothness" in _checks(jump_result)


def test_gripper_dims_excludes_ee_pose_group_named_with_a_gripper_frame_qualifier():
    # Found on unitreerobotics/G1_Dex1_Hang_Pants: its structured
    # action.left_ee_pose_gripper_base/action.right_ee_pose_gripper_torso sub-features (a
    # continuous end-effector pose *expressed in the gripper's own base/torso frame*, not a
    # gripper actuator) flatten into per-dim names like "left_ee_pose_gripper_base.x"/".roll"
    # -- gripper_dims' bare "gripper" substring search matched all six of these pose-axis
    # dims purely because "gripper" happens to appear in their shared frame-qualifier
    # prefix, not because any of them is a gripper.
    ds = _gripper_snap_dataset()
    ds.action_names = [
        "left_ee_pose_gripper_base.x",
        "left_ee_pose_gripper_base.y",
        "left_ee_pose_gripper_base.z",
        "left_ee_pose_gripper_base.roll",
        "left_ee_pose_gripper_base.pitch",
        "gripper",
    ]
    ctx = DatasetContext.build(ds, Config())
    assert ctx.gripper_dims == [5]


def test_smoothness_detects_a_jump_on_an_ee_pose_dim_named_with_a_gripper_frame_qualifier():
    # Same real dataset as above: before this fix, every one of these pose-axis dims was
    # wrongly treated as a gripper dim and excluded from spike/jump detection entirely (its
    # step_p99 set to NaN) -- so a genuine jump on any of them would go completely
    # undetected, not merely miscounted.
    ds = _gripper_snap_dataset()
    ds.action_names = [
        "ee_pose_gripper_base.x",
        "ee_pose_gripper_base.y",
        "ee_pose_gripper_base.z",
        "ee_pose_gripper_base.roll",
        "ee_pose_gripper_base.pitch",
        "gripper",
    ]
    results = {r.episode_index: r for r in evaluate(ds)}

    # the genuine, unrelated jump injected on dim 1 ("ee_pose_gripper_base.y") must be caught
    jump_result = results[37]
    assert jump_result.metrics["jumps"] == 1
    assert "smoothness" in _checks(jump_result)

    # every ordinary episode (including the gripper's own real snap) stays silent
    for i in range(36):
        assert results[i].metrics["jumps"] == 0, (i, results[i].metrics)
        assert "smoothness" not in _checks(results[i]), (i, results[i].flags)


def test_stale_state(make_dataset):
    specs = clean_specs()
    specs[7] = EpisodeSpec(stale_state_from=(60, 150), seed=7)
    results = _results(make_dataset, specs)

    assert results[7].metrics["stale_state_fraction"] > 0.2
    assert "stale_state" in _checks(results[7])
    assert all("stale_state" not in _checks(r) for i, r in enumerate(results) if i != 7)


def test_stale_state_ignores_a_dataset_wide_low_rate_baseline(make_dataset):
    """A robot whose observation.state publishes slower than the logging fps holds state
    for a frame or two throughout every episode -- a hardware characteristic, not a stale
    read. Found on RoboCOIN/Galbot_G1_bag_table_trash_1: every one of 15 episodes had a
    tight, consistent ~22% stale-state fraction (confirmed correlated across all 38
    action/state dims, not isolated to the chassis), flagging every episode of an otherwise
    clean dataset. A genuinely bad episode (a real localized stale burst on top of that same
    baseline) must still be caught."""
    specs = [
        EpisodeSpec(seconds=8.0 + 0.3 * (i % 4), seed=i, state_hold_every=5) for i in range(12)
    ]
    specs.append(EpisodeSpec(seconds=8.0, seed=99, state_hold_every=5, stale_state_from=(60, 150)))
    results = _results(make_dataset, specs)

    for i in range(12):
        assert "stale_state" not in _checks(results[i]), (i, results[i].metrics, results[i].flags)
    assert "stale_state" in _checks(results[12])


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


def _moderate_corr_sync_dataset() -> Dataset:
    """A single-dim dataset where ``observation.state`` is only ever a partial echo of
    ``action`` (mixed with an unrelated motion at a different frequency), so every clean
    episode's peak correlation lands in a tight, moderate 0.3-0.4 band -- mirroring a real
    action encoding (e.g. an absolute target end-effector pose in a different
    representation/frame than observation.state) that never reaches a high correlation
    even with no desync at all. One episode's state is genuinely scrambled (a real
    recorder defect) and must still be caught.
    """
    n = 240  # 8s at FPS
    t = np.arange(n) / FPS
    f1, f2, lag, alpha = 0.6, 1.3, 1, 0.48

    def episode(idx: int, rng: np.random.Generator | None) -> Episode:
        phase1, phase2 = 0.1 * idx, 0.37 + 0.13 * idx
        action = 50.0 * np.sin(2 * np.pi * f1 * t + phase1)
        if rng is not None:
            # a real recorder bug: state is unrelated to action, not a delayed/mixed copy
            state = rng.permutation(action)
        else:
            action_lagged = np.concatenate([np.repeat(action[:1], lag), action[:-lag]])
            indep = 50.0 * np.sin(2 * np.pi * f2 * t + phase2)
            state = alpha * action_lagged + (1 - alpha) * indep
        frame_index = np.arange(n)
        return Episode(
            index=idx,
            timestamps=frame_index / FPS,
            frame_index=frame_index,
            action=action[:, None],
            state=state[:, None],
        )

    episodes = [episode(i, None) for i in range(12)]
    episodes.append(episode(12, np.random.default_rng(0)))
    return Dataset(
        root=Path("."), fps=FPS, codebase_version="v3.0", robot_type=None, episodes=episodes
    )


def test_sync_weak_correlation_needs_a_material_drop_from_baseline():
    # Found by running `demoqc score` against real Hub datasets beyond the existing
    # validation set (`lerobot/LIBERO-Cosmos3`, `lerobot/LIBERO_10-Cosmos3`): 40-60% of
    # episodes carried a "weak correlation" `sync` flag, with peak correlations spread
    # smoothly across a tight 0.30-0.53 band and no separation between any "low" and
    # "high" cluster -- the dataset's own action encoding (an absolute target
    # end-effector pose in a different representation than observation.state) just
    # never reaches a high correlation, the same fixed-absolute-threshold instability
    # already fixed for stale_state/length/ldlj_z.
    results = {r.episode_index: r for r in evaluate(_moderate_corr_sync_dataset())}

    for i in range(12):
        assert results[i].metrics["sync_peak_corr"] < Config().sync_min_peak_corr
        assert "sync" not in _checks(results[i]), (i, results[i].flags)

    # a genuine recorder defect (state unrelated to action) must still be caught
    scrambled = results[12]
    assert scrambled.metrics["sync_peak_corr"] < 0.1
    assert "sync" in _checks(scrambled)


def test_sync_short_episode_does_not_produce_an_impossible_correlation():
    # Found by running `demoqc score` against a real Hub dataset beyond the existing
    # validation set (`IPEC-COMMUNITY/kuka_lerobot`, 10 fps, many ~1s episodes): several
    # short episodes' `sync` flag printed "r=-2.00" -- a Pearson correlation outside the
    # mathematically possible [-1, 1] range -- alongside a `RuntimeWarning: invalid value
    # encountered in divide`. Root cause: `check_sync`'s own `ep.length < 10` floor admits
    # an episode whose velocity series (one frame shorter, from `np.diff`) has only 9
    # frames -- too short for `_best_lag`'s inner `len(at) < 10` floor at *every* lag in
    # its search window, so the loop never updated `best_corr`'s initial sentinel and it
    # leaked out unmodified as the "measured" correlation. The same sentinel, pulled in by
    # `_sync_corr_baseline` from every episode this short, drags a dataset's median
    # baseline down toward that impossible value -- demonstrated below by a baseline that
    # would otherwise suppress the flag on a genuinely desynced episode.
    n_long, length_long = 8, 8 * FPS
    t = np.arange(length_long) / FPS

    def clean_long(idx: int) -> Episode:
        action = 50.0 * np.sin(2 * np.pi * 0.6 * t + 0.1 * idx)
        state = np.concatenate([action[:1], action[:-1]])  # tracks action with a 1-frame lag
        frame_index = np.arange(length_long)
        return Episode(
            index=idx,
            timestamps=frame_index / FPS,
            frame_index=frame_index,
            action=action[:, None],
            state=state[:, None],
        )

    def short(idx: int) -> Episode:
        length = 10  # right at check_sync's own `ep.length < 10` floor
        action = 50.0 * np.sin(2 * np.pi * 0.6 * np.arange(length) / FPS + 0.2 * idx)
        state = np.concatenate([action[:1], action[:-1]])  # same clean 1-frame lag, just short
        frame_index = np.arange(length)
        return Episode(
            index=idx,
            timestamps=frame_index / FPS,
            frame_index=frame_index,
            action=action[:, None],
            state=state[:, None],
        )

    def scrambled(idx: int) -> Episode:
        action = 50.0 * np.sin(2 * np.pi * 0.6 * t + 0.3)
        state = np.random.default_rng(0).permutation(action)  # a real recorder defect
        frame_index = np.arange(length_long)
        return Episode(
            index=idx,
            timestamps=frame_index / FPS,
            frame_index=frame_index,
            action=action[:, None],
            state=state[:, None],
        )

    episodes = [clean_long(i) for i in range(n_long)]
    episodes += [short(n_long + i) for i in range(6)]
    episodes.append(scrambled(n_long + 6))
    ds = Dataset(
        root=Path("."), fps=FPS, codebase_version="v3.0", robot_type=None, episodes=episodes
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        results = {r.episode_index: r for r in evaluate(ds)}

    # too short to measure a correlation at all -- no bogus metric, no flag off a sentinel
    for i in range(n_long, n_long + 6):
        assert "sync_peak_corr" not in results[i].metrics
        assert "sync" not in _checks(results[i])

    # the sentinel must not have dragged the dataset's own baseline toward -2.0
    for i in range(n_long):
        assert results[i].metrics["sync_peak_corr"] > 0.8

    # with a sane baseline, the genuinely desynced episode is still caught
    scrambled_result = results[n_long + 6]
    assert scrambled_result.metrics["sync_peak_corr"] < 0.3
    assert "sync" in _checks(scrambled_result)


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


def test_gripper_chatter_detected_with_grouped_feature_names(make_dataset):
    # Real ALOHA-format datasets store action/observation.state "names" as a dict grouping
    # all dims under one motor-group key, not a flat list (see
    # test_dataset.test_grouped_feature_names_are_flattened for the full root cause). Before
    # flattening it, DatasetContext.gripper_dims searched the dict's *keys* ("motors") for
    # "gripper", found nothing, and gripper_chatter silently never fired on any such
    # dataset -- exactly the real-world teleoperated grippers it exists to catch.
    specs = clean_specs()
    specs[3] = EpisodeSpec(gripper_chatter_at=(100, 115), seed=3)
    path = make_dataset(specs, grouped_names=True)
    results = evaluate(load_dataset(path))

    assert results[3].metrics["gripper_toggles_per_window"] >= 4
    assert "gripper_chatter" in _checks(results[3])
    assert all("gripper_chatter" not in _checks(r) for i, r in enumerate(results) if i != 3)


def test_gripper_chatter_detected_with_wrapped_feature_names(make_dataset):
    # Real dataset flex-pi/self_repair_gripper_dagger stores action/observation.state
    # "names" as the flat per-dim list wrapped in an extra outer list ([[...]], see
    # test_dataset.test_wrapped_feature_names_are_flattened for the full root cause).
    # Before flattening it, DatasetContext.gripper_dims iterated that single "dim" (whose
    # "name" is the whole 6-element list) and crashed on `name.lower()` -- so this dataset
    # couldn't be scored at all, not even silently miscalibrated.
    specs = clean_specs()
    specs[3] = EpisodeSpec(gripper_chatter_at=(100, 115), seed=3)
    path = make_dataset(specs, wrapped_names=True)
    results = evaluate(load_dataset(path))

    assert results[3].metrics["gripper_toggles_per_window"] >= 4
    assert "gripper_chatter" in _checks(results[3])
    assert all("gripper_chatter" not in _checks(r) for i, r in enumerate(results) if i != 3)


def test_sync_matches_all_dims_with_grouped_feature_names(make_dataset):
    # Same root cause as the gripper_chatter case above: with "names" as an unflattened
    # dict ({"motors": [...]}), _match_sync_dims read it via enumerate()/`in`, which only
    # sees the dict's one key ("motors") as if it were a single dim 0 -- pairing away all
    # other real dims instead of matching all 6 by name. Assert directly on the matched
    # dim indices (the actual root cause) rather than through the correlation heuristic,
    # since a single-dim match can coincidentally still look "synced enough" to pass the
    # heuristic-level test above.
    from demoqc.checks import Config, DatasetContext

    path = make_dataset(clean_specs(2), grouped_names=True)
    ctx = DatasetContext.build(load_dataset(path), Config())

    assert ctx.sync_dims == (list(range(6)), list(range(6)))


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
