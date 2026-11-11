import warnings
from pathlib import Path

import numpy as np
import pytest

from demoqc.checks import Config, _angular_action_dims, evaluate
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
