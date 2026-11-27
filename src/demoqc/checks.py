"""Per-episode quality checks.

Every check is computed relative to a :class:`DatasetContext` built from the whole
dataset, so thresholds adapt to the robot's units and the operator's typical speed
instead of being hard-coded per embodiment. Thresholds were calibrated on public
LeRobot datasets (``lerobot/svla_so101_pickplace``, ``lerobot/pusht``).
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import get_type_hints

import numpy as np

from demoqc.dataset import Dataset, Episode

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

ERROR, WARN, INFO = "error", "warn", "info"


def _coerce(expected: type, value: object, where: str) -> float | int:
    if expected is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{where}: expected a number, got {value!r}")
        return float(value)
    if expected is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{where}: expected an integer, got {value!r}")
        return value
    raise ValueError(f"{where}: unsupported config field type {expected!r}")  # pragma: no cover


@dataclass
class Config:
    gap_factor: float = 1.5  # a timestep > gap_factor / fps is a gap
    idle_fraction_of_p75: float = 0.1  # idle if speed < this * dataset 75th-pct speed
    idle_smoothing_s: float = 0.2
    trim_pad_s: float = 0.25  # context kept around trimmed idle
    min_trimmable_idle_s: float = 1.0
    max_idle_fraction: float = 0.4
    max_pause_s: float = 3.0
    spike_factor: float = 6.0  # step > spike_factor * dataset p99 step
    stale_state_max_fraction: float = 0.1  # flag if an episode's stale-state fraction exceeds
    # the dataset's own baseline (DatasetContext.stale_state_baseline) by more than this
    outlier_z: float = 3.5
    min_relative_length_diff: float = 0.25
    jerk_z: float = -3.0
    min_ldlj_diff: float = 1.0  # jerkier-than-peers also needs an LDLJ gap from the median
    # of at least this many nats (~2.7x the underlying jerk integral) -- guards against a
    # z-score inflated by a tiny MAD when most episodes share a near-identical motion
    sync_max_lag_s: float = 0.5  # search window for action/state cross-correlation
    sync_min_peak_corr: float = 0.4  # below this, the lag estimate is unreliable
    sync_corr_min_drop: float = 0.25  # once a dataset has a reliable baseline peak correlation
    # (DatasetContext.sync_corr_baseline), a "weak correlation" flag also needs the episode's
    # own peak correlation to fall at least this far below it -- see check_sync
    sync_lag_outlier_frames: float = 2.0  # flag if an episode's lag differs from the median by more
    dedupe_resample_len: int = 50  # points each trimmed action trajectory is resampled to
    dedupe_z: float = 3.0  # near-duplicate if nn distance < median nn distance - this * MAD
    gripper_active_frac: float = 0.5  # episode's gripper range must exceed this * dataset scale
    gripper_hysteresis_frac: float = 0.15  # fraction of the episode's own range, debounces noise
    gripper_chatter_window_s: float = 1.0  # sliding window toggles are counted over
    gripper_chatter_min_toggles: int = (
        4  # that many open/close transitions in the window is chatter
    )
    camera_dark_mean_max: float = 0.03  # pixel mean below this (0-1 scale) is near-black
    camera_dark_std_max: float = 0.02  # pixel std below this means almost no variation either
    video_decode_max_samples: int = 20  # frames sampled per camera when --decode-video is used
    video_dark_run_min_frac: float = 0.15  # fraction of sampled frames that must be dark to flag
    video_freeze_max_abs_diff: float = 0.03  # largest per-block thumbnail diff (0-1) still "frozen"
    angular_wrap_tol: float = 0.05  # float slack past +-pi an "angular" dim's values may still hit
    angular_wrap_reach: float = 0.3  # how close to +-pi a dim must get to count as wrapping
    angular_pair_cancel_frac: float = 0.25  # gimbal-lock: x/z step cancels if their combined
    # step is below this fraction of each axis's own step
    smoothness_dim_outlier_z: float = 4.0  # exclude a dim from spike/jump detection if
    # log(step_p99) is this many robust MADs from the dataset's other action dims' own
    # median -- a joint/axis the task barely ever moves, so step_p99 reflects only noise
    unused_side_scale_ratio: float = 100.0  # exclude a bimanual robot's whole "left"/
    # "right" side from spike/jump detection if every one of its dims' own robust scale is
    # at least this many times smaller than its named counterpart on the other side -- see
    # _exclude_unused_side_dims
    smoothness_ramp_gap_frames: int = 10  # bridge a same-direction gap of up to this many
    # frames between two big-step frames on the same dim before counting runs -- frame-to-
    # frame noise mid-ramp shouldn't split one continuous motion into many "jumps"
    quaternion_norm_tol: float = 0.05  # a candidate group of 4 action dims (by feature name)
    # is only trusted as a real unit quaternion if its dataset-wide sum-of-squares stays
    # within this of 1.0 -- rejects a false grouping from naming coincidence alone

    def with_overrides(self, values: dict[str, object], source: str = "--config") -> Config:
        """Return a copy with ``values`` applied, validating keys and types.

        Every key must name an existing field (a typo or a renamed field fails loudly
        instead of being silently ignored) and every value must match that field's type
        (all fields are ``float`` or ``int``).
        """
        hints = get_type_hints(type(self))
        known = {f.name for f in fields(self)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"{source}: unknown config key(s): {', '.join(unknown)}")
        changes = {k: _coerce(hints[k], v, f"{source}: {k}") for k, v in values.items()}
        return replace(self, **changes)

    @classmethod
    def from_file(cls, path: str | Path) -> Config:
        """Load threshold/penalty overrides from a TOML file (see README for keys)."""
        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(f"{path}: invalid TOML: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected a table of config keys")
        return cls().with_overrides(data, source=str(path))


@dataclass
class Flag:
    check: str
    severity: str
    message: str
    penalty: float = 0.0


@dataclass
class EpisodeResult:
    episode_index: int
    length: int
    duration_s: float
    task: str | None
    metrics: dict[str, float] = field(default_factory=dict)
    flags: list[Flag] = field(default_factory=list)
    trim: tuple[int, int] | None = None  # suggested [start, end) frame range

    @property
    def score(self) -> float:
        return float(max(0.0, 100.0 - sum(f.penalty for f in self.flags)))


def _match_sync_dims(
    action_names: list[str] | None,
    state_names: list[str] | None,
    action_width: int | None,
    state_width: int | None,
) -> tuple[list[int], list[int]] | None:
    """Pair up action and observation.state dims to cross-correlate for the sync check.

    Matches by feature name when both streams have names, so a reordered or extra dim
    (e.g. an added sensor) doesn't disable the check; falls back to comparing same-width
    streams positionally when names aren't available.
    """
    if action_names and state_names:
        state_pos = {name: i for i, name in enumerate(state_names) if name}
        pairs = [(i, state_pos[name]) for i, name in enumerate(action_names) if name in state_pos]
        if not pairs:
            return None  # names present but nothing in common; don't guess positionally
        a_idx, s_idx = zip(*pairs, strict=True)
        return list(a_idx), list(s_idx)
    if action_width is None or state_width is None or action_width != state_width:
        return None
    return list(range(action_width)), list(range(state_width))


@dataclass
class DatasetContext:
    fps: float
    action_scale: np.ndarray | None  # robust std per action dim
    step_p99: np.ndarray | None  # 99th pct |delta action| per dim, in scaled units
    idle_speed: float  # speed threshold, scaled units / s
    gripper_dims: list[int] = field(default_factory=list)  # action dims named per
    # _GRIPPER_NAME_KEYWORDS ("*gripper*", or a VR-teleop controller's "*trigger*"/"*squeeze*")
    sync_dims: tuple[list[int], list[int]] | None = None  # matched (action, state) dim indices
    angular_dims: np.ndarray | None = None  # action dims that wrap as a radian angle in [-pi, pi]
    angular_pairs: list[tuple[int, int]] = field(default_factory=list)  # x/z axis dim pairs of
    # the same Euler-angle rotation triple, which can trade off at a gimbal-lock singularity
    quaternion_groups: list[tuple[int, int, int, int]] = field(default_factory=list)  # groups of
    # 4 action dims (w, x, y, z) that together form a unit quaternion -- see
    # _quaternion_action_groups
    half_quaternion_groups: list[tuple[int, int, float]] = field(default_factory=list)  # (w, a,
    # scale) pairs of action dims that together form a *half* unit quaternion (a single-axis
    # rotation, e.g. a wheeled base's yaw) -- see _half_quaternion_action_groups
    stale_state_baseline: float = 0.0  # dataset's own median per-episode stale-state fraction;
    # some robots (e.g. a humanoid whose chassis/arm state publishes slower than the logging
    # fps) hold observation.state for a frame or two on every episode as a hardware artifact,
    # not a recorder bug -- see check_staleness.
    sync_corr_baseline: float | None = None  # dataset's own median per-episode sync peak
    # correlation (None below 5 qualifying episodes); some action encodings (e.g. an absolute
    # target end-effector pose in a different frame/representation than observation.state, as
    # opposed to a direct per-joint command) cap out at a moderate correlation for every
    # episode even with no desync at all -- see check_sync.
    # --decode-video support (check_video_freeze); video_root is only set for a local
    # dataset (v2.x or v3.0), since videos are never downloaded from the Hub (see
    # video.resolve_v2_video_path/resolve_v3_video_path).
    video_root: Path | None = None
    video_path_template: str | None = None
    chunks_size: int = 1000
    camera_names: list[str] = field(default_factory=list)

    @classmethod
    def build(cls, ds: Dataset, cfg: Config, decode_video: bool = False) -> DatasetContext:
        action_names = ds.action_names or []
        gripper_dims = [
            i
            for i, name in enumerate(action_names)
            if name
            and any(kw in name.lower() for kw in _GRIPPER_NAME_KEYWORDS)
            and not _is_pose_axis_group_member(name, action_names)
        ]
        a_width = next((e.action.shape[1] for e in ds.episodes if e.action is not None), None)
        s_width = next((e.state.shape[1] for e in ds.episodes if e.state is not None), None)
        sync_dims = _match_sync_dims(ds.action_names, ds.state_names, a_width, s_width)
        sync_corr_baseline = _sync_corr_baseline(ds.episodes, sync_dims, cfg.sync_max_lag_s, ds.fps)
        video_root = ds.root if decode_video and ds.repo_id is None else None
        video_ctx = dict(
            video_root=video_root,
            video_path_template=ds.video_path_template,
            chunks_size=ds.chunks_size,
            camera_names=ds.camera_names,
        )
        actions = [e.action for e in ds.episodes if e.action is not None and len(e.action) > 1]
        if not actions:
            return cls(
                ds.fps,
                None,
                None,
                0.0,
                gripper_dims,
                sync_dims,
                sync_corr_baseline=sync_corr_baseline,
                **video_ctx,
            )
        allA = np.concatenate(actions)
        # nan-aware: a single non-finite value (sensor glitch, corrupted write) in one
        # episode must not poison the scale for every other episode in the dataset. A
        # plain np.percentile/.std propagates NaN across the *whole* column it appears in,
        # which would silently disable idle/smoothness/gripper_chatter on that dimension
        # dataset-wide instead of only on the one corrupted episode (see check_integrity).
        q75, q25 = np.nanpercentile(allA, [75, 25], axis=0)
        scale = (q75 - q25) / 1.349
        std = np.nanstd(allA, axis=0)
        scale = np.where(scale > 1e-9, scale, std)
        scale = np.where(scale > 1e-9, scale, np.nan)  # constant dims are ignored
        angular_dims = _angular_action_dims(allA, cfg)
        angular_pairs = _angular_pair_axes(ds.action_names, angular_dims)
        quaternion_groups = _quaternion_action_groups(ds.action_names, allA, cfg)
        corrected_actions = [_correct_quaternion_signs(a, quaternion_groups) for a in actions]
        steps = np.concatenate(
            [np.abs(_angular_diff(a, angular_dims)) / scale for a in corrected_actions]
        )
        speeds = np.concatenate([_speed(a, scale, ds.fps) for a in actions])
        p75 = float(np.nanpercentile(speeds, 75)) if np.isfinite(speeds).any() else 0.0
        # a dim with scale already NaN'd out above (constant dataset-wide) makes its whole
        # `steps` column NaN too; only run nanpercentile on the live columns so that doesn't
        # raise numpy's "All-NaN slice encountered" on a dataset where some/all dims never
        # vary at all (e.g. a single-episode dataset with a frozen action stream).
        live = np.isfinite(scale)
        step_p99 = np.full(scale.shape, np.nan)
        if live.any():
            step_p99[live] = np.nanpercentile(steps[:, live], 99, axis=0)
        # dims that almost never change (e.g. a binary gripper) have no step scale;
        # leave them out of spike detection rather than flag every toggle
        step_p99 = np.where(step_p99 > 1e-9, step_p99, np.nan)
        # an unused side's dims are dropped from both the live step_p99 array *and* the
        # population the next two tests build their own median/MAD from (a separate copy,
        # scale_for_outliers -- ctx.action_scale itself must stay intact for every other
        # check that uses it), so a real but smaller defect elsewhere isn't masked by a
        # whole frozen limb skewing the dataset's own median -- see _unused_side_dims.
        unused_side_dims = _unused_side_dims(ds.action_names, scale, cfg.unused_side_scale_ratio)
        step_p99[unused_side_dims] = np.nan
        scale_for_outliers = scale.copy()
        scale_for_outliers[unused_side_dims] = np.nan
        step_p99 = _exclude_noise_dominated_dims(step_p99, cfg.smoothness_dim_outlier_z)
        step_p99 = _exclude_scale_outlier_dims(
            step_p99, scale_for_outliers, cfg.smoothness_dim_outlier_z
        )
        step_p99 = _exclude_gripper_dims(step_p99, gripper_dims)
        # half-quaternion pairs (e.g. base_orient_quat_z/w) would otherwise have just been
        # excluded above as a scale/noise outlier (their raw components' own IQR-based
        # scale really is tiny) -- override with the physically meaningful rotation-angle
        # step instead of leaving them excluded or using the misleading component-based one.
        half_quaternion_stats = _half_quaternion_dataset_stats(ds.action_names, allA, actions, cfg)
        for w, a, _hscale, p99 in half_quaternion_stats:
            step_p99[w] = p99
            step_p99[a] = p99
        half_quaternion_groups = [(w, a, hscale) for w, a, hscale, _p99 in half_quaternion_stats]
        idle_speed = cfg.idle_fraction_of_p75 * p75
        stale_fracs = [
            float(_stale_mask(e.action, e.state, scale, idle_speed, ds.fps).mean())
            for e in ds.episodes
            if e.action is not None and e.state is not None and len(e.action) >= 3
        ]
        stale_state_baseline = float(np.median(stale_fracs)) if stale_fracs else 0.0
        return cls(
            fps=ds.fps,
            action_scale=scale,
            step_p99=step_p99,
            idle_speed=idle_speed,
            gripper_dims=gripper_dims,
            sync_dims=sync_dims,
            angular_dims=angular_dims,
            angular_pairs=angular_pairs,
            quaternion_groups=quaternion_groups,
            half_quaternion_groups=half_quaternion_groups,
            stale_state_baseline=stale_state_baseline,
            sync_corr_baseline=sync_corr_baseline,
            **video_ctx,
        )


def _exclude_noise_dominated_dims(step_p99: np.ndarray, z_thresh: float) -> np.ndarray:
    """Exclude (set to NaN) action dims whose ``step_p99`` is a wild outlier among this
    dataset's *other* dims, in log space (dims span orders of magnitude multiplicatively).

    Found on a real dataset (``RoboCOIN/AI2_Alphabot_2_insert_water_pipe``): its torso
    joints sit still for virtually the entire task (dataset-wide range under 0.08 rad,
    vs. 0.17+ for every arm/eef/gripper dim), so their ``step_p99`` reflects nothing but
    sensor/controller noise around a fixed setpoint -- and that noise, scattered as many
    separate single-frame blips across almost every episode (not one sustained ramp, so
    the existing run-counting fix doesn't help), was flagged as a dataset-wide pile of
    "abrupt action jump(s)": 19/20 episodes carried a `smoothness` flag, several hitting
    the check's max penalty, on a dataset whose other checks looked clean. A genuinely
    driven dim's ``step_p99`` (99th-pct per-frame step, already normalised by the dim's own
    scale) clusters tightly with every other genuinely driven dim regardless of its
    physical units, since it reflects control/teleop dynamics (how many frames a typical
    motion takes), not the unit itself -- so comparing a dim's ``step_p99`` to the
    *dataset's own other dims'* median, via a robust z-score, needs no per-unit tuning and
    is a no-op on a dataset where every dim is actually used (confirmed on
    ``zekaiwang/trex_dataset``, a 58-action-dim dexterous-hand dataset whose dims
    naturally spread over a much wider range of ``step_p99`` than a simple arm: every
    genuinely moved finger dim there stays within z=+-3, while two already-known
    barely-used finger dims (range 0.02-0.03 rad, an order of magnitude below every other
    finger's 0.4-1.1 rad) sit at z=+5.4/+5.9 and are correctly excluded instead).
    Skipped entirely below 5 live dims, where a robust z-score isn't meaningful.
    """
    valid = np.isfinite(step_p99) & (step_p99 > 0)
    if valid.sum() < 5:
        return step_p99
    idx = np.flatnonzero(valid)
    logp = np.log(step_p99[idx])
    med = np.median(logp)
    mad = np.median(np.abs(logp - med)) * 1.4826
    if mad < 1e-9:
        return step_p99
    z = np.abs(logp - med) / mad
    out = step_p99.copy()
    out[idx[z > z_thresh]] = np.nan
    return out


def _exclude_scale_outlier_dims(
    step_p99: np.ndarray, scale: np.ndarray, z_thresh: float
) -> np.ndarray:
    """Exclude (set to NaN) action dims whose own range (robust ``scale``) is a wild
    low outlier among this dataset's *other* dims, in log space.

    A dim that moves a tiny amount all episode (e.g. a torso/neck joint the task barely
    uses) can still pass ``_exclude_noise_dominated_dims``: that check compares each dim's
    ``step_p99`` -- the 99th-pct per-frame step *already divided by the dim's own scale* --
    across dims, and dividing a dim's own sensor noise by its own (equally tiny) scale can
    land that ratio in the same range as a genuinely driven dim's ratio, hiding exactly the
    dim it's meant to catch. The dim's raw ``scale`` itself, compared directly to every
    other dim's ``scale`` (not run through that same per-dim normalisation), has no such
    blind spot.

    Found on a real dataset (``RoboCOIN/AI2_Alphabot_2_insert_water_pipe``): its
    ``neck_joint_1/2_rad`` and ``torso_joint_4_rad`` dims have a dataset-wide range under
    0.0002 rad -- scale outlier z=4.1-5.1 against the dataset's other dims' own
    median/MAD of log-scale -- yet their ``step_p99`` outlier z (against other dims'
    ``step_p99``) only reaches 1.1-2.1, well short of ``smoothness_dim_outlier_z``'s 4.0,
    so they stayed in spike/jump detection and kept contributing scattered single-frame
    noise blips to the dataset's `smoothness` flags. Every genuinely driven dim in the
    same dataset sits within scale outlier z=+-1.4 of the median (confirmed directly), and
    on ``zekaiwang/trex_dataset`` (58 action dims spanning a naturally wide range of motion
    scale) every dim stays within z=+-2.2, so this needs no per-unit tuning either. Only a
    *low*-side outlier (a barely-used dim) is excluded -- a dim with an unusually *large*
    scale (e.g. a literal gripper's wide open/closed range) is a legitimate signal, not
    this artifact, and gripper dims are already excluded by name via
    ``_exclude_gripper_dims``. Skipped entirely below 5 live dims, same as
    ``_exclude_noise_dominated_dims``.
    """
    valid = np.isfinite(scale) & (scale > 0)
    if valid.sum() < 5:
        return step_p99
    idx = np.flatnonzero(valid)
    logs = np.log(scale[idx])
    med = np.median(logs)
    mad = np.median(np.abs(logs - med)) * 1.4826
    if mad < 1e-9:
        return step_p99
    z = (logs - med) / mad
    out = step_p99.copy()
    out[idx[z < -z_thresh]] = np.nan
    return out


def _unused_side_dims(names: list[str] | None, scale: np.ndarray, min_ratio: float) -> list[int]:
    """Return the action dim indices of a bimanual robot's "left" or "right" side when
    that whole side is never actually commanded for the whole dataset.

    Found on real Hub datasets beyond the existing validation set
    (``RoboCOIN/AI2_Alphabot_2_package_fruit``, ``RoboCOIN/AI2_Alphabot_2_stamp_document``,
    both single-arm tasks on a dual-arm robot): 10+ of ~30 action dims -- the *entire*
    unused ``left_arm_joint_1..7_rad``/``left_eef_pos_x/y/z_m``/``left_eef_rot_euler_x/y/
    z_rad`` group -- sit at a near-frozen setpoint for every episode of the whole dataset
    (dataset-wide robust ``scale`` under 4e-5, vs. 0.02-0.3 for the matching ``right_*``
    dim actually driving the task). This cluster, left in the dim population either of the
    two existing per-dim exclusions (``_exclude_noise_dominated_dims``,
    ``_exclude_scale_outlier_dims``) compares against, is itself ~a third of the dataset's
    dims -- too large a minority for their dataset-wide median/MAD not to shift toward it,
    which both masks the frozen left arm itself (every one of its members' own outlier z
    caps at 3.6-3.8, just under ``smoothness_dim_outlier_z``'s 4.0) *and*, on exactly this
    dataset, masks the real, smaller defect underneath: the genuinely torso-noise-dominated
    ``torso_joint_1/2/3_rad`` dims (the same ``RoboCOIN/AI2_Alphabot_2_insert_water_pipe``
    pattern the scale-outlier test was built for) land at scale outlier z~=0 instead of the
    4+ they'd get without the frozen left arm pulling the dataset's median scale down toward
    its own near-zero range -- confirmed directly: removing the frozen left arm from the
    population first, then recomputing, correctly surfaces `torso_joint_1/2/3_rad` as the
    scale outliers they are. The result before this fix: 10-14/20 episodes of each dataset
    carried a `smoothness` "abrupt action jump" flag from scattered torso noise (confirmed
    by inspecting the flagged dim/frame directly), on datasets whose actually-driven
    ``right_*`` dims and other checks were clean.

    Unlike those two dataset-wide-median tests, this doesn't need the frozen cluster to be
    a small minority of *all* dims to find it in the first place: it only compares each
    "left"/"right"-named dim directly against its own named counterpart (matched by
    replacing "left" with "right" in the feature name, e.g. ``"left_arm_joint_3_rad"`` <->
    ``"right_arm_joint_3_rad"``), so a whole-limb cluster is no harder to catch than a
    single dim. A whole side is only returned when *every* matched pair agrees, both in
    which side is the tiny one and by a decisive margin (``min_ratio``, 100x): a real
    bimanual task, even a lopsided one, has no reason to move one arm literally two orders
    of magnitude less than the other on *every single joint and pose axis at once* --
    that's the signature of a control channel never being commanded at all, not real
    asymmetric motion. Needs at least 4 matched pairs, so a dataset with only one or two
    "left"/"right"-named dims (too little evidence either way) is left alone. A dataset
    where both sides are genuinely driven (every dataset in the existing validation set,
    plus every ALOHA/Agilex two-arm dataset from earlier fixes) has near-1 ratios on every
    pair and returns nothing.
    """
    if not names:
        return []
    lower = [n.lower() if n else "" for n in names]
    name_to_idx = {n: i for i, n in enumerate(lower) if n}
    pairs = []
    for i, name in enumerate(lower):
        if "left" not in name:
            continue
        j = name_to_idx.get(name.replace("left", "right"))
        if j is not None:
            pairs.append((i, j))
    if len(pairs) < 4:
        return []
    left_idx = np.array([p[0] for p in pairs])
    right_idx = np.array([p[1] for p in pairs])
    valid = np.isfinite(scale[left_idx]) & np.isfinite(scale[right_idx])
    valid &= (scale[left_idx] > 0) & (scale[right_idx] > 0)
    if valid.sum() < 4:
        return []
    ratio = np.log(scale[left_idx[valid]]) - np.log(scale[right_idx[valid]])
    log_min_ratio = np.log(min_ratio)
    if np.all(ratio < -log_min_ratio):
        unused_side = "left"
    elif np.all(ratio > log_min_ratio):
        unused_side = "right"
    else:
        return []
    return [i for i, name in enumerate(lower) if unused_side in name]


_GRIPPER_NAME_KEYWORDS = ("gripper", "trigger", "squeeze", "hand_joint")
# "trigger"/"squeeze" cover VR-controller teleop naming (e.g. a Quest-style controller's
# index-trigger/grip-squeeze axis driving a humanoid hand's open/close), the same
# near-binary actuator pattern as a literal "*gripper*" dim -- see _exclude_gripper_dims.
# "hand_joint" covers a dexterous multi-finger hand replacing a simple gripper (e.g. a
# Unitree G1's per-finger joints, "left_hand_joint_1_rad".."left_hand_joint_7_rad"): each
# finger joint sits at a steady open/closed setpoint almost the whole episode and snaps
# through its range in 1-2 frames during a deliberate grasp/release, exactly the gripper
# pattern -- "hand_joint" rather than bare "hand" so an eef/wrist pose dim incidentally
# named "*_hand_pos_*"/"*_hand_rot_*" (continuous, not near-binary) isn't swept in too.

_POSE_AXIS_SUFFIXES = frozenset({"x", "y", "z", "roll", "pitch", "yaw"})


def _is_pose_axis_group_member(name: str, all_names: list[str]) -> bool:
    """True if ``name`` is one axis of a >=3-dim position/orientation group sharing a
    dotted prefix (e.g. ``"left_ee_pose_gripper_base.x"``, one of that group's
    ``.x``/``.y``/``.z``/``.roll``/``.pitch``/``.yaw`` siblings).

    Found on a real Hub dataset beyond the existing validation set
    (``unitreerobotics/G1_Dex1_Hang_Pants``, found via ``list_datasets(filter="LeRobot",
    sort="downloads")``): its structured ``action.left_ee_pose_gripper_base``/
    ``action.right_ee_pose_gripper_torso`` sub-features (a continuous end-effector pose
    *expressed in the gripper's own base/torso frame*, not a gripper actuator) get flattened
    into per-dim names like ``"left_ee_pose_gripper_base.x"``/``".roll"`` by
    ``dataset._structured_subfeature_names``, which prefixes each axis with the
    sub-feature's own dotted key. ``gripper_dims``' bare ``"gripper" in name.lower()``
    search matches every one of these 24 dims purely because "gripper" happens to appear in
    that shared *reference-frame* qualifier, not because any of them is a gripper -- unlike
    a real gripper/trigger/squeeze/hand_joint actuator, whose own per-dim name (not just a
    shared group prefix) encodes the keyword (``"gripper_pos"``, ``"left_trigger"``,
    ``"left_hand_joint_1_rad"``). Confirmed on the real dataset: these dims are continuous,
    high-cardinality position/orientation signals (1100+ distinct values across ~1700
    frames of one episode, smoothly varying throughout), not a near-binary actuator -- and
    excluding them from spike/jump detection (as every other ``gripper_dims`` member is)
    silently disabled smoothness checking on all 24 of them with no visible symptom, exactly
    the "silently disabled, no error" failure mode this tool already treats as seriously as
    a false positive. A group is only recognised this way with 3+ of its dims actually
    sharing one of the six canonical pose-axis suffixes, so a coincidental single dim
    literally named e.g. ``"some_gripper.roll"`` (no x/y/z/pitch/yaw siblings) is left alone.
    """
    if "." not in name:
        return False
    prefix, suffix = name.rsplit(".", 1)
    if suffix.lower() not in _POSE_AXIS_SUFFIXES:
        return False
    sibling_prefix = prefix + "."
    siblings = {
        n.rsplit(".", 1)[1].lower() for n in all_names if n.startswith(sibling_prefix) and "." in n
    }
    return len(siblings & _POSE_AXIS_SUFFIXES) >= 3


def _exclude_gripper_dims(step_p99: np.ndarray, gripper_dims: list[int]) -> np.ndarray:
    """Exclude (set to NaN) gripper action dims from spike/jump detection.

    Found on a real Hub dataset beyond the existing validation set
    (``allenai/MolmoAct2-BimanualYAM-Dataset``): its gripper dims sit static almost the
    whole episode (holding open or closed) and snap through their full 0..1 range in just
    1-4 frames during a deliberate grasp or release -- normal, expected behaviour for a
    near-binary actuator driven to a setpoint as fast as possible, not jerkiness. Because
    these transitions are brief and rare, the dataset-wide 99th-percentile step for a
    gripper dim reflects the (near-zero) holding frames, not the transition itself, so
    *any* percentile-based "this step is an outlier" threshold ends up flagging every
    ordinary grasp as an "abrupt action jump": 13/20 episodes here, none of them a real
    defect, on a dataset whose arm joints (continuously driven, so their own `step_p99`
    already reflects normal fast motion) carried no smoothness flags at all.
    ``check_gripper_chatter`` already owns detecting genuinely bad gripper behaviour
    (rapid open/close toggling) via a hysteresis-debounced counter that doesn't depend on
    this percentile baseline, so this generic jump/spike detector adds no value for
    gripper dims and only miscalibrates on exactly the actuator type it's least suited to.

    The same pattern recurs on VR-controller-teleoperated humanoids under a different
    name: found on ``BitRobot/HIW-500-LeRobot`` (a Unitree G1 teleoperated via a VR
    controller), whose ``right_trigger``/``right_squeeze`` action dims (0..10 and 0..1
    ranges) snap between two steady setpoints exactly like a literal gripper dim, but
    were invisible to the "gripper" name match -- 19/20 episodes carried spurious "abrupt
    action jump(s)" flags, almost entirely on those two dims. See
    ``_GRIPPER_NAME_KEYWORDS``.

    Also recurs on a humanoid with a dexterous multi-finger hand instead of a simple
    gripper: found on ``RoboCOIN/Unitree_G1edu_u3_place_bread_1``, whose
    ``left/right_hand_joint_1..7_rad`` action dims (14 per-finger joints) each sit at a
    steady setpoint for most of an episode and snap through their own range in 1-2 frames
    during a grasp/release, same as a literal gripper dim -- 5/10 episodes carried spurious
    "abrupt action jump(s)" flags, every flagged frame on a ``*_hand_joint_*`` dim.
    """
    if not gripper_dims:
        return step_p99
    out = step_p99.copy()
    out[gripper_dims] = np.nan
    return out


def _angular_action_dims(allA: np.ndarray, cfg: Config) -> np.ndarray:
    """Which action dims are a radian angle wrapped to [-pi, pi] (e.g. ``eef_rot_euler_z``).

    Detected from the data, not the feature name: a dim counts as wrapping only if every
    value for it, across the whole dataset, stays within ``+-(pi + angular_wrap_tol)`` *and*
    it actually reaches within ``angular_wrap_reach`` of both +pi and -pi -- i.e. it uses the
    full wrap range, not just a joint whose physical limits happen to sit under pi. A
    position (metres) or a joint with a narrower range never satisfies both at once, so this
    only ever matches a genuine wrapped angle.
    """
    lo = np.nanmin(allA, axis=0)
    hi = np.nanmax(allA, axis=0)
    bounded = (hi <= np.pi + cfg.angular_wrap_tol) & (lo >= -np.pi - cfg.angular_wrap_tol)
    reach = cfg.angular_wrap_reach
    reaches_both_ends = (hi >= np.pi - reach) & (lo <= -np.pi + reach)
    return bounded & reaches_both_ends


def _angular_diff(action: np.ndarray, angular: np.ndarray | None) -> np.ndarray:
    """``np.diff(action, axis=0)``, with wrapping angle dims taken as their shortest step.

    A wrapped angle (values confined to [-pi, pi]) jumps by ~2*pi on an ordinary frame that
    crosses the wrap boundary during otherwise continuous rotation -- indistinguishable from
    real motion to a plain diff. Found on a real dataset
    (RoboCOIN/Agilex_Cobot_Magic_heat_burger): its ``eef_rot_euler_x/z_rad`` dims wrap during
    normal wrist motion, and the plain diff below used to flag that crossing as a
    single-frame "abrupt action jump" on nearly every episode. Taking the equivalent diff in
    (-pi, pi] instead removes exactly that artifact and nothing else -- it's a no-op for any
    dim whose steps never approach +-pi.
    """
    d = np.diff(action, axis=0)
    if angular is not None and angular.any():
        d = d.copy()
        d[:, angular] = _wrap_pi(d[:, angular])
    return d


def _wrap_pi(x: np.ndarray) -> np.ndarray:
    """Wrap radian value(s) to the shortest equivalent in ``(-pi, pi]``."""
    return (x + np.pi) % (2 * np.pi) - np.pi


_AXIS_SUFFIX = re.compile(r"^(.+)_([a-zA-Z])_rad$")


def _angular_pair_axes(
    action_names: list[str] | None, angular_dims: np.ndarray | None
) -> list[tuple[int, int]]:
    """Pairs of wrapped-angle dims that are the x/z axes of the same Euler-angle rotation
    triple (same feature-name prefix, e.g. ``eef_rot_euler_x_rad``/``eef_rot_euler_z_rad``).

    Found on a real dataset (``RoboCOIN/Agilex_Cobot_Magic_place_towel_flat``): when the
    middle (``_y_rad``, pitch) axis nears its own +-pi/2 gimbal-lock singularity, a tiny real
    rotation can swing the x and z axes' individual values by a large, precisely correlated
    amount even though the end effector barely moved -- each axis read in isolation looks
    like an abrupt jump. The exact combination that stays (near) constant through this --
    ``x - z`` or ``x + z``, depending on sign convention -- is checked directly in
    ``check_smoothness`` rather than relying on locating the pitch axis itself, so this needs
    no assumption about which Euler convention or axis order produced the data.
    """
    if action_names is None or angular_dims is None:
        return []
    groups: dict[str, dict[str, int]] = {}
    for i in np.nonzero(angular_dims)[0]:
        name = action_names[i] if i < len(action_names) else None
        m = _AXIS_SUFFIX.match(name) if name else None
        if m:
            groups.setdefault(m.group(1), {})[m.group(2).lower()] = int(i)
    return [(g["x"], g["z"]) for g in groups.values() if "x" in g and "z" in g]


def _suppress_gimbal_lock_pairs(
    action: np.ndarray, big: np.ndarray, pairs: list[tuple[int, int]], cancel_frac: float
) -> None:
    """Clear a spike/jump flag shared by a paired rotation-axis dim when it's a gimbal-lock
    representation artifact rather than real motion (see ``_angular_pair_axes``), by testing
    whether ``a - b`` or ``a + b`` barely moved while each axis's own step is large. Mutates
    ``big`` in place.
    """
    for a, b in pairs:
        flagged = big[:, a] | big[:, b]
        if not flagged.any():
            continue
        da = _wrap_pi(np.diff(action[:, a]))
        db = _wrap_pi(np.diff(action[:, b]))
        combo_minus = np.abs(_wrap_pi(np.diff(action[:, a] - action[:, b])))
        combo_plus = np.abs(_wrap_pi(np.diff(action[:, a] + action[:, b])))
        best_combo = np.minimum(combo_minus, combo_plus)
        dominant = np.maximum(np.abs(da), np.abs(db))
        cancels = flagged & (best_combo < cancel_frac * dominant)
        big[cancels, a] = False
        big[cancels, b] = False


_QUATERNION_SUFFIX = re.compile(r"^(.+?)[._-]?q(?:uat)?[._-]?([wxyz])$", re.IGNORECASE)
# matches both a bare "q"+letter suffix ("left_ee.qw") and a spelled-out "quat"+letter
# suffix ("base_orient_quat_w") -- the latter found on a real Hub dataset (see
# _half_quaternion_action_groups below).


def _quaternion_action_groups(
    action_names: list[str] | None, allA: np.ndarray, cfg: Config
) -> list[tuple[int, int, int, int]]:
    """Groups of 4 action dims that together form a unit quaternion (e.g. an end-effector
    orientation stored as ``left_ee.qw``/``left_ee.qx``/``left_ee.qy``/``left_ee.qz``).

    Candidate groups are found by feature name (four dims sharing a prefix, each ending in
    a ``q``+w/x/y/z suffix), then confirmed from the data itself: a real quaternion's
    components always sum-of-squares to 1 for every frame, so a false grouping from naming
    coincidence alone is rejected rather than trusted -- the same "data decides, name only
    narrows the search" approach as ``_angular_action_dims``.
    """
    if action_names is None:
        return []
    groups: dict[str, dict[str, int]] = {}
    for i, name in enumerate(action_names):
        m = _QUATERNION_SUFFIX.match(name) if name else None
        if m:
            groups.setdefault(m.group(1), {})[m.group(2).lower()] = i
    out = []
    for g in groups.values():
        if not all(c in g for c in "wxyz") or len(g) != 4:
            continue
        cols = (g["w"], g["x"], g["y"], g["z"])
        norm_sq = np.nansum(allA[:, cols] ** 2, axis=1)
        finite = np.isfinite(norm_sq)
        if finite.sum() < 10:
            continue
        if np.median(np.abs(norm_sq[finite] - 1.0)) > cfg.quaternion_norm_tol:
            continue
        out.append(cols)
    return out


def _half_quaternion_action_groups(
    action_names: list[str] | None, allA: np.ndarray, cfg: Config
) -> list[tuple[int, int]]:
    """Pairs of action dims (``qw`` plus exactly one of ``qx``/``qy``/``qz``) that together
    form a *half* unit quaternion: a pure single-axis (planar) rotation, where the other two
    quaternion components are always exactly 0 and never logged as a feature at all.

    Found on a real Hub dataset beyond the existing validation set
    (``RoboCOIN/Galbot_G1_load_floor_trash_3``, a wheeled-base humanoid): its base's yaw is
    stored as ``base_orient_quat_z``/``base_orient_quat_w`` -- no ``qx``/``qy`` feature in
    sight, confirmed directly that ``z**2 + w**2`` is 1.0 (to float precision) across every
    frame of every episode. ``w`` sits within 0.045 of 1.0 for virtually the whole dataset
    (the robot rarely rotates its base), so its own IQR-based ``scale`` is tiny; a genuine,
    modest, perfectly smooth base turn still moves ``w`` (and ``z``) by many multiples of
    that tiny scale, well past ``smoothness_dim_outlier_z``'s 4.0 exclusion threshold. Unlike
    the already-fixed noise-dominated/scale-outlier torso-joint case, this dim is not noise
    around a fixed setpoint that should simply be excluded -- it carries a real, meaningful
    signal (how much the base turned) that's just encoded non-linearly (``w = cos(theta/2)``)
    in a way raw per-component diffing can't see: near ``theta=0`` the component values barely
    move per radian turned, so per-component ``step_p99`` reflects almost no real motion at
    all. Candidate pairs are found by feature name (the same ``q``/``quat`` suffix convention
    as ``_quaternion_action_groups``, just with only 2 of the 4 letters present), then
    confirmed from the data the same "name narrows the search, data decides" way: a
    coincidental 2-dim naming match whose sum-of-squares isn't near 1 is rejected. See
    ``_half_quaternion_step`` for how this is turned into a real, well-scaled step instead.
    """
    if action_names is None:
        return []
    groups: dict[str, dict[str, int]] = {}
    for i, name in enumerate(action_names):
        m = _QUATERNION_SUFFIX.match(name) if name else None
        if m:
            groups.setdefault(m.group(1), {})[m.group(2).lower()] = i
    out = []
    for g in groups.values():
        axis_letters = [c for c in "xyz" if c in g]
        if "w" not in g or len(g) != 2 or len(axis_letters) != 1:
            continue
        w, a = g["w"], g[axis_letters[0]]
        norm_sq = np.nansum(allA[:, [w, a]] ** 2, axis=1)
        finite = np.isfinite(norm_sq)
        if finite.sum() < 10:
            continue
        if np.median(np.abs(norm_sq[finite] - 1.0)) > cfg.quaternion_norm_tol:
            continue
        out.append((w, a))
    return out


def _half_quaternion_angle(action: np.ndarray, w: int, a: int) -> np.ndarray:
    """The rotation half-angle (radians, ``theta/2``) a half-quaternion ``(w, a)`` pair
    encodes, recovered directly via ``atan2`` instead of the nonlinear raw components."""
    return np.arctan2(action[:, a], action[:, w])


def _half_quaternion_step(action: np.ndarray, w: int, a: int) -> np.ndarray:
    """Signed per-frame rotation step (radians) for a half-quaternion ``(w, a)`` pair: the
    shortest-path circular diff of its recovered half-angle, well-scaled and linear in the
    real rotation regardless of where ``w``/``a`` individually sit -- unlike diffing ``w``
    or ``a`` directly, which is most insensitive to real motion exactly where this
    representation spends almost all its time (``theta`` near 0, i.e. ``w`` near +-1).
    """
    return _wrap_pi(np.diff(_half_quaternion_angle(action, w, a)))


def _half_quaternion_dataset_stats(
    action_names: list[str] | None,
    allA: np.ndarray,
    actions: list[np.ndarray],
    cfg: Config,
) -> list[tuple[int, int, float, float]]:
    """For each detected half-quaternion pair (``_half_quaternion_action_groups``), the
    dataset-wide scale (robust IQR of the recovered rotation half-angle, see
    ``_half_quaternion_angle``) and 99th-percentile normalised step -- computed the exact
    same way ``DatasetContext.build`` computes every other action dim's ``scale``/
    ``step_p99``, just on the derived angle instead of the raw, nonlinear components.
    Returns ``(w, a, scale, step_p99)`` tuples; a pair whose angle never varies at all
    across the dataset (the base never rotates) is left out, same as any other constant
    dim.
    """
    out = []
    for w, a in _half_quaternion_action_groups(action_names, allA, cfg):
        half_theta = _half_quaternion_angle(allA, w, a)
        finite = np.isfinite(half_theta)
        if finite.sum() < 2:
            continue
        q75, q25 = np.nanpercentile(half_theta[finite], [75, 25])
        scale = (q75 - q25) / 1.349
        std = float(np.nanstd(half_theta[finite]))
        scale = scale if scale > 1e-9 else std
        if scale <= 1e-9:
            continue
        dtheta = np.concatenate(
            [_half_quaternion_step(act, w, a) for act in actions if len(act) > 1]
        )
        if dtheta.size == 0:
            continue
        p99 = float(np.nanpercentile(np.abs(dtheta) / scale, 99))
        if p99 <= 1e-9:
            continue
        out.append((w, a, scale, p99))
    return out


def _apply_half_quaternion_steps(
    step: np.ndarray, action: np.ndarray, groups: list[tuple[int, int, float]]
) -> np.ndarray:
    """Replace a half-quaternion pair's own component steps in ``step`` with the shared,
    physically meaningful rotation-angle step (see ``_half_quaternion_step``), scaled by
    the dataset-wide spread of that angle (``DatasetContext.build``) instead of either raw
    component's own (misleadingly tiny or nonlinear) scale.
    """
    if not groups:
        return step
    out = step.copy()
    for w, a, scale in groups:
        dtheta = _half_quaternion_step(action, w, a) / scale
        out[:, w] = dtheta
        out[:, a] = dtheta
    return out


def _correct_quaternion_signs(
    action: np.ndarray, groups: list[tuple[int, int, int, int]]
) -> np.ndarray:
    """Return a copy of ``action`` with each quaternion group's sign made continuous
    frame-to-frame.

    Found on a real Hub dataset beyond the existing validation set
    (``adityx23/icl-dataset``, a bimanual arm teleoperated via end-effector pose): its
    ``left/right_ee.qw/qx/qy/qz`` action dims flip sign outright between two consecutive
    frames (confirmed directly: the dot product of the two frames' quaternions is exactly
    -1.0), which plain per-dim diffing reads as all four dims jumping by up to their full
    range in a single frame -- but a unit quaternion ``q`` and its negation ``-q`` represent
    the *exact same* orientation (the standard double-cover property of the representation),
    so nothing about the robot's pose actually changed. For each group, the sign of every
    frame after the first is flipped (cumulatively) whenever its raw dot product with the
    previous frame is negative, which removes exactly this artifact and nothing else: two
    genuinely close orientations already have a positive dot product and are untouched.
    """
    if not groups:
        return action
    out = action.copy()
    for cols in groups:
        q = out[:, cols]
        if len(q) < 2:
            continue
        raw_dot = np.sum(q[:-1] * q[1:], axis=1)
        sign = np.where(raw_dot >= 0, 1.0, -1.0)
        sign = np.where(np.isfinite(raw_dot), sign, 1.0)  # a non-finite frame: don't flip
        s = np.concatenate([[1.0], np.cumprod(sign)])
        out[:, cols] = q * s[:, None]
    return out


def _bridge_sustained_motion_gaps(big: np.ndarray, step: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill a short, same-direction gap between two big-step frames on the same dim.

    Found on a real dataset (``RoboCOIN/Galbot_G1_load_floor_trash_3``, a wheeled-base
    humanoid): its base orientation is logged as a quaternion z/w pair, and ``w`` sits
    within 0.045 of 1.0 for virtually the entire dataset (the robot rarely rotates its
    base), so its own IQR-based ``scale`` -- and therefore ``step_p99`` -- is tiny. A
    genuine, continuous, smooth base rotation then produces a long run of frames whose
    step ratio exceeds ``spike_factor * step_p99``, exactly the already-handled "sustained
    ramp" case (see ``check_smoothness``) -- but ordinary frame-to-frame measurement noise
    occasionally drops one or two consecutive frames' ratio back under that threshold mid-
    turn, since the motion isn't perfectly uniform. ``_runs`` only merges *literally
    consecutive* big frames, so what is physically one continuous rotation was split into
    8-22 separate "jumps" per episode (several hitting the check's max penalty) instead of
    the one real event it is. Bridges a gap of up to ``max_gap`` frames between two big
    frames on the *same dim* only when the step's sign doesn't change across the gap, so
    two genuinely separate, oppositely-signed excursions close together are never merged
    (confirmed on the real dataset: every genuinely distinct rotation event there is
    hundreds of frames from the next, far past any reasonable gap tolerance).
    """
    if max_gap <= 0:
        return big
    out = big.copy()
    for d in range(big.shape[1]):
        idx = np.flatnonzero(big[:, d])
        if len(idx) < 2:
            continue
        gaps = np.diff(idx)
        bridge = (gaps > 1) & (gaps <= max_gap + 1)
        for start, end in zip(idx[:-1][bridge], idx[1:][bridge], strict=True):
            if np.sign(step[start, d]) == np.sign(step[end, d]):
                out[start + 1 : end, d] = True
    return out


def _speed(action: np.ndarray, scale: np.ndarray, fps: float) -> np.ndarray:
    """RMS speed across (scaled) action dims, one value per transition.

    Dims with ``scale`` set to NaN (constant across the whole dataset, excluded from
    spike/idle thresholds by ``DatasetContext.build``) are dropped per frame. A frame
    where *every* dim is excluded this way -- every action dim is constant dataset-wide,
    i.e. nothing in the dataset ever moves -- used to come out as ``nanmean`` of an
    all-NaN row: a ``RuntimeWarning`` and a ``NaN`` speed. That NaN then compared False to
    every idle threshold, so ``check_idle``'s absolute "no motion detected" fallback
    silently never fired on a dataset that's completely dead. "Every tracked dim held
    still" is a real zero speed, not an unknown one, so return 0 for that frame instead.
    """
    v = np.diff(action, axis=0) / scale * fps
    valid = np.isfinite(v)
    sq = np.where(valid, v * v, 0.0)
    counts = np.maximum(valid.sum(axis=1), 1)
    return np.sqrt(sq.sum(axis=1) / counts)


def _robust_z(x: float, population: np.ndarray) -> float:
    med = float(np.median(population))
    mad = float(np.median(np.abs(population - med))) * 1.4826
    if mad < 1e-9:
        return 0.0
    return (x - med) / mad


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of consecutive True runs."""
    if not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def check_timing(ep: Episode, ctx: DatasetContext, cfg: Config, out: EpisodeResult) -> None:
    if ep.length < 2:
        return
    expected = 1.0 / ctx.fps
    dt = np.diff(ep.timestamps)
    non_monotonic = int((dt <= 0).sum())
    gaps = dt > cfg.gap_factor * expected
    dropped = int(np.round(dt[gaps] / expected).sum() - gaps.sum()) if gaps.any() else 0
    fi_steps = np.diff(ep.frame_index)
    fi_missing = int(np.clip(fi_steps - 1, 0, None).sum())
    out.metrics.update(
        dropped_frames=dropped,
        dropped_fraction=dropped / (ep.length + dropped),
        max_gap_s=float(dt.max()),
        non_monotonic_steps=non_monotonic,
        frame_index_missing=fi_missing,
    )
    if non_monotonic:
        out.flags.append(
            Flag("timing", ERROR, f"{non_monotonic} non-increasing timestamp step(s)", 30)
        )
    if dropped:
        frac = out.metrics["dropped_fraction"]
        out.flags.append(
            Flag(
                "timing",
                WARN if frac < 0.05 else ERROR,
                f"~{dropped} dropped frame(s) ({frac:.1%}), largest gap {dt.max():.3f}s",
                min(40.0, 300 * frac + 5),
            )
        )
    if fi_missing and not dropped:
        out.flags.append(Flag("timing", WARN, f"{fi_missing} missing frame_index value(s)", 10))


def check_metadata(ep: Episode, ctx: DatasetContext, cfg: Config, out: EpisodeResult) -> None:
    if ep.meta_length is not None and ep.meta_length != ep.length:
        out.flags.append(
            Flag(
                "metadata",
                ERROR,
                f"metadata says {ep.meta_length} frames, data has {ep.length}",
                20,
            )
        )


def check_integrity(ep: Episode, ctx: DatasetContext, cfg: Config, out: EpisodeResult) -> None:
    """Non-finite (NaN/Inf) value(s) in ``action`` or ``observation.state``: a corrupted
    write or sensor glitch. Every other check's thresholds come from robust statistics
    computed across the whole dataset (``DatasetContext.build``); without this check, a
    single bad frame in one episode is invisible on its own (``nanmean``/``nanpercentile``
    elsewhere just quietly drop it), yet the episode itself is unusable for training as-is."""
    bad = []
    for name, arr in (("action", ep.action), ("observation.state", ep.state)):
        if arr is None or arr.size == 0:
            continue
        n_bad = int((~np.isfinite(arr)).any(axis=1).sum())
        if n_bad:
            bad.append(f"{name} ({n_bad} frame(s))")
    if bad:
        out.flags.append(
            Flag("integrity", ERROR, "non-finite (NaN/Inf) value(s) in " + ", ".join(bad), 60.0)
        )


def check_video_sync(ep: Episode, ctx: DatasetContext, cfg: Config, out: EpisodeResult) -> None:
    tolerance = cfg.gap_factor / ctx.fps
    data_duration = ep.length / ctx.fps
    bad = []
    for cam, (start, end) in ep.video_spans.items():
        if abs((end - start) - data_duration) > tolerance:
            bad.append(f"{cam} {end - start:.2f}s")
    if bad:
        out.flags.append(
            Flag(
                "video_sync",
                ERROR,
                f"video span != data duration {data_duration:.2f}s: " + ", ".join(bad),
                20,
            )
        )


def check_camera(ep: Episode, ctx: DatasetContext, cfg: Config, out: EpisodeResult) -> None:
    """A camera that's near-black for the whole episode: a disconnected camera, a lens cap
    left on, or video that failed to record. Uses the per-episode pixel mean/std LeRobot
    already writes to its own metadata (``meta/episodes`` on v3.0, ``episodes_stats.jsonl``
    on v2.x); no video file is ever opened."""
    if not ep.camera_stats:
        return
    dead = [
        f"{cam} (mean={mean:.3f}, std={std:.3f})"
        for cam, (mean, std) in sorted(ep.camera_stats.items())
        if mean <= cfg.camera_dark_mean_max and std <= cfg.camera_dark_std_max
    ]
    if dead:
        out.metrics["camera_dead_count"] = len(dead)
        out.flags.append(
            Flag(
                "camera",
                ERROR,
                "near-black for the whole episode (disconnected camera or lens cap?): "
                + "; ".join(dead),
                min(50.0, 25.0 * len(dead)),
            )
        )


def check_video_freeze(ep: Episode, ctx: DatasetContext, cfg: Config, out: EpisodeResult) -> None:
    """A camera that goes black, or a video file that fails to decode, partway through an
    episode: real, training-blocking defects that the metadata-only ``camera`` check can't
    see because it only has one mean/std for the whole episode. Opt-in (``--decode-video``):
    needs the optional ``av`` dependency and a local, already-downloaded dataset (v2.x or
    v3.0; see ``DatasetContext.video_root``), so ``ctx.video_root`` is ``None`` and this is
    a no-op for every other run, including the default one."""
    if ctx.video_root is None:
        return
    from demoqc import video

    for camera in ctx.camera_names:
        mean, _std = ep.camera_stats.get(camera, (1.0, 1.0))
        if mean <= cfg.camera_dark_mean_max:
            continue  # whole-episode dead camera; the cheaper `camera` check already flags it
        time_range = None
        if camera in ep.video_chunk_file:
            chunk_index, file_index = ep.video_chunk_file[camera]
            path = video.resolve_v3_video_path(
                ctx.video_root, ctx.video_path_template, chunk_index, file_index, camera
            )
            time_range = ep.video_spans.get(camera)
        else:
            path = video.resolve_v2_video_path(
                ctx.video_root, ctx.video_path_template, ctx.chunks_size, ep.index, camera
            )
        if not path.exists():
            continue
        samples = video.decode_samples(path, cfg.video_decode_max_samples, time_range=time_range)
        if not samples:
            out.flags.append(
                Flag("video_freeze", ERROR, f"{camera}: video file failed to decode", 40.0)
            )
            continue
        run = video.longest_dark_run(
            samples, cfg.camera_dark_mean_max, cfg.camera_dark_std_max, cfg.video_dark_run_min_frac
        )
        if run is not None:
            start_t, end_t = samples[run[0]].t, samples[run[1] - 1].t
            out.flags.append(
                Flag(
                    "video_freeze",
                    WARN,
                    f"{camera}: goes near-black from ~{start_t:.1f}s to ~{end_t:.1f}s "
                    "mid-episode (disconnected or occluded camera?)",
                    20.0,
                )
            )
            continue
        # A frozen-but-not-dark feed (stalled camera/recorder repeating the last frame) looks
        # just like a legitimately static scene -- e.g. the lead/trail idle time every episode
        # has -- unless the robot was actually moving while it happened. Only flag that case;
        # a frozen clip during real idle time isn't distinguishable from a fine, boring one.
        frozen = video.longest_frozen_run(
            samples, cfg.video_freeze_max_abs_diff, cfg.video_dark_run_min_frac
        )
        if frozen is not None:
            start_t, end_t = samples[frozen[0]].t, samples[frozen[1] - 1].t
            base = time_range[0] if time_range is not None else 0.0
            if _episode_moved(ep, ctx, start_t - base, end_t - base):
                out.flags.append(
                    Flag(
                        "video_freeze",
                        ERROR,
                        f"{camera}: frozen (no change) from ~{start_t:.1f}s to ~{end_t:.1f}s "
                        "while the robot kept moving (dropped frames or a stalled feed?)",
                        35.0,
                    )
                )


def _episode_moved(ep: Episode, ctx: DatasetContext, start_s: float, end_s: float) -> bool:
    """Whether the commanded action shows real motion for the *entire* ``[start_s, end_s)``
    window of ``ep`` (episode-relative seconds) -- the same "clearly moving" bar
    ``check_staleness`` uses, so a frozen camera during genuine idle time (common and
    harmless) isn't confused with one that stalled while the robot kept going (a real,
    missed-frame defect). Requires the *minimum* speed in the window, not just its max:
    a frozen-frame window is coarse (only a handful of sampled frames, seconds apart), and
    real validation found one spanning an episode's idle-to-moving transition -- almost
    entirely idle, but with the tail end just over the "moving" bar -- flagged by a max-based
    check even though the camera wasn't frozen while the robot was actually moving.
    """
    if ep.action is None or ctx.action_scale is None or ep.length < 3:
        return False
    speed = _speed(ep.action, ctx.action_scale, ctx.fps)
    lo = max(0, int(start_s * ctx.fps))
    hi = min(len(speed), int(end_s * ctx.fps) + 1)
    if hi <= lo:
        return False
    return bool(np.min(speed[lo:hi]) > max(ctx.idle_speed, 1e-9) * 5)


def check_idle(ep: Episode, ctx: DatasetContext, cfg: Config, out: EpisodeResult) -> None:
    if ep.action is None or ctx.action_scale is None:
        return
    if ep.length < 3:
        # Too short to compute the smoothed idle/lead/trail profile below (needs at least
        # a couple of real transitions), but a 1- or 2-frame episode is the most degenerate
        # "no motion" case there is -- an aborted recording that barely started -- and the
        # absolute no-motion fallback further down still applies to it. Returning here
        # unconditionally (the old behaviour) silently skipped it instead: a real so100_stacking
        # episode with a single frame scored 90.0 with no `idle` flag at all, while a 14-frame
        # episode with the exact same bit-identical (zero-motion) action correctly scored 20.0
        # with an `idle: no motion detected` error -- the shortest, most broken recordings were
        # the only ones this check stayed silent on.
        speed = _speed(ep.action, ctx.action_scale, ctx.fps)
        if len(speed) == 0 or bool(np.all(speed < max(ctx.idle_speed, 1e-9))):
            out.metrics.update(idle_fraction=1.0, lead_idle_s=ep.length / ctx.fps, trail_idle_s=0.0)
            out.flags.append(Flag("idle", ERROR, "no motion detected in the whole episode", 60))
        return
    speed = _speed(ep.action, ctx.action_scale, ctx.fps)
    win = max(1, int(round(cfg.idle_smoothing_s * ctx.fps)))
    if win > 1 and len(speed) >= win:
        speed = np.convolve(speed, np.ones(win) / win, mode="same")
    # floored like check_staleness/_episode_moved: idle_speed is 0.1 * the dataset's own
    # p75 speed, which is exactly 0 when the dataset has (almost) no real motion anywhere
    # -- "speed < 0" can then never be true, so a completely dead episode would otherwise
    # never trip the "no motion detected" fallback below.
    idle = speed < max(ctx.idle_speed, 1e-9)
    fps = ctx.fps
    moving = np.flatnonzero(~idle)
    if len(moving) == 0:
        out.metrics.update(idle_fraction=1.0, lead_idle_s=ep.length / fps, trail_idle_s=0.0)
        out.flags.append(Flag("idle", ERROR, "no motion detected in the whole episode", 60))
        return

    lead, trail = int(moving[0]), int(len(idle) - 1 - moving[-1])
    interior = idle[moving[0] : moving[-1] + 1]
    longest_pause = max((e - s for s, e in _runs(interior)), default=0)
    out.metrics.update(
        idle_fraction=float(idle.mean()),
        lead_idle_s=lead / fps,
        trail_idle_s=trail / fps,
        longest_pause_s=longest_pause / fps,
    )

    trimmable = (lead + trail) / fps
    if trimmable >= cfg.min_trimmable_idle_s:
        pad = int(round(cfg.trim_pad_s * fps))
        start = max(0, lead - pad)
        end = min(ep.length, ep.length - trail + pad)
        out.trim = (start, end)
        out.flags.append(
            Flag(
                "idle",
                INFO,
                f"{trimmable:.1f}s idle at start/end (lead {lead / fps:.1f}s, "
                f"trail {trail / fps:.1f}s); trim to frames [{start}, {end})",
                min(10.0, 2 * trimmable),
            )
        )
    if out.metrics["idle_fraction"] > cfg.max_idle_fraction:
        frac = out.metrics["idle_fraction"]
        out.flags.append(
            Flag(
                "idle", WARN, f"idle {frac:.0%} of the episode", (frac - cfg.max_idle_fraction) * 50
            )
        )
    if longest_pause / fps > cfg.max_pause_s:
        out.flags.append(
            Flag("idle", WARN, f"operator paused {longest_pause / fps:.1f}s mid-episode", 5)
        )


def check_smoothness(ep: Episode, ctx: DatasetContext, cfg: Config, out: EpisodeResult) -> None:
    if ep.action is None or ctx.action_scale is None or ep.length < 3:
        return
    corrected_action = _correct_quaternion_signs(ep.action, ctx.quaternion_groups)
    step = _angular_diff(corrected_action, ctx.angular_dims) / ctx.action_scale
    if ctx.half_quaternion_groups:
        step = _apply_half_quaternion_steps(step, corrected_action, ctx.half_quaternion_groups)
    limit = cfg.spike_factor * ctx.step_p99
    big = np.abs(step) > limit
    if ctx.angular_pairs:
        _suppress_gimbal_lock_pairs(ep.action, big, ctx.angular_pairs, cfg.angular_pair_cancel_frac)
    any_big = big.any(axis=1)
    # a glitch is a large step immediately reversed by a large step on the same dim. This
    # must run on the *un-bridged* big frames: bridging below fills in frames whose own
    # step is small (that's the point), and running reversal detection on those too would
    # read ordinary small-amplitude noise at a bridged frame as a brand-new spike that
    # didn't exist before bridging.
    reversal = big[:-1] & big[1:] & (np.sign(step[:-1]) != np.sign(step[1:]))
    frame_reversal = reversal.any(axis=1)
    spikes = int(frame_reversal.sum())
    # each spike's up/down pair of frames is consumed by that spike, not also counted as
    # a jump. Of what's left, a *sustained run* of consecutive big frames in the same
    # direction (e.g. a multi-fingered hand closing quickly: many frames in a row with a
    # step far above this dim's usual near-zero idle step, but each one continuing smoothly
    # from the last) is one continuous fast motion, not one "abrupt jump" per frame it
    # spans -- found on a real dataset (zekaiwang/trex_dataset) whose dexterous-hand finger
    # targets sit still almost the whole episode, so step_p99 for those dims is tiny, and a
    # single smooth, monotonic grasp closing over ~20 frames was reported as "20 abrupt
    # action jumps" even though none of those steps is actually discontinuous. A gap of a
    # frame or two below the big-step threshold, in the middle of an otherwise continuous,
    # same-direction run (ordinary measurement noise, not a pause), is bridged the same way
    # before counting runs (see ``_bridge_sustained_motion_gaps``).
    bridged = _bridge_sustained_motion_gaps(big, step, cfg.smoothness_ramp_gap_frames)
    any_bridged = bridged.any(axis=1)
    consumed = np.zeros_like(any_big)
    consumed[:-1] |= frame_reversal
    consumed[1:] |= frame_reversal
    jumps = len(_runs(any_bridged & ~consumed))
    out.metrics.update(spikes=spikes, jumps=jumps)
    if spikes:
        out.flags.append(
            Flag(
                "smoothness",
                WARN,
                f"{spikes} single-frame action spike(s) (sensor glitch?)",
                min(25.0, 8.0 * spikes),
            )
        )
    if jumps > 0:
        out.flags.append(
            Flag("smoothness", WARN, f"{jumps} abrupt action jump(s)", min(20.0, 5.0 * jumps))
        )

    start, end = out.trim or (0, ep.length)
    seg = corrected_action[start:end] / ctx.action_scale
    seg = seg[:, np.isfinite(ctx.action_scale)]
    if len(seg) >= 10:
        out.metrics["ldlj"] = _ldlj(seg, ctx.fps)


def _ldlj(pos: np.ndarray, fps: float) -> float:
    """Log dimensionless jerk of a trajectory; higher (closer to 0) is smoother."""
    dt = 1.0 / fps
    vel = np.diff(pos, axis=0) / dt
    jerk = np.diff(pos, n=3, axis=0) / dt**3
    v_peak = float(np.linalg.norm(vel, axis=1).max())
    if v_peak < 1e-9:
        return float("nan")
    duration = len(pos) * dt
    integral = float((np.linalg.norm(jerk, axis=1) ** 2).sum() * dt)
    return float(-np.log(duration**3 / v_peak**2 * integral + 1e-12))


def _stale_mask(
    action: np.ndarray, state: np.ndarray, scale: np.ndarray, idle_speed: float, fps: float
) -> np.ndarray:
    """Per-transition mask: ``observation.state`` repeated bit-for-bit while ``action`` moves."""
    repeated = np.all(np.diff(state, axis=0) == 0, axis=1)
    moving = _speed(action, scale, fps) > max(idle_speed, 1e-9) * 5
    return repeated & moving


def check_staleness(ep: Episode, ctx: DatasetContext, cfg: Config, out: EpisodeResult) -> None:
    """Observation repeated bit-for-bit while the commanded action moves: a stale read.

    Flags an episode only if its stale fraction materially *exceeds the dataset's own
    baseline* (``ctx.stale_state_baseline``), not a fixed absolute fraction. Found on a real
    Hub dataset (``RoboCOIN/Galbot_G1_bag_table_trash_1``, a wheeled-base humanoid): every one
    of its 15 episodes had `observation.state` bit-identical to the previous frame on a tight,
    consistent ~22% of moving frames (confirmed across every one of the 38 action/state dims,
    correlated together, not just the chassis/base dims) -- the state message is evidently
    published at a lower rate than the 30 fps logging rate and repeated to fill in-between
    frames, a hardware/recorder characteristic of this platform, not a localized sensor glitch.
    The old fixed `stale_state_max_fraction` (0.1) flagged every episode (mean score 77.3 on an
    otherwise-clean dataset) for exactly the same reason the `length`/`ldlj_z` checks already
    guard against a tiny-spread false positive, just via an absolute threshold instead of a
    MAD this time. A dataset with no such baseline (``ctx.stale_state_baseline`` near 0, true
    of every dataset in the existing validation set) makes this a no-op: ``excess`` reduces to
    the old ``frac`` and the comparison is unchanged.
    """
    if ep.state is None or ep.action is None or ctx.action_scale is None or ep.length < 3:
        return
    stale = _stale_mask(ep.action, ep.state, ctx.action_scale, ctx.idle_speed, ctx.fps)
    frac = float(stale.mean())
    out.metrics["stale_state_fraction"] = frac
    excess = frac - ctx.stale_state_baseline
    if excess > cfg.stale_state_max_fraction:
        longest = max((e - s for s, e in _runs(stale)), default=0)
        baseline_note = (
            f"; dataset baseline {ctx.stale_state_baseline:.0%}"
            if ctx.stale_state_baseline > 1e-9
            else ""
        )
        out.flags.append(
            Flag(
                "stale_state",
                WARN,
                f"observation.state frozen on {frac:.0%} of moving frames "
                f"(longest run {longest} frames){baseline_note}",
                min(30.0, 100 * (excess - cfg.stale_state_max_fraction) + 10),
            )
        )


def _best_lag(a: np.ndarray, s: np.ndarray, max_lag: int) -> tuple[int, float]:
    """Lag (in frames) and peak Pearson correlation that best aligns ``s[t + lag]`` with
    ``a[t]``, averaged over dims with enough variance in both signals. ``a`` and ``s`` are
    per-dim velocity series of equal length; a positive lag means ``s`` trails ``a``.

    Returns ``nan`` for the correlation when no lag in ``[-max_lag, max_lag]`` had enough
    overlapping samples to measure one at all (a very short episode, close to this
    function's own ``len(at) < 10`` floor, can fail that floor at *every* lag). Found on a
    real Hub dataset (``IPEC-COMMUNITY/kuka_lerobot``, 10 fps, many ~1s episodes): episodes
    right at `check_sync`'s own ``ep.length < 10`` minimum have a velocity series of only 9
    frames, shorter than 10 at every lag, so the loop below never updated the initial
    ``best_corr`` sentinel and it leaked out as the "measured" correlation -- an impossible
    value outside [-1, 1] (confirmed directly: `demoqc score` printed `r=-2.00` in a `sync`
    flag message). The same sentinel also polluted `_sync_corr_baseline`'s per-episode
    correlation list, dragging a dataset's median baseline down to near 0 and silently
    making the "weak correlation" check far too lenient dataset-wide once several episodes
    this short exist.
    """
    n = len(a)
    valid_a = a.std(axis=0) > 1e-9
    if not valid_a.any():
        return 0, 0.0
    best_lag, best_corr = 0, float("-inf")
    for lag in range(-max_lag, max_lag + 1):
        at, st = (a[: n - lag], s[lag:]) if lag >= 0 else (a[-lag:], s[: n + lag])
        if len(at) < 10:
            continue
        at, st = at[:, valid_a], st[:, valid_a]
        ok = st.std(axis=0) > 1e-9
        if not ok.any():
            continue
        at, st = at[:, ok] - at[:, ok].mean(axis=0), st[:, ok] - st[:, ok].mean(axis=0)
        num = (at * st).sum(axis=0)
        den = np.sqrt((at**2).sum(axis=0) * (st**2).sum(axis=0))
        avg_corr = float(np.mean(np.divide(num, den, out=np.zeros_like(num), where=den > 1e-9)))
        if avg_corr > best_corr:
            best_lag, best_corr = lag, avg_corr
    if best_corr == float("-inf"):
        return 0, float("nan")
    return best_lag, best_corr


def _sync_corr_baseline(
    episodes: list[Episode],
    sync_dims: tuple[list[int], list[int]] | None,
    max_lag_s: float,
    fps: float,
) -> float | None:
    """Dataset's own median per-episode sync peak correlation, or ``None`` below 5
    qualifying episodes (the same gate every other dataset-relative check uses).

    Found on real Hub datasets (``lerobot/LIBERO-Cosmos3``, ``lerobot/LIBERO_10-Cosmos3``):
    their ``action`` feature is an absolute target end-effector pose (position + a 6D
    rotation) in a different representation/frame than ``observation.state`` (position +
    axis-angle), rather than a direct per-joint command -- so even with no desync at all,
    the two streams' matched dims (position and a binary gripper) only ever reach a
    moderate correlation, confirmed directly: every one of 20 sampled episodes landed in a
    tight 0.30-0.53 band, with no separation between any "low" and "high" cluster, yet
    40-60% of episodes (whichever happened to fall under the fixed 0.4 cutoff) carried a
    "weak correlation" flag -- the exact fixed-absolute-threshold instability already fixed
    for ``stale_state``/``length``/``ldlj_z``, just not yet extended to this check.
    """
    if sync_dims is None:
        return None
    a_idx, s_idx = sync_dims
    max_lag = max(1, int(round(max_lag_s * fps)))
    corrs = []
    for ep in episodes:
        if ep.action is None or ep.state is None or ep.length < 10:
            continue
        if max(a_idx) >= ep.action.shape[1] or max(s_idx) >= ep.state.shape[1]:
            continue
        _, corr = _best_lag(
            np.diff(ep.action[:, a_idx], axis=0), np.diff(ep.state[:, s_idx], axis=0), max_lag
        )
        if not np.isnan(corr):
            corrs.append(corr)
    return float(np.median(corrs)) if len(corrs) >= 5 else None


def check_sync(ep: Episode, ctx: DatasetContext, cfg: Config, out: EpisodeResult) -> None:
    """Estimate the lag between commanded ``action`` and observed ``observation.state``.

    A USB latency spike or a recorder bug desyncs the two streams for part of a recording
    session; that episode's best-fit lag will differ from its peers' even though each
    stream looks fine on its own.

    The "weak correlation" flag also requires a material drop from the dataset's own
    baseline correlation (``ctx.sync_corr_baseline``) once there's a reliable one -- some
    action encodings cap out at a moderate correlation for every episode with no desync at
    all; see ``_sync_corr_baseline``. A dataset with no such baseline (too few episodes, or
    unmatched/positionally-matched dims) falls back to the plain fixed cutoff, unchanged.
    """
    if ep.action is None or ep.state is None or ep.length < 10 or ctx.sync_dims is None:
        return
    a_idx, s_idx = ctx.sync_dims
    if max(a_idx) >= ep.action.shape[1] or max(s_idx) >= ep.state.shape[1]:
        return  # this episode's streams are narrower than the dataset's matched dims
    action, state = ep.action[:, a_idx], ep.state[:, s_idx]
    max_lag = max(1, int(round(cfg.sync_max_lag_s * ctx.fps)))
    lag, corr = _best_lag(np.diff(action, axis=0), np.diff(state, axis=0), max_lag)
    if np.isnan(corr):
        return  # too short, at every lag, to measure a correlation at all
    out.metrics.update(sync_lag_frames=float(lag), sync_lag_s=lag / ctx.fps, sync_peak_corr=corr)
    baseline = ctx.sync_corr_baseline
    if corr < cfg.sync_min_peak_corr and (
        baseline is None or baseline - corr > cfg.sync_corr_min_drop
    ):
        baseline_note = f" (dataset baseline r={baseline:.2f})" if baseline is not None else ""
        out.flags.append(
            Flag(
                "sync",
                WARN,
                f"action/observation.state correlation is weak (r={corr:.2f}); "
                f"lag estimate unreliable{baseline_note}",
                10,
            )
        )


def _hysteresis_state(pos: np.ndarray, mid: float, band: float) -> np.ndarray:
    """Binary open/closed state with hysteresis: only crosses when past ``mid +/- band``,
    so noise sitting near the midpoint doesn't register as a toggle."""
    state = np.empty(len(pos), dtype=bool)
    cur = bool(pos[0] > mid)
    for i, v in enumerate(pos):
        if v > mid + band:
            cur = True
        elif v < mid - band:
            cur = False
        state[i] = cur
    return state


def _max_in_window(times: np.ndarray, window_s: float) -> int:
    """Largest count of ``times`` falling within any ``window_s``-second span."""
    j, best = 0, 0
    for i in range(len(times)):
        while times[i] - times[j] > window_s:
            j += 1
        best = max(best, i - j + 1)
    return best


def check_gripper_chatter(
    ep: Episode, ctx: DatasetContext, cfg: Config, out: EpisodeResult
) -> None:
    """Rapid open/close toggling on a gripper dim: an anxious-squeeze teleop artifact, not
    a deliberate grasp. Gripper dims are found by feature name (``names`` in
    ``meta/info.json``); datasets without feature names skip this check."""
    if not ctx.gripper_dims or ep.action is None or ctx.action_scale is None or ep.length < 5:
        return
    worst_count, worst_dim = 0, None
    for dim in ctx.gripper_dims:
        pos = ep.action[:, dim]
        scale = ctx.action_scale[dim]
        span = float(pos.max() - pos.min())
        if not np.isfinite(scale) or span < cfg.gripper_active_frac * scale:
            continue  # gripper barely moved this episode; nothing to debounce or flag
        mid = (float(pos.max()) + float(pos.min())) / 2
        state = _hysteresis_state(pos, mid, cfg.gripper_hysteresis_frac * span)
        edges = np.flatnonzero(np.diff(state.astype(np.int8)) != 0) + 1
        if len(edges) < cfg.gripper_chatter_min_toggles:
            continue
        count = _max_in_window(ep.timestamps[edges], cfg.gripper_chatter_window_s)
        if count > worst_count:
            worst_count, worst_dim = count, dim
    if worst_dim is None:
        return
    out.metrics["gripper_toggles_per_window"] = worst_count
    if worst_count >= cfg.gripper_chatter_min_toggles:
        out.flags.append(
            Flag(
                "gripper_chatter",
                WARN,
                f"gripper toggled {worst_count}x within {cfg.gripper_chatter_window_s:.1f}s "
                f"(dim {worst_dim}); chatter rather than a deliberate grasp?",
                min(20.0, 3.0 * worst_count),
            )
        )


def check_relative(
    results: list[EpisodeResult],
    ctx: DatasetContext,
    cfg: Config,
    disabled_checks: frozenset[str] = frozenset(),
) -> None:
    """Checks that compare an episode to its peers in the same dataset."""
    if len(results) < 5:
        return
    # Many recorders use a fixed episode duration, so compare active (trimmed) time and
    # require a material relative difference, not just a large z on a tiny spread.
    if "length" not in disabled_checks:
        active = np.array([_active_s(r) for r in results])
        median = float(np.median(active))
        for r, value in zip(results, active, strict=True):
            r.metrics["active_s"] = value
            z = _robust_z(value, active)
            r.metrics["active_z"] = z
            material = abs(value - median) > cfg.min_relative_length_diff * median
            if z < -cfg.outlier_z and material:
                r.flags.append(
                    Flag(
                        "length",
                        WARN,
                        f"unusually short: {value:.1f}s of activity vs median "
                        f"{median:.1f}s (z={z:.1f}); aborted demo?",
                        10,
                    )
                )
            elif z > cfg.outlier_z and material:
                r.flags.append(
                    Flag(
                        "length",
                        INFO,
                        f"unusually long: {value:.1f}s of activity vs median "
                        f"{median:.1f}s (z={z:.1f})",
                        5,
                    )
                )

    ldlj = np.array([r.metrics.get("ldlj", np.nan) for r in results])
    valid = ldlj[np.isfinite(ldlj)]
    if len(valid) >= 5:
        # LDLJ is a log, so a fixed gap in nats is a fixed *relative* change in the
        # underlying (always-positive) dimensionless jerk integral, regardless of this
        # dataset's own absolute LDLJ range -- unlike the length check above, there's no
        # natural "percent of the median" floor for a quantity that can be negative. A
        # task whose episodes are all a close variation on the same motion (most real
        # datasets) makes the *majority* cluster tightly, so the population MAD reflects
        # only that majority's tiny spread even when a handful of episodes sit well above
        # it (e.g. a few notably smoother demos) -- found on a real Hub dataset
        # (EPITECH-LILLE/eval_training): one episode (z=-3.6 on a 20-episode peer group)
        # had an LDLJ gap of only ~0.6 nats from the median, and the same episode, scored
        # against the full 40-episode dataset instead, had z=-0.68 (not flagged) for an
        # almost identical raw LDLJ -- the z-score, not the episode, had changed. Real
        # jerkier-than-peers episodes already in the validated benchmark set (e.g.
        # shylee/so100_cube, jccj/so100_block_in_cup) all have gaps of 1.4+ nats, well
        # clear of this floor.
        median_ldlj = float(np.median(valid))
        for r in results:
            value = r.metrics.get("ldlj")
            if value is None or not np.isfinite(value):
                continue
            z = _robust_z(value, valid)
            r.metrics["ldlj_z"] = z
            if z < cfg.jerk_z and median_ldlj - value > cfg.min_ldlj_diff:
                r.flags.append(
                    Flag(
                        "smoothness",
                        WARN,
                        f"much jerkier than peers (smoothness z={z:.1f})",
                        min(20.0, 4 * -z),
                    )
                )

    # only episodes whose own action/state correlation is trustworthy inform (and are
    # eligible for) the lag-outlier check; a low-corr episode is already flagged on its own
    confident = [
        r for r in results if r.metrics.get("sync_peak_corr", 0.0) >= cfg.sync_min_peak_corr
    ]
    if len(confident) < 5:
        return
    lags = np.array([r.metrics["sync_lag_frames"] for r in confident])
    median_lag = float(np.median(lags))
    for r in confident:
        lag = r.metrics["sync_lag_frames"]
        if abs(lag - median_lag) > cfg.sync_lag_outlier_frames:
            r.flags.append(
                Flag(
                    "sync",
                    WARN,
                    f"action/observation.state lag is {lag:.0f} frames vs dataset median "
                    f"{median_lag:.0f} (r={r.metrics['sync_peak_corr']:.2f}); possible bad sync",
                    min(25.0, 5 * abs(lag - median_lag)),
                )
            )


def _active_s(r: EpisodeResult) -> float:
    if "lead_idle_s" not in r.metrics:
        return r.duration_s
    return max(0.0, r.duration_s - r.metrics["lead_idle_s"] - r.metrics["trail_idle_s"])


EPISODE_CHECKS = [
    ("timing", check_timing),
    ("metadata", check_metadata),
    ("integrity", check_integrity),
    ("video_sync", check_video_sync),
    ("camera", check_camera),
    ("video_freeze", check_video_freeze),
    ("idle", check_idle),
    ("smoothness", check_smoothness),
    ("stale_state", check_staleness),
    ("sync", check_sync),
    ("gripper_chatter", check_gripper_chatter),
]

# Every name that can appear as a Flag.check, usable with --disable-check. "length" and
# "duplicate" aren't in EPISODE_CHECKS: "length" is only ever emitted by check_relative
# (no per-episode half to skip) and "duplicate" comes from dedupe.find_duplicates/
# flag_duplicates, run separately by the CLI.
CHECK_NAMES = tuple(name for name, _ in EPISODE_CHECKS) + ("length", "duplicate")


def evaluate(
    ds: Dataset,
    cfg: Config | None = None,
    disabled_checks: frozenset[str] = frozenset(),
    decode_video: bool = False,
) -> list[EpisodeResult]:
    cfg = cfg or Config()
    ctx = DatasetContext.build(ds, cfg, decode_video=decode_video)
    checks = [(name, fn) for name, fn in EPISODE_CHECKS if name not in disabled_checks]
    results = []
    for ep in ds.episodes:
        r = EpisodeResult(ep.index, ep.length, ep.length / ds.fps, ep.task)
        for _, check in checks:
            check(ep, ctx, cfg, r)
        results.append(r)
    check_relative(results, ctx, cfg, disabled_checks)
    return results
