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
    stale_state_max_fraction: float = 0.1
    outlier_z: float = 3.5
    min_relative_length_diff: float = 0.25
    jerk_z: float = -3.0
    min_ldlj_diff: float = 1.0  # jerkier-than-peers also needs an LDLJ gap from the median
    # of at least this many nats (~2.7x the underlying jerk integral) -- guards against a
    # z-score inflated by a tiny MAD when most episodes share a near-identical motion
    sync_max_lag_s: float = 0.5  # search window for action/state cross-correlation
    sync_min_peak_corr: float = 0.4  # below this, the lag estimate is unreliable
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
    gripper_dims: list[int] = field(default_factory=list)  # action dims named "*gripper*"
    sync_dims: tuple[list[int], list[int]] | None = None  # matched (action, state) dim indices
    angular_dims: np.ndarray | None = None  # action dims that wrap as a radian angle in [-pi, pi]
    angular_pairs: list[tuple[int, int]] = field(default_factory=list)  # x/z axis dim pairs of
    # the same Euler-angle rotation triple, which can trade off at a gimbal-lock singularity
    # --decode-video support (check_video_freeze); video_root is only set for a local
    # dataset (v2.x or v3.0), since videos are never downloaded from the Hub (see
    # video.resolve_v2_video_path/resolve_v3_video_path).
    video_root: Path | None = None
    video_path_template: str | None = None
    chunks_size: int = 1000
    camera_names: list[str] = field(default_factory=list)

    @classmethod
    def build(cls, ds: Dataset, cfg: Config, decode_video: bool = False) -> DatasetContext:
        gripper_dims = [
            i for i, name in enumerate(ds.action_names or []) if name and "gripper" in name.lower()
        ]
        a_width = next((e.action.shape[1] for e in ds.episodes if e.action is not None), None)
        s_width = next((e.state.shape[1] for e in ds.episodes if e.state is not None), None)
        sync_dims = _match_sync_dims(ds.action_names, ds.state_names, a_width, s_width)
        video_root = ds.root if decode_video and ds.repo_id is None else None
        video_ctx = dict(
            video_root=video_root,
            video_path_template=ds.video_path_template,
            chunks_size=ds.chunks_size,
            camera_names=ds.camera_names,
        )
        actions = [e.action for e in ds.episodes if e.action is not None and len(e.action) > 1]
        if not actions:
            return cls(ds.fps, None, None, 0.0, gripper_dims, sync_dims, **video_ctx)
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
        steps = np.concatenate([np.abs(_angular_diff(a, angular_dims)) / scale for a in actions])
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
        step_p99 = _exclude_noise_dominated_dims(step_p99, cfg.smoothness_dim_outlier_z)
        return cls(
            fps=ds.fps,
            action_scale=scale,
            step_p99=step_p99,
            idle_speed=cfg.idle_fraction_of_p75 * p75,
            gripper_dims=gripper_dims,
            sync_dims=sync_dims,
            angular_dims=angular_dims,
            angular_pairs=angular_pairs,
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
    step = _angular_diff(ep.action, ctx.angular_dims) / ctx.action_scale
    limit = cfg.spike_factor * ctx.step_p99
    big = np.abs(step) > limit
    if ctx.angular_pairs:
        _suppress_gimbal_lock_pairs(ep.action, big, ctx.angular_pairs, cfg.angular_pair_cancel_frac)
    any_big = big.any(axis=1)
    # a glitch is a large step immediately reversed by a large step on the same dim
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
    # action jumps" even though none of those steps is actually discontinuous.
    consumed = np.zeros_like(any_big)
    consumed[:-1] |= frame_reversal
    consumed[1:] |= frame_reversal
    jumps = len(_runs(any_big & ~consumed))
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
    seg = ep.action[start:end] / ctx.action_scale
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


def check_staleness(ep: Episode, ctx: DatasetContext, cfg: Config, out: EpisodeResult) -> None:
    """Observation repeated bit-for-bit while the commanded action moves: a stale read."""
    if ep.state is None or ep.action is None or ctx.action_scale is None or ep.length < 3:
        return
    repeated = np.all(np.diff(ep.state, axis=0) == 0, axis=1)
    moving = _speed(ep.action, ctx.action_scale, ctx.fps) > max(ctx.idle_speed, 1e-9) * 5
    stale = repeated & moving
    frac = float(stale.mean())
    out.metrics["stale_state_fraction"] = frac
    if frac > cfg.stale_state_max_fraction:
        longest = max((e - s for s, e in _runs(stale)), default=0)
        out.flags.append(
            Flag(
                "stale_state",
                WARN,
                f"observation.state frozen on {frac:.0%} of moving frames "
                f"(longest run {longest} frames)",
                min(30.0, 100 * (frac - cfg.stale_state_max_fraction) + 10),
            )
        )


def _best_lag(a: np.ndarray, s: np.ndarray, max_lag: int) -> tuple[int, float]:
    """Lag (in frames) and peak Pearson correlation that best aligns ``s[t + lag]`` with
    ``a[t]``, averaged over dims with enough variance in both signals. ``a`` and ``s`` are
    per-dim velocity series of equal length; a positive lag means ``s`` trails ``a``.
    """
    n = len(a)
    valid_a = a.std(axis=0) > 1e-9
    if not valid_a.any():
        return 0, 0.0
    best_lag, best_corr = 0, -2.0
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
        avg_corr = float(np.mean(np.where(den > 1e-9, num / den, 0.0)))
        if avg_corr > best_corr:
            best_lag, best_corr = lag, avg_corr
    return best_lag, best_corr


def check_sync(ep: Episode, ctx: DatasetContext, cfg: Config, out: EpisodeResult) -> None:
    """Estimate the lag between commanded ``action`` and observed ``observation.state``.

    A USB latency spike or a recorder bug desyncs the two streams for part of a recording
    session; that episode's best-fit lag will differ from its peers' even though each
    stream looks fine on its own.
    """
    if ep.action is None or ep.state is None or ep.length < 10 or ctx.sync_dims is None:
        return
    a_idx, s_idx = ctx.sync_dims
    if max(a_idx) >= ep.action.shape[1] or max(s_idx) >= ep.state.shape[1]:
        return  # this episode's streams are narrower than the dataset's matched dims
    action, state = ep.action[:, a_idx], ep.state[:, s_idx]
    max_lag = max(1, int(round(cfg.sync_max_lag_s * ctx.fps)))
    lag, corr = _best_lag(np.diff(action, axis=0), np.diff(state, axis=0), max_lag)
    out.metrics.update(sync_lag_frames=float(lag), sync_lag_s=lag / ctx.fps, sync_peak_corr=corr)
    if corr < cfg.sync_min_peak_corr:
        out.flags.append(
            Flag(
                "sync",
                WARN,
                f"action/observation.state correlation is weak (r={corr:.2f}); "
                "lag estimate unreliable",
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
