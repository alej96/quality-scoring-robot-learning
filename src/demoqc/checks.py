"""Per-episode quality checks.

Every check is computed relative to a :class:`DatasetContext` built from the whole
dataset, so thresholds adapt to the robot's units and the operator's typical speed
instead of being hard-coded per embodiment. Thresholds were calibrated on public
LeRobot datasets (``lerobot/svla_so101_pickplace``, ``lerobot/pusht``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from demoqc.dataset import Dataset, Episode

ERROR, WARN, INFO = "error", "warn", "info"


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


@dataclass
class DatasetContext:
    fps: float
    action_scale: np.ndarray | None  # robust std per action dim
    step_p99: np.ndarray | None  # 99th pct |delta action| per dim, in scaled units
    idle_speed: float  # speed threshold, scaled units / s

    @classmethod
    def build(cls, ds: Dataset, cfg: Config) -> DatasetContext:
        actions = [e.action for e in ds.episodes if e.action is not None and len(e.action) > 1]
        if not actions:
            return cls(ds.fps, None, None, 0.0)
        allA = np.concatenate(actions)
        q75, q25 = np.percentile(allA, [75, 25], axis=0)
        scale = (q75 - q25) / 1.349
        std = allA.std(axis=0)
        scale = np.where(scale > 1e-9, scale, std)
        scale = np.where(scale > 1e-9, scale, np.nan)  # constant dims are ignored
        steps = np.concatenate([np.abs(np.diff(a, axis=0)) / scale for a in actions])
        speeds = np.concatenate([_speed(a, scale, ds.fps) for a in actions])
        p75 = float(np.nanpercentile(speeds, 75)) if np.isfinite(speeds).any() else 0.0
        step_p99 = np.nanpercentile(steps, 99, axis=0)
        # dims that almost never change (e.g. a binary gripper) have no step scale;
        # leave them out of spike detection rather than flag every toggle
        step_p99 = np.where(step_p99 > 1e-9, step_p99, np.nan)
        return cls(
            fps=ds.fps,
            action_scale=scale,
            step_p99=step_p99,
            idle_speed=cfg.idle_fraction_of_p75 * p75,
        )


def _speed(action: np.ndarray, scale: np.ndarray, fps: float) -> np.ndarray:
    """RMS speed across (scaled) action dims, one value per transition."""
    v = np.diff(action, axis=0) / scale * fps
    return np.sqrt(np.nanmean(v**2, axis=1))


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


def check_idle(ep: Episode, ctx: DatasetContext, cfg: Config, out: EpisodeResult) -> None:
    if ep.action is None or ctx.action_scale is None or ep.length < 3:
        return
    speed = _speed(ep.action, ctx.action_scale, ctx.fps)
    win = max(1, int(round(cfg.idle_smoothing_s * ctx.fps)))
    if win > 1 and len(speed) >= win:
        speed = np.convolve(speed, np.ones(win) / win, mode="same")
    idle = speed < ctx.idle_speed
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
    step = np.diff(ep.action, axis=0) / ctx.action_scale
    limit = cfg.spike_factor * ctx.step_p99
    big = np.abs(step) > limit
    # a glitch is a large step immediately reversed by a large step on the same dim
    reversal = big[:-1] & big[1:] & (np.sign(step[:-1]) != np.sign(step[1:]))
    spikes = int(reversal.any(axis=1).sum())
    # each spike accounts for two big steps; whatever remains is a one-way jump
    jumps = max(0, int(big.any(axis=1).sum()) - 2 * spikes)
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


def check_relative(results: list[EpisodeResult], ctx: DatasetContext, cfg: Config) -> None:
    """Checks that compare an episode to its peers in the same dataset."""
    if len(results) < 5:
        return
    # Many recorders use a fixed episode duration, so compare active (trimmed) time and
    # require a material relative difference, not just a large z on a tiny spread.
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
                    f"unusually long: {value:.1f}s of activity vs median {median:.1f}s (z={z:.1f})",
                    5,
                )
            )

    ldlj = np.array([r.metrics.get("ldlj", np.nan) for r in results])
    valid = ldlj[np.isfinite(ldlj)]
    if len(valid) < 5:
        return
    for r in results:
        value = r.metrics.get("ldlj")
        if value is None or not np.isfinite(value):
            continue
        z = _robust_z(value, valid)
        r.metrics["ldlj_z"] = z
        if z < cfg.jerk_z:
            r.flags.append(
                Flag(
                    "smoothness",
                    WARN,
                    f"much jerkier than peers (smoothness z={z:.1f})",
                    min(20.0, 4 * -z),
                )
            )


def _active_s(r: EpisodeResult) -> float:
    if "lead_idle_s" not in r.metrics:
        return r.duration_s
    return max(0.0, r.duration_s - r.metrics["lead_idle_s"] - r.metrics["trail_idle_s"])


EPISODE_CHECKS = [check_timing, check_metadata, check_idle, check_smoothness, check_staleness]


def evaluate(ds: Dataset, cfg: Config | None = None) -> list[EpisodeResult]:
    cfg = cfg or Config()
    ctx = DatasetContext.build(ds, cfg)
    results = []
    for ep in ds.episodes:
        r = EpisodeResult(ep.index, ep.length, ep.length / ds.fps, ep.task)
        for check in EPISODE_CHECKS:
            check(ep, ctx, cfg, r)
        results.append(r)
    check_relative(results, ctx, cfg)
    return results
