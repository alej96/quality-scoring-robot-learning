# Roadmap

The objective: the tool a LeRobot user runs before training. It should **score, trim,
flag and dedupe** teleoperation episodes, export a cleaned dataset, and integrate with the
Hugging Face Hub. Every item must solve a real problem in real community datasets.

Work top-down through **Next up**. Each item should ship in one focused session
(implementation, tests and docs). Split big items into slices that are useful on their own.

## Next up

- [x] **Action/state sync check.** Estimate per-episode lag between `action` and
      `observation.state` (cross-correlate per-dim velocities over ±0.5 s; only dims present
      in both with matching names or shapes). Flag episodes whose lag deviates from the
      dataset median by more than 2 frames, or whose peak correlation is low. Catches "bad
      syncs" from USB latency spikes and recorder bugs. Add to the report and README table.
      Shipped as the `sync` check (`sync_lag_frames`, `sync_lag_s`, `sync_peak_corr` metrics).
      Follow-up: dims are currently matched by shape only (`action.shape[1] == state.shape[1]`)
      because the loader doesn't read per-feature names from `meta/info.json` yet; once it
      does, match by name instead so a reordered or extra dim doesn't disable the check.
- [ ] **Near-duplicate episodes (`demoqc dedupe`, plus a `duplicates` section in `score`).**
      Resample each episode's scaled action trajectory (trimmed) to a fixed length, compute
      pairwise distances (start with L2 on the resampled trajectories, then DTW if needed),
      and group near-duplicates with a threshold relative to the dataset's
      nearest-neighbour distance distribution. Also catch exact duplicates (identical data
      hashes), which happen when uploads are re-run. Validate on community datasets and
      report what you find.
- [ ] **Export a cleaned dataset (slice 1): `demoqc export --keep-from report.json`.**
      Write a filtered LeRobot v3.0 dataset (tabular data and meta only): drop rejected
      episodes, re-index episodes and frames, recompute `meta/info.json` totals and
      `meta/episodes`. Must load with LeRobot's own loader (add a test that does so if
      `lerobot` is installed, skipped otherwise).
- [ ] **Export slice 2: apply trims.** Cut idle frames using `trim`, shift timestamps and
      update video `from_timestamp`/`to_timestamp` so videos stay aligned without
      re-encoding.
- [ ] **Hub integration: `demoqc push-report <repo_id>`.** Upload the JSON report plus a
      Markdown summary to the dataset repo as a Hub pull request (`create_pr=True`), so
      dataset owners get actionable feedback. Generate a dataset-card snippet with a badge.
- [ ] **Gripper chatter check.** Detect gripper dims by feature name (`gripper` in
      `names`) and flag rapid open/close toggling, a common teleop artifact.
- [ ] **Configurable thresholds.** `--config demoqc.toml` with per-check thresholds and
      penalty weights, plus `--disable-check NAME`. Document the defaults.
- [ ] **Markdown report output (`--md`)** for pasting into dataset cards or PRs.
- [ ] **Scale to large datasets.** Stream per data file instead of loading everything at
      once; add `--episodes 0-99` and `--max-episodes`. Target: DROID-scale metadata without
      running out of memory.
- [ ] **Video checks (optional `[video]` extra).** Sample frames with PyAV at low
      resolution to detect frozen or black camera frames and decode errors. Never required
      for the core path.
- [ ] **Benchmark the community.** `scripts/benchmark.py` scores the N most-downloaded
      community LeRobot datasets and writes `docs/benchmarks.md` (scores, idle seconds,
      top flags). Refresh it periodically; it doubles as a regression test for the
      heuristics.

## Needs the owner

- Publish to PyPI (needs a PyPI account and trusted-publisher setup).
- Announce on the LeRobot Discord or the Hugging Face forums once export and dedupe ship.

## Known limitations

- LeRobot writes timestamps as `frame_index / fps`, so the timing check rarely fires on
  LeRobot-recorded data. The `stale_state`, `video_sync` and `idle` checks carry most of
  the signal there.
- Scores are relative within a dataset for the smoothness and length checks. A dataset
  where every demo is bad is caught only by the absolute checks.

## Log

- 2026-10-09: v0.1.0. LeRobot v2.x/v3.0 loader (local or Hub, no videos), checks for
  timing, metadata, video_sync, idle/trim, smoothness, stale_state and length, plus
  JSON/CSV reports, the `demoqc score` CLI, CI and auto-merge. Validated on
  lerobot/svla_so101_pickplace, lerobot/pusht and 5 community SO-100 datasets.
- 2026-10-09: Added the `sync` check (action/observation.state lag via per-dim
  cross-correlation, ±0.5s window); flags a weak peak correlation or a lag that deviates
  from the dataset's median by more than 2 frames. Validated with synthetic-defect tests
  only; the Hugging Face Hub was unreachable from this environment (network policy blocks
  huggingface.co) so real-dataset validation is a follow-up.
