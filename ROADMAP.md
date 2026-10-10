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
- [x] **Near-duplicate episodes (`demoqc dedupe`, plus a `duplicates` section in `score`).**
      Resamples each episode's scaled action trajectory (trimmed) to a fixed length and
      groups near-duplicates with a threshold relative to the dataset's nearest-neighbour
      distance distribution (robust z vs. the dataset's own nn-distance spread). Exact
      duplicates (identical action/state bytes) are caught separately by content hash.
      Shipped as `src/demoqc/dedupe.py`, the `demoqc dedupe` subcommand, and a
      `duplicate_groups` summary plus per-episode `duplicate` flags in `demoqc score`.
      Follow-up: pairwise L2 is O(n²) in episode count and holds the whole distance matrix
      in memory; fine up to a few thousand episodes, but DTW (for duplicates recorded at
      different speeds beyond what resampling absorbs) and a chunked/approximate
      nearest-neighbour search (for huge datasets) are still open.
- [x] **Export a cleaned dataset (slice 1): `demoqc export --keep-from report.json`.**
      Write a filtered LeRobot v3.0 dataset (tabular data and meta only): drop rejected
      episodes, re-index episodes and frames, recompute `meta/info.json` totals and
      `meta/episodes`. Must load with LeRobot's own loader (add a test that does so if
      `lerobot` is installed, skipped otherwise).
      Shipped as `src/demoqc/export.py` and the `demoqc export` subcommand. Also rewrites
      `meta/stats.json` (exact for tabular features, count-weighted combination of the
      per-episode stats for video features) and the per-episode `stats/index` and
      `stats/episode_index` columns, which would otherwise go stale.
      Follow-ups: (1) **v2.x export.** v3.0 only for now, but most community datasets
      (e.g. the so100 ones used for validation) are v2.1, so this is the most useful next
      slice of export (per-episode parquet and videos, `episodes.jsonl`, `episodes_stats.jsonl`).
      (2) An option to copy or symlink the source `videos/` directory for local sources (they
      are never read today; the export only notes that they must be copied). (3) `--min-score`
      on `export` itself, so a separate `score --json` run isn't required. (4) Single `train`
      split only: other `splits` entries in `info.json` are collapsed into `train`.
- [x] **Export slice 2: apply trims.** Cut idle frames using `trim`, shift timestamps and
      update video `from_timestamp`/`to_timestamp` so videos stay aligned without
      re-encoding.
      Shipped as `demoqc export --apply-trim`: `frame_index`/`index` are renumbered,
      `timestamp` is shifted by the number of leading frames cut, and each camera's video
      span is narrowed by the same frame counts (video bytes are never touched). `length` in
      `meta/episodes` is now always rewritten to match the written frame count (previously it
      only warned on a pre-existing mismatch and left the stale value, which would have made
      a trimmed export self-inconsistent).
      Follow-up: v2.x export (including trims) is still not implemented; this only covers
      v3.0, same as slice 1.
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
- 2026-10-09: Added exact and near-duplicate episode detection (`src/demoqc/dedupe.py`,
  `demoqc dedupe`, `duplicate_groups` in `score`). Validated on
  lerobot/svla_so101_pickplace (50 ep), masato-ka/so100_cutlery_handling_simple (50 ep),
  aaronsu11/so100_lego (100 ep) and shylee/so100_cube (200 ep): no duplicates and no false
  positives on any of them. Synthetic tests cover exact (byte-identical) and near
  (same seed, different duration) duplicates, plus a 40-episode independent-seed check for
  false positives.
- 2026-10-09: Added `demoqc export SOURCE --keep-from report.json -o DIR` (LeRobot v3.0 only,
  `src/demoqc/export.py`). Validated on lerobot/svla_so101_pickplace (50 ep) and
  lerobot/pusht (206 ep), the only v3.0 datasets in the validation set (the so100 community
  ones are v2.1, which export rejects with a clear error). Keeping every episode reproduces
  the data and `meta/episodes` tables exactly and `stats.json` to within 3e-5 (including the
  combined video stats); dropping ~2/7 of the episodes gives contiguous indices, correct
  `dataset_from/to_index` ranges and stats that match a recomputation from the kept frames.
  All four exports load with `LeRobotDataset` from lerobot 0.6.1 (with placeholder files
  for the videos, which LeRobot checks for) and `episodes=[...]` subsets resolve correctly.
  `tests/test_export.py` covers this on synthetic data, including a 3-file layout where one
  file contains only dropped episodes; the lerobot loader test is skipped when lerobot isn't
  installed (CI doesn't install it).
- 2026-10-10: Added `demoqc export --apply-trim`, which cuts each kept episode's idle
  start/end frames per the report's `trim` suggestion, renumbers `frame_index`/`index`, shifts
  `timestamp`, and narrows the per-camera `videos/*/from_timestamp`/`to_timestamp` window by the
  same frame counts (videos are still never read, copied or re-encoded). Also fixed `length` in
  `meta/episodes` to always match the written frame count instead of only warning when it
  disagreed. Validated on lerobot/svla_so101_pickplace (50 ep, v3.0): the Hugging Face Hub was
  reachable this run. 46/50 episodes had a suggested trim; after `--apply-trim`, total frames
  dropped from 11939 to 10528 (1411 trimmed), re-scoring the export dropped `trimmable idle`
  from 71.6s/47 flags to 3.6s/3 flags (the small remainder is the check's own padding, by
  design) and mean score rose from 97.1 to 99.8 with no new flags. Verified per-episode that
  the trimmed frame count and both video-span edges exactly match `start`/`end` from the
  report. Also validated on lerobot/pusht (206 ep, v3.0, no idle episodes): `--apply-trim`
  trimmed 0 frames and the re-scored export is byte-for-byte the same report (mean 100.0, no
  flags), confirming no false positives when there is nothing to trim. The so100 community
  datasets used for earlier validation (masato-ka/so100_cutlery_handling_simple etc.) are
  v2.1, so `export` still rejects them with the existing "v2.x not implemented" error.
