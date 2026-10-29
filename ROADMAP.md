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
      Follow-up (resolved 2026-10-25): that "fine up to a few thousand episodes" claim
      undercounted the actual memory use by ~2-3 orders of magnitude -- see the Log entry.
      DTW and a chunked/approximate search for datasets beyond what the (n, n) distance
      matrix itself can hold are still open.
- [x] **Export a cleaned dataset (slice 1): `demoqc export --keep-from report.json`.**
      Write a filtered LeRobot v3.0 dataset (tabular data and meta only): drop rejected
      episodes, re-index episodes and frames, recompute `meta/info.json` totals and
      `meta/episodes`. Must load with LeRobot's own loader (add a test that does so if
      `lerobot` is installed, skipped otherwise).
      Shipped as `src/demoqc/export.py` and the `demoqc export` subcommand. Also rewrites
      `meta/stats.json` (exact for tabular features, count-weighted combination of the
      per-episode stats for video features) and the per-episode `stats/index` and
      `stats/episode_index` columns, which would otherwise go stale.
      Follow-ups: (1) ~~v2.x export~~ shipped 2026-10-19 (`_export_v2` in `src/demoqc/export.py`;
      see the Log entry) — slice 1 only (drop/renumber, no `--apply-trim`).
      (2) ~~An option to copy the source `videos/` directory for local sources~~ shipped
      2026-10-23 as `--copy-videos` (see the Log entry). (3) ~~`--min-score` on
      `export` itself~~ shipped 2026-10-20 (see the Log entry). (4) Single `train`
      split only: other `splits` entries in `info.json` are collapsed into `train`.
      (5) ~~v2.x `--apply-trim`~~ shipped 2026-10-21 (see the Log entry) — a v2.x video has no
      per-episode time span to narrow like v3.0's, so the export warns that the video itself
      must also be cut to match instead of narrowing a stored span.
      (6) ~~`--copy-videos` only copies; a `--symlink-videos` variant~~ shipped 2026-10-24
      (see the Log entry).
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
      Skipped this run in favour of the `--md` item below: there is no `HF_TOKEN` in this
      environment, so a Hub PR can't be created or validated end to end, and opening PRs on
      real community dataset repos isn't something to ship untested. The Markdown body this
      needs (`format_markdown`) is now done; `push-report` is mostly `HfApi.create_commit` /
      `create_pull_request` plumbing once a token is available.
- [x] **Gripper chatter check.** Detect gripper dims by feature name (`gripper` in
      `names`) and flag rapid open/close toggling, a common teleop artifact.
      Shipped as the `gripper_chatter` check. `dataset.py` now reads `info["features"]["action"]["names"]`
      onto `Dataset.action_names` (previously unused); `DatasetContext.gripper_dims` finds dims whose
      name contains "gripper" (case-insensitive). A dim counts as active only if its per-episode range
      exceeds half the dataset's own robust scale for that dim (debounces noise on an unused or barely-
      moved gripper); state changes use hysteresis (15% of the episode's own range) before counting
      toggles, then flags 4+ toggles within any 1-second sliding window.
      Follow-up (resolved 2026-10-12): `check_sync` now matches action/state dims by name
      instead of requiring `action.shape[1] == state.shape[1]`, so a reordered or extra dim
      (e.g. an added force sensor) no longer disables the sync check. See the Log entry.
- [x] **Configurable thresholds (slice 1): `--config demoqc.toml`.** Override any subset of
      `Config`'s threshold/penalty fields from a TOML file, for `score` and `dedupe`.
      Shipped as `Config.from_file`/`Config.with_overrides` (`src/demoqc/checks.py`) and
      `--config FILE.toml` on both subcommands. Unknown keys or wrong-typed values raise a
      clear error (naming the key) instead of being silently ignored or miscalibrating the
      run; `tomli` is pulled in as a dependency only on Python < 3.11 (stdlib `tomllib`
      covers 3.11+).
      Follow-up (shipped 2026-10-15 as `--disable-check NAME`; see the Log entry): a
      per-check name registry (`checks.CHECK_NAMES`) shared between `EPISODE_CHECKS` and the
      names `check_relative` emits, so a noisy check (e.g. `sync` on a dataset with an
      unsynced third camera clock) can be turned off entirely instead of re-tuned.
- [x] **Markdown report output (`--md`)** for pasting into dataset cards or PRs.
      Shipped as `demoqc score --md report.md` and `src/demoqc/report.py::format_markdown`:
      a score/flags table, duplicate-group counts, and a table of the lowest-scoring
      episodes with their flags, matching `format_summary`'s content but as GitHub/Hub-
      flavoured Markdown tables. Table cells escape `|` in flag messages so an unusual
      message can't break a row.
      Follow-up: this is the Markdown half of the `push-report` item above; once that ships,
      it should reuse `format_markdown` as the PR/dataset-card body instead of generating its
      own text.
- [x] **Scale to large datasets (slice 1): `--episodes` and `--max-episodes`.** Load only
      some of a dataset's episodes. From the Hub, only the data file(s) that contain the
      requested episodes are downloaded; locally, data files that contain none of them are
      never opened.
      Shipped as `demoqc score`/`demoqc dedupe --episodes SPEC`/`--max-episodes N`
      (`load_dataset(..., episodes=, max_episodes=)` in `src/demoqc/dataset.py`). Metadata
      (`meta/*`, always small) is downloaded first, then `meta/episodes`'
      `data/chunk_index`/`data/file_index` columns (v3.0) or the `episode_<n>.parquet`
      filename convention (v2.x) map the requested episodes to the exact data file(s) to
      fetch or open; narrowing is best-effort and falls back to reading everything if a
      layout can't be mapped from metadata alone, so it's never less correct, only
      sometimes not faster.
      Follow-up: this still doesn't bound peak memory when scoring *all* of a huge
      dataset — `_read_frames` still concatenates every needed file into one DataFrame.
      True per-file streaming (bounding memory independent of `--episodes`) is still open.
      ~~Extending `--episodes`/`--max-episodes` to `demoqc export`~~ shipped 2026-10-22; see
      the Log entry.
- [x] **Video checks (slice 1): black/dead camera from metadata alone.** LeRobot already
      writes each episode's per-camera pixel mean/std into `meta/episodes` (v3.0) or
      `meta/episodes_stats.jsonl` (v2.x) when it records a dataset. Read it and flag a
      camera that's near-black for the whole episode (mean and std both near zero): a
      disconnected camera, a lens cap left on, or video that failed to record. No video
      file is ever opened, so this needed no new dependency and stays in the core path.
      Shipped as the `camera` check (`src/demoqc/dataset.py`'s `Episode.camera_stats`,
      `checks.check_camera`, `Config.camera_dark_mean_max`/`camera_dark_std_max`).
      Follow-up (resolved 2026-10-26): detecting a camera going black or a video failing to
      decode mid-episode shipped as the opt-in `video_freeze` check
      (`--decode-video`, `demoqc[video]`); see the Log entry. Follow-up (resolved
      2026-10-27): `video_freeze` now also supports LeRobot v3.0 (resolving a camera's local
      video file via its `meta/episodes` chunk/file index, and narrowing decode to that
      episode's own `from_timestamp`/`to_timestamp` span since a v3.0 video file holds
      several episodes back to back); see the Log entry. Follow-up (resolved 2026-10-28):
      `video_freeze` now also detects a literally frozen (repeated) frame on a camera that
      never goes dark; see the Log entry.
- [x] **Benchmark the community.** `scripts/benchmark.py` scores a curated list of real
      community LeRobot datasets and writes `docs/benchmarks.md` (scores, trimmable idle,
      duplicate groups, top flags). Refresh it periodically; it doubles as a regression
      test for the heuristics.
      Shipped as `scripts/benchmark.py` and `docs/benchmarks.md`. Scores each dataset in
      isolation and catches any exception (unreachable repo, unrecognised format, a bug in
      a check) into an error row instead of stopping the whole run, so one bad dataset
      doesn't lose the rest. Took a curated list (`DEFAULT_REPOS`: the reference and
      community so100 datasets already used for validation) rather than the Hub's actual
      most-downloaded LeRobot datasets, which are enormous OpenX/DROID re-releases
      (hundreds of GB of tabular data even with videos excluded) this tool isn't calibrated
      for and that would make the benchmark slow and its numbers meaningless; `--max-episodes`
      (default 200) caps runtime further. `--repo` adds/replaces the list, for anyone who
      wants to benchmark a different set.
      Follow-up: querying the Hub for real most-downloaded small/medium teleoperation
      datasets (e.g. filtering `list_datasets(filter="LeRobot", sort="downloads")` by
      dataset size or episode count rather than a hand-picked list) is still open.

- [x] **Data-integrity check (NaN/Inf in `action`/`observation.state`).** Found while
      reviewing `checks.py` for the daily session: `DatasetContext.build` computed the
      dataset-wide `action_scale` (and `step_p99`) with plain `np.percentile`/`.std`, which
      propagate `NaN` across an *entire column* when even one value in it is non-finite.
      A single corrupted frame in one episode's `action` (a sensor glitch or a partial,
      corrupted write) therefore silently set that whole dimension's scale to `NaN` for
      *every* episode in the dataset, not just the bad one -- disabling `idle`,
      `smoothness` (spike/jump detection on that dim) and, worst, `gripper_chatter`
      (which requires a finite scale and only ever watches one or two gripper dims, so one
      bad frame anywhere in the dataset's gripper column could blind that check dataset-wide)
      with no visible symptom pointing at the actual cause. Fixed `DatasetContext.build` to
      use `np.nanpercentile`/`np.nanstd`, and added the `integrity` check, which flags the
      actual corrupted episode (ERROR, "non-finite (NaN/Inf) value(s) in action/
      observation.state") so the defect is visible on its own instead of only as collateral
      damage elsewhere. Also made `dedupe._trimmed_action` skip a non-finite trajectory
      instead of feeding `NaN` into `np.interp` and corrupting that episode's row of the
      near-duplicate distance matrix. New tests in `tests/test_checks.py`: `integrity` fires
      on an injected NaN in `action` and separately in `observation.state` (and stays silent
      on every other episode), plus a regression test that reproduces the exact scenario
      above (one episode with a NaN in its gripper dim, another with a real
      `gripper_chatter` defect) and confirms `gripper_chatter` still fires on the second
      episode; all three fail without this change (confirmed via `git stash` on just
      `src/demoqc/checks.py`). Validated on real data: `lerobot/svla_so101_pickplace` (50 ep,
      mean 97.1), `aaronsu11/so100_lego` (100 ep, mean 92.2) and `shylee/so100_cube` (200 ep,
      mean 99.2) all reproduced their existing validated scores and flag counts exactly, with
      zero `integrity` flags on any of them -- `nanpercentile`/`nanstd` are mathematically
      identical to `percentile`/`.std` when there's nothing non-finite to ignore, so this is
      a no-op on real, clean data and only changes behavior when a dataset is actually
      corrupted.

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
- 2026-10-11: Added the `gripper_chatter` check (rapid open/close toggling via feature-name
  lookup + hysteresis-debounced sliding-window toggle counting). Validated on
  lerobot/svla_so101_pickplace (50 ep, max 2 toggles/window, well under the 4-toggle
  threshold; every normal grasp is 1-2 toggles), masato-ka/so100_cutlery_handling_simple
  (50 ep) and aaronsu11/so100_lego (100 ep), both also maxing out at 2 toggles/window, and
  shylee/so100_cube (200 ep), where the gripper barely moves (0 episodes pass the active-range
  filter) — no false positives on any of them. Also ran on lerobot/pusht (206 ep, no gripper
  feature at all) to confirm the check degrades to a no-op rather than erroring when a dataset
  has no feature names or no gripper dim. Synthetic test injects a 15-frame toggle burst on one
  episode's gripper dim and checks it fires there only; confirmed the test fails with
  `KeyError` before the implementation.
- 2026-10-12: `check_sync` now matches action/observation.state dims by feature name
  (`Dataset.state_names`, newly loaded from `info["features"]["observation.state"]["names"]`)
  instead of requiring the two streams to have equal width. Previously, any dataset where
  the two streams had different widths (e.g. an extra, unmatched sensor dim on one side)
  silently disabled the sync check for every episode; a dataset where both streams had the
  same width but a different dim order would get wrong lag/correlation estimates from
  comparing unrelated dims position-by-position. Falls back to positional matching when
  either stream lacks feature names, so nothing changes for datasets without them. Two new
  synthetic tests confirm both bugs: one reorders `observation.state`'s columns relative to
  `action`'s (previously computed near-zero peak correlation for a dataset that is in fact
  tightly synced, risking a false `sync` flag; now correctly reports r>0.8), the other adds
  an action-only `force_sensor` dim with no state counterpart (previously the sync check
  produced no metrics at all; now it still detects an injected 6-frame lag). Both tests fail
  with the pre-fix code (`KeyError` / an assertion on `sync_peak_corr`). Validated on
  lerobot/svla_so101_pickplace (50 ep), masato-ka/so100_cutlery_handling_simple (50 ep) and
  aaronsu11/so100_lego (100 ep): all three have identically-ordered, identically-named
  action/state features, so matched dims are unchanged (`[0..5]`/`[0..5]`) and scores, sync
  metrics and flags are byte-for-byte the same as before this change — no regression.
- 2026-10-13: Added `demoqc score --md report.md` (`format_markdown` in `report.py`), a
  Markdown rendering of the same summary `format_summary` prints to the terminal, for
  pasting into a dataset card or a Hub/GitHub PR. Flag messages are escaped (`|` -> `\|`)
  so a stray pipe can't break a table row. Validated on lerobot/svla_so101_pickplace
  (50 ep, v3.0) and masato-ka/so100_cutlery_handling_simple (50 ep, v2.1): both render as
  clean GitHub-flavoured Markdown tables with no escaping artifacts. Skipped the
  higher-priority `push-report` item this run — no `HF_TOKEN` in this environment to create
  or validate a real Hub PR — and left a note on it pointing back to `format_markdown` as
  the body it should reuse.
- 2026-10-14: Added `--config demoqc.toml` to `demoqc score`/`demoqc dedupe`
  (`Config.from_file`/`Config.with_overrides`), overriding any subset of the threshold and
  penalty fields in `Config` so a different embodiment, fps or operator style doesn't need a
  code change. Unknown keys and wrong-typed values are rejected with an error naming the
  key. `huggingface.co` was reachable this run but there is still no `HF_TOKEN`, so
  `push-report` (which needs to open a Hub PR on someone else's dataset repo) stays blocked;
  picked this item instead. Validated on lerobot/svla_so101_pickplace (50 ep, v3.0: mean
  score 97.13 at defaults, 97.14 with a config loosening `max_idle_fraction`,
  `gripper_chatter_min_toggles` and `sync_min_peak_corr`) and
  masato-ka/so100_cutlery_handling_simple (50 ep, v2.1: 92.92 at defaults, 93.47 tuned),
  confirming thresholds actually move scores end to end and a bad config file exits 2 with a
  clear message on both. New `tests/test_config.py` (8 cases: override/validation on
  `Config` directly, TOML parsing, a synthetic gripper-chatter episode that only flags once
  `gripper_chatter_min_toggles` is lowered via config, and both CLI paths) fails on 7/8
  cases without this change.
- 2026-10-15: Added **`--disable-check NAME`** to `demoqc score` (the follow-up noted under
  the `--config` item). Lets a check that's noise for a particular dataset (no gripper but a
  jittery `sync` from an unsynced third camera clock; intentional near-identical demos that
  shouldn't count as `duplicate`) be turned off instead of fought with thresholds. Split
  `check_metadata` into `check_metadata` (episode length) and the new `check_video_sync`
  (video span) so they're independently disableable; added `checks.CHECK_NAMES` (the
  registry the original follow-up asked for, shared between `EPISODE_CHECKS` and
  `check_relative`'s own `length` flag) and threaded `disabled_checks` through
  `evaluate()`/`check_relative()`. `length` has no `EPISODE_CHECKS` entry (it's peer-only)
  and `duplicate` isn't part of `checks.py` at all (the CLI calls
  `dedupe.find_duplicates`/`flag_duplicates` separately), so both are special-cased in
  `evaluate()` and `cli._score()` respectively; disabling a check removes its metrics too,
  not just its flags. argparse `choices=CHECK_NAMES` rejects a typo'd name up front. 5 new
  tests (disabling suppresses flags+metrics on a real sync defect, the metadata/video_sync
  split stays independent, `length`'s peer-only path, the CLI's `duplicate` path, and the
  argparse rejection) all fail without this change. Validated on
  lerobot/svla_so101_pickplace (50 ep): disabling `gripper_chatter`+`sync` left
  `flags_by_check` and mean score (97.13) byte-for-byte identical to the baseline (neither
  check fires on this dataset) and left no `sync_*`/`gripper_*` metrics in the report. On
  masato-ka/so100_cutlery_handling_simple (50 ep): disabling `duplicate`+`length` dropped
  the 2 `length` flags from `flags_by_check` (`idle: 52, length: 2` -> `idle: 52`) and raised
  mean score from 92.92 to 93.32, with `duplicate_groups` empty either way (none to
  suppress) — confirms the flag removed is exactly the one disabled, nothing else moves.
- 2026-10-16: Added **`--episodes SPEC`/`--max-episodes N`** to `demoqc score`/`demoqc
  dedupe` (the roadmap's "scale to large datasets" slice 1), so trying the tool on a huge
  dataset doesn't require loading all of it first. Validated on
  masato-ka/so100_cutlery_handling_simple (50 ep, v2.1): a full `score` downloads all 54
  files (588 KiB, ~24s on this run); `--episodes 0-2` downloads only 4 meta files + 3
  episode parquet files (204 KiB, ~1.7s) and scores exactly episodes 0-2. Confirmed the
  score/metrics for those 3 episodes match a full-dataset load almost exactly (idle penalty
  6.13 vs 6.07 for episode 0) — the tiny difference is `DatasetContext`'s robust stats
  (e.g. `idle_speed`) being computed over the loaded subset rather than all 50 episodes,
  which is the documented "thresholds are relative to the dataset" behavior, not a bug;
  noted as a caveat in the README. Also validated on lerobot/svla_so101_pickplace (50 ep,
  v3.0, single data file since the dataset is smaller than one chunk): `--episodes 0-2`
  loads exactly 3 episodes; narrowing is a no-op here since there's only one file to begin
  with, which is the expected fallback, not a bug. New tests in `tests/test_dataset.py`
  corrupt the data file(s) a requested `--episodes` subset doesn't need (both v3.0
  multi-file and v2.1 per-episode-file layouts) and confirm loading still succeeds with
  exactly the requested episodes, proving those files are never opened; plus CLI spec
  parsing, `--max-episodes`, combining both, and an unknown-episode error. 10 of the 14 new
  tests fail with `TypeError`/`SystemExit` without this change (confirmed via `git stash`
  on just the implementation files).
- 2026-10-17: Added the **`camera` check** (video checks slice 1), flagging a camera
  that's near-black for an entire episode using the per-episode pixel mean/std LeRobot
  already writes to its own metadata (`meta/episodes` on v3.0, `meta/episodes_stats.jsonl`
  on v2.x) — no video file is ever downloaded or opened, so this needed no new dependency
  and the `push-report`/"benchmark" items below stayed next in line, not this. `HF_TOKEN`
  is still unset in this environment, so `push-report` stays blocked. Validated on
  lerobot/svla_so101_pickplace (50 ep, v3.0, 2 cameras), masato-ka/so100_cutlery_handling_simple
  (50 ep, v2.1, 1 camera), aaronsu11/so100_lego (100 ep, v2.1, 2 cameras) and shylee/so100_cube
  (200 ep, v2.1, 3 cameras): per-episode pixel mean/std load correctly from both metadata
  formats (confirmed by printing `Episode.camera_stats`) and no episode on any of the four
  is flagged — all real cameras here are well above the near-black threshold (lowest
  real per-episode mean seen: 0.45; lowest std: 0.14), and mean scores are unchanged from
  before this change (97.1 / 92.9 / 92.2 / 99.2). New `tests/test_checks.py::test_camera_dead`
  (parametrized over v3.0 and v2.1) injects a near-black `camera_dead` episode via a new
  `EpisodeSpec` field and `tests/conftest.py` now also writes `meta/episodes_stats.jsonl`
  for synthetic v2.x datasets (it previously wrote none, so this check's v2.x path was
  untestable); both parametrizations fail with `AssertionError` before this change. Also
  fixed a bug caught by the v3.0 test while building this: parquet round-trips the nested
  per-channel video stats as a ragged (non-rectangular) array, which `np.asarray` rejects
  with `ValueError`, so the first version of `_image_stat_scalar` silently returned `None`
  for every v3.0 episode; it now flattens the nesting recursively instead.
- 2026-10-18: Added `scripts/benchmark.py` and `docs/benchmarks.md` (the "benchmark the
  community" item). Scores a curated list of real datasets and renders a Markdown table
  (episodes, frames, mean/median score, trimmable idle, duplicate groups, top flags);
  each dataset is scored independently and a failure becomes an error row instead of
  aborting the run. Ran it end to end against all 6 datasets in `DEFAULT_REPOS`
  (lerobot/svla_so101_pickplace, lerobot/pusht, masato-ka/so100_cutlery_handling_simple,
  aaronsu11/so100_lego, shylee/so100_cube, jccj/so100_block_in_cup): all 6 scored with no
  errors and no new flags beyond what earlier validation runs already found (mean scores
  97.1/100.0/92.9/92.2/99.2/95.2), confirming the script reproduces the existing,
  already-validated numbers rather than introducing new false positives. `tests/test_benchmark.py`
  covers `score_repo` on synthetic local datasets (clean, `--max-episodes`-capped, a real
  injected idle defect, and a missing/unreadable dataset producing an error row instead of
  raising) and `format_benchmark_markdown` on both a normal and an error row, all against
  the existing `EpisodeSpec` fixtures — no network needed for the test suite.
- 2026-10-19: Added **v2.x export** (`demoqc export`'s highest-priority follow-up: most
  community datasets, including every so100 one used for validation, are v2.1, and export
  previously rejected them outright). `src/demoqc/export.py` now dispatches on
  `codebase_version`: `_export_v3` is the existing (refactored, behaviour-unchanged) v3.0 path,
  and the new `_export_v2` handles v2.x's different layout (one parquet per episode, JSONL
  episode/stats metadata instead of `meta/episodes` parquet columns) — renumbering episodes,
  rewriting `meta/episodes.jsonl` and `meta/episodes_stats.jsonl` (including each episode's own
  `index`/`episode_index` stats, which describe the new numbering), updating `meta/info.json`
  totals (`total_chunks`, `total_videos` included), and recomputing `meta/stats.json` (exact for
  tabular features, count-weighted combination of per-episode stats for video features, same as
  v3.0) when the source has one — some real datasets ship none, which export now detects and
  leaves a warning for instead of guessing. The stats-combination math was extracted into a
  shared `_combine_from_arrays` helper used by both formats, so the v3.0 behaviour is provably
  unchanged (all pre-existing v3.0 export tests still pass unmodified). This is slice 1 only: no
  `--apply-trim` yet (v2.x needs its own trim implementation; passing `--apply-trim` on a v2.x
  source now errors out clearly instead of silently misbehaving) — tracked as a new follow-up
  above. Also surfaced and documented a real correctness gap videos never need: a v2.x video
  file is named after its episode's index directly, so renumbering episodes invalidates
  `videos/`'s filenames (unlike v3.0, where chunk/file indices in `meta/episodes` still point at
  the right file); `demoqc export` now prints a warning with the exact renaming rule instead of
  the (wrong, for v2.x) "copy videos/ unchanged" message. Validated on real datasets: scoring
  and exporting masato-ka/so100_cutlery_handling_simple (50 ep, `--min-score 90`, 45 kept, no
  `meta/stats.json` in the source — export correctly warns and writes none) and re-scoring the
  export reproduced the original per-episode scores almost exactly (max diff 0.07, from
  `DatasetContext`'s robust stats over a smaller episode set, the same documented caveat as
  `--episodes`); aaronsu11/so100_lego (100 ep, `--min-score 95`, 27 kept, this dataset *does*
  ship `meta/stats.json`, with `mean`/`std`/`min`/`max`/`q01`/`q99` but no `count` key — a real-
  world stats-key variant the synthetic fixtures didn't cover, confirmed it recomputes
  correctly anyway since exact tabular recompute doesn't need `count`) — the recomputed
  `action` mean/min/max matched a direct recomputation from the exported frames, `index`/
  `episode_index` maxed out at the correct new values (14985 / 26), and `info.json`'s
  `total_videos` (54 = 27 kept episodes * 2 cameras) and `total_chunks` (1) were both right;
  also ran on shylee/so100_cube (200 ep, `--min-score 97`, 191 kept). New
  `tests/test_export.py` cases (mirroring the existing v3.0 coverage) fail without this change;
  `tests/conftest.py`'s synthetic v2.x writer also had its `data_path`/`video_path` templates
  fixed to the real v2.x convention (`episode_chunk`/`episode_index` placeholders; it had been
  silently reusing v3.0's template, untested until this export path needed it).
- 2026-10-20: Added **`--min-score N`** to `demoqc export` as an alternative to `--keep-from`
  (an export follow-up: previously every export needed a separate `demoqc score --json` run
  just to produce the keep list). `--keep-from`/`--min-score` are now a mutually exclusive,
  required argparse group; `--min-score` scores the dataset in-process at default thresholds
  (`evaluate` + duplicate detection + `build_report`, the same path `demoqc score` uses) and
  feeds the resulting keep/trim lists straight into `export_dataset`, reusing the existing
  report-derived logic via two new pure helpers (`keep_from_report_dict`/`trim_from_report_dict`
  in `src/demoqc/export.py`) that the file-based `keep_from_report`/`trim_from_report` now also
  call. `push-report` stayed blocked again this run (`HF_TOKEN` is still unset in this
  environment) even though the Hugging Face Hub itself was reachable. Validated on
  masato-ka/so100_cutlery_handling_simple (50 ep, v2.1): `export --min-score 90` and
  `export --keep-from <score --json output> --min-score 90` produced byte-for-byte identical
  output directories (`diff -rq`), both dropping the same 5 episodes. Also validated on
  lerobot/svla_so101_pickplace (50 ep, v3.0) with `--apply-trim` added: both paths trimmed the
  same 1411 idle frames and again produced byte-identical directories. New tests in
  `tests/test_export.py` (`test_export_command_min_score_matches_a_separate_score_json_run`,
  `test_export_command_requires_exactly_one_keep_source`) fail without this change (confirmed
  via `git stash` on just `src/demoqc/cli.py`/`src/demoqc/export.py`).
- 2026-10-21: Added **v2.x `--apply-trim`** (the export follow-up: v2.x is most of the real
  validation set, and `--apply-trim` previously errored out on it unconditionally, so no v2.x
  export could ever cut idle frames). `_export_v2` now accepts the same `trim` mapping as
  `_export_v3`: for each kept episode it slices the parquet rows to the suggested `[start, end)`
  range, renumbers `frame_index`/`index` and shifts `timestamp` by `start`, and rewrites
  `meta/episodes.jsonl`'s `length` to the trimmed count (this now also always matches the data
  even when a trim isn't applied, the same correctness fix `--apply-trim` made for v3.0's
  `meta/episodes` `length` on 2026-10-10). Unlike v3.0, a v2.x video file has no per-episode
  `from_timestamp`/`to_timestamp` span in its metadata to narrow -- it's simply the whole
  original episode -- so trimming only the tabular data would leave a trimmed export's (still
  uncut) video longer than its data; `demoqc export` now appends a warning to the existing
  video-renaming one spelling this out, naming the `trim` frame range from the report as what
  the video itself must be cut to. Validated on two real v2.x datasets:
  masato-ka/so100_cutlery_handling_simple (50 ep, 47/50 episodes with a suggested trim) --
  `--apply-trim` dropped total frames from 29853 to 25052 (4801 trimmed, 160.0s at 30 fps), and
  re-scoring the export gave mean score 100.0 with 0 remaining trimmable idle and no new flags;
  aaronsu11/so100_lego (100 ep, `--min-score 95`, 27 kept) -- trimmed 978 idle frames (32.6s),
  re-scoring the export gave mean score 99.75 (up from the dataset's usual ~92, per the
  2026-10-19 Log entry) with 0 remaining trimmable idle. Also re-ran the existing v3.0
  `--apply-trim` validation on lerobot/svla_so101_pickplace (50 ep) to confirm no regression:
  still 1411 frames / 47.0s trimmed, byte-identical to the 2026-10-10 numbers. New tests in
  `tests/test_export.py` (`test_v2_apply_trim_cuts_frames_shifts_timestamps_and_updates_length`,
  `test_v2_apply_trim_rejects_out_of_range_trims`, `test_v2_apply_trim_for_a_dropped_episode_is_ignored`,
  `test_v2_export_command_apply_trim_warns_about_cutting_videos`) replace the old
  "rejected with a clear message" tests and fail without this change (confirmed via `git stash`
  on just `src/demoqc/export.py`/`src/demoqc/cli.py`).
- 2026-10-22: Added **`--episodes`/`--max-episodes` to `demoqc export`** (the "scale to large
  datasets" follow-up: previously only `score`/`dedupe` could narrow a huge dataset, so trying
  `export` on one meant downloading and scoring every episode first). `cli.py`'s export
  subparser now calls the existing `_add_episode_args` helper; `_load()` already threaded
  `episodes`/`max_episodes` through to `load_dataset` for every subcommand, so no change was
  needed in `export.py` itself — episodes outside the requested subset are simply never loaded,
  so neither `--min-score` scoring nor a `--keep-from` report can keep them, and `export_dataset`
  then runs against whatever got downloaded/read for that subset. Validated on real datasets:
  `demoqc export masato-ka/so100_cutlery_handling_simple --episodes 0-4 --min-score 0 -o out`
  (v2.1) downloaded only 9 files (5 episode parquets + metadata) instead of all 54 and wrote
  exactly 5 episodes / 2981 frames, matching a direct check of the source; also ran on
  `lerobot/svla_so101_pickplace --episodes 0-9 --min-score 80` (v3.0, single data file, so no
  download savings there since everything lives in one chunk — the documented best-effort
  fallback, not a bug) and got 10 kept episodes as expected. New tests in `tests/test_export.py`
  cover `--episodes`/`--max-episodes` restricting the keepable set even with a permissive
  `--min-score`, a `--keep-from` report rejected when it doesn't match the `--episodes` subset,
  and (mirroring `test_dataset.py`'s narrowing tests) a v2.x export succeeding when the data
  files for episodes outside `--episodes` are missing entirely, simulating a Hub download that
  was itself narrowed; all 4 fail with `SystemExit`/"unrecognized arguments" before this change.
- 2026-10-23: Added **`demoqc export --copy-videos`** (the export follow-up: previously every
  export left videos untouched and printed a warning telling the user to copy/rename them by
  hand, so a cleaned v2.x export wasn't directly trainable without a manual step). For v3.0,
  `export_dataset(..., copy_videos=True)` copies the source's `videos/` directory into the
  export unchanged (file indices stay valid, same as the existing manual instructions). For
  v2.x, it copies and renames each kept episode's video file (every camera) from
  `episode_<old index>.mp4` to `episode_<new index>.mp4` using `meta/info.json`'s `video_path`
  template, replacing the old "copy this by hand" warning entirely; a trimmed v2.x export still
  warns that the copied video's content itself (not just its filename) must be cut to the
  `trim` range, since this tool copies bytes but never decodes or re-encodes video. The CLI
  rejects `--copy-videos` for a Hub source up front (`ds.repo_id is not None`), since videos are
  never downloaded by this tool and there would be nothing on disk to copy. Video bytes are
  always copied as-is (`shutil.copyfile`/`copytree`), so this needed no new dependency and
  stays in the core path. Validated on two real datasets: `masato-ka/so100_cutlery_handling_simple`
  (50 ep, v2.1, `--min-score 90` dropping 5) — a script comparing every copied, renamed video's
  bytes against the original file for its old index confirmed all 45 kept videos matched
  exactly and the 5 dropped episodes' videos were not carried over under any name; and
  `lerobot/svla_so101_pickplace` (50 ep, v3.0, `--min-score 80`, keeping all 50) — `diff -rq`
  between the source and exported `videos/` directories reported no differences. New tests in
  `tests/test_export.py` (8 cases covering the v3.0 whole-directory copy, the v2.x rename
  mapping, a missing video file producing a warning instead of a crash, both CLI paths, the
  `--apply-trim` + `--copy-videos` interaction, and the Hub-source rejection) all fail without
  this change (confirmed via `git stash` on just `src/demoqc/export.py`/`src/demoqc/cli.py`).
- 2026-10-24: Added **`demoqc export --symlink-videos`** (the `--copy-videos` follow-up: on a
  large dataset, `--copy-videos` doubles the on-disk video size, which can be prohibitive). Does
  the same job as `--copy-videos` (copy the whole `videos/` directory for v3.0 since it's
  unmodified by export; copy-and-rename each kept episode's video for v2.x) but with symlinks
  (`Path.symlink_to`, resolved to the source's absolute real path) instead of byte copies, so it
  uses no extra disk space. Mutually exclusive with `--copy-videos` at the CLI (argparse
  mutually-exclusive group) and defensively in `export_dataset` itself (`ValueError` if both are
  passed to the library function directly). Added `ExportResult.videos_symlinked` alongside the
  existing `videos_copied`, and parallel warnings (missing video file, trimmed-but-uncut v2.x
  video content) worded for symlinks. Validated on two real datasets with their actual videos
  downloaded locally (`snapshot_download`, not just the metadata+parquet this tool normally
  fetches): `masato-ka/so100_cutlery_handling_simple` (50 ep, v2.1, `--min-score 90` dropping 5)
  — `--symlink-videos` produced a 1.8 MiB export vs. 325 MiB for the same export with
  `--copy-videos`, and a script comparing every one of the 45 kept, renamed videos' bytes against
  the `--copy-videos` output found zero mismatches, confirming the symlinks resolve to the right
  (renamed) content; and `lerobot/svla_so101_pickplace` (50 ep, v3.0, `--min-score 90
  --apply-trim`) — the single `videos/` symlink resolves to the source's `videos/` directory
  exactly (`Path.resolve()` equality) and the export is 448 KiB total vs. the source's own 28 KiB
  `du` for `videos/` alone (no extra video bytes either way). New tests in `tests/test_export.py`
  (9 cases mirroring the existing `--copy-videos` coverage, plus mutual-exclusion checks at both
  the library and CLI level) fail without this change — 8/9 with `SystemExit`/`AttributeError`,
  the 9th incidentally passing either way since an unrecognized flag and a mutual-exclusion error
  both exit 2 (confirmed via `git stash` on just `src/demoqc/export.py`/`src/demoqc/cli.py`).
- 2026-10-25: Fixed a near-duplicate-detection memory bug (the `duplicate` check's own
  follow-up, which claimed the pairwise distance computation was "fine up to a few
  thousand episodes" -- it undercounted by 2-3 orders of magnitude). `find_duplicates`
  built an `(n, n, D)` broadcast intermediate (`mat[:, None] - mat[None, :]`) to get the
  `(n, n)` distance matrix, where `D` is the resample length times active action dims
  (hundreds, typically); at a "few thousand episodes" that intermediate alone is tens of
  GB, long before the actual `(n, n)` result is a problem, so `demoqc score`/`demoqc dedupe`
  would OOM on any dataset big enough for this tool's own `--episodes`/`--max-episodes`
  support to matter. `dedupe.py` now has `_pairwise_mean_sq_l2`, which computes the same
  values via the Gram-matrix identity `||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b` (one BLAS
  matrix multiply), cutting peak memory from O(n²D) to O(n²+nD) with no change to the
  result. New `tests/test_dedupe.py` cases: one checks the new helper against a brute-force
  broadcast on a small matrix (values match to 1e-6; the tiny diagonal float-cancellation
  noise never matters since the caller overwrites the diagonal with `inf` right after); the
  other uses `tracemalloc` to assert peak memory for a 300x300 input stays under 20 MB --
  confirmed the naive broadcast actually peaks at ~413 MB for the same input (measured
  directly, not estimated), and the test fails both that way and via `ImportError` on the
  unmodified module (confirmed via `git stash` on just `src/demoqc/dedupe.py`). Validated on
  all 4 real datasets in the existing validation set (lerobot/svla_so101_pickplace,
  masato-ka/so100_cutlery_handling_simple, aaronsu11/so100_lego, shylee/so100_cube): mean
  scores (97.1/92.9/92.2/99.2) and `duplicate_groups` (0 on all 4) are byte-for-byte
  unchanged from the existing validated numbers, confirming the optimization changed no
  output.
- 2026-10-26: Added the **`video_freeze` check** (`--decode-video`, the `camera` check's
  highest-priority follow-up: it only ever reads LeRobot's own per-episode pixel mean/std,
  so a camera that goes black or a video that fails to decode *partway through* an episode
  -- a loose cable, a brief occlusion, a corrupted download -- averages out across the whole
  episode and is invisible to it). New optional `demoqc[video]` extra (`av`/PyAV, lazily
  imported only when `--decode-video` is passed and never touched by the default code
  path) and `src/demoqc/video.py`: samples up to 20 evenly-spaced frames per camera,
  downscaled to grayscale, and flags a contiguous near-black run covering >=15% of the
  samples (`video_dark_run_min_frac` in `Config`), or a file that raises on decode at all,
  as a decode error. Skips a camera the (cheaper, metadata-only) `camera` check already
  flagged dead for the whole episode, so the same defect isn't penalized twice. Scoped to
  **local LeRobot v2.x datasets only** this round -- a v2.x video's path is deterministic
  from its episode index (`Dataset.video_path_template`/`chunks_size`, now loaded from
  `info.json`), while v3.0 needs its own per-camera chunk/file-index lookup, not yet
  implemented (new follow-up on the `camera` check item above). The CLI rejects
  `--decode-video` for a Hub source (videos are still never downloaded, even with this
  flag) and for a missing `av` install, and warns and no-ops on a v3.0 dataset rather than
  erroring. Validated on a real dataset: downloaded `masato-ka/so100_cutlery_handling_simple`
  episodes 0-4's actual video files (`av1`-encoded, decoded fine via `libdav1d`) and ran
  `demoqc score --episodes 0-4 --decode-video`; the report was byte-for-byte identical to
  the same run without `--decode-video` (mean score 90.4 either way) -- no false positives
  on real, clean footage. Also truncated one real episode's video file to 2000 bytes to
  simulate a corrupted download: `video_freeze` fired on exactly that episode (ERROR
  severity, "failed to decode") and every other episode scored unaffected, confirming a
  bad video file doesn't abort the whole `score` run. New `tests/test_video.py` (13 cases):
  pure-Python tests of `longest_dark_run`/`resolve_v2_video_path` run in CI with no
  dependency; `av`-dependent tests (real tiny synthetic `.mp4`s, skipped via
  `importorskip("av")` when the extra isn't installed, which is always true in CI since
  `.[dev]` doesn't pull it in) cover a mid-episode dark run, an undecodable file, the
  check staying silent by default (`decode_video=False`), not double-flagging a
  `camera`-caught episode, and the CLI's three validation paths (Hub rejection, `av`
  missing, v3.0 warning). All fail without this change (confirmed via `git stash` on the
  implementation files, including an `ImportError` for the module itself).
- 2026-10-27: Added **LeRobot v3.0 support to `video_freeze`** (`--decode-video`'s
  highest-priority follow-up from yesterday: v3.0 datasets, including the project's own
  reference dataset, couldn't use the check at all). Unlike v2.x (one video file per
  episode, resolved directly from the episode index), a v3.0 video file holds several
  consecutive episodes' footage back to back, addressed by the `chunk_index`/`file_index`
  each episode's `meta/episodes` row names per camera (`videos/{camera}/chunk_index`/
  `file_index`, now loaded onto `Episode.video_chunk_file` in `dataset.py`) and resolved to
  a path with the new `video.resolve_v3_video_path`. `video.decode_samples` gained an
  optional `time_range` to narrow decoding to one episode's own
  `from_timestamp`/`to_timestamp` span within that shared file (seeking near the start
  first) instead of decoding the whole thing; `check_video_freeze` picks this path whenever
  an episode has a `video_chunk_file` entry for a camera, falling back to the existing v2.x
  per-episode-file path otherwise. The CLI's old "v3.0 not supported yet" warning is gone;
  `DatasetContext.build` now sets `video_root` for any local dataset, not just v2.x.
  Validated on a real dataset with its actual videos downloaded locally
  (`snapshot_download`, not just the metadata+parquet this tool normally fetches):
  `lerobot/svla_so101_pickplace` (50 ep, v3.0, 2 cameras, one shared video file per camera
  holding all 50 episodes) — `demoqc score --decode-video` produced a report byte-for-byte
  identical to the same run without `--decode-video` (mean score 97.13 either way, 0
  `video_freeze` flags) in 59s, confirming no false positives on real, clean footage and
  that the new per-episode seek-and-decode path doesn't change anything when there's
  nothing to find. Then truncated one camera's shared video file to 1/3 of its original
  byte length to simulate a corrupted/incomplete download: `video_freeze` fired a decode
  error on exactly the 33 episodes (17-49) whose `from_timestamp` falls past the truncation
  point, scoring each down by the check's fixed 40-point penalty, while all 17 earlier
  episodes sharing that same (now-truncated) file scored byte-for-byte identical to the
  clean baseline — confirming the per-episode time window is resolved and decoded
  independently, a corrupted tail doesn't propagate backward to episodes whose footage is
  still intact in the same file, and a bad shared file still doesn't abort the whole `score`
  run. New `tests/test_video.py` cases (`resolve_v3_video_path` pure-path tests,
  `decode_samples`'s `time_range` narrowing on a real tiny synthetic multi-segment `.mp4`,
  and a full two-episode-sharing-one-file `check_video_freeze` integration test) all fail
  without this change (confirmed via `git stash` on the implementation files -- the
  integration test fails with `AttributeError` since `Episode` had no `video_chunk_file`
  yet, and the CLI no-video-files test fails because the old code still printed and exited
  via the removed v3.0 warning path).
- 2026-10-28: Added **frozen-frame detection to `video_freeze`** (the `camera`/`video_freeze`
  check's last open follow-up: a camera or recorder that stalls and repeats the same frame
  without ever going dark -- a dropped/duplicated frame, not a disconnect -- previously had
  normal-looking pixel mean/std and no partial dark run, so it was invisible to every
  existing check). `video.FrameSample` now also carries a 32x32 block-averaged thumbnail of
  each sampled frame; `video.longest_frozen_run` finds the longest run of consecutive
  samples whose largest per-block difference stays below `Config.video_freeze_max_abs_diff`
  (0.03). `check_video_freeze` only flags a frozen run when the dataset's own action data
  shows the robot was *continuously* commanding motion throughout that span (the same
  "clearly moving" bar `check_staleness` uses via the new `_episode_moved` helper): a frozen
  frame during genuine idle time (the lead/trail idle almost every demo has) looks identical
  and isn't a defect on its own, so flagging it regardless would mean flagging ordinary,
  well-compressed footage. Dark and frozen detection are mutually exclusive per
  camera/episode (frozen is only checked when no dark run fired) so the same incident is
  never flagged twice.
  Validating this on real data caught two real false-positive bugs before they shipped, both
  found by running `--decode-video` against `masato-ka/so100_cutlery_handling_simple` (the
  dataset this project already uses for calibration) and diffing against the same run
  without the flag, exactly as this file asks: (1) the first version used mean squared
  thumbnail difference on an 8x8 thumbnail, which flagged 4 of the first 5 episodes --
  a camera positioned so the moving robot occupies only a small part of the frame makes the
  *average* pixel change negligible over a couple of seconds even with real, confirmed motion
  (checked directly: the `max abs diff` on the actual frames was 0.97-1.0, while the mean was
  as low as 7e-7). Switched to the *maximum* per-block difference on a 32x32 thumbnail
  (small moving regions still move at least one block noticeably; real validation found the
  smallest real-motion max-diff at this resolution was 0.058, comfortably above the 0.03
  threshold). (2) Even after that fix, one episode still false-positived: the frozen-run
  window (coarse, only ~3 of the 20 sampled frames) happened to span an episode's
  idle-to-moving transition, and checking the *max* speed anywhere in that window caught the
  tail end just crossing into motion even though the window was almost entirely idle.
  Switched `_episode_moved` to the *minimum* speed in the window, requiring motion for the
  *entire* span, not just somewhere in it -- re-validated and the false positive was gone
  with no loss of detection on the synthetic defect test. Final validation, after both
  fixes: `masato-ka/so100_cutlery_handling_simple` (20 episodes, their actual downloaded
  video files) and `aaronsu11/so100_lego` (10 episodes) both produced byte-for-byte identical
  reports with and without `--decode-video` (mean scores 94.59 and 88.6 unchanged); re-ran
  the full `lerobot/svla_so101_pickplace` (50 ep, v3.0, 2 cameras) validation from the
  2026-10-27 Log entry and got the same byte-for-byte identical report (mean 97.13) and the
  same 0 `video_freeze` flags on this clean reference dataset, confirming no regression to
  the existing dark-run/decode-error detection. Runtime on that dataset rose from the
  2026-10-27 baseline's 59s to 140s (`--decode-video` is opt-in and already the slow path;
  `_thumbnail` is a vectorized `np.add.reduceat` block-reduce, not a per-cell Python loop,
  to keep this overhead down). New `tests/test_video.py` cases: pure tests of
  `longest_frozen_run` (partial run, too-short run, no run, a whole-clip run, and the
  no-thumbnail fallback), an integration test with a real synthetic video frozen mid-episode
  while the (synthetic) robot keeps moving throughout (flags `video_freeze` as an error), and
  one with the same frozen frame placed during the episode's idle lead-in instead (stays
  silent) -- proving the false-positive guard this whole investigation was about. All of the
  pure tests and the moving-while-frozen integration test fail without this change (confirmed
  via `git stash` on the implementation files); the idle-guard test passes either way by
  design, since it's asserting the *absence* of a flag. Existing tests' "clean" synthetic
  video fixtures (constant solid-colour frames, fine when only pixel mean/std mattered) had
  to gain frame-to-frame variation (`_varying`, a 7-frame period chosen to not alias with
  `decode_samples`' sampling stride) so they aren't themselves mistaken for a frozen feed by
  the new detection -- real footage is never pixel-for-pixel static, so this is a fixture gap
  this change exposed, not new unrelated noise.
- 2026-10-29: Added the **`integrity` check** and fixed a real silent-corruption bug found
  while reviewing `checks.py` (the "Data-integrity check" item above). `DatasetContext.build`
  used plain `np.percentile`/`.std` to compute the dataset-wide `action_scale`; both
  propagate `NaN` across an entire column when even one value in it is non-finite, so a
  single corrupted frame in one episode's `action` (sensor glitch, truncated write) silently
  set that whole dimension's scale to `NaN` for *every other* episode too, not just the bad
  one -- quietly disabling `idle`/`smoothness` on that dimension and, worst, `gripper_chatter`
  (needs a finite scale, watches only one or two dims) dataset-wide, with nothing in the
  report pointing at why. Switched to `np.nanpercentile`/`np.nanstd` and added `integrity`
  (flags the actual corrupted episode: ERROR, non-finite value count per stream) so the real
  defect is visible on its own rather than only as unexplained collateral damage elsewhere;
  also made `dedupe._trimmed_action` skip a non-finite trajectory instead of feeding `NaN`
  into `np.interp`. New `tests/test_checks.py` cases (`integrity` on an injected NaN in
  `action`, separately in `observation.state`, plus a regression test reproducing the exact
  cross-episode `gripper_chatter` blinding above) all fail without this change (confirmed via
  `git stash` on `src/demoqc/checks.py`). Validated on real data: `lerobot/svla_so101_pickplace`
  (50 ep), `aaronsu11/so100_lego` (100 ep) and `shylee/so100_cube` (200 ep) all reproduced
  their existing validated mean scores and flag counts exactly (97.1 / 92.2 / 99.2) with zero
  `integrity` flags, confirming this is a no-op on real, clean data.
