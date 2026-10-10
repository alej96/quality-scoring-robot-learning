# demoqc: quality scoring for robot-learning datasets

Community teleoperation datasets are full of jerky demos, dropped frames, idle time,
bad syncs and near-duplicate episodes. Training on them wastes compute and teaches
policies to hesitate. `demoqc` scores every episode of a
[LeRobot](https://github.com/huggingface/lerobot)-format dataset, explains what is wrong,
and tells you what to trim or drop.

- Works on a local dataset directory or straight from the Hugging Face Hub
- Reads only metadata and parquet data; videos are never downloaded
- Supports LeRobot v2.x (per-episode files) and v3.0 (consolidated files)
- Thresholds adapt to each dataset (robust statistics across its episodes), so the same
  defaults work for an SO-101 arm at 30 fps and a 2-D PushT agent at 10 fps

## Install

```bash
pip install git+https://github.com/alej96/quality-scoring-robot-learning
```

## Usage

```bash
demoqc score lerobot/svla_so101_pickplace
demoqc score ./my_dataset --json report.json --csv report.csv --min-score 80
demoqc score user/dataset --fail-under 85   # non-zero exit for CI gates
demoqc dedupe user/dataset --json dupes.json  # exact and near-duplicate episodes

# write a cleaned copy (LeRobot v3.0) without the episodes below --min-score
demoqc score user/dataset --json report.json --min-score 80
demoqc export user/dataset --keep-from report.json -o ./clean_dataset

# also cut each kept episode's idle start/end frames
demoqc export user/dataset --keep-from report.json --apply-trim -o ./clean_dataset
```

Example (real output, abridged):

```
masato-ka/so100_cutlery_handling_simple  (LeRobot v2.1, so100, 30 fps)
50 episodes, 29853 frames
score: mean 93.3, median 94.1; 0 below 70
trimmable idle: 181.6s
flags: idle x52

lowest-scoring episodes (top 3):
  ep    37  score  80.8    19.9s
      [info] idle: 10.2s idle at start/end (lead 1.2s, trail 9.0s); trim to frames [28, 336)
      [warn] idle: idle 58% of the episode
```

The JSON report (`schema_version: 1`) contains, per episode, a 0-100 `score`, every
`flag` with its severity and penalty, raw `metrics`, and a suggested `trim` frame range.
`summary.keep` lists the episodes at or above `--min-score`.

## Exporting a cleaned dataset

`demoqc export <source> --keep-from report.json -o <dir>` writes a LeRobot **v3.0** dataset
that contains only the episodes in the report's `summary.keep`. It does what filtering rows
by hand gets wrong: episodes are renumbered `0..k-1`, the global `index` stays contiguous,
`meta/episodes` gets the new indices and `dataset_from_index`/`dataset_to_index` ranges,
`meta/info.json` totals and `splits` are updated, and `meta/stats.json` (and the per-episode
`stats/index` and `stats/episode_index` columns) are recomputed for the kept episodes, so the
normalisation statistics no longer include the episodes you dropped. Tabular features are
recomputed exactly; video features are combined from the per-episode stats.

Only tabular data and metadata are written; videos are never read or copied. Their time spans
and file indices are unchanged, so copy the source `videos/` directory into the output to
train on it. The report must come from the same dataset (episode and frame counts are
checked). v2.x export is not implemented yet.

`--apply-trim` additionally cuts each kept episode's idle start/end frames, using the same
`trim` range `demoqc score` suggests: `frame_index` and `index` are renumbered, `timestamp` is
shifted so the episode still starts near 0, and `meta/episodes`' `length` and each camera's
`videos/*/from_timestamp`/`to_timestamp` are narrowed by the same number of frames, so the
(uncopied, unmodified) video file's window for that episode still lines up with the trimmed
data. Episodes with no suggested trim are kept in full.

## What it checks

| Check | Detects | How |
|---|---|---|
| `timing` | dropped frames, non-monotonic or missing timestamps | timestamp gaps vs `1/fps`, `frame_index` continuity |
| `metadata` | episode length in metadata ≠ frames in data | `meta/episodes` vs data |
| `video_sync` | video segment duration ≠ data duration (v3.0) | `videos/*/from_timestamp`, `to_timestamp` |
| `idle` | dead time at start/end (trimmable), long pauses, mostly-idle demos | action speed below 10% of the dataset's 75th-percentile speed |
| `smoothness` | single-frame glitches, abrupt jumps, demos far jerkier than their peers | per-dimension step size vs dataset p99; log dimensionless jerk (LDLJ) robust z-score |
| `stale_state` | `observation.state` frozen while the action moves (stale sensor reads) | bit-identical consecutive states during motion |
| `length` | demos with far less (aborted?) or far more activity than peers | robust z-score of active (trimmed) duration |
| `sync` | bad action/state sync (USB latency spikes, recorder bugs) | cross-correlate per-dim velocities over ±0.5s; flag low peak correlation or a lag far from the dataset median |
| `duplicate` | exact (re-run upload) and near-duplicate (repeated demo) episodes | hash of raw action/state for exact matches; L2 distance of resampled, trimmed, scaled action trajectories vs. the dataset's own nearest-neighbour distance distribution for near matches |

The score starts at 100 and subtracts each flag's penalty. Info-level flags (such as
trimmable idle) cost little because they can be fixed automatically.

`demoqc score` includes a `duplicate_groups` summary and flags every episode in a group
except the lowest-indexed (kept) one. `demoqc dedupe <source>` runs duplicate detection on
its own and prints the groups (`--json` writes them out).

## Development

```bash
uv venv && uv pip install -e ".[dev]"
ruff check . && ruff format --check . && pytest -q
pytest -m network   # end-to-end against public Hub datasets
```

See [ROADMAP.md](ROADMAP.md) for what's next (applying trims on export, pushing reports to
the Hub, gripper chatter, configurable thresholds).

## License

MIT
