# demoqc: quality scoring for robot-learning datasets

Community teleoperation datasets are full of jerky demos, dropped frames, idle time,
bad syncs and near-duplicate episodes. Training on them wastes compute and teaches
policies to hesitate. `demoqc` scores every episode of a
[LeRobot](https://github.com/huggingface/lerobot)-format dataset, explains what is wrong,
and tells you what to trim or drop.

- Works on a local dataset directory or straight from the Hugging Face Hub
- Reads only metadata and parquet data; videos are never downloaded, except for the
  opt-in `--decode-video` check on an already-local dataset
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
demoqc score user/dataset --md report.md    # Markdown summary for a dataset card or PR
demoqc dedupe user/dataset --json dupes.json  # exact and near-duplicate episodes

# write a cleaned copy (LeRobot v3.0) without the episodes below --min-score
demoqc score user/dataset --json report.json --min-score 80
demoqc export user/dataset --keep-from report.json -o ./clean_dataset

# or skip the separate score --json run and score+export in one step
demoqc export user/dataset --min-score 80 -o ./clean_dataset

# also cut each kept episode's idle start/end frames
demoqc export user/dataset --keep-from report.json --apply-trim -o ./clean_dataset

# also copy (and, for v2.x, rename) the source's video files -- local datasets only
demoqc export ./my_dataset --min-score 80 --copy-videos -o ./clean_dataset

# or symlink them instead, to avoid duplicating video data on disk
demoqc export ./my_dataset --min-score 80 --symlink-videos -o ./clean_dataset
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

`--md report.md` writes the same summary as a Markdown file: score and flag tables, the
duplicate-group counts, and a table of the lowest-scoring episodes, ready to paste into a
Hugging Face dataset card or a GitHub/Hub pull request.

## Exporting a cleaned dataset

`demoqc export <source> --keep-from report.json -o <dir>` writes a LeRobot dataset (**v3.0**
or **v2.x**, matching the source) that contains only the episodes in the report's
`summary.keep`. Pass `--min-score N` instead of `--keep-from` to score the dataset in the
same command (at default thresholds) and keep everything scoring `N` or above, without a
separate `demoqc score --json` run first; `--keep-from` and `--min-score` are mutually
exclusive and exactly one is required. It does what filtering rows by hand gets wrong:
episodes are renumbered `0..k-1`, the global `index` stays contiguous, per-episode metadata
gets the new indices,
`meta/info.json` totals and `splits` are updated, and `meta/stats.json` (plus any per-episode
`index`/`episode_index` stats) is recomputed for the kept episodes, so the normalisation
statistics no longer include the episodes you dropped. Tabular features are recomputed
exactly; video features are combined from the per-episode stats, when the source has them
(some real datasets ship no `meta/stats.json` at all; export leaves a warning and writes none,
rather than guessing).

Only tabular data and metadata are written by default; videos are never read or copied. For
**v3.0**, their time spans and file indices are unchanged, so copy the source `videos/`
directory into the output to train on it. For **v2.x**, a video file is named after its
episode's index directly (`episode_<n>.mp4`), so renumbering episodes means each kept
episode's video files (every camera) must be renamed from `episode_<old index>.mp4` to
`episode_<new index>.mp4` before training on the export; `demoqc export` prints a warning
with the exact rule (new index = position in `kept`) since it never touches the video files
itself. The report must come from the same dataset (episode and frame counts are checked).

Pass `--copy-videos` to have `demoqc export` do this itself for a **local** source: it copies
(and, for v2.x, renames) each video file next to the export, as raw bytes — never reading,
decoding or re-encoding them, so no video dependency is needed. It's a no-op (with a warning)
if the source has no `videos/` directory on disk, and the CLI rejects it for a Hub source
up front, since videos are never downloaded by this tool. With `--apply-trim`, a copied v3.0
video is already aligned (its per-episode time span was narrowed in the metadata); a copied
v2.x video still covers the whole original episode and must still be cut to the `trim` frame
range from the report, as `demoqc export` warns.

`--symlink-videos` does the same job (copy for v3.0, copy+rename for v2.x) but with symlinks
instead of byte copies: a single `videos/` symlink for v3.0 (its directory is unchanged, so one
link is enough), or one symlink per kept-and-renamed video file for v2.x. This uses no extra
disk space — useful on a large dataset where `--copy-videos` would double the video size on
disk — but makes the export depend on the source directory staying where it is; move or delete
the source and the exported videos stop resolving. `--copy-videos` and `--symlink-videos` are
mutually exclusive.

`--apply-trim` additionally cuts each kept episode's idle start/end frames, using the same
`trim` range `demoqc score` suggests: `frame_index` and `index` are renumbered and `timestamp`
is shifted so the episode still starts near 0. Episodes with no suggested trim are kept in
full. For **v3.0**, `meta/episodes`' `length` and each camera's
`videos/*/from_timestamp`/`to_timestamp` are also narrowed by the same number of frames, so the
(uncopied, unmodified) video file's window for that episode still lines up with the trimmed
data. For **v2.x**, `meta/episodes.jsonl`'s `length` is updated the same way, but a v2.x video
file has no per-episode time span stored in its metadata to narrow instead -- it simply covers
the whole original episode -- so `demoqc export` prints an additional warning: the video itself
must be cut to the same `trim` frame range (on top of being renamed) before training on the
export, or it will run longer than the now-trimmed data.

## What it checks

| Check | Detects | How |
|---|---|---|
| `timing` | dropped frames, non-monotonic or missing timestamps | timestamp gaps vs `1/fps`, `frame_index` continuity |
| `metadata` | episode length in metadata ≠ frames in data | `meta/episodes` vs data |
| `video_sync` | video segment duration ≠ data duration (v3.0) | `videos/*/from_timestamp`, `to_timestamp` |
| `camera` | a camera near-black for a whole episode (disconnected, lens cap) | per-episode pixel mean/std LeRobot already writes to its metadata; no video is ever opened |
| `video_freeze` | a camera going black, or a video file failing to decode, partway through an episode | decode sampled frames and look for a partial near-black run or a decode error; opt-in (`--decode-video`), see below |
| `idle` | dead time at start/end (trimmable), long pauses, mostly-idle demos | action speed below 10% of the dataset's 75th-percentile speed |
| `smoothness` | single-frame glitches, abrupt jumps, demos far jerkier than their peers | per-dimension step size vs dataset p99; log dimensionless jerk (LDLJ) robust z-score |
| `stale_state` | `observation.state` frozen while the action moves (stale sensor reads) | bit-identical consecutive states during motion |
| `length` | demos with far less (aborted?) or far more activity than peers | robust z-score of active (trimmed) duration |
| `sync` | bad action/state sync (USB latency spikes, recorder bugs) | match action/`observation.state` dims by feature name (falls back to position if either stream lacks names), cross-correlate per-dim velocities over ±0.5s; flag low peak correlation or a lag far from the dataset median |
| `duplicate` | exact (re-run upload) and near-duplicate (repeated demo) episodes | hash of raw action/state for exact matches; L2 distance of resampled, trimmed, scaled action trajectories vs. the dataset's own nearest-neighbour distance distribution for near matches |
| `gripper_chatter` | rapid open/close toggling (anxious squeeze, not a deliberate grasp) | find gripper dim(s) by feature name (`gripper` in `meta/info.json`'s `names`); hysteresis-debounced open/closed state, flag 4+ toggles within any 1s window |

The score starts at 100 and subtracts each flag's penalty. Info-level flags (such as
trimmable idle) cost little because they can be fixed automatically.

`demoqc score` includes a `duplicate_groups` summary and flags every episode in a group
except the lowest-indexed (kept) one. `demoqc dedupe <source>` runs duplicate detection on
its own and prints the groups (`--json` writes them out).

## Configurable thresholds

Every threshold and penalty is a field of `demoqc.checks.Config`
(`src/demoqc/checks.py`), calibrated on public SO-100/SO-101 datasets. A different
embodiment, fps or operator style can warrant different values, so `score` and `dedupe`
take `--config FILE.toml` to override any subset of them:

```toml
# demoqc.toml
max_idle_fraction = 0.6          # this robot's task legitimately has long idle holds
gripper_chatter_min_toggles = 6  # this gripper is driven by a noisy analog stick
```

```bash
demoqc score user/dataset --config demoqc.toml
```

Unknown keys or values of the wrong type are rejected with an error naming the key, so a
typo fails loudly instead of being silently ignored. Keys not listed in the file keep their
default.

To turn off a check entirely (e.g. a dataset with no gripper but a noisy `sync` check
from an unsynced third camera clock, or duplicate detection on a dataset where every demo
is intentionally similar), pass `--disable-check NAME` to `demoqc score` once per check:

```bash
demoqc score user/dataset --disable-check gripper_chatter --disable-check sync
```

Valid names are `timing`, `metadata`, `video_sync`, `camera`, `video_freeze`, `idle`,
`smoothness`, `stale_state`, `sync`, `gripper_chatter`, `length` and `duplicate`; an
unknown name is rejected up front.
Disabling a check removes both its flags and its metrics from the report, and (for
`duplicate`) skips `duplicate_groups` and the duplicate-detection pass entirely.

## Scoring a subset of episodes

`demoqc score`, `demoqc dedupe` and `demoqc export` take `--episodes SPEC` (e.g. `0-99` or
`5,10,20-30`, comma-separated indices and/or ranges) and `--max-episodes N` to load only some
of a dataset's episodes, for trying the tool on a huge dataset (DROID-scale and up) without
loading all of it:

```bash
demoqc score user/huge-dataset --episodes 0-199 --json report.json
demoqc score user/huge-dataset --max-episodes 500   # just the first 500
demoqc export user/huge-dataset --episodes 0-199 --min-score 80 -o preview  # export just these
```

On `export`, episodes outside `--episodes`/`--max-episodes` are dropped unconditionally, even
if `--min-score` or a `--keep-from` report would otherwise keep them (they were never loaded,
so there is nothing to keep); a `--keep-from` report must have been produced by a `demoqc score`
run over the same subset, or it's rejected as made for a different dataset.

From the Hugging Face Hub, only the data file(s) that actually contain the requested
episodes are downloaded (metadata is always small and downloaded in full); locally, data
files that contain none of the requested episodes are never opened. On a v2.x dataset
(one file per episode) this is close to proportional: scoring 3 of 50 episodes of
`masato-ka/so100_cutlery_handling_simple` downloads 3 episode files instead of 50. The
narrowing is best-effort and falls back to reading everything if a dataset's layout can't
be mapped from its metadata alone; either way the episodes loaded are exactly the ones
requested.

Checks and thresholds are relative to whatever is loaded (see "Configurable thresholds"
above), so a score or flag computed from a small `--episodes` subset is relative to that
subset, not the full dataset, and can differ slightly from a full-dataset run; checks that
need 5+ episodes (`length`, smoothness/sync outlier comparisons) are silently skipped
below that.

## Frame-level video checks (`--decode-video`)

Every other check reads only metadata and tabular data; videos are never downloaded or
opened by default. `demoqc score --decode-video` is the one opt-in exception: it decodes
each episode's video to catch a camera that goes black, or a video file that fails to
decode, **partway through** an episode -- a disconnected cable or a brief occlusion that
the metadata-only `camera` check can't see, since that check only has one pixel mean/std
for the *whole* episode.

```bash
pip install 'demoqc[video]'   # pulls in the optional `av` (PyAV) dependency
demoqc score ./my_dataset --decode-video
```

It needs a **local** dataset (LeRobot **v2.x or v3.0**) with its video files already on
disk (videos are still never downloaded from the Hub, even with this flag); scoring a Hub
repo id with `--decode-video` errors out, since there's no local source to read from. For
each camera, up to 20 evenly-spaced frames are sampled and downscaled to grayscale; for
v3.0, where several episodes share one video file, decoding is narrowed to that episode's
own `from_timestamp`/`to_timestamp` span first. A contiguous near-black run covering at
least 15% of the samples is flagged as `video_freeze`, and a video that can't be decoded at
all (corrupted or truncated) is flagged as a decode error. A camera already flagged dead
for the whole episode by the `camera` check is skipped, so the same defect isn't penalized
twice.

## Development

```bash
uv venv && uv pip install -e ".[dev]"
ruff check . && ruff format --check . && pytest -q
pytest -m network   # end-to-end against public Hub datasets
```

`scripts/benchmark.py` scores a curated list of real community datasets and writes
[docs/benchmarks.md](docs/benchmarks.md) (mean/median score, trimmable idle, duplicate
groups, top flags per dataset). Re-run it after changing a check or threshold and diff
the file against the previous commit to see exactly which real datasets moved, and by
how much — it doubles as a regression check for the heuristics:

```bash
python scripts/benchmark.py                        # default dataset list -> docs/benchmarks.md
python scripts/benchmark.py --repo user/dataset     # add/replace datasets (repeatable)
python scripts/benchmark.py --max-episodes 0        # disable the per-dataset episode cap
```

See [ROADMAP.md](ROADMAP.md) for what's next (pushing reports to the Hub, disabling
individual checks, scaling to large datasets).

## License

MIT
