# CLAUDE.md

Guidance for AI agents (including the daily autonomous maintainer) working in this repo.

## What this project is

`demoqc` scores the quality of robot-learning teleoperation datasets in LeRobot format.
It is a Python CLI plus Hugging Face Hub integration. Its users are people about to train
an imitation-learning policy on community data who want to know which episodes to trim,
drop or dedupe. Read README.md for features and ROADMAP.md for priorities.

## Layout

- `src/demoqc/dataset.py`: load local or Hub datasets (v2.x and v3.0), metadata and parquet only
- `src/demoqc/checks.py`: `Config`, `DatasetContext` (dataset-wide robust stats), one
  `check_*` function per check, `evaluate()`
- `src/demoqc/report.py`: JSON/CSV report (`SCHEMA_VERSION`) and terminal summary
- `src/demoqc/cli.py`: argparse CLI, one subcommand per feature
- `tests/conftest.py`: synthetic dataset builder with injectable defects (`EpisodeSpec`)

## Rules

- **Real problems only.** Every change must make the tool better at finding or fixing
  real defects in real datasets, or make it easier to use. No cosmetic churn, filler
  commits, or speculative abstractions.
- **Tests for every behavior change.** A new check needs a synthetic-defect test showing it
  fires on the defect and stays silent on clean episodes (see `tests/test_checks.py`).
  Make sure the test fails without your change.
- **Validate on real data.** When changing loading or scoring, run
  `demoqc score lerobot/svla_so101_pickplace` and at least one community dataset (e.g.
  `masato-ka/so100_cutlery_handling_simple`, `aaronsu11/so100_lego`, `shylee/so100_cube`)
  and note what changed in the commit body. Watch for false positives: a check that flags
  most episodes of a good dataset is miscalibrated.
- **Thresholds are relative to the dataset** (robust stats in `DatasetContext`) unless the
  defect is absolute (e.g. timestamps going backwards). Put new thresholds in `Config`.
- **Report schema.** Additive changes are fine. Breaking changes bump `SCHEMA_VERSION`.
- **Dependencies:** keep the core to numpy, pandas, pyarrow and huggingface_hub. Heavy
  things (video decoding, lerobot itself) go behind optional extras.
- Never modify `.github/workflows/`. Changes there block auto-merge and need a human.
- Never download videos in the default code path.

## Commands

```bash
uv venv && uv pip install -e ".[dev]"     # or: pip install -e ".[dev]"
ruff check . && ruff format --check . && pytest -q   # must pass before pushing
pytest -m network                          # optional, needs Hugging Face access
```

## Shipping

Work on a branch named `claude/<yyyy-mm-dd>-<slug>` and push it. The `auto-merge` workflow
runs CI and rebase-merges the branch into `main` when CI is green. Write commit subjects in
the imperative mood (under 72 characters) with a body explaining the problem and how you
verified the change. The first commit's subject becomes the PR title. Update ROADMAP.md
(check items off, add follow-ups, append to the Log) in the same branch.
