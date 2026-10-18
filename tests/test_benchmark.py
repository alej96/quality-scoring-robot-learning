import datetime as dt

from scripts.benchmark import BenchmarkRow, format_benchmark_markdown, score_repo
from tests.conftest import EpisodeSpec, clean_specs


def test_score_repo_local_dataset(make_dataset):
    # score_repo must work on a plain local path, not just a Hub repo id, since that's
    # what load_dataset dispatches to when the source isn't a Hub repo id
    root = make_dataset(clean_specs())
    row = score_repo(str(root), max_episodes=None)

    assert row.error is None
    assert row.episodes == 12
    assert row.mean_score > 95
    assert row.duplicate_groups == 0


def test_score_repo_max_episodes(make_dataset):
    root = make_dataset(clean_specs(n=20))
    row = score_repo(str(root), max_episodes=5)

    assert row.error is None
    assert row.episodes == 5


def test_score_repo_reports_error_without_raising(tmp_path):
    # a dataset that can't be loaded (missing meta/info.json) must come back as an error
    # row, not raise, so one broken dataset in the list doesn't stop the rest of the run
    missing = tmp_path / "not-a-dataset"
    missing.mkdir()
    row = score_repo(str(missing), max_episodes=None)

    assert row.error is not None
    assert row.episodes == 0


def test_score_repo_flags_a_real_defect(make_dataset):
    specs = clean_specs()
    specs[3] = EpisodeSpec(seconds=10, lead_idle_s=3.0, seed=3)
    root = make_dataset(specs)
    row = score_repo(str(root), max_episodes=None)

    assert row.error is None
    assert row.flags_by_check == {"idle": 1}


def test_format_benchmark_markdown_renders_both_kinds_of_row():
    rows = [
        BenchmarkRow(
            repo_id="good/dataset",
            episodes=50,
            frames=1000,
            mean_score=97.1,
            median_score=98.0,
            trimmable_idle_s=12.3,
            duplicate_groups=1,
            flags_by_check={"idle": 5},
        ),
        BenchmarkRow(repo_id="bad/dataset", error="FileNotFoundError: nope"),
    ]
    out = format_benchmark_markdown(rows, dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc), 200)

    assert "good/dataset" in out
    assert "| 50 | 1000 | 97.1 | 98.0 | 12.3s | 1 | idle x5 |" in out
    assert "bad/dataset" in out
    assert "error: FileNotFoundError: nope" in out
    assert "capped at 200 episodes/dataset" in out


def test_format_benchmark_markdown_no_flags_says_none():
    rows = [BenchmarkRow(repo_id="clean/dataset", episodes=1, frames=1, flags_by_check={})]
    out = format_benchmark_markdown(rows, dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc), None)

    assert "none |" in out
    assert "capped at" not in out
