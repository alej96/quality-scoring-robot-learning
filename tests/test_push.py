"""Tests for demoqc.push and the `demoqc push-report` CLI command.

There is no HF_TOKEN in CI or this development environment, so the Hub write path
(``HfApi.create_commit``) is exercised with a mock rather than a real Hub call.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from demoqc.checks import Config, DatasetContext, evaluate
from demoqc.cli import main
from demoqc.dataset import load_dataset
from demoqc.dedupe import find_duplicates, flag_duplicates
from demoqc.push import badge_color, badge_markdown, push_report
from demoqc.report import build_report
from tests.conftest import EpisodeSpec, clean_specs


def _make_report(make_dataset, specs=None, min_score=70.0):
    src = make_dataset(specs or clean_specs(6))
    ds = load_dataset(src)
    cfg = Config()
    results = evaluate(ds, cfg)
    ctx = DatasetContext.build(ds, cfg)
    groups = find_duplicates(ds, results, ctx, cfg)
    flag_duplicates(results, groups)
    return ds, build_report(ds, results, min_score=min_score, duplicate_groups=groups)


def test_badge_color_thresholds():
    assert badge_color(100.0) == "brightgreen"
    assert badge_color(90.0) == "brightgreen"
    assert badge_color(89.9) == "yellow"
    assert badge_color(70.0) == "yellow"
    assert badge_color(69.9) == "red"


def test_badge_markdown_links_to_the_uploaded_report(make_dataset):
    _, report = _make_report(make_dataset)
    badge = badge_markdown(report, "user/dataset", revision="main")
    assert f"{report['summary']['mean_score']:.1f}" in badge
    assert "img.shields.io/badge/" in badge
    assert "https://huggingface.co/datasets/user/dataset/blob/main/demoqc/report.md" in badge


def test_push_report_uploads_json_and_markdown_and_returns_pr_url(make_dataset, monkeypatch):
    _, report = _make_report(make_dataset)
    captured = {}

    def fake_create_commit(self, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(pr_url="https://huggingface.co/datasets/user/dataset/discussions/1")

    monkeypatch.setattr("huggingface_hub.HfApi.create_commit", fake_create_commit)

    pr_url = push_report("user/dataset", report, revision="main", top=5)

    assert pr_url == "https://huggingface.co/datasets/user/dataset/discussions/1"
    assert captured["repo_id"] == "user/dataset"
    assert captured["repo_type"] == "dataset"
    assert captured["create_pr"] is True
    assert captured["revision"] == "main"
    paths = {op.path_in_repo for op in captured["operations"]}
    assert paths == {"demoqc/report.json", "demoqc/report.md"}
    json_op = next(op for op in captured["operations"] if op.path_in_repo == "demoqc/report.json")
    assert (
        json.loads(json_op.path_or_fileobj.decode())["dataset"]["episodes"]
        == report["dataset"]["episodes"]
    )
    md_op = next(op for op in captured["operations"] if op.path_in_repo == "demoqc/report.md")
    assert b"demoqc report" in md_op.path_or_fileobj
    # the dataset-card badge snippet is included in the PR description for the owner to copy
    assert "img.shields.io/badge/" in captured["commit_description"]


def test_push_report_raises_if_no_pr_url_is_returned(make_dataset, monkeypatch):
    _, report = _make_report(make_dataset)
    monkeypatch.setattr(
        "huggingface_hub.HfApi.create_commit", lambda self, **kw: SimpleNamespace(pr_url=None)
    )
    with pytest.raises(RuntimeError, match="no pull request URL"):
        push_report("user/dataset", report)


def test_cli_push_report_dry_run_writes_files_without_a_hub_call(
    make_dataset, monkeypatch, tmp_path, capsys
):
    src = make_dataset(clean_specs(6))
    ds = load_dataset(src)
    ds.repo_id = "user/some-hub-dataset"  # simulate a dataset loaded from the Hub
    monkeypatch.setattr("demoqc.cli.load_dataset", lambda *a, **k: ds)
    # if the CLI tried to call the Hub despite --dry-run, this would raise instead of returning
    monkeypatch.setattr(
        "huggingface_hub.HfApi.create_commit",
        lambda self, **kw: (_ for _ in ()).throw(AssertionError("Hub should not be called")),
    )

    code = main(
        [
            "push-report",
            "user/some-hub-dataset",
            "--dry-run",
            "-o",
            str(tmp_path),
            "--min-score",
            "0",
        ]
    )

    assert code == 0
    report_json = tmp_path / "demoqc" / "report.json"
    report_md = tmp_path / "demoqc" / "report.md"
    assert report_json.exists()
    assert report_md.exists()
    data = json.loads(report_json.read_text())
    assert data["dataset"]["episodes"] == 6
    assert "demoqc report" in report_md.read_text()
    assert "wrote" in capsys.readouterr().out


def test_cli_push_report_opens_a_pull_request(make_dataset, monkeypatch, capsys):
    src = make_dataset(clean_specs(6))
    ds = load_dataset(src)
    ds.repo_id = "user/some-hub-dataset"
    monkeypatch.setattr("demoqc.cli.load_dataset", lambda *a, **k: ds)
    monkeypatch.setattr(
        "huggingface_hub.HfApi.create_commit",
        lambda self, **kw: SimpleNamespace(
            pr_url="https://huggingface.co/datasets/user/some-hub-dataset/discussions/3"
        ),
    )

    code = main(["push-report", "user/some-hub-dataset", "--min-score", "0"])

    assert code == 0
    assert "discussions/3" in capsys.readouterr().out


def test_cli_push_report_requires_a_hub_source(make_dataset, capsys):
    src = make_dataset(clean_specs(6))

    code = main(["push-report", str(src), "--min-score", "0"])

    assert code == 2
    assert "needs a Hub repo id" in capsys.readouterr().err


def test_cli_push_report_reports_a_clean_error_on_hub_failure(make_dataset, monkeypatch, capsys):
    src = make_dataset(clean_specs(6))
    ds = load_dataset(src)
    ds.repo_id = "user/some-hub-dataset"
    monkeypatch.setattr("demoqc.cli.load_dataset", lambda *a, **k: ds)

    def raise_no_token(self, **kw):
        raise RuntimeError("no token found")

    monkeypatch.setattr("huggingface_hub.HfApi.create_commit", raise_no_token)

    code = main(["push-report", "user/some-hub-dataset", "--min-score", "0"])

    assert code == 2
    assert "failed to push report" in capsys.readouterr().err


def test_push_report_badge_matches_dataset_score_color(make_dataset):
    bad_specs = clean_specs(6) + [
        EpisodeSpec(seconds=8.0, lead_idle_s=6.0, seed=99)  # mostly idle -> low score
    ]
    _, report = _make_report(make_dataset, specs=bad_specs, min_score=0.0)
    badge = badge_markdown(report, "user/dataset")
    mean = report["summary"]["mean_score"]
    assert badge_color(mean) in badge
