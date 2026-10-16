import csv
import json

import pytest

from demoqc.cli import main
from tests.conftest import EpisodeSpec, clean_specs


def test_score_writes_reports(make_dataset, tmp_path, capsys):
    specs = clean_specs()
    specs[0] = EpisodeSpec(lead_idle_s=2.0, spikes_at=[150], seed=0)
    root = make_dataset(specs)
    out_json, out_csv = tmp_path / "r.json", tmp_path / "r.csv"

    assert main(["score", str(root), "--json", str(out_json), "--csv", str(out_csv)]) == 0

    report = json.loads(out_json.read_text())
    assert report["schema_version"] == 1
    assert report["dataset"]["episodes"] == len(specs)
    worst = min(report["episodes"], key=lambda e: e["score"])
    assert worst["episode_index"] == 0
    assert worst["trim"] is not None
    assert 0 in report["summary"]["keep"] or worst["score"] < 70

    rows = list(csv.DictReader(out_csv.open()))
    assert len(rows) == len(specs)
    assert "ep     0" in capsys.readouterr().out


def test_score_writes_markdown(make_dataset, tmp_path):
    specs = clean_specs()
    specs[0] = EpisodeSpec(lead_idle_s=2.0, spikes_at=[150], seed=0)
    root = make_dataset(specs)
    out_md = tmp_path / "r.md"

    assert main(["score", str(root), "--md", str(out_md), "--quiet"]) == 0

    md = out_md.read_text()
    assert md.startswith("### demoqc report:")
    assert "| mean score |" in md
    assert "idle" in md


def test_fail_under(make_dataset):
    root = make_dataset(clean_specs())
    assert main(["score", str(root), "--quiet", "--fail-under", "50"]) == 0
    assert main(["score", str(root), "--quiet", "--fail-under", "101"]) == 1


def test_bad_source(tmp_path, capsys):
    assert main(["score", str(tmp_path)]) == 2
    assert "info.json" in capsys.readouterr().err


def test_score_reports_duplicates(make_dataset, tmp_path, capsys):
    specs = clean_specs()
    specs[9] = specs[2]  # byte-identical re-run
    root = make_dataset(specs)
    out_json = tmp_path / "r.json"

    assert main(["score", str(root), "--json", str(out_json), "--quiet"]) == 0

    report = json.loads(out_json.read_text())
    groups = report["summary"]["duplicate_groups"]
    assert len(groups) == 1 and groups[0]["kind"] == "exact"
    dup = next(e for e in report["episodes"] if e["episode_index"] == 9)
    assert any(f["check"] == "duplicate" for f in dup["flags"])


def test_disable_check_duplicate_suppresses_group_and_flags(make_dataset, tmp_path, capsys):
    specs = clean_specs()
    specs[9] = specs[2]  # byte-identical re-run
    root = make_dataset(specs)
    out_json = tmp_path / "r.json"

    assert (
        main(
            [
                "score",
                str(root),
                "--disable-check",
                "duplicate",
                "--json",
                str(out_json),
                "--quiet",
            ]
        )
        == 0
    )

    report = json.loads(out_json.read_text())
    assert report["summary"]["duplicate_groups"] == []
    dup = next(e for e in report["episodes"] if e["episode_index"] == 9)
    assert all(f["check"] != "duplicate" for f in dup["flags"])


def test_disable_check_rejects_unknown_name(make_dataset, capsys):
    root = make_dataset(clean_specs())
    with pytest.raises(SystemExit) as exc:
        main(["score", str(root), "--disable-check", "not_a_check"])
    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_dedupe_command(make_dataset, tmp_path, capsys):
    specs = clean_specs()
    specs[9] = specs[2]
    root = make_dataset(specs)
    out_json = tmp_path / "dups.json"

    assert main(["dedupe", str(root), "--json", str(out_json)]) == 0

    assert "exact duplicates" in capsys.readouterr().out
    groups = json.loads(out_json.read_text())
    assert len(groups) == 1
    assert sorted(groups[0]["episodes"]) == [2, 9]


def test_score_episodes_spec_restricts_dataset(make_dataset, tmp_path):
    root = make_dataset(clean_specs(6))
    out_json = tmp_path / "r.json"

    assert (
        main(["score", str(root), "--episodes", "1,4-5", "--json", str(out_json), "--quiet"]) == 0
    )

    report = json.loads(out_json.read_text())
    assert sorted(e["episode_index"] for e in report["episodes"]) == [1, 4, 5]
    assert report["dataset"]["episodes"] == 3


def test_score_max_episodes(make_dataset, tmp_path):
    root = make_dataset(clean_specs(6))
    out_json = tmp_path / "r.json"

    assert (
        main(["score", str(root), "--max-episodes", "2", "--json", str(out_json), "--quiet"]) == 0
    )

    report = json.loads(out_json.read_text())
    assert sorted(e["episode_index"] for e in report["episodes"]) == [0, 1]


def test_score_episodes_rejects_bad_spec(make_dataset, capsys):
    root = make_dataset(clean_specs())
    with pytest.raises(SystemExit) as exc:
        main(["score", str(root), "--episodes", "not-a-spec"])
    assert exc.value.code == 2
    assert "--episodes" in capsys.readouterr().err


def test_score_episodes_rejects_unknown_episode(make_dataset, capsys):
    root = make_dataset(clean_specs(3))
    assert main(["score", str(root), "--episodes", "99", "--quiet"]) == 2
    assert "99" in capsys.readouterr().err


def test_dedupe_episodes_spec(make_dataset, tmp_path):
    specs = clean_specs(6)
    specs[5] = specs[1]  # byte-identical re-run
    root = make_dataset(specs)
    out_json = tmp_path / "dups.json"

    assert main(["dedupe", str(root), "--episodes", "0-3", "--json", str(out_json)]) == 0

    # the duplicate pair (1, 5) isn't fully in the loaded subset, so it isn't found
    groups = json.loads(out_json.read_text())
    assert groups == []
