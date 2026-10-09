import csv
import json

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


def test_fail_under(make_dataset):
    root = make_dataset(clean_specs())
    assert main(["score", str(root), "--quiet", "--fail-under", "50"]) == 0
    assert main(["score", str(root), "--quiet", "--fail-under", "101"]) == 1


def test_bad_source(tmp_path, capsys):
    assert main(["score", str(tmp_path)]) == 2
    assert "info.json" in capsys.readouterr().err
