from demoqc.checks import Config, DatasetContext, evaluate
from demoqc.dataset import load_dataset
from demoqc.dedupe import find_duplicates, flag_duplicates
from demoqc.report import build_report, format_markdown
from tests.conftest import EpisodeSpec, clean_specs


def _report(root, min_score=70.0):
    ds = load_dataset(str(root))
    cfg = Config()
    results = evaluate(ds, cfg)
    ctx = DatasetContext.build(ds, cfg)
    groups = find_duplicates(ds, results, ctx, cfg)
    flag_duplicates(results, groups)
    return build_report(ds, results, min_score=min_score, duplicate_groups=groups)


def test_format_markdown_has_tables_and_flags(make_dataset):
    specs = clean_specs()
    specs[0] = EpisodeSpec(lead_idle_s=2.0, spikes_at=[150], seed=0)
    root = make_dataset(specs)
    report = _report(root)

    md = format_markdown(report, top=5)

    assert md.startswith(f"### demoqc report: `{report['dataset']['source']}`")
    assert "| mean score |" in md
    assert "**Flags**" in md
    assert "idle" in md
    assert "| Episode | Score | Duration | Flags |" in md
    assert "| 0 |" in md


def test_format_markdown_escapes_pipes_in_flag_messages(make_dataset):
    specs = clean_specs()
    root = make_dataset(specs)
    report = _report(root)
    # inject a flag message containing a pipe to confirm the table cell stays intact
    report["episodes"][0]["flags"] = [
        {"check": "fake", "severity": "warn", "message": "a | b", "penalty": 1.0}
    ]
    report["episodes"][0]["score"] = 10.0

    md = format_markdown(report, top=3)

    row = next(line for line in md.splitlines() if line.startswith("| 0 |"))
    assert "a \\| b" in row
    assert row.count("|") == 6  # 4 column separators + 2 escaped pipes, no extra columns


def test_format_markdown_no_flags_omits_sections(make_dataset):
    root = make_dataset(clean_specs())
    report = _report(root)
    for e in report["episodes"]:
        e["flags"] = []
    report["summary"]["flags_by_check"] = {}
    report["summary"]["duplicate_groups"] = []

    md = format_markdown(report, top=5)

    assert "**Flags**" not in md
    assert "Lowest-scoring" not in md
    assert "Duplicates" not in md
