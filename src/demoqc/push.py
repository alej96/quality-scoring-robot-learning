"""Push a demoqc quality report to the dataset's own Hugging Face Hub repo.

Scoring a dataset only helps its owner if they see the result. ``demoqc push-report
<repo_id>`` scores the dataset and opens a Hub pull request (``create_pr=True``, so nothing
lands on the default branch without the owner's review) adding ``demoqc/report.json`` and
``demoqc/report.md`` to the repo, with a dataset-card badge snippet in the PR description
the owner can paste into their ``README.md``.

This is the one place in demoqc that writes to the Hub instead of only reading from it, so
it needs a token with write access to ``repo_id`` (``HF_TOKEN``, a cached ``huggingface-cli
login``, or an explicit ``token``).
"""

from __future__ import annotations

import json

from demoqc.report import format_markdown

REPORT_DIR = "demoqc"


def badge_color(mean_score: float) -> str:
    """shields.io color name for a mean score: green >= 90, yellow >= 70, red otherwise."""
    if mean_score >= 90:
        return "brightgreen"
    if mean_score >= 70:
        return "yellow"
    return "red"


def badge_markdown(report: dict, repo_id: str, revision: str = "main") -> str:
    """A dataset-card badge linking to the uploaded report, for pasting into ``README.md``."""
    mean = report["summary"]["mean_score"]
    badge_url = f"https://img.shields.io/badge/demoqc%20score-{mean:.1f}-{badge_color(mean)}"
    report_url = f"https://huggingface.co/datasets/{repo_id}/blob/{revision}/{REPORT_DIR}/report.md"
    return f"[![demoqc score]({badge_url})]({report_url})"


def push_report(
    repo_id: str,
    report: dict,
    *,
    token: str | None = None,
    revision: str | None = None,
    top: int = 10,
) -> str:
    """Upload ``report`` (JSON) and its Markdown rendering to ``repo_id`` as a Hub PR.

    Returns the PR's URL.
    """
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=token)
    json_bytes = (json.dumps(report, indent=2) + "\n").encode()
    md_bytes = (format_markdown(report, top=top) + "\n").encode()
    operations = [
        CommitOperationAdd(path_in_repo=f"{REPORT_DIR}/report.json", path_or_fileobj=json_bytes),
        CommitOperationAdd(path_in_repo=f"{REPORT_DIR}/report.md", path_or_fileobj=md_bytes),
    ]
    mean = report["summary"]["mean_score"]
    badge = badge_markdown(report, repo_id, revision=revision or "main")
    commit_info = api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        revision=revision,
        create_pr=True,
        commit_message=f"Add demoqc quality report (mean score {mean:.1f})",
        commit_description=(
            "Automated quality report from "
            "[demoqc](https://github.com/alej96/quality-scoring-robot-learning): "
            f"mean score {mean:.1f}, {report['summary']['episodes_below_threshold']} "
            f"episode(s) below {report['summary']['min_score_threshold']:g}. See "
            f"`{REPORT_DIR}/report.md` in this PR for the full summary.\n\n"
            "Paste this into your dataset card to show the score:\n\n"
            f"{badge}"
        ),
    )
    if commit_info.pr_url is None:
        raise RuntimeError(
            f"{repo_id}: commit succeeded but no pull request URL was returned "
            "(create_pr=True should always produce one)"
        )
    return commit_info.pr_url
