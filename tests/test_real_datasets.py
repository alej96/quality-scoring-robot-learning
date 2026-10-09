"""End-to-end checks against public Hub datasets. Run with: pytest -m network"""

import pytest

from demoqc.checks import evaluate
from demoqc.dataset import load_dataset

pytestmark = pytest.mark.network


def test_svla_so101_pickplace_v3():
    ds = load_dataset("lerobot/svla_so101_pickplace")
    assert ds.codebase_version == "v3.0"
    assert len(ds.episodes) == 50
    assert ds.total_frames == 11939
    results = evaluate(ds)
    # teleop recordings start and end at rest: most episodes should get a trim
    assert sum(r.trim is not None for r in results) >= 30
    assert all(r.score > 70 for r in results)


def test_community_v21_dataset():
    ds = load_dataset("masato-ka/so100_cutlery_handling_simple")
    assert ds.codebase_version == "v2.1"
    results = {r.episode_index: r for r in evaluate(ds)}
    # episode 37 has ~9s of trailing idle (recording left running)
    assert results[37].metrics["trail_idle_s"] > 7
