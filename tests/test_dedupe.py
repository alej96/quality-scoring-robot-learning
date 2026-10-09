from demoqc.checks import Config, DatasetContext, evaluate
from demoqc.dataset import load_dataset
from demoqc.dedupe import find_duplicates, flag_duplicates
from tests.conftest import EpisodeSpec, clean_specs


def _find(make_dataset, specs):
    ds = load_dataset(make_dataset(specs))
    cfg = Config()
    results = evaluate(ds, cfg)
    ctx = DatasetContext.build(ds, cfg)
    return results, find_duplicates(ds, results, ctx, cfg)


def test_clean_dataset_has_no_duplicates(make_dataset):
    _, groups = _find(make_dataset, clean_specs())
    assert groups == []


def test_exact_duplicate_detected(make_dataset):
    # a re-run upload appends a byte-identical copy of an existing episode
    specs = clean_specs()
    specs[9] = specs[2]
    _, groups = _find(make_dataset, specs)

    exact = [g for g in groups if g.kind == "exact"]
    assert len(exact) == 1
    assert set(exact[0].episodes) == {2, 9}
    assert exact[0].max_distance == 0.0


def test_near_duplicate_detected(make_dataset):
    # same demonstration (seed) repeated with a slightly different duration: same
    # trajectory shape, not byte-identical
    specs = clean_specs()
    specs[9] = EpisodeSpec(seed=2, seconds=specs[2].seconds + 0.1)
    _, groups = _find(make_dataset, specs)

    assert all(g.kind != "exact" for g in groups)
    near = [g for g in groups if g.kind == "near"]
    assert any(set(g.episodes) == {2, 9} for g in near)


def test_independent_episodes_are_not_near_duplicates(make_dataset):
    # distinct seeds (and so distinct trajectories) must never be grouped
    _, groups = _find(make_dataset, clean_specs(20))
    assert groups == []


def test_flag_duplicates_penalizes_all_but_the_kept_episode(make_dataset):
    specs = clean_specs()
    specs[9] = specs[2]
    results, groups = _find(make_dataset, specs)
    flag_duplicates(results, groups)

    by_idx = {r.episode_index: r for r in results}
    assert not any(f.check == "duplicate" for f in by_idx[2].flags)
    dup_flags = [f for f in by_idx[9].flags if f.check == "duplicate"]
    assert len(dup_flags) == 1
    assert dup_flags[0].severity == "error"
    assert by_idx[9].score < 60
