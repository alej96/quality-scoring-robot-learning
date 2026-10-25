import tracemalloc

import numpy as np

from demoqc.checks import Config, DatasetContext, evaluate
from demoqc.dataset import load_dataset
from demoqc.dedupe import _pairwise_mean_sq_l2, find_duplicates, flag_duplicates
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


def test_pairwise_mean_sq_l2_matches_brute_force():
    rng = np.random.default_rng(0)
    mat = rng.normal(size=(8, 5))
    brute = np.sqrt(np.mean((mat[:, None, :] - mat[None, :, :]) ** 2, axis=2))
    np.testing.assert_allclose(_pairwise_mean_sq_l2(mat), brute, atol=1e-6)


def test_pairwise_mean_sq_l2_does_not_materialize_an_n_by_n_by_d_tensor():
    # The dataset's near-duplicate check resamples each episode to a few hundred values
    # (dedupe_resample_len * active action dims), so for a few hundred episodes the naive
    # broadcast `mat[:, None] - mat[None, :]` intermediate is tens to hundreds of MB larger
    # than the (n, n) distance matrix it computes -- and is what actually limited this
    # check's dataset size, not the (n, n) matrix itself. n=300, d=300 here makes that
    # intermediate ~206 MB; the Gram-matrix formulation should need only a few MB.
    rng = np.random.default_rng(0)
    mat = rng.normal(size=(300, 300))

    tracemalloc.start()
    try:
        _pairwise_mean_sq_l2(mat)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert peak < 20 * 1024 * 1024  # a naive broadcast would need ~206 MB here
