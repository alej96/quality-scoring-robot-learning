"""Config.from_file / with_overrides: TOML-driven thresholds and penalties."""

import pytest

from demoqc.checks import Config, evaluate
from demoqc.cli import main
from demoqc.dataset import load_dataset
from tests.conftest import EpisodeSpec, clean_specs


def test_with_overrides_changes_requested_fields_only():
    cfg = Config().with_overrides({"gripper_chatter_min_toggles": 2, "spike_factor": 3.0})
    assert cfg.gripper_chatter_min_toggles == 2
    assert cfg.spike_factor == 3.0
    assert cfg.idle_fraction_of_p75 == Config().idle_fraction_of_p75  # untouched default


def test_with_overrides_rejects_unknown_key():
    with pytest.raises(ValueError, match="unknown config key.*typo_field"):
        Config().with_overrides({"typo_field": 1})


def test_with_overrides_rejects_wrong_type():
    with pytest.raises(ValueError, match="gap_factor.*expected a number"):
        Config().with_overrides({"gap_factor": "big"})
    with pytest.raises(ValueError, match="dedupe_resample_len.*expected an integer"):
        Config().with_overrides({"dedupe_resample_len": 2.5})


def test_from_file_parses_toml(tmp_path):
    path = tmp_path / "demoqc.toml"
    path.write_text("gripper_chatter_min_toggles = 2\nmax_idle_fraction = 0.6\n")
    cfg = Config.from_file(path)
    assert cfg.gripper_chatter_min_toggles == 2
    assert cfg.max_idle_fraction == 0.6


def test_from_file_bad_toml_raises(tmp_path):
    path = tmp_path / "demoqc.toml"
    path.write_text("not valid toml {{{")
    with pytest.raises(ValueError, match="invalid TOML"):
        Config.from_file(path)


def test_config_changes_check_outcome(make_dataset):
    # a 3-toggle burst, below the default 4-toggle chatter threshold
    specs = clean_specs()
    specs[3] = EpisodeSpec(gripper_chatter_at=(100, 103), seed=3)
    root = make_dataset(specs)
    ds = load_dataset(root)

    default_flags = {f.check for f in evaluate(ds, Config())[3].flags}
    assert "gripper_chatter" not in default_flags

    tuned_flags = {f.check for f in evaluate(ds, Config(gripper_chatter_min_toggles=3))[3].flags}
    assert "gripper_chatter" in tuned_flags


def test_cli_score_accepts_config_file(make_dataset, tmp_path, capsys):
    specs = clean_specs()
    specs[3] = EpisodeSpec(gripper_chatter_at=(100, 103), seed=3)
    root = make_dataset(specs)
    config_path = tmp_path / "demoqc.toml"
    config_path.write_text("gripper_chatter_min_toggles = 3\n")

    assert main(["score", str(root), "--config", str(config_path), "--quiet"]) == 0


def test_cli_rejects_bad_config_file(make_dataset, tmp_path, capsys):
    root = make_dataset(clean_specs())
    config_path = tmp_path / "demoqc.toml"
    config_path.write_text("not_a_real_field = 1\n")

    assert main(["score", str(root), "--config", str(config_path), "--quiet"]) == 2
    assert "unknown config key" in capsys.readouterr().err
