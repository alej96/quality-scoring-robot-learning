# Roadmap

The objective: the tool a LeRobot user runs before training. It should **score, trim,
flag and dedupe** teleoperation episodes, export a cleaned dataset, and integrate with the
Hugging Face Hub. Every item must solve a real problem in real community datasets.

Work top-down through **Next up**. Each item should ship in one focused session
(implementation, tests and docs). Split big items into slices that are useful on their own.

## Next up

- [x] **Fix `demoqc dedupe`/`score` flagging nearly every episode of a dataset with no
      `action`/`observation.state` feature as an exact duplicate.** Found by running
      `demoqc score` against a real Hub dataset beyond the existing validation set
      (`yaak-ai/L2D`, a car-driving dataset found via `list_datasets(filter="LeRobot",
      sort="downloads")`): 14/15 episodes carried a `duplicate` "exact duplicate of episode
      0" ERROR (mean score 53.3), on a dataset whose other checks were completely silent.
      Root cause: this dataset's `meta/info.json` has no literal `action` or
      `observation.state` feature at all -- it stores a mixed continuous/discrete action
      and a structured per-signal state under dotted per-key names instead
      (`action.continuous`, `action.discrete`, `observation.state.vehicle`,
      `observation.state.waypoints`, etc., a representation `dataset.py`'s loader doesn't
      read), so `Episode.action`/`Episode.state` loaded as `None` for every single episode
      (confirmed directly: `load_dataset(...).episodes[i].action is None` for all `i`).
      `dedupe._content_hash` hashes whichever of `(ep.action, ep.state)` isn't `None` and
      skips the rest, so every episode with both `None` produced the exact same
      (effectively empty) SHA-1 digest -- `find_duplicates` then grouped all of them as one
      "exact duplicate" set purely because none of them had anything to compare, not
      because any content actually matched. Fixed by excluding an episode with both
      `action` and `state` `None` from the exact-duplicate hash grouping entirely
      (`find_duplicates` in `src/demoqc/dedupe.py`), and by adding a `Dataset.warnings`
      entry in `dataset.py` (`load_dataset`) when `meta/info.json` has no literal
      `action`/`observation.state` feature, naming the problem instead of leaving every
      check silently empty with no explanation. New
      `tests/test_dedupe.py::test_episodes_with_no_action_or_state_are_not_exact_duplicates`
      and `tests/test_dataset.py::test_structured_action_state_names_load_as_missing_with_a_warning`
      (a new `omit_action_state` flag on `tests/conftest.py`'s `write_dataset` writes no
      `action`/`observation.state` columns or features at all, reproducing the real
      dataset's schema shape); both fail without this change (confirmed via `git stash` on
      `src/demoqc/dedupe.py`/`dataset.py` -- the dedupe test asserts `[DuplicateGroup(...)]
      == []` and gets a 5-episode exact group, the dataset test asserts `ds.warnings == []`
      instead of the new message, reproducing the real dataset's false positive and silent
      cause exactly). Validated on the real dataset: `demoqc score yaak-ai/L2D
      --max-episodes 15` went from mean 53.3 (14/15 episodes, `duplicate` x14, no
      explanation printed anywhere) to mean 100.0 (0 flags, plus a `demoqc: warning: no
      action or observation.state feature in meta/info.json; ...` line so the all-clean
      score isn't mistaken for "nothing to find" rather than "nothing was checked"). Re-ran
      the full existing validation set (`lerobot/svla_so101_pickplace`,
      `masato-ka/so100_cutlery_handling_simple`, `aaronsu11/so100_lego`,
      `shylee/so100_cube`, `jccj/so100_block_in_cup`) and got byte-identical mean scores and
      flag counts to every prior Log entry (97.1 / 92.9 / 92.2 / 99.2 / 95.2), confirming
      this is a no-op whenever a dataset actually has `action`/`observation.state` data to
      hash.
      Follow-up (resolved 2026-11-14): `load_dataset` now reconstructs a flat `action`/
      `observation.state` array from a dataset's numeric dotted sub-features, so checks
      beyond `duplicate` get real signal too; see the Log entry.
- [x] **Fix `check_smoothness`/`check_gripper_chatter` not recognizing a VR-teleop
      controller's `trigger`/`squeeze` axis as the gripper-equivalent dim it is.** Found
      by running `demoqc score` against a real Hub dataset beyond the existing validation
      set (`BitRobot/HIW-500-LeRobot`, a Unitree G1 humanoid teleoperated via a VR
      controller, found via `list_datasets(filter="LeRobot", sort="downloads")`): 19/20
      episodes carried a `smoothness` "abrupt action jump(s)" flag (mean score 76.0, 2/20
      below 70), several with 20-140 "jumps". Per-dim inspection of the flagged frames
      found the cause: `right_trigger` (range 0..10) and `right_squeeze` (range 0..1) --
      the controller axes mapped to the hand's open/close command -- sit at a steady
      setpoint almost the whole episode and snap across their full range in 1-3 frames
      during a deliberate grasp/release, exactly the already-fixed literal-`*gripper*`-dim
      pattern (`_exclude_gripper_dims`/`check_gripper_chatter`), just under VR-controller
      naming instead of "gripper" -- so `gripper_dims`' `"gripper" in name.lower()` match
      never found them, and they were invisible to both the exclusion and to
      `gripper_chatter`'s own (correct) way of catching real anxious-squeeze toggling.
      Fixed by widening the name match to a shared `checks._GRIPPER_NAME_KEYWORDS =
      ("gripper", "trigger", "squeeze")` tuple, used everywhere `gripper_dims` is built.
      New `tests/test_checks.py::test_gripper_dims_matches_vr_controller_trigger_squeeze_names`
      (asserts `DatasetContext.build(...).gripper_dims` directly) and
      `test_smoothness_ignores_vr_controller_trigger_snap` (the full false-positive
      reproduction, reusing the existing `_gripper_snap_dataset` fixture with its last dim
      renamed to `"right_trigger"`) both fail without this change (confirmed via `git
      stash` on just `src/demoqc/checks.py` -- `gripper_dims == []` and every "clean"
      episode asserts `1 == 0` on `jumps`, reproducing the real dataset's false positives
      exactly). Validated on the real dataset: `demoqc score BitRobot/HIW-500-LeRobot
      --max-episodes 20` went from mean 76.0 (`smoothness` x27, 2/20 below 70) to mean 74.2
      (`smoothness` x23, `gripper_chatter` x6, 7/20 below 70) -- the mean score itself drops,
      but for the right reason: per-episode `jumps` counts on the affected dims fell
      sharply (e.g. one episode's right_trigger/right_squeeze jump count went from 68 to 0),
      confirming the spurious smoothness signal is gone, while `gripper_chatter` now
      correctly fires on genuine rapid full-range toggling on those same dims (confirmed by
      inspecting the raw trigger values at the flagged frames: repeated 0<->10 swings
      within a handful of frames, not sensor jitter) that was previously completely
      invisible to the tool -- a more accurate diagnosis, not a new false positive. Re-ran
      the full existing validation set (`lerobot/svla_so101_pickplace`,
      `masato-ka/so100_cutlery_handling_simple`, `aaronsu11/so100_lego`,
      `shylee/so100_cube`, `jccj/so100_block_in_cup`) and every gripper-bearing dataset from
      the earlier gripper-snap/chatter fixes (`RoboCOIN/Agilex_Cobot_Magic_heat_burger`,
      `_place_towel_flat`, `lerobot/aloha_static_cups_open`,
      `RoboCOIN/AI2_Alphabot_2_insert_water_pipe`, `lerobot/aloha_static_battery`,
      `zekaiwang/trex_dataset`, `allenai/MolmoAct2-BimanualYAM-Dataset`) via `git stash`
      before/after for an exact apples-to-apples diff: byte-identical mean scores and flag
      counts on all twelve, confirming this is a no-op whenever a dataset has no dim named
      with any of the three keywords.
- [x] **Fix `check_staleness` flagging every episode of a dataset whose state publishes
      slower than the logging fps.** Found by running `demoqc score` against a real Hub
      dataset beyond the existing validation set
      (`RoboCOIN/Galbot_G1_bag_table_trash_1`, a wheeled-base humanoid, found via
      `list_datasets(filter="LeRobot", sort="downloads")`): all 15/15 episodes carried a
      `stale_state` flag (mean score 77.3, 2 episodes below 70), on a dataset whose other
      checks (`idle`, `metadata`, `timing`) looked clean. Per-episode inspection of
      `observation.state` found the cause: every episode had a tight, consistent ~21.7-22.7%
      fraction of frames where all 38 action/state dims were simultaneously bit-identical to
      the previous frame while the commanded action kept moving -- confirmed not isolated to
      the chassis/base dims (which would suggest a slow base-odometry sensor specifically):
      every one of the 38 dims, including the actively-driven arm joints, showed a
      substantial per-dim freeze fraction (25-98%) individually, and the *combined* ~22%
      rate (all 38 dims frozen on the same frame) matches a state message published as one
      atomic unit at a rate below the 30 fps logging rate, with the last reading repeated to
      fill in-between frames -- a platform/recorder characteristic of this humanoid, not a
      localized stale-read bug. `check_staleness` compared this fraction against a fixed
      absolute `stale_state_max_fraction` (0.1), so the fraction being nearly identical on
      every episode (and far above 0.1) flagged the entire dataset -- the exact MAD/threshold
      -instability pattern the `length` and `ldlj_z` checks were already guarded against
      ("require a material difference from a dataset-relative baseline, not just a large
      value against a fixed cutoff"), just not extended to `stale_state`. Fixed by adding
      `DatasetContext.stale_state_baseline` (`checks.DatasetContext.build`, via the new
      `_stale_mask` helper shared with `check_staleness` itself): the dataset's own median
      per-episode stale fraction, computed the same way `check_staleness` already did per
      episode. `check_staleness` now flags only when an episode's own fraction exceeds that
      baseline by more than `stale_state_max_fraction`, instead of comparing the raw fraction
      to a fixed cutoff -- a no-op (baseline near 0) on every dataset in the existing
      validation set, where `excess` reduces to the old `frac`. New
      `tests/test_checks.py::test_stale_state_ignores_a_dataset_wide_low_rate_baseline` builds
      a 13-episode synthetic dataset by hand (12 "clean" episodes each with a uniform
      `state_hold_every=5` low-rate-publish pattern throughout, via a new
      `EpisodeSpec.state_hold_every` field in `tests/conftest.py`, plus one more episode with
      the same baseline pattern *and* an additional genuine localized stale burst layered on
      top) and asserts every baseline-only episode stays silent while the genuinely worse
      episode is still flagged; fails without this change (confirmed via `git stash` on just
      `src/demoqc/checks.py` -- episode 0 asserts `'stale_state' not in {'stale_state'}`,
      reproducing the real dataset's per-episode false positive exactly). Validated on the
      real dataset: `demoqc score RoboCOIN/Galbot_G1_bag_table_trash_1 --max-episodes 15`
      went from mean 77.3 (15/15 episodes, `stale_state` x15) to mean 97.8 (0 `stale_state`
      flags, only the pre-existing `smoothness` ones remain, unrelated to this check). Also
      confirmed two other real humanoid datasets from the same search
      (`RoboCOIN/Unitree_G1edu_u3_place_bread`, `RoboCOIN/Galaxea_R1_Lite_classify_object_green_tablecloth`)
      have a baseline of 0.0 (no inherent state-hold characteristic) and were already clean
      before this change, so this fix is specific to the one platform that actually exhibits
      the artifact, not a blanket loosening. Re-ran the full existing validation set
      (`lerobot/svla_so101_pickplace`, `masato-ka/so100_cutlery_handling_simple`,
      `aaronsu11/so100_lego`, `shylee/so100_cube`, `jccj/so100_block_in_cup`) and got
      byte-identical mean scores and flag counts to every prior Log entry (97.1 / 92.9 / 92.2
      / 99.2 / 95.2), confirming this is a no-op whenever a dataset has no dataset-wide
      stale-state baseline to begin with.
- [x] **Fix `check_smoothness` flagging an ordinary gripper grasp/release as an abrupt
      action jump.** Found by running `demoqc score` against a real Hub dataset beyond
      the existing validation set (`allenai/MolmoAct2-BimanualYAM-Dataset`, a bimanual
      YAM-arm manipulation dataset found via `list_datasets(filter="LeRobot", sort=
      "downloads")`): 13/20 episodes carried a `smoothness` "abrupt action jump(s)" flag
      (mean score 88.2), on a dataset whose other checks looked clean. Per-dim inspection
      of every flagged frame found the cause: every single one, with no exception, was on
      `left_gripper.pos` or `right_gripper.pos` (confirmed by printing the flagged
      dim/frame/raw-action directly) -- never on any of the 12 arm joint dims. A gripper
      sits at a steady open or closed setpoint for almost the whole episode and snaps
      across its full `0..1` range in 1-4 frames during a single, deliberate grasp or
      release; because that transition is brief and rare relative to an episode's total
      length (a couple of frames out of 1500-2100), the dataset-wide 99th-percentile step
      for the gripper dim reflects only its idle-frame jitter, not the transition --
      so *any* percentile-based "this step is an outlier" threshold ends up flagging every
      ordinary grasp, the exact opposite of a joint dim (continuously driven, so its own
      `step_p99` already reflects normal fast motion and a real outlier still stands out).
      This is a different root cause from the already-fixed noise-dominated-dim case
      (an unused torso/neck joint moved by nothing but sensor noise): the gripper dim here
      isn't a `step_p99` outlier among the dataset's other dims at all (its scale and
      `step_p99` sit in the same order of magnitude as every joint's), so
      `_exclude_noise_dominated_dims` doesn't catch it -- the problem is specific to how a
      near-binary actuator's *rare, large, deliberate* transitions interact with a
      percentile baseline, not an unusually small or noisy scale. Fixed by adding
      `checks._exclude_gripper_dims`, called from `DatasetContext.build` right after the
      existing noise-dominated-dim exclusion: every dim in `ctx.gripper_dims` (already
      detected by feature name, the same list `check_gripper_chatter` uses) has its
      `step_p99` set to NaN, excluding it from spike/jump detection via the same mechanism
      that already excludes a constant or noise-dominated dim; `check_gripper_chatter`
      itself is unaffected (it uses hysteresis-debounced toggle counting, not `step_p99`,
      and already owns detecting genuinely bad gripper behaviour like rapid toggling).
      New `tests/test_checks.py::test_smoothness_ignores_gripper_snap` builds a
      38-episode synthetic dataset by hand (36 "clean" episodes each with a normal
      single-frame full-range gripper snap plus five continuously-moving joint dims, one
      more episode with the same ordinary snap, and one episode with a genuine, unrelated
      jump on a joint dim) and asserts every snap-only episode stays silent while the
      genuine-jump episode is still flagged; fails without this change (confirmed via
      `git stash` on just `src/demoqc/checks.py` -- every "clean" episode asserts
      `1 == 0` on `jumps`, reproducing the real dataset's per-episode false positive
      exactly). Validated on the real dataset: `demoqc score
      allenai/MolmoAct2-BimanualYAM-Dataset --max-episodes 20` went from mean 88.2
      (13/20 episodes, `smoothness` x13) to mean 92.9 (0 `smoothness` flags, only the
      pre-existing `idle` ones remain). Re-ran the full existing validation set
      (`lerobot/svla_so101_pickplace`, `masato-ka/so100_cutlery_handling_simple`,
      `aaronsu11/so100_lego`, `shylee/so100_cube`, `jccj/so100_block_in_cup`) and got
      byte-identical mean scores and flag counts to every prior Log entry (97.1 / 92.9 /
      92.2 / 99.2 / 95.2) -- none of their gripper dims ever cross the spike threshold, so
      this is a no-op there. Also re-ran every dataset from the smoothness-fix series that
      has a gripper dim (`RoboCOIN/Agilex_Cobot_Magic_heat_burger`, `_place_towel_flat`,
      `lerobot/aloha_static_cups_open`, `RoboCOIN/AI2_Alphabot_2_insert_water_pipe`) plus
      the two without one (`lerobot/aloha_static_battery`, `zekaiwang/trex_dataset`): all
      six reproduced their existing validated mean scores exactly (94.1 / 93.1 / 100.0 /
      94.1 / 95.0 / 95.2); `heat_burger` had exactly one gripper-dim jump frame riding
      alongside several real Euler-angle jump frames in the same episode, and losing it
      didn't move that episode's score or flag count at all, confirming this only removes
      flags that were never real to begin with.
- [x] **Fix the "jerkier than peers" (`ldlj_z`) smoothness flag firing on an
      insignificant LDLJ gap when the peer group's spread happens to be tiny.** Found by
      running `demoqc score` against a real Hub dataset beyond the existing validation set
      (`EPITECH-LILLE/eval_training`, found via `list_datasets(filter="LeRobot", sort=
      "downloads")`): episode 37, scored with `--max-episodes 20`, carried a `smoothness`
      "much jerkier than peers (smoothness z=-3.6)" flag; the same episode, re-scored
      against the dataset's full 40 episodes instead, got z=-0.68 (not flagged) for an
      almost identical raw LDLJ value (-19.68 vs -19.65). The z-score, not the episode,
      had changed: most of this dataset's episodes are a close variation on the same
      motion, so `_robust_z`'s MAD reflects only that tight majority's own tiny spread
      (0.185 with 20 peers) even with a handful of episodes sitting notably smoother on
      the other side, inflating a moderate, probably-insignificant ~0.6-nat LDLJ gap from
      the median into an extreme, peer-group-dependent z-score -- the exact MAD-instability
      pattern `check_relative`'s own `length` check was already guarded against
      ("require a material relative difference, not just a large z on a tiny spread"),
      just not extended to `ldlj_z`. Fixed by requiring the LDLJ gap from the median to
      also exceed a new `Config.min_ldlj_diff` (1.0 nat, ~2.7x the underlying dimensionless
      jerk integral) before flagging -- a fixed gap in nats is a fixed *relative* change
      in the always-positive jerk integral regardless of the dataset's own absolute LDLJ
      range, so, unlike `length`'s check, this needed no dataset-relative floor. New
      `tests/test_checks.py::test_smoothness_jerkier_than_peers_needs_a_material_ldlj_gap`
      builds a 22-episode synthetic dataset by hand (a tight cluster of near-identical
      motions, a few notably smoother ones, one moderately jerkier episode that trips the
      raw z-score threshold with only a ~0.7-nat gap, one genuinely/drastically jerkier
      episode) and asserts the moderate episode stays silent while the drastic one is
      still flagged; fails without this change (confirmed via `git stash` on
      `src/demoqc/checks.py` -- the moderate episode asserts a `smoothness` flag is absent
      and gets one instead). Validated on the real dataset: `demoqc score
      EPITECH-LILLE/eval_training --max-episodes 20` went from mean 94.1 (1/20 flagged,
      the inconsistent z=-3.6 flag) to mean 99.3 (0 `smoothness` flags), now agreeing with
      the `--max-episodes 40` run's own 0 `smoothness` flags (previously 85.6 vs 98.5
      depending only on how many episodes happened to be loaded). Re-ran the full existing
      validation set (`lerobot/svla_so101_pickplace`, `masato-ka/so100_cutlery_handling_simple`,
      `aaronsu11/so100_lego`, `shylee/so100_cube`, `jccj/so100_block_in_cup`) and got
      byte-identical mean scores and flag counts to every prior Log entry (97.1 / 92.9 /
      92.2 / 99.2 / 95.2), including `shylee/so100_cube`'s and `jccj/so100_block_in_cup`'s
      own existing `ldlj_z` true positives (LDLJ gaps 1.4-2.7 nats, well clear of the new
      1.0-nat floor) -- confirming this only removes flags whose own gap is too small to
      be material, not the real ones.
- [x] **Fix `check_smoothness` flagging a noise-dominated dim (e.g. an unused torso/neck
      joint) as abrupt action jumps.** Found by running `demoqc score` against a real Hub
      dataset beyond the existing validation set
      (`RoboCOIN/AI2_Alphabot_2_insert_water_pipe`, found via `list_datasets(filter=
      "LeRobot", sort="downloads")`): 19/20 episodes carried a `smoothness` "abrupt action
      jump(s)" flag (mean score 85.1, 1 episode below 70), several hitting the check's max
      20-point penalty, on a dataset whose other checks looked clean. Per-dim inspection of
      the flagged action dims (`torso_joint_1/2/3_rad`) found the root cause: this robot's
      torso barely moves at all through the whole task (dataset-wide range 0.06-0.08 rad,
      under 5 degrees, vs. 0.17+ rad/m for every genuinely driven arm/eef/gripper dim), so
      `step_p99` for those dims reflects nothing but sensor/controller noise around a fixed
      setpoint, not real motion -- and that noise, scattered as separate single-frame blips
      across almost every episode (not one sustained ramp, so the existing run-counting fix
      doesn't help: these aren't consecutive), each independently exceeded
      `spike_factor * step_p99` and was counted as its own "jump". A genuinely driven dim's
      `step_p99` (already normalised by the dim's own robust scale) clusters tightly with
      every other genuinely driven dim regardless of physical units, since it reflects
      control/teleop dynamics (how many frames a typical motion takes), not the unit
      itself. Fixed by adding `checks._exclude_noise_dominated_dims`, called from
      `DatasetContext.build` right after `step_p99` is computed: a dim is excluded from
      spike/jump detection (its `step_p99` set to NaN, reusing the same mechanism that
      already excludes a fully-constant dim) if `log(step_p99)` is more than
      `smoothness_dim_outlier_z` (4.0, new in `Config`) robust MADs from the *median of the
      dataset's other dims' own* `log(step_p99)` -- skipped entirely below 5 live dims,
      where a robust z-score isn't meaningful. New
      `tests/test_checks.py::test_smoothness_ignores_a_noise_dominated_dim` builds a
      41-episode synthetic dataset by hand (5 normal, actively-moving joint dims with a
      natural spread of `step_p99`, plus a 6th dim that sits almost frozen except for
      scattered single-frame blips in 6 of 40 "clean" episodes, plus one episode with a
      genuine jump on dim 1) and asserts every blip episode is silent while the genuine-jump
      episode still flags; fails without this change (confirmed via `git stash` on just
      `src/demoqc/checks.py` -- each blip episode asserts `1 == 0` on `jumps`, reproducing
      the real dataset's per-episode overcounting exactly). Validated on the real dataset:
      `demoqc score RoboCOIN/AI2_Alphabot_2_insert_water_pipe --max-episodes 20` went from
      mean 85.1 (1/20 below 70, `smoothness` x19 episodes) to mean 94.1 (0/20 below 70,
      `smoothness` x4 episodes) -- the 4 remaining `smoothness` episodes are a smaller,
      related residual on `torso_joint_4_rad`/`neck_joint_1_rad`/`neck_joint_2_rad` (the
      *high*-side mirror of this same defect: both their `scale` and `step_p99` are
      dominated by noise around a near-zero-range dim, but their outlier z-score, 2.5-3.0,
      falls short of the 4.0 threshold -- chosen conservatively so as not to risk excluding
      a real, legitimately low-`step_p99` dim elsewhere, see below), tracked as a follow-up.
      Re-ran the full existing validation set (`lerobot/svla_so101_pickplace`,
      `masato-ka/so100_cutlery_handling_simple`, `aaronsu11/so100_lego`,
      `shylee/so100_cube`, `jccj/so100_block_in_cup`), both ALOHA joint-space datasets
      (`lerobot/aloha_static_cups_open`, `lerobot/aloha_static_battery`), both Agilex
      datasets from the gimbal-lock fix's own validation
      (`RoboCOIN/Agilex_Cobot_Magic_heat_burger`, `_place_towel_flat`, via `git stash`
      before/after for an exact apples-to-apples diff since the full-dataset episode counts
      differ from their original Log entries) and the sustained-ramp fix's own
      `zekaiwang/trex_dataset` (a 58-action-dim dexterous-hand dataset, the real test of
      this fix's safety: its dims naturally spread over a much wider `step_p99` range than
      a simple arm) and got byte-for-byte identical mean scores and flag counts to every
      prior Log entry on all seven -- confirming this is a no-op whenever a dataset's dims
      are all genuinely driven, including a dataset with 58 dims of naturally varied
      motion scale.
      Follow-up: `torso_joint_4_rad`/`neck_joint_1/2_rad` on the same real dataset are the
      same defect with a less extreme outlier z-score (2.5-3.0); lowering the threshold to
      catch them loses the safety margin against `trex_dataset`'s real dims (max z=2.94)
      and needs either a smarter combined signal or more real-dataset evidence first.
- [x] **Fix `check_smoothness` counting every frame of one sustained, smooth motion as a
      separate "abrupt action jump".** Found by running `demoqc score` against a real Hub
      dataset beyond the existing validation set (`zekaiwang/trex_dataset`, a dual-arm
      dexterous-hand (22-DOF-per-hand) manipulation dataset found via
      `list_datasets(filter="LeRobot", sort="downloads")`): 19/50 episodes carried a
      `smoothness` "abrupt action jump" flag (mean score 94.3), several reporting 10-32
      "jumps" each and hitting the check's max 20-point penalty, on a dataset whose other
      checks (`idle`, `metadata`, `timing`) looked clean. Per-episode inspection of the
      flagged action dims (e.g. `left_hand_target_q_21`) found the actual cause: a finger
      joint sits still for almost the whole episode (it only moves to grasp), so its
      dataset-wide `step_p99` baseline is tiny; printing the raw values around the flagged
      frames showed a perfectly smooth, monotonic, single-direction ramp (0.18 -> 0.69 over
      ~20 frames, no reversal, no discontinuity) -- a fast but genuinely smooth grasp-
      closing motion, not jerkiness. `check_smoothness` counted `big.any(axis=1).sum()`
      (the number of *frames* where any dim's step exceeds `spike_factor * step_p99`) as
      the number of "abrupt action jumps", with no notion that many *consecutive* big
      frames on the same dim, all stepping the same direction, are one continuous motion,
      not N separate discontinuities -- exactly the "demos far jerkier than their peers"
      detector overcounting a single fast-but-smooth event because the dim's "normal" step
      is tiny the rest of the time. Fixed by counting contiguous *runs* of leftover big
      frames (after excluding the two frames each already-detected spike consumes) via the
      existing `_runs` helper, instead of counting every big frame in the run separately;
      an isolated single-frame jump (one-frame run) is unaffected, two well-separated
      jumps still count as two, and a single sustained ramp now counts as the one event it
      is. New `tests/test_checks.py::test_smoothness_counts_a_sustained_ramp_as_one_jump`
      builds a 122-episode synthetic dataset by hand (120 clean episodes establishing a
      tiny dataset-wide step baseline for a "finger" dim, one smooth 21-frame ramp episode,
      one episode with two genuinely separate single-frame jumps) and asserts the ramp
      episode gets exactly 1 jump (not 20) while the two-jump episode still gets 2 and
      every clean episode stays silent; fails without this change (confirmed via
      `git stash` on just `src/demoqc/checks.py` -- the ramp episode asserts `20 == 1`,
      reproducing the real dataset's per-frame overcounting exactly). Validated on the real
      dataset: `demoqc score zekaiwang/trex_dataset --max-episodes 50` went from mean 94.3
      (`smoothness` jump counts up to 32 per episode) to mean 95.2 (same 19 episodes still
      flagged -- this fix doesn't claim the remaining, smaller jump counts are false
      positives, only that a single continuous motion shouldn't be counted once per frame
      it spans -- but with accurate, much smaller counts, e.g. episode 12's 32 "jumps"
      -> 8 real contiguous runs). Re-ran the full existing validation set
      (`lerobot/svla_so101_pickplace`, `masato-ka/so100_cutlery_handling_simple`,
      `aaronsu11/so100_lego`, `shylee/so100_cube`, `jccj/so100_block_in_cup`) and both ALOHA
      joint-space datasets (`lerobot/aloha_static_cups_open`, `lerobot/aloha_static_battery`)
      and got byte-for-byte identical mean scores and flag counts to every prior Log entry
      on all but `aloha_static_battery`: there, the flag *count* stayed at 46 (same 46
      frames/episodes implicated) but mean score rose from 93.3 to 95.0 on the full
      49-episode dataset, since several of its real jump events also span multiple
      consecutive frames and were previously over-penalized by the same per-frame
      overcounting this fix corrects -- a strictly more accurate severity, not a different
      diagnosis.
- [x] **Fix `check_smoothness` flagging an Euler-angle gimbal-lock flip as a jump.** Found
      by running `demoqc score` against real Hub datasets beyond the existing validation
      set, in the same family as (and one episode of) the dataset the wrapped-angle fix
      above was found on: `RoboCOIN/Agilex_Cobot_Magic_place_towel_flat` (mean score 90.8,
      3 episodes carrying a `smoothness` "abrupt action jump" flag) and
      `RoboCOIN/Agilex_Cobot_Magic_heat_burger` itself, whose 1 residual `smoothness`
      episode this run's own Log entry below had assumed was "real jerkiness, not
      wrapping" -- it wasn't quite right either (see below). Both episodes'
      `left/right_eef_rot_euler_x/z_rad` dims -- already known to wrap, and already
      excluded from the wrap-crossing false positive by the fix above -- still swung by a
      huge, precisely correlated amount on the exact frame flagged: e.g.
      `right_eef_rot_euler_x_rad` jumped -0.97 rad while `right_eef_rot_euler_z_rad` jumped
      -0.93 rad in the very same single frame, with their *difference* (`x - z`) staying
      almost exactly constant (-0.584 before, -0.628 after -- a 0.044 rad drift versus each
      axis's own ~0.95 rad swing). That's the textbook signature of an Euler-angle
      gimbal-lock singularity (here, the un-exported pitch/`_y_rad` axis passing near its
      own +-pi/2): the end effector's real orientation barely changed, but the particular
      pair of axes that becomes degenerate at that singularity (`x - z`, or `x + z`
      depending on sign convention) swaps a huge, matched amount between them to represent
      the same orientation. `check_smoothness`'s existing per-dim wrap-aware diff has no
      notion of a *pair* of dims trading off against each other, so it read each axis's own
      swing as an independent "abrupt action jump", exactly like the already-fixed
      wrap-boundary case but for a different root cause. Fixed by adding
      `checks._angular_pair_axes` (`DatasetContext.build`): pairs of already-detected
      wrapped-angle dims that share a feature-name prefix differing only in the axis letter
      before `_rad` (e.g. `eef_rot_euler_x_rad`/`eef_rot_euler_z_rad`) are grouped as a
      "coupled rotation pair". `checks._suppress_gimbal_lock_pairs`, called from
      `check_smoothness` right after the existing spike/jump detection, then clears a
      flagged frame on *both* dims of a pair whenever the wrapped difference of `x - z` (or
      `x + z`, whichever is smaller -- checking both makes this robust to the convention's
      sign without needing to know it) is below `angular_pair_cancel_frac` (0.25, new in
      `Config`) of each axis's own raw step; a frame where only one paired axis jumps, or
      both jump by unrelated amounts, never satisfies this and stays flagged. This needs no
      assumption about Euler convention or axis order and no attempt to locate the pitch
      axis itself -- it's a direct test on the two dims that actually jump. New
      `tests/test_checks.py::test_smoothness_ignores_gimbal_lock_pair_cancellation` builds a
      small synthetic dataset by hand (paired `_x_rad`/`_z_rad` dims that both wrap
      dataset-wide, one episode where both jump by the *same* amount at the same frame, one
      where only one of them jumps) and asserts the matched-jump episode gets zero
      smoothness flags while the lone-jump episode still gets one; fails without this
      change (confirmed via `git stash` on just `src/demoqc/checks.py` -- the matched-jump
      episode asserts `1 == 0` on `jumps`, reproducing the real datasets exactly). Validated
      on the real datasets: `RoboCOIN/Agilex_Cobot_Magic_place_towel_flat` (15 ep) went from
      mean 90.8 (3 `smoothness` flags) to 93.1 (0 `smoothness` flags, only the pre-existing
      `idle` ones remain); `RoboCOIN/Agilex_Cobot_Magic_heat_burger` (20 ep, the wrapped-angle
      fix's own validation dataset) went from mean 88.9 (1/20 below 70, `smoothness` x9) to
      94.1 (0/20 below 70, `smoothness` x1) -- confirmed the one remaining flag is a genuine,
      unrelated jump on ordinary joint dims (`left_arm_joint_3/5/6_rad`, far outside this
      fix's scope), not another instance of this artifact. Also re-ran
      `lerobot/aloha_static_cups_open`/`_battery` (joint-space, no Euler-angle dims) and got
      byte-identical scores and flag counts to the existing validated numbers, confirming
      this is a no-op on datasets without a coupled rotation pair. Re-ran the full existing
      validation set (`lerobot/svla_so101_pickplace`, `masato-ka/so100_cutlery_handling_simple`,
      `aaronsu11/so100_lego`, `shylee/so100_cube`, `jccj/so100_block_in_cup`): byte-identical
      mean scores and flag counts to every prior Log entry (97.1 / 92.9 / 92.2 / 99.2 / 95.2),
      confirming no regression -- none of their action dims are Euler angles.
- [x] **Fix `check_smoothness` flagging normal wrapped-angle rotation as a jump/spike.**
      Found by running `demoqc score` against a real Hub dataset beyond the existing
      validation set (`RoboCOIN/Agilex_Cobot_Magic_heat_burger`, a dual-arm teleop
      dataset found via `list_datasets(filter="LeRobot", sort="downloads")`): nearly
      every episode (27/20 sampled, mean score 71.0, 6/20 below 70) carried multiple
      `smoothness` "abrupt action jump"/"single-frame action spike" flags, on a dataset
      whose other checks (`idle`, `metadata`, `timing`) looked clean. Per-dim inspection
      of the raw action (`action.shape == [26]`, names like `left_eef_rot_euler_x_rad`)
      found the actual cause: two of its 26 dims are Euler angles wrapped to `[-pi, pi]`
      (confirmed: `min`/`max` across episodes sit at -3.14/3.142, essentially exactly
      `+-pi`), and the robot's wrist rotates through the wrap boundary during perfectly
      smooth, continuous motion -- every other dim (joints in radians with a narrower
      range, positions in metres, a 0-1 gripper) never comes close to `+-pi` and has a
      per-frame step two to three orders of magnitude smaller (p99 0.03-0.18 rad vs. a
      ~6.28-rad wrap jump). `check_smoothness`'s plain `np.diff` has no notion of the
      dim being circular, so a wrap crossing -- functionally identical to a tiny real
      rotation -- read as a single-frame step tens of times the dataset's own
      `step_p99` for that dim, well past `spike_factor`'s 6x threshold. Fixed by adding
      `checks._angular_action_dims` (`DatasetContext.build`): a dim counts as a wrapped
      angle only if every value for it, across the whole dataset, stays within
      `+-(pi + angular_wrap_tol)` *and* it actually reaches within `angular_wrap_reach`
      of both `+pi` and `-pi` -- derived from the data, not the feature name, so a joint
      or position that merely happens to sit under `pi` (every dim in the existing
      validation set) never qualifies, and the two conditions together only ever match
      a dim that genuinely uses the full wrap range. `checks._angular_diff` then takes
      the equivalent diff in `(-pi, pi]` for those dims only, in both
      `DatasetContext.build`'s `step_p99` baseline and `check_smoothness` itself, so the
      wrap crossing is seen as the small real rotation it is -- a no-op for every other
      dim and every dataset without this representation. New
      `tests/test_checks.py::test_smoothness_ignores_wrapped_angle_crossings` builds a
      small synthetic dataset by hand (one angle dim that wraps across several episodes,
      one ordinary joint dim, one episode with a real, unrelated jump on the joint dim)
      and asserts the wrap episode gets zero smoothness flags while the real-jump episode
      still gets one; fails without this change (confirmed via `git stash` on just
      `src/demoqc/checks.py` -- the wrap episode gets 2 spurious jumps). A second new test,
      `test_angular_action_dims_requires_reaching_both_wrap_boundaries`, locks in the
      detection's three-way boundary logic directly. Validated on the real dataset:
      `demoqc score RoboCOIN/Agilex_Cobot_Magic_heat_burger --max-episodes 20` went from
      mean 71.0 (median 72.3, 6/20 below 70, `smoothness` x27) to mean 88.9 (median 92.1,
      1/20 below 70, `smoothness` x9) -- confirmed via `DatasetContext.build` that only the
      four actual `*_rot_euler_x/z_rad` dims (left and right arm) are detected as angular,
      not the 22 other action dims; the remaining 9 `smoothness` flags are on different
      episodes/dims (real jerkiness, not wrapping) and unaffected by this change. Also ran
      `lerobot/aloha_static_cups_open` (50 ep) and `lerobot/aloha_static_battery` (49 ep,
      joint-space action, no wrapped dims) to confirm a no-op on datasets without this
      representation: byte-identical mean scores (100.0, 93.3) and flag counts
      (`smoothness` x46 on `battery`, unchanged) before and after. Re-ran the full existing
      validation set (`lerobot/svla_so101_pickplace`, `masato-ka/so100_cutlery_handling_simple`,
      `aaronsu11/so100_lego`, `shylee/so100_cube`): byte-identical mean scores to every
      prior Log entry (97.1 / 92.9 / 92.2 / 99.2), confirming no regression -- none of
      their action dims are bounded anywhere near `+-pi`.
- [x] **Fix `check_idle` silently skipping 1- and 2-frame episodes.** Found by running
      `demoqc score` against a real Hub dataset beyond the existing validation set
      (`lerobot/svla_so100_stacking`): episode 20 (exactly 1 frame) scored 90.0 with only
      an unrelated "unusually short" `length` warning, while episodes 37, 39 and 55 --
      slightly longer (14, 434 and 141 frames) but with the exact same bit-identical,
      zero-motion action -- correctly scored 20.0-30.0 with `idle: no motion detected in
      the whole episode` (ERROR, penalty 60). Root cause: `check_idle` had
      `if ep.action is None or ctx.action_scale is None or ep.length < 3: return` as its
      very first line -- the `ep.length < 3` guard exists because the smoothed lead/trail
      idle profile further down needs a couple of real transitions to be meaningful, but
      it also skipped the function's own absolute "no motion detected" fallback, which
      needs no such minimum. The shortest, most obviously broken episodes in a dataset
      (an aborted recording cut off almost immediately) were therefore the *only* ones
      this check stayed completely silent on, while a merely-short-but-not-that-short
      zero-motion episode was correctly flagged. Fixed by narrowing the early return to
      just the action/scale guard, and handling `ep.length < 3` as its own branch that
      still calls `_speed()` (safe at any length: `np.diff` on a 1-row array yields a
      0-row result) and flags the same `idle` ERROR whenever there's no speed data at all
      or every available speed sample is below `idle_speed`, before returning (too short
      to say anything about a lead/trail trim either way). New
      `tests/test_checks.py::test_very_short_frozen_episode_flags_no_motion` builds a
      5-episode synthetic dataset (varied normal motion) with one 1-frame and one
      2-frame frozen-action episode and asserts both get the `idle: no motion detected`
      flag while the other three don't; fails without this change (confirmed via
      `git stash` on just `src/demoqc/checks.py` -- the 1-frame case asserts `AssertionError`
      with only a `length` flag present, reproducing the real dataset exactly). Validated
      on the real dataset: `demoqc score lerobot/svla_so100_stacking` now scores episode 20
      at 40.0 with the `idle` error, consistent with its three siblings that have the
      identical defect (previously the only one of the four scoring as if nothing were
      wrong). Re-ran the full existing validation set
      (`lerobot/svla_so101_pickplace`, `masato-ka/so100_cutlery_handling_simple`,
      `aaronsu11/so100_lego`, `shylee/so100_cube`) and got byte-identical mean scores to
      every prior Log entry (97.1 / 92.9 / 92.2 / 99.2), confirming this is a no-op on
      every episode with 3+ frames -- it only changes behavior for the 1- and 2-frame
      degenerate case.
- [x] **Flatten grouped `names` in `meta/info.json` (ALOHA-format datasets).** Found by
      running `demoqc score` against real Hub datasets beyond the existing validation set
      (`lerobot/aloha_static_cups_open`, `lerobot/aloha_static_battery`): `gripper_chatter`
      never fired on either (despite real `left_gripper`/`right_gripper` dims), and `sync`
      fired a "weak correlation" warning on roughly half of every ALOHA dataset's episodes
      (7/15 on `cups_open`, 6/15 on `battery`) with no obvious reason -- the episodes looked
      fine by eye. Root cause: ALOHA-format datasets store `meta/info.json`'s per-feature
      `names` as a dict grouping all dims under one motor-group key
      (`{"motors": ["left_waist", ..., "right_gripper"]}`) instead of a flat list of
      per-dim strings, which `demoqc`'s loader read as-is. `DatasetContext.gripper_dims`'
      `"gripper" in name.lower()` search over `enumerate(ds.action_names or [])` then
      iterated the dict's *keys* ("motors"), never finding "gripper" at all -- silently
      disabling gripper-chatter detection dataset-wide, no error, no warning, nothing in
      the report pointing at why. `_match_sync_dims` (feature-name dim matching for the
      `sync` check) hit the same shape mismatch from the other side: given a dict it still
      entered its name-matching branch, but `enumerate()`/`in` on a 1-key dict treated that
      single key as if it were dim 0, "matching" only one real dim out of 14 and silently
      dropping the other 13 -- so `sync` was comparing one single joint's velocity (whichever
      the dict's iteration order put first) instead of both arms, both correlating poorly on
      its own and blind to a real defect on any of the other 13 dims. Fixed by flattening
      `names` once, in `dataset.py`, right where `meta/info.json` is read: a dict's values
      (lists) are concatenated in their stored order into the flat per-dim list both
      `gripper_dims` and `_match_sync_dims` already expected; a flat list (every other
      dataset in the existing validation set) passes through unchanged. New tests:
      `tests/test_dataset.py::test_grouped_feature_names_are_flattened` (the loader output
      directly) and `tests/test_checks.py::test_gripper_chatter_detected_with_grouped_feature_names`/
      `test_sync_matches_all_dims_with_grouped_feature_names` (the two silently-broken
      checks, the latter asserting `DatasetContext.build`'s matched dim indices directly
      since a single-dim sync match can coincidentally still pass a correlation-threshold
      heuristic); `tests/conftest.py`'s `write_dataset` gained a `grouped_names` flag that
      writes the real `{"motors": [...]}` shape. All three fail without this change
      (confirmed via `git stash` on just `src/demoqc/dataset.py`). Validated on the real
      datasets that found the bug: `lerobot/aloha_static_cups_open` (15 ep) went from mean
      score 95.3 (7 spurious `sync` warnings, r=0.11-0.35) to 100.0 (0 flags, peak corr now
      implicitly computed over all 14 dims); `lerobot/aloha_static_battery` (15 ep) went
      from mean 90.7 (6 spurious `sync` warnings alongside 13 real `smoothness` ones) to
      94.7 (`sync_peak_corr` now a consistent 0.70-0.78 across every episode, all 6 spurious
      warnings gone, the 13 real `smoothness` flags unchanged) -- in both cases the fix only
      removed warnings that didn't correspond to anything wrong in the data, it introduced
      no new flags. Re-ran the full existing flat-`names` validation set
      (`lerobot/svla_so101_pickplace`, `masato-ka/so100_cutlery_handling_simple`,
      `aaronsu11/so100_lego`, `shylee/so100_cube`): byte-identical mean scores to every
      prior Log entry (97.1 / 92.9 / 92.2 / 99.2), confirming this is a no-op when `names`
      is already flat.
- [x] **Fix `check_idle` missing a dataset-wide frozen action stream.** Found by running
      `demoqc score` against a real Hub dataset beyond the existing validation set
      (`omkarmayekar555/robot-learning-tutorial-data`, found via `list_datasets(filter=
      "LeRobot", sort="downloads")`): every action dim was bit-identical for the episode's
      entire 1776 frames (a broken teleop/recorder session, not real motion), yet
      `demoqc score` gave it 90.0 with no flag at all pointing at the actual defect -- only
      an unrelated weak `sync` warning -- plus a `RuntimeWarning: Mean of empty slice` and
      `All-NaN slice encountered` printed to stderr. Root cause: `DatasetContext.build`'s
      `action_scale` collapses to NaN for every dim when nothing in the dataset ever varies
      (by design, so a genuinely constant dim like an unused sensor doesn't get flagged for
      spike detection); `_speed()` then hit `np.nanmean` on an all-NaN row, which is both a
      `RuntimeWarning` and a `NaN` speed, and `NaN < idle_speed` is always `False` --
      `check_idle`'s absolute "no motion detected" fallback (meant for exactly this case)
      silently never fired. A second, independent issue in the same function: `idle_speed`
      itself is `0.1 * the dataset's own p75 speed`, which is exactly `0.0` when the dataset
      has (almost) no real motion anywhere, so even a correctly-computed `speed == 0.0`
      frame failed `speed < idle_speed` (`0 < 0` is `False`) -- `check_staleness` and
      `_episode_moved` already guarded this exact case with `max(ctx.idle_speed, 1e-9)`,
      but `check_idle` itself, the one place that most needed it, didn't. Fixed `_speed()`
      to treat a frame where every tracked dim is excluded as a real zero speed instead of
      an unknown (NaN) one, added the same `max(idle_speed, 1e-9)` floor to `check_idle`,
      and made `DatasetContext.build` skip `nanpercentile` on `step_p99` dims that are
      already all-NaN (same root cause, a separate warning site) instead of calling it and
      immediately discarding a warning-raising NaN result. New
      `tests/test_checks.py::test_frozen_action_whole_dataset_flags_no_motion` builds a
      5-episode synthetic dataset where every episode shares the exact same frozen action
      (reproducing the real "nothing varies dataset-wide" condition, not just within one
      episode) with `warnings.simplefilter("error", RuntimeWarning)` around the whole
      `evaluate()` call, and fails with the pre-fix code exactly as the real dataset did
      (`idle` missing, only `sync` in the flags) -- confirmed via `git stash` on just
      `src/demoqc/checks.py`. Validated on the real dataset: `demoqc score
      omkarmayekar555/robot-learning-tutorial-data` now scores it 30.0 with `[error] idle:
      no motion detected in the whole episode`, no RuntimeWarning on stderr. Re-ran the full
      existing validation set (`lerobot/svla_so101_pickplace`, `masato-ka/so100_cutlery_handling_simple`,
      `aaronsu11/so100_lego`, `shylee/so100_cube`) and got byte-identical mean scores to
      every prior Log entry (97.1 / 92.9 / 92.2 / 99.2), confirming this is a no-op on
      normal data -- it only changes behavior for the degenerate all-dims-constant case.
      Also tried `jchandru08/so100_press_red_button` (29 ep, v2.1) from the same Hub search:
      7 episodes correctly flag a real `metadata` length mismatch (declared length off by
      1-3 frames from the data, both directions) -- the existing `metadata` check already
      handles this correctly, no bug found there.
- [x] **Detect duplicate `episode_index` across a dataset's own metadata files.** Found
      by running `demoqc score` against a real community dataset (benchmark expansion):
      a second recording/upload session had renumbered episodes from 0 instead of
      continuing the existing numbering, so some `episode_index` values appeared in two
      different metadata chunk files. The loader grouped frames by `episode_index` alone
      (its whole mechanism for staying agnostic to v2.x/v3.0 file layouts) and silently
      concatenated the two unrelated recordings into one nonsense "episode" per affected
      index, which then failed almost every other check for confusing, misleading reasons.
      Shipped: `dataset.load_dataset` now detects any `episode_index` that appears more
      than once in `meta/episodes/*.parquet` (v3.0) or `meta/episodes.jsonl` (v2.x), drops
      every episode under those indices (no way to tell which recording is real) instead of
      merging or guessing, and surfaces a `Dataset.warnings` entry (printed by the CLI,
      and in the JSON report's `dataset.warnings`) naming the affected indices. Explicitly
      requesting one of those indices via `--episodes` now raises a specific error instead
      of the generic "not in the dataset" one. See the Log entry for the real-data numbers.
- [x] **Action/state sync check.** Estimate per-episode lag between `action` and
      `observation.state` (cross-correlate per-dim velocities over ±0.5 s; only dims present
      in both with matching names or shapes). Flag episodes whose lag deviates from the
      dataset median by more than 2 frames, or whose peak correlation is low. Catches "bad
      syncs" from USB latency spikes and recorder bugs. Add to the report and README table.
      Shipped as the `sync` check (`sync_lag_frames`, `sync_lag_s`, `sync_peak_corr` metrics).
      Follow-up: dims are currently matched by shape only (`action.shape[1] == state.shape[1]`)
      because the loader doesn't read per-feature names from `meta/info.json` yet; once it
      does, match by name instead so a reordered or extra dim doesn't disable the check.
- [x] **Near-duplicate episodes (`demoqc dedupe`, plus a `duplicates` section in `score`).**
      Resamples each episode's scaled action trajectory (trimmed) to a fixed length and
      groups near-duplicates with a threshold relative to the dataset's nearest-neighbour
      distance distribution (robust z vs. the dataset's own nn-distance spread). Exact
      duplicates (identical action/state bytes) are caught separately by content hash.
      Shipped as `src/demoqc/dedupe.py`, the `demoqc dedupe` subcommand, and a
      `duplicate_groups` summary plus per-episode `duplicate` flags in `demoqc score`.
      Follow-up: pairwise L2 is O(n²) in episode count and holds the whole distance matrix
      in memory; fine up to a few thousand episodes, but DTW (for duplicates recorded at
      different speeds beyond what resampling absorbs) and a chunked/approximate
      nearest-neighbour search (for huge datasets) are still open.
      Follow-up (resolved 2026-10-25): that "fine up to a few thousand episodes" claim
      undercounted the actual memory use by ~2-3 orders of magnitude -- see the Log entry.
      DTW and a chunked/approximate search for datasets beyond what the (n, n) distance
      matrix itself can hold are still open.
      Follow-up (resolved 2026-10-30): the second half -- the (n, n) distance matrix itself
      no longer fitting in memory at a large enough episode count -- is fixed; see the Log
      entry. DTW (for duplicates recorded at different speeds beyond what resampling
      absorbs) is still open.
- [x] **Export a cleaned dataset (slice 1): `demoqc export --keep-from report.json`.**
      Write a filtered LeRobot v3.0 dataset (tabular data and meta only): drop rejected
      episodes, re-index episodes and frames, recompute `meta/info.json` totals and
      `meta/episodes`. Must load with LeRobot's own loader (add a test that does so if
      `lerobot` is installed, skipped otherwise).
      Shipped as `src/demoqc/export.py` and the `demoqc export` subcommand. Also rewrites
      `meta/stats.json` (exact for tabular features, count-weighted combination of the
      per-episode stats for video features) and the per-episode `stats/index` and
      `stats/episode_index` columns, which would otherwise go stale.
      Follow-ups: (1) ~~v2.x export~~ shipped 2026-10-19 (`_export_v2` in `src/demoqc/export.py`;
      see the Log entry) — slice 1 only (drop/renumber, no `--apply-trim`).
      (2) ~~An option to copy the source `videos/` directory for local sources~~ shipped
      2026-10-23 as `--copy-videos` (see the Log entry). (3) ~~`--min-score` on
      `export` itself~~ shipped 2026-10-20 (see the Log entry). (4) Single `train`
      split only: other `splits` entries in `info.json` are collapsed into `train`.
      (5) ~~v2.x `--apply-trim`~~ shipped 2026-10-21 (see the Log entry) — a v2.x video has no
      per-episode time span to narrow like v3.0's, so the export warns that the video itself
      must also be cut to match instead of narrowing a stored span.
      (6) ~~`--copy-videos` only copies; a `--symlink-videos` variant~~ shipped 2026-10-24
      (see the Log entry).
- [x] **Export slice 2: apply trims.** Cut idle frames using `trim`, shift timestamps and
      update video `from_timestamp`/`to_timestamp` so videos stay aligned without
      re-encoding.
      Shipped as `demoqc export --apply-trim`: `frame_index`/`index` are renumbered,
      `timestamp` is shifted by the number of leading frames cut, and each camera's video
      span is narrowed by the same frame counts (video bytes are never touched). `length` in
      `meta/episodes` is now always rewritten to match the written frame count (previously it
      only warned on a pre-existing mismatch and left the stale value, which would have made
      a trimmed export self-inconsistent).
      Follow-up: v2.x export (including trims) is still not implemented; this only covers
      v3.0, same as slice 1.
- [x] **Hub integration: `demoqc push-report <repo_id>`.** Upload the JSON report plus a
      Markdown summary to the dataset repo as a Hub pull request (`create_pr=True`), so
      dataset owners get actionable feedback. Generate a dataset-card snippet with a badge.
      Shipped as `src/demoqc/push.py` (`push_report`, `badge_markdown`) and the `demoqc
      push-report <repo_id>` subcommand: scores the repo in-process (same `--config`/
      `--disable-check`/`--episodes`/`--min-score` options as `score`), then opens a Hub PR
      adding `demoqc/report.json` and `demoqc/report.md` via a single `HfApi.create_commit(...,
      create_pr=True)`, with a shields.io score badge (green/yellow/red at 90/70) linking to
      the uploaded `report.md` in the PR description for the owner to paste into their dataset
      card. `--dry-run -o DIR` writes the same two files locally instead of calling the Hub,
      for previewing the report or running somewhere with no write access. `HF_TOKEN` is still
      unset in this environment (confirmed via `HfApi().whoami()` raising
      `LocalTokenNotFoundError`, same as every prior Log entry that needed one), so the actual
      PR-opening call can't be validated against a real repo here -- attempting it against a
      real public dataset repo was in fact blocked by this environment's own sandbox policy
      (a write/PR-opening call against public infrastructure), which is the expected behavior
      given there's no token anyway. Validated everything that doesn't need write access: new
      `tests/test_push.py` (9 cases) mocks `huggingface_hub.HfApi.create_commit` to check the
      uploaded operations' paths/content, the `create_pr=True`/`repo_type="dataset"` kwargs,
      the PR-url return value, a `RuntimeError` if the mocked call ever returned no PR url, the
      CLI's Hub-vs-local-source rejection, and a clean error message on a Hub failure; all 9
      fail with `ModuleNotFoundError` without this change (confirmed via moving
      `src/demoqc/push.py` aside and `git stash` on `cli.py`). Ran `demoqc push-report
      lerobot/svla_so101_pickplace --dry-run` for real (the Hub was reachable this run): wrote
      `demoqc/report.json`/`report.md` with the dataset's existing validated mean score (97.1)
      and badge color (`brightgreen`), confirming the in-process scoring path and file content
      match `demoqc score`'s own output exactly.
- [x] **Gripper chatter check.** Detect gripper dims by feature name (`gripper` in
      `names`) and flag rapid open/close toggling, a common teleop artifact.
      Shipped as the `gripper_chatter` check. `dataset.py` now reads `info["features"]["action"]["names"]`
      onto `Dataset.action_names` (previously unused); `DatasetContext.gripper_dims` finds dims whose
      name contains "gripper" (case-insensitive). A dim counts as active only if its per-episode range
      exceeds half the dataset's own robust scale for that dim (debounces noise on an unused or barely-
      moved gripper); state changes use hysteresis (15% of the episode's own range) before counting
      toggles, then flags 4+ toggles within any 1-second sliding window.
      Follow-up (resolved 2026-10-12): `check_sync` now matches action/state dims by name
      instead of requiring `action.shape[1] == state.shape[1]`, so a reordered or extra dim
      (e.g. an added force sensor) no longer disables the sync check. See the Log entry.
- [x] **Configurable thresholds (slice 1): `--config demoqc.toml`.** Override any subset of
      `Config`'s threshold/penalty fields from a TOML file, for `score` and `dedupe`.
      Shipped as `Config.from_file`/`Config.with_overrides` (`src/demoqc/checks.py`) and
      `--config FILE.toml` on both subcommands. Unknown keys or wrong-typed values raise a
      clear error (naming the key) instead of being silently ignored or miscalibrating the
      run; `tomli` is pulled in as a dependency only on Python < 3.11 (stdlib `tomllib`
      covers 3.11+).
      Follow-up (shipped 2026-10-15 as `--disable-check NAME`; see the Log entry): a
      per-check name registry (`checks.CHECK_NAMES`) shared between `EPISODE_CHECKS` and the
      names `check_relative` emits, so a noisy check (e.g. `sync` on a dataset with an
      unsynced third camera clock) can be turned off entirely instead of re-tuned.
- [x] **Markdown report output (`--md`)** for pasting into dataset cards or PRs.
      Shipped as `demoqc score --md report.md` and `src/demoqc/report.py::format_markdown`:
      a score/flags table, duplicate-group counts, and a table of the lowest-scoring
      episodes with their flags, matching `format_summary`'s content but as GitHub/Hub-
      flavoured Markdown tables. Table cells escape `|` in flag messages so an unusual
      message can't break a row.
      Follow-up: this is the Markdown half of the `push-report` item above; once that ships,
      it should reuse `format_markdown` as the PR/dataset-card body instead of generating its
      own text.
- [x] **Scale to large datasets (slice 1): `--episodes` and `--max-episodes`.** Load only
      some of a dataset's episodes. From the Hub, only the data file(s) that contain the
      requested episodes are downloaded; locally, data files that contain none of them are
      never opened.
      Shipped as `demoqc score`/`demoqc dedupe --episodes SPEC`/`--max-episodes N`
      (`load_dataset(..., episodes=, max_episodes=)` in `src/demoqc/dataset.py`). Metadata
      (`meta/*`, always small) is downloaded first, then `meta/episodes`'
      `data/chunk_index`/`data/file_index` columns (v3.0) or the `episode_<n>.parquet`
      filename convention (v2.x) map the requested episodes to the exact data file(s) to
      fetch or open; narrowing is best-effort and falls back to reading everything if a
      layout can't be mapped from metadata alone, so it's never less correct, only
      sometimes not faster.
      Follow-up: this still doesn't bound peak memory when scoring *all* of a huge
      dataset — `_read_frames` still concatenates every needed file into one DataFrame.
      True per-file streaming (bounding memory independent of `--episodes`) is still open.
      ~~Extending `--episodes`/`--max-episodes` to `demoqc export`~~ shipped 2026-10-22; see
      the Log entry.
- [x] **Video checks (slice 1): black/dead camera from metadata alone.** LeRobot already
      writes each episode's per-camera pixel mean/std into `meta/episodes` (v3.0) or
      `meta/episodes_stats.jsonl` (v2.x) when it records a dataset. Read it and flag a
      camera that's near-black for the whole episode (mean and std both near zero): a
      disconnected camera, a lens cap left on, or video that failed to record. No video
      file is ever opened, so this needed no new dependency and stays in the core path.
      Shipped as the `camera` check (`src/demoqc/dataset.py`'s `Episode.camera_stats`,
      `checks.check_camera`, `Config.camera_dark_mean_max`/`camera_dark_std_max`).
      Follow-up (resolved 2026-10-26): detecting a camera going black or a video failing to
      decode mid-episode shipped as the opt-in `video_freeze` check
      (`--decode-video`, `demoqc[video]`); see the Log entry. Follow-up (resolved
      2026-10-27): `video_freeze` now also supports LeRobot v3.0 (resolving a camera's local
      video file via its `meta/episodes` chunk/file index, and narrowing decode to that
      episode's own `from_timestamp`/`to_timestamp` span since a v3.0 video file holds
      several episodes back to back); see the Log entry. Follow-up (resolved 2026-10-28):
      `video_freeze` now also detects a literally frozen (repeated) frame on a camera that
      never goes dark; see the Log entry.
- [x] **Benchmark the community.** `scripts/benchmark.py` scores a curated list of real
      community LeRobot datasets and writes `docs/benchmarks.md` (scores, trimmable idle,
      duplicate groups, top flags). Refresh it periodically; it doubles as a regression
      test for the heuristics.
      Shipped as `scripts/benchmark.py` and `docs/benchmarks.md`. Scores each dataset in
      isolation and catches any exception (unreachable repo, unrecognised format, a bug in
      a check) into an error row instead of stopping the whole run, so one bad dataset
      doesn't lose the rest. Took a curated list (`DEFAULT_REPOS`: the reference and
      community so100 datasets already used for validation) rather than the Hub's actual
      most-downloaded LeRobot datasets, which are enormous OpenX/DROID re-releases
      (hundreds of GB of tabular data even with videos excluded) this tool isn't calibrated
      for and that would make the benchmark slow and its numbers meaningless; `--max-episodes`
      (default 200) caps runtime further. `--repo` adds/replaces the list, for anyone who
      wants to benchmark a different set.
      Follow-up: querying the Hub for real most-downloaded small/medium teleoperation
      datasets (e.g. filtering `list_datasets(filter="LeRobot", sort="downloads")` by
      dataset size or episode count rather than a hand-picked list) is still open.

- [x] **Data-integrity check (NaN/Inf in `action`/`observation.state`).** Found while
      reviewing `checks.py` for the daily session: `DatasetContext.build` computed the
      dataset-wide `action_scale` (and `step_p99`) with plain `np.percentile`/`.std`, which
      propagate `NaN` across an *entire column* when even one value in it is non-finite.
      A single corrupted frame in one episode's `action` (a sensor glitch or a partial,
      corrupted write) therefore silently set that whole dimension's scale to `NaN` for
      *every* episode in the dataset, not just the bad one -- disabling `idle`,
      `smoothness` (spike/jump detection on that dim) and, worst, `gripper_chatter`
      (which requires a finite scale and only ever watches one or two gripper dims, so one
      bad frame anywhere in the dataset's gripper column could blind that check dataset-wide)
      with no visible symptom pointing at the actual cause. Fixed `DatasetContext.build` to
      use `np.nanpercentile`/`np.nanstd`, and added the `integrity` check, which flags the
      actual corrupted episode (ERROR, "non-finite (NaN/Inf) value(s) in action/
      observation.state") so the defect is visible on its own instead of only as collateral
      damage elsewhere. Also made `dedupe._trimmed_action` skip a non-finite trajectory
      instead of feeding `NaN` into `np.interp` and corrupting that episode's row of the
      near-duplicate distance matrix. New tests in `tests/test_checks.py`: `integrity` fires
      on an injected NaN in `action` and separately in `observation.state` (and stays silent
      on every other episode), plus a regression test that reproduces the exact scenario
      above (one episode with a NaN in its gripper dim, another with a real
      `gripper_chatter` defect) and confirms `gripper_chatter` still fires on the second
      episode; all three fail without this change (confirmed via `git stash` on just
      `src/demoqc/checks.py`). Validated on real data: `lerobot/svla_so101_pickplace` (50 ep,
      mean 97.1), `aaronsu11/so100_lego` (100 ep, mean 92.2) and `shylee/so100_cube` (200 ep,
      mean 99.2) all reproduced their existing validated scores and flag counts exactly, with
      zero `integrity` flags on any of them -- `nanpercentile`/`nanstd` are mathematically
      identical to `percentile`/`.std` when there's nothing non-finite to ignore, so this is
      a no-op on real, clean data and only changes behavior when a dataset is actually
      corrupted.

## Needs the owner

- Publish to PyPI (needs a PyPI account and trusted-publisher setup).
- Announce on the LeRobot Discord or the Hugging Face forums once export and dedupe ship.

## Known limitations

- LeRobot writes timestamps as `frame_index / fps`, so the timing check rarely fires on
  LeRobot-recorded data. The `stale_state`, `video_sync` and `idle` checks carry most of
  the signal there.
- Scores are relative within a dataset for the smoothness and length checks. A dataset
  where every demo is bad is caught only by the absolute checks.

## Log

- 2026-10-09: v0.1.0. LeRobot v2.x/v3.0 loader (local or Hub, no videos), checks for
  timing, metadata, video_sync, idle/trim, smoothness, stale_state and length, plus
  JSON/CSV reports, the `demoqc score` CLI, CI and auto-merge. Validated on
  lerobot/svla_so101_pickplace, lerobot/pusht and 5 community SO-100 datasets.
- 2026-10-09: Added the `sync` check (action/observation.state lag via per-dim
  cross-correlation, ±0.5s window); flags a weak peak correlation or a lag that deviates
  from the dataset's median by more than 2 frames. Validated with synthetic-defect tests
  only; the Hugging Face Hub was unreachable from this environment (network policy blocks
  huggingface.co) so real-dataset validation is a follow-up.
- 2026-10-09: Added exact and near-duplicate episode detection (`src/demoqc/dedupe.py`,
  `demoqc dedupe`, `duplicate_groups` in `score`). Validated on
  lerobot/svla_so101_pickplace (50 ep), masato-ka/so100_cutlery_handling_simple (50 ep),
  aaronsu11/so100_lego (100 ep) and shylee/so100_cube (200 ep): no duplicates and no false
  positives on any of them. Synthetic tests cover exact (byte-identical) and near
  (same seed, different duration) duplicates, plus a 40-episode independent-seed check for
  false positives.
- 2026-10-09: Added `demoqc export SOURCE --keep-from report.json -o DIR` (LeRobot v3.0 only,
  `src/demoqc/export.py`). Validated on lerobot/svla_so101_pickplace (50 ep) and
  lerobot/pusht (206 ep), the only v3.0 datasets in the validation set (the so100 community
  ones are v2.1, which export rejects with a clear error). Keeping every episode reproduces
  the data and `meta/episodes` tables exactly and `stats.json` to within 3e-5 (including the
  combined video stats); dropping ~2/7 of the episodes gives contiguous indices, correct
  `dataset_from/to_index` ranges and stats that match a recomputation from the kept frames.
  All four exports load with `LeRobotDataset` from lerobot 0.6.1 (with placeholder files
  for the videos, which LeRobot checks for) and `episodes=[...]` subsets resolve correctly.
  `tests/test_export.py` covers this on synthetic data, including a 3-file layout where one
  file contains only dropped episodes; the lerobot loader test is skipped when lerobot isn't
  installed (CI doesn't install it).
- 2026-10-10: Added `demoqc export --apply-trim`, which cuts each kept episode's idle
  start/end frames per the report's `trim` suggestion, renumbers `frame_index`/`index`, shifts
  `timestamp`, and narrows the per-camera `videos/*/from_timestamp`/`to_timestamp` window by the
  same frame counts (videos are still never read, copied or re-encoded). Also fixed `length` in
  `meta/episodes` to always match the written frame count instead of only warning when it
  disagreed. Validated on lerobot/svla_so101_pickplace (50 ep, v3.0): the Hugging Face Hub was
  reachable this run. 46/50 episodes had a suggested trim; after `--apply-trim`, total frames
  dropped from 11939 to 10528 (1411 trimmed), re-scoring the export dropped `trimmable idle`
  from 71.6s/47 flags to 3.6s/3 flags (the small remainder is the check's own padding, by
  design) and mean score rose from 97.1 to 99.8 with no new flags. Verified per-episode that
  the trimmed frame count and both video-span edges exactly match `start`/`end` from the
  report. Also validated on lerobot/pusht (206 ep, v3.0, no idle episodes): `--apply-trim`
  trimmed 0 frames and the re-scored export is byte-for-byte the same report (mean 100.0, no
  flags), confirming no false positives when there is nothing to trim. The so100 community
  datasets used for earlier validation (masato-ka/so100_cutlery_handling_simple etc.) are
  v2.1, so `export` still rejects them with the existing "v2.x not implemented" error.
- 2026-10-11: Added the `gripper_chatter` check (rapid open/close toggling via feature-name
  lookup + hysteresis-debounced sliding-window toggle counting). Validated on
  lerobot/svla_so101_pickplace (50 ep, max 2 toggles/window, well under the 4-toggle
  threshold; every normal grasp is 1-2 toggles), masato-ka/so100_cutlery_handling_simple
  (50 ep) and aaronsu11/so100_lego (100 ep), both also maxing out at 2 toggles/window, and
  shylee/so100_cube (200 ep), where the gripper barely moves (0 episodes pass the active-range
  filter) — no false positives on any of them. Also ran on lerobot/pusht (206 ep, no gripper
  feature at all) to confirm the check degrades to a no-op rather than erroring when a dataset
  has no feature names or no gripper dim. Synthetic test injects a 15-frame toggle burst on one
  episode's gripper dim and checks it fires there only; confirmed the test fails with
  `KeyError` before the implementation.
- 2026-10-12: `check_sync` now matches action/observation.state dims by feature name
  (`Dataset.state_names`, newly loaded from `info["features"]["observation.state"]["names"]`)
  instead of requiring the two streams to have equal width. Previously, any dataset where
  the two streams had different widths (e.g. an extra, unmatched sensor dim on one side)
  silently disabled the sync check for every episode; a dataset where both streams had the
  same width but a different dim order would get wrong lag/correlation estimates from
  comparing unrelated dims position-by-position. Falls back to positional matching when
  either stream lacks feature names, so nothing changes for datasets without them. Two new
  synthetic tests confirm both bugs: one reorders `observation.state`'s columns relative to
  `action`'s (previously computed near-zero peak correlation for a dataset that is in fact
  tightly synced, risking a false `sync` flag; now correctly reports r>0.8), the other adds
  an action-only `force_sensor` dim with no state counterpart (previously the sync check
  produced no metrics at all; now it still detects an injected 6-frame lag). Both tests fail
  with the pre-fix code (`KeyError` / an assertion on `sync_peak_corr`). Validated on
  lerobot/svla_so101_pickplace (50 ep), masato-ka/so100_cutlery_handling_simple (50 ep) and
  aaronsu11/so100_lego (100 ep): all three have identically-ordered, identically-named
  action/state features, so matched dims are unchanged (`[0..5]`/`[0..5]`) and scores, sync
  metrics and flags are byte-for-byte the same as before this change — no regression.
- 2026-10-13: Added `demoqc score --md report.md` (`format_markdown` in `report.py`), a
  Markdown rendering of the same summary `format_summary` prints to the terminal, for
  pasting into a dataset card or a Hub/GitHub PR. Flag messages are escaped (`|` -> `\|`)
  so a stray pipe can't break a table row. Validated on lerobot/svla_so101_pickplace
  (50 ep, v3.0) and masato-ka/so100_cutlery_handling_simple (50 ep, v2.1): both render as
  clean GitHub-flavoured Markdown tables with no escaping artifacts. Skipped the
  higher-priority `push-report` item this run — no `HF_TOKEN` in this environment to create
  or validate a real Hub PR — and left a note on it pointing back to `format_markdown` as
  the body it should reuse.
- 2026-10-14: Added `--config demoqc.toml` to `demoqc score`/`demoqc dedupe`
  (`Config.from_file`/`Config.with_overrides`), overriding any subset of the threshold and
  penalty fields in `Config` so a different embodiment, fps or operator style doesn't need a
  code change. Unknown keys and wrong-typed values are rejected with an error naming the
  key. `huggingface.co` was reachable this run but there is still no `HF_TOKEN`, so
  `push-report` (which needs to open a Hub PR on someone else's dataset repo) stays blocked;
  picked this item instead. Validated on lerobot/svla_so101_pickplace (50 ep, v3.0: mean
  score 97.13 at defaults, 97.14 with a config loosening `max_idle_fraction`,
  `gripper_chatter_min_toggles` and `sync_min_peak_corr`) and
  masato-ka/so100_cutlery_handling_simple (50 ep, v2.1: 92.92 at defaults, 93.47 tuned),
  confirming thresholds actually move scores end to end and a bad config file exits 2 with a
  clear message on both. New `tests/test_config.py` (8 cases: override/validation on
  `Config` directly, TOML parsing, a synthetic gripper-chatter episode that only flags once
  `gripper_chatter_min_toggles` is lowered via config, and both CLI paths) fails on 7/8
  cases without this change.
- 2026-10-15: Added **`--disable-check NAME`** to `demoqc score` (the follow-up noted under
  the `--config` item). Lets a check that's noise for a particular dataset (no gripper but a
  jittery `sync` from an unsynced third camera clock; intentional near-identical demos that
  shouldn't count as `duplicate`) be turned off instead of fought with thresholds. Split
  `check_metadata` into `check_metadata` (episode length) and the new `check_video_sync`
  (video span) so they're independently disableable; added `checks.CHECK_NAMES` (the
  registry the original follow-up asked for, shared between `EPISODE_CHECKS` and
  `check_relative`'s own `length` flag) and threaded `disabled_checks` through
  `evaluate()`/`check_relative()`. `length` has no `EPISODE_CHECKS` entry (it's peer-only)
  and `duplicate` isn't part of `checks.py` at all (the CLI calls
  `dedupe.find_duplicates`/`flag_duplicates` separately), so both are special-cased in
  `evaluate()` and `cli._score()` respectively; disabling a check removes its metrics too,
  not just its flags. argparse `choices=CHECK_NAMES` rejects a typo'd name up front. 5 new
  tests (disabling suppresses flags+metrics on a real sync defect, the metadata/video_sync
  split stays independent, `length`'s peer-only path, the CLI's `duplicate` path, and the
  argparse rejection) all fail without this change. Validated on
  lerobot/svla_so101_pickplace (50 ep): disabling `gripper_chatter`+`sync` left
  `flags_by_check` and mean score (97.13) byte-for-byte identical to the baseline (neither
  check fires on this dataset) and left no `sync_*`/`gripper_*` metrics in the report. On
  masato-ka/so100_cutlery_handling_simple (50 ep): disabling `duplicate`+`length` dropped
  the 2 `length` flags from `flags_by_check` (`idle: 52, length: 2` -> `idle: 52`) and raised
  mean score from 92.92 to 93.32, with `duplicate_groups` empty either way (none to
  suppress) — confirms the flag removed is exactly the one disabled, nothing else moves.
- 2026-10-16: Added **`--episodes SPEC`/`--max-episodes N`** to `demoqc score`/`demoqc
  dedupe` (the roadmap's "scale to large datasets" slice 1), so trying the tool on a huge
  dataset doesn't require loading all of it first. Validated on
  masato-ka/so100_cutlery_handling_simple (50 ep, v2.1): a full `score` downloads all 54
  files (588 KiB, ~24s on this run); `--episodes 0-2` downloads only 4 meta files + 3
  episode parquet files (204 KiB, ~1.7s) and scores exactly episodes 0-2. Confirmed the
  score/metrics for those 3 episodes match a full-dataset load almost exactly (idle penalty
  6.13 vs 6.07 for episode 0) — the tiny difference is `DatasetContext`'s robust stats
  (e.g. `idle_speed`) being computed over the loaded subset rather than all 50 episodes,
  which is the documented "thresholds are relative to the dataset" behavior, not a bug;
  noted as a caveat in the README. Also validated on lerobot/svla_so101_pickplace (50 ep,
  v3.0, single data file since the dataset is smaller than one chunk): `--episodes 0-2`
  loads exactly 3 episodes; narrowing is a no-op here since there's only one file to begin
  with, which is the expected fallback, not a bug. New tests in `tests/test_dataset.py`
  corrupt the data file(s) a requested `--episodes` subset doesn't need (both v3.0
  multi-file and v2.1 per-episode-file layouts) and confirm loading still succeeds with
  exactly the requested episodes, proving those files are never opened; plus CLI spec
  parsing, `--max-episodes`, combining both, and an unknown-episode error. 10 of the 14 new
  tests fail with `TypeError`/`SystemExit` without this change (confirmed via `git stash`
  on just the implementation files).
- 2026-10-17: Added the **`camera` check** (video checks slice 1), flagging a camera
  that's near-black for an entire episode using the per-episode pixel mean/std LeRobot
  already writes to its own metadata (`meta/episodes` on v3.0, `meta/episodes_stats.jsonl`
  on v2.x) — no video file is ever downloaded or opened, so this needed no new dependency
  and the `push-report`/"benchmark" items below stayed next in line, not this. `HF_TOKEN`
  is still unset in this environment, so `push-report` stays blocked. Validated on
  lerobot/svla_so101_pickplace (50 ep, v3.0, 2 cameras), masato-ka/so100_cutlery_handling_simple
  (50 ep, v2.1, 1 camera), aaronsu11/so100_lego (100 ep, v2.1, 2 cameras) and shylee/so100_cube
  (200 ep, v2.1, 3 cameras): per-episode pixel mean/std load correctly from both metadata
  formats (confirmed by printing `Episode.camera_stats`) and no episode on any of the four
  is flagged — all real cameras here are well above the near-black threshold (lowest
  real per-episode mean seen: 0.45; lowest std: 0.14), and mean scores are unchanged from
  before this change (97.1 / 92.9 / 92.2 / 99.2). New `tests/test_checks.py::test_camera_dead`
  (parametrized over v3.0 and v2.1) injects a near-black `camera_dead` episode via a new
  `EpisodeSpec` field and `tests/conftest.py` now also writes `meta/episodes_stats.jsonl`
  for synthetic v2.x datasets (it previously wrote none, so this check's v2.x path was
  untestable); both parametrizations fail with `AssertionError` before this change. Also
  fixed a bug caught by the v3.0 test while building this: parquet round-trips the nested
  per-channel video stats as a ragged (non-rectangular) array, which `np.asarray` rejects
  with `ValueError`, so the first version of `_image_stat_scalar` silently returned `None`
  for every v3.0 episode; it now flattens the nesting recursively instead.
- 2026-10-18: Added `scripts/benchmark.py` and `docs/benchmarks.md` (the "benchmark the
  community" item). Scores a curated list of real datasets and renders a Markdown table
  (episodes, frames, mean/median score, trimmable idle, duplicate groups, top flags);
  each dataset is scored independently and a failure becomes an error row instead of
  aborting the run. Ran it end to end against all 6 datasets in `DEFAULT_REPOS`
  (lerobot/svla_so101_pickplace, lerobot/pusht, masato-ka/so100_cutlery_handling_simple,
  aaronsu11/so100_lego, shylee/so100_cube, jccj/so100_block_in_cup): all 6 scored with no
  errors and no new flags beyond what earlier validation runs already found (mean scores
  97.1/100.0/92.9/92.2/99.2/95.2), confirming the script reproduces the existing,
  already-validated numbers rather than introducing new false positives. `tests/test_benchmark.py`
  covers `score_repo` on synthetic local datasets (clean, `--max-episodes`-capped, a real
  injected idle defect, and a missing/unreadable dataset producing an error row instead of
  raising) and `format_benchmark_markdown` on both a normal and an error row, all against
  the existing `EpisodeSpec` fixtures — no network needed for the test suite.
- 2026-10-19: Added **v2.x export** (`demoqc export`'s highest-priority follow-up: most
  community datasets, including every so100 one used for validation, are v2.1, and export
  previously rejected them outright). `src/demoqc/export.py` now dispatches on
  `codebase_version`: `_export_v3` is the existing (refactored, behaviour-unchanged) v3.0 path,
  and the new `_export_v2` handles v2.x's different layout (one parquet per episode, JSONL
  episode/stats metadata instead of `meta/episodes` parquet columns) — renumbering episodes,
  rewriting `meta/episodes.jsonl` and `meta/episodes_stats.jsonl` (including each episode's own
  `index`/`episode_index` stats, which describe the new numbering), updating `meta/info.json`
  totals (`total_chunks`, `total_videos` included), and recomputing `meta/stats.json` (exact for
  tabular features, count-weighted combination of per-episode stats for video features, same as
  v3.0) when the source has one — some real datasets ship none, which export now detects and
  leaves a warning for instead of guessing. The stats-combination math was extracted into a
  shared `_combine_from_arrays` helper used by both formats, so the v3.0 behaviour is provably
  unchanged (all pre-existing v3.0 export tests still pass unmodified). This is slice 1 only: no
  `--apply-trim` yet (v2.x needs its own trim implementation; passing `--apply-trim` on a v2.x
  source now errors out clearly instead of silently misbehaving) — tracked as a new follow-up
  above. Also surfaced and documented a real correctness gap videos never need: a v2.x video
  file is named after its episode's index directly, so renumbering episodes invalidates
  `videos/`'s filenames (unlike v3.0, where chunk/file indices in `meta/episodes` still point at
  the right file); `demoqc export` now prints a warning with the exact renaming rule instead of
  the (wrong, for v2.x) "copy videos/ unchanged" message. Validated on real datasets: scoring
  and exporting masato-ka/so100_cutlery_handling_simple (50 ep, `--min-score 90`, 45 kept, no
  `meta/stats.json` in the source — export correctly warns and writes none) and re-scoring the
  export reproduced the original per-episode scores almost exactly (max diff 0.07, from
  `DatasetContext`'s robust stats over a smaller episode set, the same documented caveat as
  `--episodes`); aaronsu11/so100_lego (100 ep, `--min-score 95`, 27 kept, this dataset *does*
  ship `meta/stats.json`, with `mean`/`std`/`min`/`max`/`q01`/`q99` but no `count` key — a real-
  world stats-key variant the synthetic fixtures didn't cover, confirmed it recomputes
  correctly anyway since exact tabular recompute doesn't need `count`) — the recomputed
  `action` mean/min/max matched a direct recomputation from the exported frames, `index`/
  `episode_index` maxed out at the correct new values (14985 / 26), and `info.json`'s
  `total_videos` (54 = 27 kept episodes * 2 cameras) and `total_chunks` (1) were both right;
  also ran on shylee/so100_cube (200 ep, `--min-score 97`, 191 kept). New
  `tests/test_export.py` cases (mirroring the existing v3.0 coverage) fail without this change;
  `tests/conftest.py`'s synthetic v2.x writer also had its `data_path`/`video_path` templates
  fixed to the real v2.x convention (`episode_chunk`/`episode_index` placeholders; it had been
  silently reusing v3.0's template, untested until this export path needed it).
- 2026-10-20: Added **`--min-score N`** to `demoqc export` as an alternative to `--keep-from`
  (an export follow-up: previously every export needed a separate `demoqc score --json` run
  just to produce the keep list). `--keep-from`/`--min-score` are now a mutually exclusive,
  required argparse group; `--min-score` scores the dataset in-process at default thresholds
  (`evaluate` + duplicate detection + `build_report`, the same path `demoqc score` uses) and
  feeds the resulting keep/trim lists straight into `export_dataset`, reusing the existing
  report-derived logic via two new pure helpers (`keep_from_report_dict`/`trim_from_report_dict`
  in `src/demoqc/export.py`) that the file-based `keep_from_report`/`trim_from_report` now also
  call. `push-report` stayed blocked again this run (`HF_TOKEN` is still unset in this
  environment) even though the Hugging Face Hub itself was reachable. Validated on
  masato-ka/so100_cutlery_handling_simple (50 ep, v2.1): `export --min-score 90` and
  `export --keep-from <score --json output> --min-score 90` produced byte-for-byte identical
  output directories (`diff -rq`), both dropping the same 5 episodes. Also validated on
  lerobot/svla_so101_pickplace (50 ep, v3.0) with `--apply-trim` added: both paths trimmed the
  same 1411 idle frames and again produced byte-identical directories. New tests in
  `tests/test_export.py` (`test_export_command_min_score_matches_a_separate_score_json_run`,
  `test_export_command_requires_exactly_one_keep_source`) fail without this change (confirmed
  via `git stash` on just `src/demoqc/cli.py`/`src/demoqc/export.py`).
- 2026-10-21: Added **v2.x `--apply-trim`** (the export follow-up: v2.x is most of the real
  validation set, and `--apply-trim` previously errored out on it unconditionally, so no v2.x
  export could ever cut idle frames). `_export_v2` now accepts the same `trim` mapping as
  `_export_v3`: for each kept episode it slices the parquet rows to the suggested `[start, end)`
  range, renumbers `frame_index`/`index` and shifts `timestamp` by `start`, and rewrites
  `meta/episodes.jsonl`'s `length` to the trimmed count (this now also always matches the data
  even when a trim isn't applied, the same correctness fix `--apply-trim` made for v3.0's
  `meta/episodes` `length` on 2026-10-10). Unlike v3.0, a v2.x video file has no per-episode
  `from_timestamp`/`to_timestamp` span in its metadata to narrow -- it's simply the whole
  original episode -- so trimming only the tabular data would leave a trimmed export's (still
  uncut) video longer than its data; `demoqc export` now appends a warning to the existing
  video-renaming one spelling this out, naming the `trim` frame range from the report as what
  the video itself must be cut to. Validated on two real v2.x datasets:
  masato-ka/so100_cutlery_handling_simple (50 ep, 47/50 episodes with a suggested trim) --
  `--apply-trim` dropped total frames from 29853 to 25052 (4801 trimmed, 160.0s at 30 fps), and
  re-scoring the export gave mean score 100.0 with 0 remaining trimmable idle and no new flags;
  aaronsu11/so100_lego (100 ep, `--min-score 95`, 27 kept) -- trimmed 978 idle frames (32.6s),
  re-scoring the export gave mean score 99.75 (up from the dataset's usual ~92, per the
  2026-10-19 Log entry) with 0 remaining trimmable idle. Also re-ran the existing v3.0
  `--apply-trim` validation on lerobot/svla_so101_pickplace (50 ep) to confirm no regression:
  still 1411 frames / 47.0s trimmed, byte-identical to the 2026-10-10 numbers. New tests in
  `tests/test_export.py` (`test_v2_apply_trim_cuts_frames_shifts_timestamps_and_updates_length`,
  `test_v2_apply_trim_rejects_out_of_range_trims`, `test_v2_apply_trim_for_a_dropped_episode_is_ignored`,
  `test_v2_export_command_apply_trim_warns_about_cutting_videos`) replace the old
  "rejected with a clear message" tests and fail without this change (confirmed via `git stash`
  on just `src/demoqc/export.py`/`src/demoqc/cli.py`).
- 2026-10-22: Added **`--episodes`/`--max-episodes` to `demoqc export`** (the "scale to large
  datasets" follow-up: previously only `score`/`dedupe` could narrow a huge dataset, so trying
  `export` on one meant downloading and scoring every episode first). `cli.py`'s export
  subparser now calls the existing `_add_episode_args` helper; `_load()` already threaded
  `episodes`/`max_episodes` through to `load_dataset` for every subcommand, so no change was
  needed in `export.py` itself — episodes outside the requested subset are simply never loaded,
  so neither `--min-score` scoring nor a `--keep-from` report can keep them, and `export_dataset`
  then runs against whatever got downloaded/read for that subset. Validated on real datasets:
  `demoqc export masato-ka/so100_cutlery_handling_simple --episodes 0-4 --min-score 0 -o out`
  (v2.1) downloaded only 9 files (5 episode parquets + metadata) instead of all 54 and wrote
  exactly 5 episodes / 2981 frames, matching a direct check of the source; also ran on
  `lerobot/svla_so101_pickplace --episodes 0-9 --min-score 80` (v3.0, single data file, so no
  download savings there since everything lives in one chunk — the documented best-effort
  fallback, not a bug) and got 10 kept episodes as expected. New tests in `tests/test_export.py`
  cover `--episodes`/`--max-episodes` restricting the keepable set even with a permissive
  `--min-score`, a `--keep-from` report rejected when it doesn't match the `--episodes` subset,
  and (mirroring `test_dataset.py`'s narrowing tests) a v2.x export succeeding when the data
  files for episodes outside `--episodes` are missing entirely, simulating a Hub download that
  was itself narrowed; all 4 fail with `SystemExit`/"unrecognized arguments" before this change.
- 2026-10-23: Added **`demoqc export --copy-videos`** (the export follow-up: previously every
  export left videos untouched and printed a warning telling the user to copy/rename them by
  hand, so a cleaned v2.x export wasn't directly trainable without a manual step). For v3.0,
  `export_dataset(..., copy_videos=True)` copies the source's `videos/` directory into the
  export unchanged (file indices stay valid, same as the existing manual instructions). For
  v2.x, it copies and renames each kept episode's video file (every camera) from
  `episode_<old index>.mp4` to `episode_<new index>.mp4` using `meta/info.json`'s `video_path`
  template, replacing the old "copy this by hand" warning entirely; a trimmed v2.x export still
  warns that the copied video's content itself (not just its filename) must be cut to the
  `trim` range, since this tool copies bytes but never decodes or re-encodes video. The CLI
  rejects `--copy-videos` for a Hub source up front (`ds.repo_id is not None`), since videos are
  never downloaded by this tool and there would be nothing on disk to copy. Video bytes are
  always copied as-is (`shutil.copyfile`/`copytree`), so this needed no new dependency and
  stays in the core path. Validated on two real datasets: `masato-ka/so100_cutlery_handling_simple`
  (50 ep, v2.1, `--min-score 90` dropping 5) — a script comparing every copied, renamed video's
  bytes against the original file for its old index confirmed all 45 kept videos matched
  exactly and the 5 dropped episodes' videos were not carried over under any name; and
  `lerobot/svla_so101_pickplace` (50 ep, v3.0, `--min-score 80`, keeping all 50) — `diff -rq`
  between the source and exported `videos/` directories reported no differences. New tests in
  `tests/test_export.py` (8 cases covering the v3.0 whole-directory copy, the v2.x rename
  mapping, a missing video file producing a warning instead of a crash, both CLI paths, the
  `--apply-trim` + `--copy-videos` interaction, and the Hub-source rejection) all fail without
  this change (confirmed via `git stash` on just `src/demoqc/export.py`/`src/demoqc/cli.py`).
- 2026-10-24: Added **`demoqc export --symlink-videos`** (the `--copy-videos` follow-up: on a
  large dataset, `--copy-videos` doubles the on-disk video size, which can be prohibitive). Does
  the same job as `--copy-videos` (copy the whole `videos/` directory for v3.0 since it's
  unmodified by export; copy-and-rename each kept episode's video for v2.x) but with symlinks
  (`Path.symlink_to`, resolved to the source's absolute real path) instead of byte copies, so it
  uses no extra disk space. Mutually exclusive with `--copy-videos` at the CLI (argparse
  mutually-exclusive group) and defensively in `export_dataset` itself (`ValueError` if both are
  passed to the library function directly). Added `ExportResult.videos_symlinked` alongside the
  existing `videos_copied`, and parallel warnings (missing video file, trimmed-but-uncut v2.x
  video content) worded for symlinks. Validated on two real datasets with their actual videos
  downloaded locally (`snapshot_download`, not just the metadata+parquet this tool normally
  fetches): `masato-ka/so100_cutlery_handling_simple` (50 ep, v2.1, `--min-score 90` dropping 5)
  — `--symlink-videos` produced a 1.8 MiB export vs. 325 MiB for the same export with
  `--copy-videos`, and a script comparing every one of the 45 kept, renamed videos' bytes against
  the `--copy-videos` output found zero mismatches, confirming the symlinks resolve to the right
  (renamed) content; and `lerobot/svla_so101_pickplace` (50 ep, v3.0, `--min-score 90
  --apply-trim`) — the single `videos/` symlink resolves to the source's `videos/` directory
  exactly (`Path.resolve()` equality) and the export is 448 KiB total vs. the source's own 28 KiB
  `du` for `videos/` alone (no extra video bytes either way). New tests in `tests/test_export.py`
  (9 cases mirroring the existing `--copy-videos` coverage, plus mutual-exclusion checks at both
  the library and CLI level) fail without this change — 8/9 with `SystemExit`/`AttributeError`,
  the 9th incidentally passing either way since an unrecognized flag and a mutual-exclusion error
  both exit 2 (confirmed via `git stash` on just `src/demoqc/export.py`/`src/demoqc/cli.py`).
- 2026-10-25: Fixed a near-duplicate-detection memory bug (the `duplicate` check's own
  follow-up, which claimed the pairwise distance computation was "fine up to a few
  thousand episodes" -- it undercounted by 2-3 orders of magnitude). `find_duplicates`
  built an `(n, n, D)` broadcast intermediate (`mat[:, None] - mat[None, :]`) to get the
  `(n, n)` distance matrix, where `D` is the resample length times active action dims
  (hundreds, typically); at a "few thousand episodes" that intermediate alone is tens of
  GB, long before the actual `(n, n)` result is a problem, so `demoqc score`/`demoqc dedupe`
  would OOM on any dataset big enough for this tool's own `--episodes`/`--max-episodes`
  support to matter. `dedupe.py` now has `_pairwise_mean_sq_l2`, which computes the same
  values via the Gram-matrix identity `||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b` (one BLAS
  matrix multiply), cutting peak memory from O(n²D) to O(n²+nD) with no change to the
  result. New `tests/test_dedupe.py` cases: one checks the new helper against a brute-force
  broadcast on a small matrix (values match to 1e-6; the tiny diagonal float-cancellation
  noise never matters since the caller overwrites the diagonal with `inf` right after); the
  other uses `tracemalloc` to assert peak memory for a 300x300 input stays under 20 MB --
  confirmed the naive broadcast actually peaks at ~413 MB for the same input (measured
  directly, not estimated), and the test fails both that way and via `ImportError` on the
  unmodified module (confirmed via `git stash` on just `src/demoqc/dedupe.py`). Validated on
  all 4 real datasets in the existing validation set (lerobot/svla_so101_pickplace,
  masato-ka/so100_cutlery_handling_simple, aaronsu11/so100_lego, shylee/so100_cube): mean
  scores (97.1/92.9/92.2/99.2) and `duplicate_groups` (0 on all 4) are byte-for-byte
  unchanged from the existing validated numbers, confirming the optimization changed no
  output.
- 2026-10-26: Added the **`video_freeze` check** (`--decode-video`, the `camera` check's
  highest-priority follow-up: it only ever reads LeRobot's own per-episode pixel mean/std,
  so a camera that goes black or a video that fails to decode *partway through* an episode
  -- a loose cable, a brief occlusion, a corrupted download -- averages out across the whole
  episode and is invisible to it). New optional `demoqc[video]` extra (`av`/PyAV, lazily
  imported only when `--decode-video` is passed and never touched by the default code
  path) and `src/demoqc/video.py`: samples up to 20 evenly-spaced frames per camera,
  downscaled to grayscale, and flags a contiguous near-black run covering >=15% of the
  samples (`video_dark_run_min_frac` in `Config`), or a file that raises on decode at all,
  as a decode error. Skips a camera the (cheaper, metadata-only) `camera` check already
  flagged dead for the whole episode, so the same defect isn't penalized twice. Scoped to
  **local LeRobot v2.x datasets only** this round -- a v2.x video's path is deterministic
  from its episode index (`Dataset.video_path_template`/`chunks_size`, now loaded from
  `info.json`), while v3.0 needs its own per-camera chunk/file-index lookup, not yet
  implemented (new follow-up on the `camera` check item above). The CLI rejects
  `--decode-video` for a Hub source (videos are still never downloaded, even with this
  flag) and for a missing `av` install, and warns and no-ops on a v3.0 dataset rather than
  erroring. Validated on a real dataset: downloaded `masato-ka/so100_cutlery_handling_simple`
  episodes 0-4's actual video files (`av1`-encoded, decoded fine via `libdav1d`) and ran
  `demoqc score --episodes 0-4 --decode-video`; the report was byte-for-byte identical to
  the same run without `--decode-video` (mean score 90.4 either way) -- no false positives
  on real, clean footage. Also truncated one real episode's video file to 2000 bytes to
  simulate a corrupted download: `video_freeze` fired on exactly that episode (ERROR
  severity, "failed to decode") and every other episode scored unaffected, confirming a
  bad video file doesn't abort the whole `score` run. New `tests/test_video.py` (13 cases):
  pure-Python tests of `longest_dark_run`/`resolve_v2_video_path` run in CI with no
  dependency; `av`-dependent tests (real tiny synthetic `.mp4`s, skipped via
  `importorskip("av")` when the extra isn't installed, which is always true in CI since
  `.[dev]` doesn't pull it in) cover a mid-episode dark run, an undecodable file, the
  check staying silent by default (`decode_video=False`), not double-flagging a
  `camera`-caught episode, and the CLI's three validation paths (Hub rejection, `av`
  missing, v3.0 warning). All fail without this change (confirmed via `git stash` on the
  implementation files, including an `ImportError` for the module itself).
- 2026-10-27: Added **LeRobot v3.0 support to `video_freeze`** (`--decode-video`'s
  highest-priority follow-up from yesterday: v3.0 datasets, including the project's own
  reference dataset, couldn't use the check at all). Unlike v2.x (one video file per
  episode, resolved directly from the episode index), a v3.0 video file holds several
  consecutive episodes' footage back to back, addressed by the `chunk_index`/`file_index`
  each episode's `meta/episodes` row names per camera (`videos/{camera}/chunk_index`/
  `file_index`, now loaded onto `Episode.video_chunk_file` in `dataset.py`) and resolved to
  a path with the new `video.resolve_v3_video_path`. `video.decode_samples` gained an
  optional `time_range` to narrow decoding to one episode's own
  `from_timestamp`/`to_timestamp` span within that shared file (seeking near the start
  first) instead of decoding the whole thing; `check_video_freeze` picks this path whenever
  an episode has a `video_chunk_file` entry for a camera, falling back to the existing v2.x
  per-episode-file path otherwise. The CLI's old "v3.0 not supported yet" warning is gone;
  `DatasetContext.build` now sets `video_root` for any local dataset, not just v2.x.
  Validated on a real dataset with its actual videos downloaded locally
  (`snapshot_download`, not just the metadata+parquet this tool normally fetches):
  `lerobot/svla_so101_pickplace` (50 ep, v3.0, 2 cameras, one shared video file per camera
  holding all 50 episodes) — `demoqc score --decode-video` produced a report byte-for-byte
  identical to the same run without `--decode-video` (mean score 97.13 either way, 0
  `video_freeze` flags) in 59s, confirming no false positives on real, clean footage and
  that the new per-episode seek-and-decode path doesn't change anything when there's
  nothing to find. Then truncated one camera's shared video file to 1/3 of its original
  byte length to simulate a corrupted/incomplete download: `video_freeze` fired a decode
  error on exactly the 33 episodes (17-49) whose `from_timestamp` falls past the truncation
  point, scoring each down by the check's fixed 40-point penalty, while all 17 earlier
  episodes sharing that same (now-truncated) file scored byte-for-byte identical to the
  clean baseline — confirming the per-episode time window is resolved and decoded
  independently, a corrupted tail doesn't propagate backward to episodes whose footage is
  still intact in the same file, and a bad shared file still doesn't abort the whole `score`
  run. New `tests/test_video.py` cases (`resolve_v3_video_path` pure-path tests,
  `decode_samples`'s `time_range` narrowing on a real tiny synthetic multi-segment `.mp4`,
  and a full two-episode-sharing-one-file `check_video_freeze` integration test) all fail
  without this change (confirmed via `git stash` on the implementation files -- the
  integration test fails with `AttributeError` since `Episode` had no `video_chunk_file`
  yet, and the CLI no-video-files test fails because the old code still printed and exited
  via the removed v3.0 warning path).
- 2026-10-28: Added **frozen-frame detection to `video_freeze`** (the `camera`/`video_freeze`
  check's last open follow-up: a camera or recorder that stalls and repeats the same frame
  without ever going dark -- a dropped/duplicated frame, not a disconnect -- previously had
  normal-looking pixel mean/std and no partial dark run, so it was invisible to every
  existing check). `video.FrameSample` now also carries a 32x32 block-averaged thumbnail of
  each sampled frame; `video.longest_frozen_run` finds the longest run of consecutive
  samples whose largest per-block difference stays below `Config.video_freeze_max_abs_diff`
  (0.03). `check_video_freeze` only flags a frozen run when the dataset's own action data
  shows the robot was *continuously* commanding motion throughout that span (the same
  "clearly moving" bar `check_staleness` uses via the new `_episode_moved` helper): a frozen
  frame during genuine idle time (the lead/trail idle almost every demo has) looks identical
  and isn't a defect on its own, so flagging it regardless would mean flagging ordinary,
  well-compressed footage. Dark and frozen detection are mutually exclusive per
  camera/episode (frozen is only checked when no dark run fired) so the same incident is
  never flagged twice.
  Validating this on real data caught two real false-positive bugs before they shipped, both
  found by running `--decode-video` against `masato-ka/so100_cutlery_handling_simple` (the
  dataset this project already uses for calibration) and diffing against the same run
  without the flag, exactly as this file asks: (1) the first version used mean squared
  thumbnail difference on an 8x8 thumbnail, which flagged 4 of the first 5 episodes --
  a camera positioned so the moving robot occupies only a small part of the frame makes the
  *average* pixel change negligible over a couple of seconds even with real, confirmed motion
  (checked directly: the `max abs diff` on the actual frames was 0.97-1.0, while the mean was
  as low as 7e-7). Switched to the *maximum* per-block difference on a 32x32 thumbnail
  (small moving regions still move at least one block noticeably; real validation found the
  smallest real-motion max-diff at this resolution was 0.058, comfortably above the 0.03
  threshold). (2) Even after that fix, one episode still false-positived: the frozen-run
  window (coarse, only ~3 of the 20 sampled frames) happened to span an episode's
  idle-to-moving transition, and checking the *max* speed anywhere in that window caught the
  tail end just crossing into motion even though the window was almost entirely idle.
  Switched `_episode_moved` to the *minimum* speed in the window, requiring motion for the
  *entire* span, not just somewhere in it -- re-validated and the false positive was gone
  with no loss of detection on the synthetic defect test. Final validation, after both
  fixes: `masato-ka/so100_cutlery_handling_simple` (20 episodes, their actual downloaded
  video files) and `aaronsu11/so100_lego` (10 episodes) both produced byte-for-byte identical
  reports with and without `--decode-video` (mean scores 94.59 and 88.6 unchanged); re-ran
  the full `lerobot/svla_so101_pickplace` (50 ep, v3.0, 2 cameras) validation from the
  2026-10-27 Log entry and got the same byte-for-byte identical report (mean 97.13) and the
  same 0 `video_freeze` flags on this clean reference dataset, confirming no regression to
  the existing dark-run/decode-error detection. Runtime on that dataset rose from the
  2026-10-27 baseline's 59s to 140s (`--decode-video` is opt-in and already the slow path;
  `_thumbnail` is a vectorized `np.add.reduceat` block-reduce, not a per-cell Python loop,
  to keep this overhead down). New `tests/test_video.py` cases: pure tests of
  `longest_frozen_run` (partial run, too-short run, no run, a whole-clip run, and the
  no-thumbnail fallback), an integration test with a real synthetic video frozen mid-episode
  while the (synthetic) robot keeps moving throughout (flags `video_freeze` as an error), and
  one with the same frozen frame placed during the episode's idle lead-in instead (stays
  silent) -- proving the false-positive guard this whole investigation was about. All of the
  pure tests and the moving-while-frozen integration test fail without this change (confirmed
  via `git stash` on the implementation files); the idle-guard test passes either way by
  design, since it's asserting the *absence* of a flag. Existing tests' "clean" synthetic
  video fixtures (constant solid-colour frames, fine when only pixel mean/std mattered) had
  to gain frame-to-frame variation (`_varying`, a 7-frame period chosen to not alias with
  `decode_samples`' sampling stride) so they aren't themselves mistaken for a frozen feed by
  the new detection -- real footage is never pixel-for-pixel static, so this is a fixture gap
  this change exposed, not new unrelated noise.
- 2026-10-29: Added the **`integrity` check** and fixed a real silent-corruption bug found
  while reviewing `checks.py` (the "Data-integrity check" item above). `DatasetContext.build`
  used plain `np.percentile`/`.std` to compute the dataset-wide `action_scale`; both
  propagate `NaN` across an entire column when even one value in it is non-finite, so a
  single corrupted frame in one episode's `action` (sensor glitch, truncated write) silently
  set that whole dimension's scale to `NaN` for *every other* episode too, not just the bad
  one -- quietly disabling `idle`/`smoothness` on that dimension and, worst, `gripper_chatter`
  (needs a finite scale, watches only one or two dims) dataset-wide, with nothing in the
  report pointing at why. Switched to `np.nanpercentile`/`np.nanstd` and added `integrity`
  (flags the actual corrupted episode: ERROR, non-finite value count per stream) so the real
  defect is visible on its own rather than only as unexplained collateral damage elsewhere;
  also made `dedupe._trimmed_action` skip a non-finite trajectory instead of feeding `NaN`
  into `np.interp`. New `tests/test_checks.py` cases (`integrity` on an injected NaN in
  `action`, separately in `observation.state`, plus a regression test reproducing the exact
  cross-episode `gripper_chatter` blinding above) all fail without this change (confirmed via
  `git stash` on `src/demoqc/checks.py`). Validated on real data: `lerobot/svla_so101_pickplace`
  (50 ep), `aaronsu11/so100_lego` (100 ep) and `shylee/so100_cube` (200 ep) all reproduced
  their existing validated mean scores and flag counts exactly (97.1 / 92.2 / 99.2) with zero
  `integrity` flags, confirming this is a no-op on real, clean data.
- 2026-10-30: Bounded `find_duplicates`' peak memory to a small constant instead of O(n^2) in
  episode count (the `duplicate` check's last open follow-up from 2026-10-25: that fix removed
  the O(n^2*D) broadcast intermediate but still built the full (n, n) distance matrix itself,
  which becomes the actual ceiling at a large enough episode count -- exactly the datasets this
  tool's own `--episodes`/`--max-episodes` support exists for). `HF_TOKEN` is still unset in
  this environment (confirmed via `HfApi().whoami()` raising `LocalTokenNotFoundError`, same as
  every prior run), so `push-report` stays blocked for the same reason as every earlier Log
  entry. `src/demoqc/dedupe.py` now has `_nearest_neighbor_distances`/`_pairs_within`, which
  compute the same Gram-matrix identity `_pairwise_mean_sq_l2` uses but one row block at a time
  (`_block_rows_for` sizes the block so each block's (block_rows, n) slab stays under a fixed
  element cap, so memory no longer grows with n at all); `find_duplicates` now calls these
  instead of building the full matrix, and only runs the O(n^2) `_pairwise_mean_sq_l2` on a
  single duplicate group's own (always small) rows, to get that group's max pairwise distance.
  This also replaces the old nested *Python* double loop (`for i: for j in range(i+1, n)`) that
  scanned the full matrix for pairs under threshold with a per-row-block vectorized scan, so
  it's faster as well as smaller, not just a memory/speed tradeoff. New tests in
  `tests/test_dedupe.py`: `_nearest_neighbor_distances`/`_pairs_within` exactly match a
  brute-force scan of the full `_pairwise_mean_sq_l2` matrix across block sizes smaller than,
  around, and far above `n*n` (so the blocking never changes the result, including at block
  boundaries); a `tracemalloc` test with a forced tiny block cap confirms peak memory for a
  2000-episode matrix stays under 4 MB against the 32 MB the full (n, n) matrix alone would
  need. All fail without this change (confirmed via `git stash` on just `src/demoqc/dedupe.py`
  -- `ImportError` on the new names). Validated on all 4 real datasets in the existing
  validation set (`lerobot/svla_so101_pickplace`, `masato-ka/so100_cutlery_handling_simple`,
  `aaronsu11/so100_lego`, `shylee/so100_cube`): ran `demoqc score --json` before and after this
  change (via `git stash` on just `dedupe.py`) and diffed the two JSON reports for all four --
  byte-for-byte identical, including `duplicate_groups` (empty on all 4, as in every earlier
  validation run), confirming the blocking changed no output, only how it's computed.
- 2026-10-31: Added **`demoqc push-report <repo_id>`** (the roadmap's top "Next up" item,
  repeatedly skipped on prior runs because there is no `HF_TOKEN` in this environment to open
  a real Hub PR). Scores the repo in-process and opens a pull request on it
  (`HfApi.create_commit(..., create_pr=True)`) adding `demoqc/report.json` and
  `demoqc/report.md`, with a shields.io score badge in the PR description for the owner to
  paste into their dataset card. `HF_TOKEN` is still unset here (confirmed via
  `HfApi().whoami()` raising `LocalTokenNotFoundError`), so the actual PR-opening call itself
  couldn't be run against a real repo -- attempting it against a real public dataset was in
  fact refused by this environment's own sandbox policy as a write/PR-opening action against
  public infrastructure, consistent with there being no token to do it with anyway. Everything
  else was validated for real: new `tests/test_push.py` mocks `HfApi.create_commit` to check
  the uploaded files, commit kwargs and PR-url handling (9 cases, all failing without this
  change), and `demoqc push-report lerobot/svla_so101_pickplace --dry-run` ran against the
  real, reachable Hub and reproduced the dataset's existing validated mean score (97.1) in the
  written report files. See the ROADMAP item above for the full validation notes.
- 2026-11-01: Fixed a real bug found while trying `demoqc score` against real Hub datasets
  beyond the existing validation set (the roadmap's "query the Hub for real most-downloaded
  small/medium teleoperation datasets" follow-up, done manually via `list_datasets(filter=
  "LeRobot", sort="downloads")` this run rather than as a shipped feature). Scoring
  `kmg0620/so101_bomb_pick_place_v1` (150 episodes, v3.0) gave a nonsensical result: 22
  episodes scored exactly 0.0 with `timing` ("hundreds of non-increasing timestamp steps"),
  `metadata` (declared length far below the data's actual length) and `video_sync` errors,
  plus bogus `length`/`smoothness`/`sync` flags riding along -- on a dataset whose other 128
  episodes were clean. Traced it to the dataset's own `meta/episodes/chunk-000/*.parquet`
  files: episode indices 81-102 each appear in *two* different meta chunk files with two
  different `length`/`data/file_index` values (confirmed by concatenating all 12 meta files
  and sorting by `episode_index` -- a second recording/upload session had renumbered from 0
  instead of continuing from 150's actual end). `load_dataset` groups frames by
  `episode_index` precisely so it doesn't care which physical file a frame came from, so for
  each affected index it silently summed two unrelated recordings' frames into one "episode"
  (e.g. episode 81: 564 + 380 = 944 frames, confirmed by reading both data chunk files'
  `episode_index`/`frame_index` columns directly) -- hence the "944 frames" that didn't match
  any sane `length` and the hundreds of non-increasing timestamp steps where the second
  recording's `frame_index`/`timestamp` restarted from 0 partway through. Fixed by detecting
  any `episode_index` that appears more than once across a dataset's own metadata (both the
  v3.0 `meta/episodes/*.parquet` path and the v2.x `meta/episodes.jsonl` path) and excluding
  it -- there's no principled way to tell which of the two recordings is the real one, so
  guessing or merging are both wrong; the fix surfaces a clear `Dataset.warnings` entry
  instead. New tests in `tests/test_dataset.py`
  (`test_duplicate_episode_index_v3_is_excluded_not_merged`, the v2.x equivalent, and a test
  that explicitly requesting one of the affected indices via `--episodes` raises a specific
  error) construct this exact scenario synthetically (two meta files and two data files both
  claiming `episode_index=1`) and all three fail without this change (confirmed via
  `git stash` on just `src/demoqc/dataset.py`). Validated on the real dataset: before the fix,
  `demoqc score kmg0620/so101_bomb_pick_place_v1` reported 150 episodes, mean score 83.5, 22
  episodes at 0.0; after, it reports 128 episodes (the 22 ambiguous ones correctly excluded
  with a printed warning naming them), mean score 97.9, and the *real* remaining defects on
  this dataset become visible instead of being drowned out: 7 genuine `video_sync` errors
  (video span far longer than the episode's own data duration) and a couple of real
  smoothness/length outliers. Also re-ran the full existing validation set
  (`lerobot/svla_so101_pickplace`, `masato-ka/so100_cutlery_handling_simple`) and got the same
  mean scores as every prior Log entry (97.1, 92.9) with zero new warnings, confirming this is
  a no-op on datasets that don't have the defect.
- 2026-11-02: Fixed `check_idle` silently missing a dataset-wide frozen action stream (see
  the "Next up" item above for the full root-cause and validation details). Found by
  scoring a real Hub dataset beyond the existing validation set
  (`omkarmayekar555/robot-learning-tutorial-data`, 1 episode, every action dim bit-identical
  for the whole episode): previously scored 90.0 with no flag mentioning the actual defect
  and a `RuntimeWarning` on stderr; now scores 30.0 with `[error] idle: no motion detected
  in the whole episode` and no warning. `_speed()` now returns 0 instead of NaN for a frame
  where every action dim is excluded from the dataset's scale (constant dataset-wide);
  `check_idle` now floors `idle_speed` at `1e-9` like `check_staleness`/`_episode_moved`
  already did; `DatasetContext.build` skips `nanpercentile` on `step_p99` dims that are
  already all-NaN instead of calling it and discarding a warning-raising NaN. New
  `tests/test_checks.py::test_frozen_action_whole_dataset_flags_no_motion` fails on the
  pre-fix code exactly as the real dataset did (confirmed via `git stash` on
  `src/demoqc/checks.py`). Re-validated the full existing dataset set
  (`lerobot/svla_so101_pickplace`, `masato-ka/so100_cutlery_handling_simple`,
  `aaronsu11/so100_lego`, `shylee/so100_cube`): byte-identical mean scores to every prior
  Log entry (97.1 / 92.9 / 92.2 / 99.2), confirming no regression on normal data.
- 2026-11-03: Fixed `gripper_chatter` and `sync` silently breaking on ALOHA-format datasets
  (see the "Next up" item above for the full root-cause and validation details). Found by
  scoring real Hub datasets beyond the existing validation set
  (`lerobot/aloha_static_cups_open`, `lerobot/aloha_static_battery`), whose
  `meta/info.json` groups per-dim feature names under a single motor key
  (`{"motors": [...]}`) instead of a flat list -- `demoqc`'s loader read that shape as-is,
  so `gripper_dims`' name search silently found nothing (no `gripper_chatter` detection at
  all on either dataset) and `_match_sync_dims` silently matched only 1 of 14 real dims
  (`sync` comparing a single joint instead of both arms). `dataset.py` now flattens a
  dict-shaped `names` into the flat per-dim list both checks already expected; a flat list
  passes through unchanged. `cups_open` went from mean 95.3 (7 spurious `sync` warnings)
  to 100.0; `battery` went from 90.7 (6 spurious `sync` warnings alongside 13 real
  `smoothness` ones) to 94.7 (spurious warnings gone, real ones unchanged). Re-validated
  the full flat-`names` set (`lerobot/svla_so101_pickplace`,
  `masato-ka/so100_cutlery_handling_simple`, `aaronsu11/so100_lego`, `shylee/so100_cube`):
  byte-identical mean scores to every prior Log entry (97.1 / 92.9 / 92.2 / 99.2).
- 2026-11-04: Fixed `check_idle` silently skipping 1- and 2-frame episodes (see the "Next
  up" item above for the full root-cause and validation details). Found by scoring a real
  Hub dataset beyond the existing validation set (`lerobot/svla_so100_stacking`): a
  1-frame episode (presumably an aborted recording) scored 90.0 with only an unrelated
  `length` warning, while three sibling episodes with the exact same bit-identical,
  zero-motion action but a few more frames correctly scored 20.0-30.0 with
  `idle: no motion detected`. `check_idle`'s `ep.length < 3` early return skipped its own
  absolute no-motion fallback along with the lead/trail profile it was meant to guard, so
  the shortest, most obviously broken episodes were the only ones left unflagged. New
  `tests/test_checks.py::test_very_short_frozen_episode_flags_no_motion` fails on the
  pre-fix code exactly as the real dataset did (confirmed via `git stash` on
  `src/demoqc/checks.py`). Re-validated the full existing dataset set
  (`lerobot/svla_so101_pickplace`, `masato-ka/so100_cutlery_handling_simple`,
  `aaronsu11/so100_lego`, `shylee/so100_cube`): byte-identical mean scores to every prior
  Log entry (97.1 / 92.9 / 92.2 / 99.2), confirming no regression on episodes with 3+
  frames.
- 2026-11-05: Fixed `check_smoothness` flagging a wrapped-angle rotation (an Euler angle
  crossing the `+-pi` boundary) as an abrupt action jump (see the "Next up" item above for
  the full root-cause and validation details). Found by scoring a real Hub dataset beyond
  the existing validation set (`RoboCOIN/Agilex_Cobot_Magic_heat_burger`, a dual-arm teleop
  dataset): nearly every episode carried spurious `smoothness` flags from its
  `eef_rot_euler_x/z_rad` action dims wrapping during otherwise smooth wrist motion.
  `checks.DatasetContext.build` now detects, from the data alone, which action dims are a
  radian angle wrapped to `[-pi, pi]` (bounded within it *and* reaching close to both ends),
  and both the `step_p99` baseline and `check_smoothness` itself diff those dims on the
  circle instead of linearly. New
  `tests/test_checks.py::test_smoothness_ignores_wrapped_angle_crossings` fails on the
  pre-fix code exactly as the real dataset did (confirmed via `git stash` on
  `src/demoqc/checks.py`). `demoqc score RoboCOIN/Agilex_Cobot_Magic_heat_burger
  --max-episodes 20` went from mean 71.0 (6/20 below 70) to mean 88.9 (1/20 below 70), with
  only the four actual wrapping dims detected as angular. Re-validated the full existing
  dataset set (`lerobot/svla_so101_pickplace`, `masato-ka/so100_cutlery_handling_simple`,
  `aaronsu11/so100_lego`, `shylee/so100_cube`) plus two ALOHA joint-space datasets with no
  wrapped dims (`lerobot/aloha_static_cups_open`, `lerobot/aloha_static_battery`):
  byte-identical mean scores and flag counts on all six, confirming no regression.
- 2026-11-06: Fixed `check_smoothness` flagging an Euler-angle gimbal-lock flip (a paired
  `_x_rad`/`_z_rad` axis swap near the un-exported pitch axis's own +-pi/2 singularity) as
  an abrupt action jump (see the "Next up" item above for the full root-cause and
  validation details). Found by scoring real Hub datasets beyond the existing validation
  set (`RoboCOIN/Agilex_Cobot_Magic_place_towel_flat`, and a residual flag on
  `_heat_burger` itself that yesterday's wrap fix had assumed was unrelated real
  jerkiness): both axes jump by a large, nearly identical amount in the same frame, with
  `x - z` barely moving even though each axis's own step is huge. New
  `tests/test_checks.py::test_smoothness_ignores_gimbal_lock_pair_cancellation` fails on
  the pre-fix code exactly as the real datasets did (confirmed via `git stash` on
  `src/demoqc/checks.py`). `place_towel_flat` (15 ep) went from mean 90.8 (3 `smoothness`
  flags) to 93.1 (0); `heat_burger` (20 ep) went from mean 88.9 (`smoothness` x9) to 94.1
  (`smoothness` x1, confirmed genuinely unrelated). Re-validated the full existing
  dataset set (`lerobot/svla_so101_pickplace`, `masato-ka/so100_cutlery_handling_simple`,
  `aaronsu11/so100_lego`, `shylee/so100_cube`, `jccj/so100_block_in_cup`) plus both ALOHA
  joint-space datasets: byte-identical mean scores and flag counts on all seven,
  confirming no regression.
- 2026-11-07: Fixed `check_smoothness` counting every frame of one sustained, smooth
  motion as a separate "abrupt action jump" (see the "Next up" item above for the full
  root-cause and validation details). Found by scoring a real Hub dataset beyond the
  existing validation set (`zekaiwang/trex_dataset`, a dual-arm dexterous-hand
  manipulation dataset): a finger joint sits still almost the whole episode, so its
  dataset-wide step baseline is tiny, and a single smooth, monotonic, 20-frame grasp-
  closing ramp was reported as "20 abrupt action jump(s)" even though none of those steps
  is actually discontinuous. `check_smoothness` now counts contiguous *runs* of leftover
  big-step frames (via the existing `_runs` helper) instead of every frame in the run
  separately. New `tests/test_checks.py::test_smoothness_counts_a_sustained_ramp_as_one_jump`
  fails on the pre-fix code exactly as the real dataset did (confirmed via `git stash` on
  `src/demoqc/checks.py`). `demoqc score zekaiwang/trex_dataset --max-episodes 50` went
  from mean 94.3 to 95.2 with the same 19 episodes flagged but far more accurate jump
  counts (e.g. 32 -> 8 on episode 12). Re-validated the full existing dataset set
  (`lerobot/svla_so101_pickplace`, `masato-ka/so100_cutlery_handling_simple`,
  `aaronsu11/so100_lego`, `shylee/so100_cube`, `jccj/so100_block_in_cup`) plus both ALOHA
  datasets: byte-identical on all but `aloha_static_battery`, whose mean rose from 93.3 to
  95.0 with the same 46 flags (more accurate severity, not a different diagnosis).
- 2026-11-08: Fixed `check_smoothness` flagging a noise-dominated dim (an unused torso/
  neck joint moved by nothing but its own sensor noise) as abrupt action jumps (see the
  "Next up" item above for the full root-cause and validation details). Found by scoring
  a real Hub dataset beyond the existing validation set
  (`RoboCOIN/AI2_Alphabot_2_insert_water_pipe`): its torso barely moves through the whole
  task, so `step_p99` for those dims reflects only sensor noise, and that noise -- scattered
  as separate single-frame blips across almost every episode, not one sustained ramp -- was
  flagged as a dataset-wide pile of "abrupt action jump(s)" (19/20 episodes, mean 85.1, 1
  below 70). A dim is now excluded from spike/jump detection if its `step_p99` is a robust
  log-space outlier (z > `smoothness_dim_outlier_z`, 4.0) among the dataset's *other* dims.
  New `tests/test_checks.py::test_smoothness_ignores_a_noise_dominated_dim` fails on the
  pre-fix code exactly as the real dataset did (confirmed via `git stash` on
  `src/demoqc/checks.py`). `demoqc score RoboCOIN/AI2_Alphabot_2_insert_water_pipe
  --max-episodes 20` went from mean 85.1 (1/20 below 70, `smoothness` x19 episodes) to mean
  94.1 (0/20 below 70, `smoothness` x4 episodes, a smaller related residual tracked as a
  follow-up). Re-validated the full existing dataset set
  (`lerobot/svla_so101_pickplace`, `masato-ka/so100_cutlery_handling_simple`,
  `aaronsu11/so100_lego`, `shylee/so100_cube`, `jccj/so100_block_in_cup`), both ALOHA
  datasets, both Agilex datasets (`heat_burger`, `place_towel_flat`, via `git stash`
  before/after) and `zekaiwang/trex_dataset` (58 action dims, the real safety test for a
  cross-dim outlier check): byte-identical mean scores and flag counts on all seven.
- 2026-11-09: Fixed the `ldlj_z` "jerkier than peers" smoothness flag firing on a moderate,
  probably-insignificant LDLJ gap purely because the peer group's own MAD happened to be
  tiny -- `check_relative` now also requires the gap from the median to exceed
  `Config.min_ldlj_diff` (1.0 nat) before flagging, the same "material difference, not
  just a large z on a tiny spread" guard the `length` check already had. Found on
  `EPITECH-LILLE/eval_training`: see the "Next up" entry above for the full root-cause and
  before/after numbers.
- 2026-11-10: Fixed `check_smoothness` flagging an ordinary gripper grasp/release as an
  abrupt action jump (see the "Next up" item above for the full root-cause and validation
  details). Found by scoring a real Hub dataset beyond the existing validation set
  (`allenai/MolmoAct2-BimanualYAM-Dataset`): every one of 13/20 episodes' spurious
  `smoothness` flags was on a gripper dim, never a joint. A gripper's rare, deliberate,
  full-range snap is invisible to the dataset-wide 99th-percentile step baseline (it's
  brief and rare relative to the episode), so any percentile-based outlier threshold flags
  every ordinary grasp. `DatasetContext.build` now excludes `ctx.gripper_dims` from
  spike/jump detection entirely (`checks._exclude_gripper_dims`), the same NaN-the-
  `step_p99` mechanism already used for constant and noise-dominated dims;
  `check_gripper_chatter` (hysteresis-based, not `step_p99`-based) is unaffected and still
  owns detecting real gripper misbehaviour. New
  `tests/test_checks.py::test_smoothness_ignores_gripper_snap` fails on the pre-fix code
  exactly as the real dataset did (confirmed via `git stash` on `src/demoqc/checks.py`).
  `demoqc score allenai/MolmoAct2-BimanualYAM-Dataset --max-episodes 20` went from mean
  88.2 (13/20 episodes, `smoothness` x13) to mean 92.9 (0 `smoothness` flags). Re-validated
  the full existing dataset set (`lerobot/svla_so101_pickplace`,
  `masato-ka/so100_cutlery_handling_simple`, `aaronsu11/so100_lego`, `shylee/so100_cube`,
  `jccj/so100_block_in_cup`) plus every gripper-bearing dataset from the smoothness-fix
  series (`RoboCOIN/Agilex_Cobot_Magic_heat_burger`, `_place_towel_flat`,
  `lerobot/aloha_static_cups_open`, `RoboCOIN/AI2_Alphabot_2_insert_water_pipe`) and the
  two without a gripper dim (`lerobot/aloha_static_battery`, `zekaiwang/trex_dataset`):
  byte-identical mean scores on all nine, confirming no regression.
- 2026-11-11: Fixed `check_staleness` flagging every episode of a dataset whose
  `observation.state` publishes slower than the 30 fps logging rate
  (`RoboCOIN/Galbot_G1_bag_table_trash_1`, a wheeled-base humanoid): all 15/15 episodes had
  a tight, consistent ~22% stale-state fraction (all 38 action/state dims simultaneously
  bit-identical to the previous frame on the same fraction of frames) against the check's
  fixed `stale_state_max_fraction` (0.1) -- the same tiny-spread/fixed-cutoff instability
  the `length` and `ldlj_z` checks were already guarded against, just not extended to
  `stale_state`. `DatasetContext.stale_state_baseline` (the dataset's own median per-episode
  stale fraction) now has to be exceeded by a material margin before flagging, a no-op on
  every dataset without this characteristic. `demoqc score
  RoboCOIN/Galbot_G1_bag_table_trash_1 --max-episodes 15` went from mean 77.3 (`stale_state`
  x15) to 97.8 (0 `stale_state` flags). See the Log entry under "Next up" for full details.
- 2026-11-12: Fixed `gripper_dims` (and therefore `_exclude_gripper_dims` and
  `check_gripper_chatter`) matching only a literal `*gripper*` feature name, missing a
  VR-teleop controller's `trigger`/`squeeze` axis driving the same open/close actuation
  (`BitRobot/HIW-500-LeRobot`, a Unitree G1 humanoid): 19/20 episodes carried a spurious
  `smoothness` "abrupt action jump" flag on `right_trigger`/`right_squeeze`, the same
  near-binary-snap false positive the literal-`gripper` case was already fixed for.
  `checks._GRIPPER_NAME_KEYWORDS` now also matches `trigger`/`squeeze`. `demoqc score
  BitRobot/HIW-500-LeRobot --max-episodes 20` went from mean 76.0 (`smoothness` x27) to
  mean 74.2 (`smoothness` x23, `gripper_chatter` x6 newly visible and confirmed genuine);
  byte-identical on the full existing validation set plus every other gripper-bearing
  dataset from earlier Log entries. See the Log entry under "Next up" for full details.
- 2026-11-13: Fixed `find_duplicates` flagging nearly every episode of a dataset with no
  literal `action`/`observation.state` feature as an exact duplicate of episode 0
  (`yaak-ai/L2D`, a car-driving dataset storing its data under structured per-key names
  like `action.continuous`/`observation.state.vehicle` instead): every episode's
  `action`/`state` loaded as `None`, so `_content_hash` produced the same empty digest for
  all of them. An episode with both `None` is now excluded from exact-duplicate hashing,
  and `load_dataset` adds a `Dataset.warnings` entry naming the missing feature instead of
  leaving every check silently empty. `demoqc score yaak-ai/L2D --max-episodes 15` went
  from mean 53.3 (`duplicate` x14) to 100.0 (0 flags, warning printed). See the Log entry
  under "Next up" for full details.
- 2026-11-14: Follow-up to the above: `load_dataset` now reconstructs a flat `action`/
  `observation.state` array for a dataset with no literal feature but a structured,
  dotted per-key representation (`action.continuous`/`action.discrete`,
  `observation.state.vehicle`/`.waypoints`/`.road`/...), instead of only suppressing the
  false duplicate flag and leaving every other check blind. `_structured_subfeatures`
  picks the numeric (non-`string`, non-`video`) sub-feature keys for `action`/
  `observation.state` from `meta/info.json`; `_concat_structured_columns` stacks and
  concatenates them per frame, skipping (not crashing on) any sub-feature whose declared
  `shape` doesn't actually hold row-to-row -- found on `yaak-ai/L2D` itself:
  `observation.state.waypoints` declares `shape: [10, 2]` but is stored as an Arrow
  variable-length nested list that round-trips ragged (`ValueError`/`TypeError` from
  `np.stack`/`np.asarray` without this guard, confirmed via a direct
  `load_dataset("yaak-ai/L2D", max_episodes=1)` call before the guard was added). `demoqc
  score yaak-ai/L2D --max-episodes 15` went from mean 100.0 (0 flags -- correct for
  "nothing was checked", but nothing to find either) to mean 87.4 (`idle` x15,
  `smoothness` x2, `length` x1, 3/15 below 70), all plausible real signal for a
  car-driving dataset (idle at red lights, mid-episode operator pauses, one aborted short
  demo) and no new `duplicate` false positives. New
  `tests/test_dataset.py::test_structured_action_state_is_concatenated_from_numeric_subfeatures`
  (asserts the reconstructed `ep.action`/`ep.state` and prefixed per-dim names against a
  new `structured_action_state` flag on `tests/conftest.py`'s `write_dataset`, which splits
  the existing flat action/state into two numeric dotted sub-features plus one `string`
  one, reproducing `yaak-ai/L2D`'s shape) and
  `tests/test_checks.py::test_structured_action_state_feeds_real_checks` (a frozen-action
  episode's `idle` flag now fires on a structured-feature dataset exactly as it would on a
  flat one); both fail without this change (confirmed via `git stash` on
  `src/demoqc/dataset.py` -- the dataset test gets the "no action or observation.state
  feature" warning instead of `[]`, and the checks test gets no `idle` flag since
  `ep.action` is `None`). Re-ran the full existing validation set
  (`lerobot/svla_so101_pickplace`, `masato-ka/so100_cutlery_handling_simple`,
  `aaronsu11/so100_lego`, `shylee/so100_cube`, `jccj/so100_block_in_cup`) and `pytest -m
  network`: byte-identical mean scores to every prior Log entry (97.1 / 92.9 / 92.2 / 99.2
  / 95.2), confirming this is a no-op whenever a dataset already has a literal `action`/
  `observation.state` feature.
  Follow-up: a `string`-typed structured sub-feature (road surface, lighting, etc.) is
  still never read by any check; turning a dataset's categorical state into a signal
  (e.g. a check for a stuck/constant categorical reading) is open but likely low value.
