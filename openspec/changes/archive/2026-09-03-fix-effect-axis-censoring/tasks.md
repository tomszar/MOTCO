# fix-effect-axis-censoring — Tasks

## 1. Generator censoring policy

- [x] 1.1 Pin the current clamped behavior: add a regression test that generates orientation,
      translation, and shape/relocate datasets at saturating effects with fixed seeds and asserts
      the realized surgery sizes and a dataset checksum, so `"clamp"` can later be verified
      replicate-for-replicate identical to today's generator (test passes against the unmodified
      code before any refactor).
- [x] 1.2 Add `surgery_censoring: Literal["error", "clamp"] = "error"` to
      `SemiSyntheticTrajectoryParams`, validate it in `_validate_params`, and thread it into
      `_relocate_rows` and `_translation_methyl`; verify an invalid value raises
      `SemiSyntheticTrajectoryError`.
- [x] 1.3 Implement the `"error"` policy: when the requested surgery exceeds the pool, raise
      `SemiSyntheticTrajectoryError` naming the mode, requested size, pool size, and this draw's
      saturating effect; verify with tests for orientation and translation at saturating effects
      and confirm the policy check consumes no RNG draws (the `"clamp"` path still passes 1.1).
- [x] 1.4 Record nominal size and `censored` flag beside the existing realized-size keys in the
      transform truth metadata for all three pool-limited surgeries; verify via tests that
      `censored` is true exactly when realized < nominal and that uncensored generation is
      byte-identical to the pre-change generator at the same seed.
- [x] 1.5 Update the module docstring, `SemiSyntheticTrajectoryParams` docstring, and any generator
      tests exercising e ≥ saturation to either opt into `"clamp"` or use headroom-respecting
      effects; verify `uv run pytest tests/test_semisynthetic.py -v` passes.

## 2. Enumeration headroom check

- [x] 2.1 Implement `expected_surgery_headroom(params)` (orientation and shape via the binomial
      expectation with a 3σ guard band, translation via the cached reference incidence maps) and
      unit-test it against empirical pool sizes from generated datasets across seeds.
- [x] 2.2 Add the enumeration-time validation in `study/enumerate.py` (skipped for cells whose
      generator opts into `"clamp"`), erroring with cell id, requested effect, and saturating
      effect; verify a synthetic config with an over-headroom orientation cell fails enumeration
      and a headroom-respecting config enumerates with unchanged cells and seeds.
- [x] 2.3 Verify the acceptance criterion: a Phase-4-shaped config (orientation/translation at
      0.75 and 1.00, `p_dmp=0.2`, four stages) under the default policy fails enumeration on
      exactly the censored cells (test in `tests/test_study_config.py` or a new
      `tests/test_effect_axis_censoring.py`).

## 3. Summaries and study report

- [x] 3.1 Implement `summarize_realized_surgery(records)` in `grid.py` (per-cell nominal, mean/
      min/max realized, censored fraction; fields absent for modes without pool-limited surgery)
      and export it from `motco.simulations`; verify with unit tests over synthetic records
      including a mode without surgery truth.
- [x] 3.2 Add `realized_surgery.csv` to the study report and the censored fraction to the power-
      curve frame; verify report generation on synthetic records emits the table and annotation.
- [x] 3.3 Implement the duplicate-construction flag (per same-family cell pair, fraction of
      replicate indices where both records are censored with equal realized size; flag > 5%) in
      the report; verify with synthetic records reproducing the 80/100 audit pattern that the pair
      is flagged and an uncensored pair is not.

## 4. Configs, docs, and gate

- [x] 4.1 Add `"surgery_censoring": "clamp"` to the generator block of the committed historical
      configs (`phase4_pilot_100x199.json`, `phase4_smoke.json`, `orientation_signfix_rerun`,
      `angle_pivotality_diagnostic`) and document in `examples/trajectory_power_study/README.md`
      why they carry it and that new configs must not; verify all committed configs still
      enumerate.
- [x] 4.2 Confirm the signature break is correct: test that a record produced with the old
      parameter signature refuses to resume against a new-signature cell
      (`SimulationGridError`), and that `tests/test_angle_pivotality_diagnostic.py` still passes
      with the updated config.
- [x] 4.3 Update `CLAUDE.md` (semisynthetic bullet) and `docs/phase5-readiness.md` item 4 to record
      that P2 landed; verify the pre-commit gate passes
      (`uv run ruff check src/ tests/ && uv run mypy src/motco/ &&
      MOTCO_TEST_PERMS=99 uv run pytest tests/ -m "not slow" --tb=short`).
