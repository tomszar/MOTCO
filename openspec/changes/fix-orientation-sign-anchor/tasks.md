## 1. Reproduce the defect as a failing test

- [x] 1.1 Add a test constructing a bent multi-stage trajectory whose first stage projects near zero onto its own centered principal axis, asserting two identical copies report `angle` ≈ 0; verify it FAILS against the current estimator (it should report near 180°).
- [x] 1.2 Add a repeated-perturbation test asserting the orientation sign is stable across small noise draws on that geometry; verify it FAILS against the current estimator.
- [x] 1.3 Add a null-configuration test asserting no pair of trajectories differing only by translation, uniform scale, or small noise reports an `angle` approaching 180°; verify it FAILS against the current estimator.

## 2. Fix the estimator

- [x] 2.1 Change the sign anchor in `_estimate_orientation` (`src/motco/stats/trajectory.py`) to net displacement `PC1 · (X_raw[-1] − X_raw[0])`, computing PC1 from the centered configuration as before; verify the three tests from group 1 now pass.
- [x] 2.2 Document the deviation in the docstring — the reference line it departs from (`evo_649_sm_suppmat.r:64`), that the raw anchor makes the sign translation-dependent, and the invariance net displacement provides; verify a reader can find the reason at the implementation.
- [x] 2.3 Verify reference fidelity is preserved by running the existing R-fixture regression tests against `results_example1.csv` and `results_example2.csv` and confirming they pass unchanged.

## 3. Complete the invariance contract

- [x] 3.1 Add tests for direction semantics: reversing stage order negates the orientation vector, and two-stage orientation equals the unit transition vector; verify both pass.
- [x] 3.2 Add tests for translation invariance including trajectories placed on opposite sides of the coordinate origin, and for uniform-scale invariance; verify both report zero angle within tolerance.
- [x] 3.3 Add a test recovering a constructed known angle, including a case above 90°, so the fix cannot be satisfied by collapsing all angles toward zero; verify it passes.
- [x] 3.4 Run the pre-commit gate — `uv run ruff check src/ tests/ && uv run mypy src/motco/ && MOTCO_TEST_PERMS=99 uv run pytest tests/ -m "not slow" --tb=short` — and verify all three pass.

## 4. Measure the corrected operating characteristics

- [x] 4.1 Add a re-run configuration covering the orientation, shape, and translation cells at the Phase 4 parameters and seeds, writing to its own dated results directory; verify it loads and enumerates exactly the intended cells.
- [x] 4.2 Run it to completion and merge; verify zero failed replicates.
- [x] 4.3 Recompute the observed-angle distributions per cell and verify the 150–180° mass is gone from the null cells — the direct check that the artifact is eliminated in the study path, not just in unit tests.
- [x] 4.4 Report corrected orientation `angle` power against the 0.80 floor, corrected shape-cell `angle` behavior, and the translation control's rejection rates; verify the translation control still holds nominal level at every effect size.
- [x] 4.5 Recompute the realized-geometry orientation checkpoints (population and latent) and verify whether the previously reported 81° and 57° figures change.

## 5. Record the finding

- [x] 5.1 Write a dated report under `docs/reports/` covering the defect, its mechanism, the anchor decision and why the faithful port was rejected, the four-regime comparison, fixture-equality evidence, and the corrected operating characteristics; verify every number traces to a committed test or record set.
- [x] 5.2 Mark the superseded Phase 4 orientation conclusions in `docs/reports/phase4-medium-pls-pilot-2026-08-27.md` without altering its original numbers, following the precedent set when it superseded the July pilot; verify the original report remains intact and the supersession is visible at the top.
- [x] 5.3 Update `docs/phase5-readiness.md` item 1 and `docs/roadmap.md` with the corrected diagnosis and power result; verify both link the new report.
- [x] 5.4 Update the `diagnose-angle-null-pivotality` change to record that its hypothesis was displaced by this defect and that it proceeds only if a shortfall survives the corrected re-run; verify its artifacts remain internally coherent.
- [x] 5.5 Note in `CLAUDE.md` that `trajectory.py`'s orientation sign convention deviates deliberately from the reference supplement; verify the note is accurate and brief.
