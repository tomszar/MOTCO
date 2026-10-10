# Tasks

## 1. Signature-preservation baseline

- [x] 1.1 Capture a pre-change fixture of cell ids, parameter signatures and matched seeds for **every** committed config under `examples/trajectory_power_study/` (not only the ten historical ones), committed under `tests/data/`, and add a test asserting the current code reproduces it; verify the test passes on the unchanged code

## 2. Explicit group × stage sizes (generator)

- [x] 2.1 Add `group_stage_sizes` to `SemiSyntheticTrajectoryParams` with validation (two rows, row length = `n_stages`, every cell ≥ 1, conflict error naming any non-default `n_samples`/`stage_sample_prop`/`group_ratio`); verify with unit tests for each rejection scenario in the generator delta spec
- [x] 2.2 Route sizing through the table when set, and record realized group × stage sizes in truth only when set; verify a test generates `((11, 10, 28), (9, 10, 12))` and counts exactly those cells and 80 samples
- [x] 2.3 Omit `group_stage_sizes` from the `parameter_signature` payload while it is `None` (design D1); verify the task 1.1 fixture test and `tests/test_magnitude_default_pins.py` still pass, and that an absent-table dataset and truth are byte-identical to the pre-change output at a fixed seed

## 3. Evaluation-time block selection (harness)

- [x] 3.1 Add a validated `selected_layers` resolver for `integration_params.layers` (canonical order; non-empty, known, duplicate-free); verify unit tests for the three invalid-selection cases
- [x] 3.2 Make the preprocessor fit and transform only the selected layers and record its fitted layers, and make `concatenate_blocks` follow them; verify a preprocessing test on a two-layer selection, and that transforming with a mismatched layer set raises
- [x] 3.3 Thread the selection through `concat`, `snf` and `pls` integration and through integration metadata (key recorded only when set); verify per-method tests that the outcome/latent matrices contain only selected-layer features
- [x] 3.4 Restrict `calculate_realized_geometry` to the selected layers at every checkpoint (per-block scopes for selected layers only, joint scopes over them); verify a test that a methylation + expression evaluation reports no proteomics scope
- [x] 3.5 Restrict attribution's per-layer feature indexing and original-unit scales (`evaluation.py` scale concatenation included) to the selected layers; verify an attribution test that every feature record is methylation or expression
- [x] 3.6 Verify default byte-identity: a test evaluating a fixed dataset with and without an explicit all-three selection, and against a pre-change result fixture with the key absent, all equal (excluding runtime fields)

## 4. Study configuration support

- [x] 4.1 Normalize `generator.group_stage_sizes` from nested JSON lists to tuples in the config loader and design-grid axis values; verify a round-trip test where reloading yields identical cell ids and signatures
- [x] 4.2 Confirm `evaluation.integration_params.layers` works as a design-grid axis with shared matched seeds; verify an enumeration test that two columns differing only in `layers` get identical generator params and seeds
- [x] 4.3 Verify a design grid over `generator.group_stage_sizes` enumerates one anchored power grid per table, and re-run the task 1.1 fixture test to confirm committed configs are unchanged

## 5. Phase 6 profiles

- [x] 5.1 Write `examples/trajectory_power_study/phase6_small_n_pilot.json` per design D4/D5 (baseline column only, 100 × 199, shared axis `0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0`, `magnitude_kind: joint`, new matched-seed family); verify it loads, enumerates and passes the headroom check
- [x] 5.2 Add a slow-marked smoke test that runs one replicate of every pilot cell at a few permutations end to end; verify it passes locally and that PLS CV is feasible at the 9-sample cells
- [x] 5.3 Document the profiles and the two new keys in `examples/trajectory_power_study/README.md`, `src/motco/simulations/study/README.md`, `docs/api/simulations.md` and `CLAUDE.md`; verify the docs name both keys and their byte-identity guarantee
- [x] 5.4 Run the pre-commit gate (`ruff`, `mypy`, fast tests); verify all three pass

## 6. Pilot run and effect-axis decision

- [x] 6.1 Run the pilot on `ing` (per the SLURM handbook) and merge and report it into `results/phase6-small-n-pilot-<date>/`; verify zero failures and a recorded median per-unit time
- [x] 6.2 Choose the paper-grade effect axis per mode from the pilot curves (rise and saturation of each target statistic) and decide whether the per-axis split applies; record the reasoning in `results/phase6-small-n-pilot-<date>/NOTES.md`
- [x] 6.3 Write `phase6_small_n_study.json` (and the split profile if 6.2 requires it) with the design grid (four crossed columns; see design D4), ≥ 500 × 999, Phase 5 report contract, advisory gate rules, and metadata naming the pilot evidence; verify it loads and enumerates, that the anchor is shared across split profiles, and that the cost estimate from the 6.1 unit time is recorded

## 7. Paper-grade run and findings

- [x] 7.1 Run the paper-grade study on `ing`, then merge and report; verify unit and failure accounting against the enumerated count
- [x] 7.2 Write `docs/reports/phase6-small-n-<date>.md` following `phase5_report_template.md`, with per-statistic Type I and power, the eigengap distribution with tercile-stratified orientation power, the separate cohort-size and block-count contrasts, and the per-statistic interpretability statement; verify every required item in the findings-report spec is present
- [x] 7.3 Update `docs/roadmap.md` (Phase 6 status, the small-n result, and the link to the external case-study repo and pinned versions); verify the roadmap's Phase 6 section cites the report
