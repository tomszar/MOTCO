> **Status: confirmed — proceed.** This change's hypothesis was briefly thought to be displaced by an
> orientation sign-anchor defect (`../fix-orientation-sign-anchor/`). That defect was real and is fixed, but
> the corrected re-run of 2026-08-28 shows it is **not** the cause: orientation `angle` power rose only from
> 0.65 to 0.68 against the 0.80 floor, the curve is still flat, and the near-180° null mass is unchanged at
> 21 of 700 null replicates. See `docs/reports/orientation-sign-anchor-2026-08-28.md`. The shortfall and the
> rejection inversion survive correction, so the non-pivotality hypothesis stands and this change proceeds
> as written.

## 1. Null summary in the evaluation harness

- [x] 1.1 Add a JSON-safe per-statistic null summary (retained draw count, mean, sd, q50/q90/q95/q99) to `SimulationEvaluationResult`, computed from the existing null vectors whenever `permutations > 0`; verify with a unit test in `tests/test_simulation_evaluation.py` that the summary is present, its quantiles match `numpy.quantile` on the same draws, and non-finite draws are excluded with the retained count reduced.
- [x] 1.2 Verify the summary is independent of `include_null_distributions`: a test asserting the summary is present with the flag off and the full vectors absent, and both present with the flag on.
- [x] 1.3 Verify the summary is inert: a test running the same evaluation inputs twice asserting p-values, pair statistics, and every other result field are unchanged from the pre-change values, and that `permutations=0` yields no summary.

## 2. Persistence through the grid record

- [x] 2.1 Add the additive null-summary field to `SimulationReplicateResult` in `src/motco/simulations/grid.py` and populate it in `run_simulation_replicate`; verify with a test that a completed replicate's persisted record carries the summary from the evaluation result.
- [x] 2.2 Verify backward compatibility: a test that `_replicate_result_from_dict` loads a record dict written without the field and yields an empty summary, and that a record from a `permutations=0` run is distinguishable from a missing summary.
- [x] 2.3 Verify the parameter signature is untouched: a test asserting `parameter_signature(cell)` is unchanged for an unmodified cell and that a run resumes against pre-change records without overwrite.
- [x] 2.4 Confirm the committed Phase 4 outputs are unaffected by regenerating `results/phase4-2026-08-27/report/` from its existing records and verifying the outputs are byte-identical.

## 3. Diagnostic profile

- [x] 3.1 Add `examples/trajectory_power_study/angle_pivotality_diagnostic.json`: orientation, translation, and magnitude at effect 1.00, 100 replicates, 199 permutations, `n_jobs = 1`, and every other generator/evaluation parameter matching `phase4_pilot_100x199.json`; verify it loads through `load_study_config` and `enumerate_study` produces exactly the three intended cells.
  - **Measured deviation:** `enumerate_study` produces **five** cells, not three. It unconditionally emits a `none` Type I baseline (`enumerate_type_i_grid`) and a `translation` negative control (`_negative_control_cells`) on top of the three requested power cells; the profile cannot suppress them without changing the enumerator. They are kept rather than worked around: the cross-replicate counterfactual calibrates its `z` against exactly those null-control replicates, so they are the diagnostic's reference distribution. Cost rises from ~3.7 to ~6.1 core-hours. Pinned in `tests/test_angle_pivotality_diagnostic.py::test_diagnostic_profile_enumerates_the_three_cells_plus_the_mandated_controls`.
- [x] 3.2 Verify the profile is runnable end to end at small scale by running the study shard runner with a reduced replicate override and confirming completed records carry non-empty null summaries.

## 4. Pivotality analysis

- [x] 4.1 Add `src/motco/simulations/pivotality.py` with the per-cell, per-statistic association between each replicate's observed statistic and its own null mean, sd, and q95, reported with an uncertainty measure; verify with unit tests on synthetic records covering a constructed strong-association case and a constructed no-association case.
- [x] 4.2 Add the rejection-outcome split (mean observed statistic among rejecting vs non-rejecting replicates, per cell and statistic); verify with a synthetic-record test that reproduces a known inversion.
- [x] 4.3 Add the cross-replicate standardized counterfactual: per-replicate `z = (observed − null_mean) / null_sd` calibrated against the null-control cells' `z` distribution, reporting the standardized rejection rate beside the as-specified rate for every cell including the controls; verify with a synthetic-record test that the standardized rate recovers rejections only where the null tracks the observed statistic.
- [x] 4.4 Verify explicitly that within-replicate studentization is a no-op: a test asserting the p-value is unchanged when observed and null are standardized by the same constants, so the analysis cannot be misread as recommending it.
- [x] 4.5 Add `scripts/angle_null_pivotality.py` as a thin argparse entry point reading merged JSONL via `read_replicate_results` and writing the tables; verify `--help` runs and the script produces tables from the small-scale records of task 3.2.

## 5. Run the diagnostic

- [ ] 5.1 Run the diagnostic profile to completion into its own dated output directory under `results/`, sharded, and merge; verify all three cells report zero failed replicates and every completed record carries a null summary.
- [ ] 5.2 Run the analysis over the merged records and verify all four outputs are produced: association tables, rejection split, standardized counterfactual with control rates, and the uncertainty measures.

## 6. Findings and Phase 5 decision

- [ ] 6.1 Write the dated findings report under `docs/reports/`, citing the code revision, configuration, and record set behind every number, stating the pivotality verdict, and stating whether the measured association accounts for the Phase 4 rejection inversion; verify every reported number is traceable to a committed record set.
- [ ] 6.2 State exactly one Phase 5 consequence in the report — `angle` proceeds as specified, is replaced by a pivotal or studentized statistic, or carries a revised power target — and include the explicit statement that the cross-replicate counterfactual is a diagnostic, not a deployable test.
- [ ] 6.3 Record the resolution in `docs/phase5-readiness.md` (item 1 closed with the decision, and its consequences for items 2–4 noted where they change) and in `docs/roadmap.md`; verify both documents link the new report.
- [ ] 6.4 Document the null-summary record field and the diagnostic commands in `simulations/study/README.md` and the API docs; verify the reproduction commands run as written.

## 7. Gate

- [ ] 7.1 Run the pre-commit gate — `uv run ruff check src/ tests/ && uv run mypy src/motco/ && MOTCO_TEST_PERMS=99 uv run pytest tests/ -m "not slow" --tb=short` — and verify all three pass.
