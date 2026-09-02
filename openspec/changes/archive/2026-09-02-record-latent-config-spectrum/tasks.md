# Tasks — record-latent-config-spectrum

## 1. Spectrum computation in the estimator path

- [x] 1.1 Add the configuration-spectrum computation to `src/motco/stats/trajectory.py`: from the fitted LS-mean vectors and contrast, produce per-group and pooled centered stage-mean configurations, their normalized eigenvalue spectra, and the relative eigengap, behind an opt-in keyword on `estimate_difference` whose default preserves the existing return contract exactly. Verify with unit tests: a straight trajectory yields eigengap ≈ 1, a constructed isotropic simplex yields eigengap ≈ 0, a two-stage configuration yields exactly 1, a zero-variance configuration yields the explicit undefined sentinel, and the values match an independent SVD.
- [x] 1.2 Verify the audit-definition mirror: on a balanced design, the pooled LS-mean eigengap equals (to numerical tolerance) the eigengap computed from pooled sample stage means as in the audit's reproduction snippet.
- [x] 1.3 Forward the opt-in flag through `RRPP` in `src/motco/stats/permutation.py` (serial loop and `_RRPPWorker`), returning the per-permutation pooled eigengap list alongside the three existing distributions only when requested. Verify with tests that the default call returns the pre-change tuple unchanged, and that with equal seed and `n_jobs` the null distributions are byte-identical with the flag on and off (serial and parallel).

## 2. Harness recording

- [x] 2.1 Add the observed configuration-spectrum block to `SimulationEvaluationResult` beside `null_summary`, computed on every evaluation; verify with a test in `tests/test_simulation_evaluation.py` that the block is present, JSON-safe, and matches an independent recomputation from the evaluation's design and latent matrix.
- [x] 2.2 Add the permutation eigengap summary (retained count, mean, sd, q05/q50/q95) when `permutations > 0`; verify the summary is present with permutations, absent at `permutations=0` (while the observed block remains), and that no per-permutation vectors leak into the result.
- [x] 2.3 Verify inertness: a test running identical inputs pre- and post-change paths asserting observed statistics, p-values, and every pre-existing result field are unchanged.

## 3. Grid persistence and signature

- [x] 3.1 Add the additive spectrum field to `SimulationReplicateResult`, populate it in `run_simulation_replicate`, and round-trip it through JSONL serialization; verify with a persistence test.
- [x] 3.2 Add `config_spectrum_version` to the `parameter_signature` payload (pattern of `realized_geometry_version`); verify the signature changes exactly once for an unmodified cell, is stable across enumerations under the new schema, and that resume against a pre-change record's signature is refused with the existing mismatch error.
- [x] 3.3 Verify legacy loading: `_replicate_result_from_dict` loads a record dict without the field as an empty block, distinguishable from a recorded degenerate spectrum.
- [x] 3.4 Confirm committed evidence is untouched — **substituted, see note**: regenerate `results/phase4-2026-08-27/report/` and the pivotality tables from their existing merged records and verify byte-identical outputs.

  As specified this task cannot be executed. The merged and shard JSONL under `results/` are **gitignored as
  regenerable** (`results/**/merged.jsonl`), so the "existing merged records" for Phase 4 and for the
  pivotality run are not in the repository or on the machine; and re-generating the Phase-4 records at the
  current revision would not reproduce its committed report in any case, because that report predates the
  deliberate orientation sign-anchor fix (`fab94d6`) — which is why `results/orientation-signfix-2026-08-28/`
  exists. Two substitutes were executed instead, and together they cover the claim the task is protecting:

  1. **Pre-change records, both code paths.** A record set was produced from a `git worktree` at the parent
     commit (144 records, `examples/trajectory_power_study/smoke.json`, no `config_spectrum` field) and
     reported twice — once with the pre-change code, once with this change. `specificity_matrix.csv`,
     `power_curves.csv`, `type_i_table.csv`, `acceptance_report.csv`, and `acceptance_report.json` are
     **byte-identical**; the two new tables appear alongside them.
  2. **Inertness at study scale.** The acceptance run (task 5.2, same config as the 2026-09-01 diagnostic)
     reproduces every published rate and slope of that run exactly — see
     `docs/reports/latent-config-spectrum-2026-09-02.md` §3.

## 4. Study report stratification

- [x] 4.1 Add per-cell eigengap summary columns (pooled and per-group) to the study geometry tables in `src/motco/simulations/study/` reporting; verify with a synthetic-records test.
- [x] 4.2 Add the eigengap-stratified rejection-rate table for orientation-mode power cells (within-cell terciles of the recorded pooled eigengap, per-stratum counts and Monte Carlo SEs); verify with synthetic records constructed so the strata have known rates, and verify records lacking the block yield an explicit "unavailable" rendering with all pre-existing tables unchanged.

## 5. Pivotality covariate and acceptance run

- [x] 5.1 Extend `src/motco/simulations/pivotality.py` with the per-cell, per-statistic association between recorded pooled eigengap and the replicate's own null q95 (Spearman and log–log Pearson, with the existing uncertainty treatment), reported as unavailable when spectra are absent; verify with synthetic-record tests covering a constructed negative association and a no-association case.
- [x] 5.2 Run the committed angle-pivotality diagnostic profile under the new schema into a dated `results/` directory (sharded, merged); verify zero failed replicates and every record carries the spectrum block.
- [x] 5.3 Run the pivotality analysis over the new records; verify the orientation cell reproduces the audit's association — negative and material for `angle` (audit: Spearman −0.75 at n = 100) — and record the measured values in a short dated note under `docs/reports/` citing config, revision, and record set.

  Measured: orientation cell, `angle` vs its own null q95 — Spearman **−0.749** [−0.824, −0.648], log–log
  Pearson **−0.622** (audit: −0.75 and −0.62), against +0.093 for `delta` and −0.032 for `shape` in the same
  cell. Note: `docs/reports/latent-config-spectrum-2026-09-02.md`; records:
  `results/latent-spectrum-2026-09-02/` (500 replicates, 0 failures, 500 carrying the block).

## 6. Documentation

- [x] 6.1 Document the spectrum record fields, the signature version bump and its resume consequence, and the stratified report table in `simulations/study/README.md`; verify reproduction commands run as written.
- [x] 6.2 Update `docs/phase5-readiness.md` (items 1, 3, 4: the resolvability measure is now recorded per replicate) and `docs/roadmap.md` to link the acceptance note; verify links resolve.

## 7. Gate

- [x] 7.1 Run the pre-commit gate — `uv run ruff check src/ tests/ && uv run mypy src/motco/ && MOTCO_TEST_PERMS=99 uv run pytest tests/ -m "not slow" --tb=short` — and verify all three pass.
