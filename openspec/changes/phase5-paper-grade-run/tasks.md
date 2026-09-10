## 1. Pre-flight

- [x] 1.1 Confirm the workstation tree is at the run revision on `main` with the committed profile present, and record `sha256sum examples/trajectory_power_study/phase5_power_study.json` plus `git rev-parse HEAD`; verify `git status --porcelain` shows no tracked-file modifications.
- [x] 1.2 Re-verify enumeration locally: `enumerate_study` on the committed profile returns 19 cells × 500 replicates = 9,500 units with no headroom rejection; verify the printed cell count and unit total match the config's own cost note.
- [x] 1.3 Fast-forward `~/MOTCO` on `ing` to the run revision and `uv sync`; verify `git rev-parse HEAD` matches task 1.1, `git status --porcelain` is empty, the config's sha256 matches, and `.venv/bin/python -c "import motco; print(motco.__version__)"` plus numpy/scikit-learn/scipy versions are recorded for `PROVENANCE.txt`.
- [x] 1.4 Check cluster capacity before submitting: `df -h /home1` shows enough headroom for ~250 MB of JSONL, `sinfo -p 512x1024` shows the partition up, and `MaxArraySize ≥ 100`; verify `STUDY_N_JOBS` is unset in the submitting shell.

## 2. Submit and run

- [x] 2.1 Submit the array from `~/MOTCO` on `ing`: `sbatch -p 512x1024 --cpus-per-task=1 --mem=2G --time=6:00:00 --array=0-99 --export=ALL,OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,STUDY_CONFIG=$PWD/examples/trajectory_power_study/phase5_power_study.json,STUDY_OUT=$PWD/results/phase5-<UTC date>,N_SHARDS=100 scripts/motco_study_array.sbatch`, with `STUDY_N_JOBS` unset; verify the job id is returned and record the exact launch line.
- [x] 2.2 Monitor to completion (`squeue`, then `sacct -j <id> --format=JobID,State,ExitCode,Elapsed,MaxRSS`); verify all 100 array tasks report `COMPLETED` with exit `0:0` and note the wall interval, longest task, and peak RSS.
- [x] 2.3 Resubmit only failed or timed-out array ids with identical exports if any task did not complete, and verify the resubmitted tasks complete and that resumption skipped the already-recorded replicates (shard line counts grow only by the missing units).

## 3. Merge and verify completeness

- [x] 3.1 Run `python scripts/motco_study.py merge --out-dir results/phase5-<UTC date>` on the cluster; verify the reported unit count is 9,500 with no duplicate (cell, replicate) pairs.
- [x] 3.2 Verify the record set against enumeration: all 19 enumerated cell ids present, every parameter signature matching, 0 failed records, and 0 censored surgeries (`summarize_realized_surgery` reports `censored` false everywhere with realized = nominal); record the counts for `PROVENANCE.txt`.
- [x] 3.3 Extract per-unit timings (median/min/max and total recorded core-hours, separately for the attribution-bearing orientation cells) for `PROVENANCE.txt`; verify the total is within the ~150 EPYC core-hour budget or explain the difference.
- [x] 3.4 Rsync `merged.jsonl` (and the shard JSONL) to `results/phase5-<UTC date>/` on the workstation; verify the local checksum matches the cluster copy and that both paths remain gitignored.

## 4. Report

- [x] 4.1 Generate the report on the workstation: `python scripts/motco_study.py report --config examples/trajectory_power_study/phase5_power_study.json --out-dir results/phase5-<UTC date>`; verify it exits cleanly and writes `report_contract.json`, `driver_report.csv`, the `phase4_*` gate outputs, the Type I table, power curves, specificity matrix, `eigengap_stratified_power.csv`, `realized_surgery.csv`, and the figures.
- [x] 4.2 Verify the report contract is honored in the outputs: `report_contract.json` resolves `driver_component: observed`, `cross_replicate_driver_agreement: descriptive`, `n_jobs_override: forbid`, echoes a uniform `n_jobs = 1`, and names the shared anchor's `cell_id`, `resolves_modes` and `counted_as: 1`; `driver_report.csv` carries the observed component only and no `top_k_jaccard` / `sign_agreement` columns; the attribution figure is titled for within-replicate bootstrap stability.
- [x] 4.3 Cross-check against the two pilots before writing prose (design decision D6): orientation `angle` power at e = 1.00 against 0.85 (ladder CV column) and 0.88 (design-point pilot), magnitude `delta`, shape `shape`, and the shared anchor's three rejection rates; verify each agrees within Monte Carlo error and investigate any material discrepancy before proceeding.
- [x] 4.4 Read the predeclared gate verdict from `report/phase4_gate.json` and the acceptance-target report, and record the per-rule outcome (mandatory power, mandatory control, descriptive) verbatim for the findings report.

## 5. Write up and commit

- [x] 5.1 Hand-write `results/phase5-<UTC date>/PROVENANCE.txt` to the template's field list (date, revision and clean-tree state, config path and sha256, host/partition/CPU, launch line, BLAS settings, cells/units, versions, job id, outcome counts, wall interval, unit timings, merge/report split); verify every field is filled and the launch line matches what was actually submitted.
- [x] 5.2 Write `docs/reports/phase5-paper-grade-<UTC date>.md` following `examples/trajectory_power_study/phase5_report_template.md` section for section — gate verdict, Type I, power beside the recorded eigengap distribution and `angle` null-width dispersion, specificity with orientation → `shape` stated as predeclared projection-associated cross-talk, the driver table under the observed-component contract with no cross-replicate stability claim, the anchor counted once, the n-conditional and non-ρ-invariant limitations, and reproduction commands without `--n-jobs`; verify every number cites a committed CSV/JSON path and no template section is left unaddressed.
- [ ] 5.3 Commit `results/phase5-<UTC date>/report/` and `PROVENANCE.txt`; verify `git status` shows no JSONL staged and that the committed `report/` regenerates byte-identically from the merged JSONL.
- [x] 5.4 Update `docs/roadmap.md` (Phase 5 section, "Not yet established" — retire the unrun-grid line, "Next three changes"), `docs/phase5-readiness.md` (closing status line linking the findings report), and `docs/index.md` / `examples/trajectory_power_study/README.md` where they name the run as outstanding; verify no doc still describes the paper-grade run as pending.
- [x] 5.5 Run the pre-commit gate (`uv run ruff check src/ tests/ && uv run mypy src/motco/ && MOTCO_TEST_PERMS=99 uv run pytest tests/ -m "not slow" --tb=short`) and verify all three pass.
