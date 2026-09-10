# phase5-paper-grade-run

## Why

All five items of the [Phase 5 readiness worklist](../../../docs/phase5-readiness.md) are closed (items 1–3 by the
pivotality, reflection-policy and latent-rank work; item 4 by the [design-point pilot](../../../docs/reports/phase5-design-point-pilot-2026-09-08.md);
item 5 by `phase5-report-contract-and-config` on 2026-09-09), and the roadmap's first next change is to run the
study itself. The paper-grade grid has never run: every operating characteristic the paper will report is
currently measured at pilot precision (100 replicates × 199 permutations) at a design point that was chosen
*by* those pilots, and the orientation `angle` power claim — 0.85–0.88 at ρ = 0, n = 1200, with a 0.80
predeclared floor — sits close enough to the floor that pilot-grade Monte Carlo error (SE ≈ 0.032) cannot
settle it. The config, the report contract, the gate roles and the findings-report template are all committed
and frozen; nothing is left to decide before the run, and every further Phase 5 conclusion (the exit gate,
the Phase 6 real-data configuration) is blocked behind it.

## What Changes

- **Execute the committed paper-grade study.** `examples/trajectory_power_study/phase5_power_study.json`
  unchanged and unedited: 19 cells × 500 replicates = 9,500 units (verified by `enumerate_study`, no cell
  rejected by the headroom check, so no censored surgery under the default `"error"` policy), 999
  permutations per unit, on the `ing` SLURM cluster as a 100-shard resumable array with BLAS pinned to one
  thread, `--error-policy record`, and **no** `STUDY_N_JOBS` (the contract's `n_jobs_override: forbid` makes
  the runner exit 2 on any override). Budget ≈ 150 EPYC 7662 core-hours, ~1.5–2 h wall.
- **Merge, report, and commit the outputs** under `results/phase5-<run date>/`: `report/` — including the
  contract-mandated `report_contract.json` and `driver_report.csv`, the gate's `phase4_*` outputs, the
  eigengap-stratified power table and the attribution figure — plus a hand-written `PROVENANCE.txt` carrying
  the field list the template specifies. Shard and merged JSONL stay gitignored as regenerable.
- **Record the gate decision as it comes out.** The Phase 4 gate in the config is the predeclared pass/fail:
  mandatory power on magnitude/`delta`, orientation/`angle`, shape/`shape` at a 0.80 floor; mandatory control
  on magnitude/`angle` and magnitude/`shape`; control modes `none` and translation. This change reports the
  verdict and explains every deviation from a predeclared target as a method or claim revision — **never** as
  a Monte Carlo sample-size question. It does not implement a revision: if the gate fails, the remedy is
  scoped as its own change (the roadmap's Phase 5 exit review).
- **Write the dated findings report** `docs/reports/phase5-paper-grade-<run date>.md` following
  `examples/trajectory_power_study/phase5_report_template.md` section for section, with every number citing a
  committed CSV/JSON path, orientation power read beside the recorded eigengap distribution and the `angle`
  null-width dispersion, orientation → `shape` stated as predeclared projection-associated cross-talk rather
  than a specificity failure, the shared zero-effect anchor counted as one measurement, and the
  design-point hand-offs (the n-conditional claim; the orientation surgery's non-ρ-invariant realized
  contrast) carried as stated limitations.
- **Update the docs to the post-run state.** `docs/roadmap.md` — the Phase 5 section, "Not yet established"
  (the unrun grid line is retired), and "Next three changes"; `docs/phase5-readiness.md` — a closing status
  line pointing at the findings report; `docs/index.md` and the examples README where they name the run as
  outstanding.

Out of scope: any edit to the study config, the gate roles, the acceptance targets or the report contract
(they are predeclared — editing them after seeing results would void the claim); any change to the
`delta`/`angle`/`shape` statistics, RRPP, the generator or the surgery transforms; re-running at a different
design point, rank or continuity value; implementing whatever revision a failed gate would imply; the Phase 6
real-data case study.

## Capabilities

### New Capabilities

(none)

### Modified Capabilities

(none — this change runs the committed study and records its results; no requirement, contract or behavior
changes, so `.openspec.yaml` sets `skip_specs: true`. The `trajectory-power-study` capability already
specifies everything being exercised here.)

## Impact

- `results/phase5-<run date>/` — new: `report/` (gate, acceptance, spectrum, contract, driver and attribution
  outputs) and `PROVENANCE.txt` committed; `shard_*.jsonl` and `merged.jsonl` gitignored.
- `docs/reports/phase5-paper-grade-<run date>.md` — new, the dated findings report.
- `docs/roadmap.md`, `docs/phase5-readiness.md`, `docs/index.md`, `examples/trajectory_power_study/README.md` —
  status updates only.
- Cluster — `~/MOTCO` on `cluster.ing.uc.cl` must be fast-forwarded to the run revision and its venv synced
  before submission (it currently sits one commit behind `main`, without the paper-grade profile).
- No source, test, script or config file changes; the pre-commit gate is expected to be unaffected.
