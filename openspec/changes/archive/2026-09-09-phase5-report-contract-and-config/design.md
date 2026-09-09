# Design — phase5-report-contract-and-config

## Context

See `proposal.md` — Why. Relevant current state:

- `StudyConfig` is a frozen dataclass built by `_build_config`; every sub-block (`generator`, `evaluation`, `attribution`, `matched_seeds`, `acceptance.*`) rejects unknown fields, but the *top level* silently drops keys it does not know.
- Attribution results are summarized per `(mode, effect, cell, transition, component)` in `summarize_attribution` (`phase4.py`) with components `observed`, `pls_captured`, `residual`; the CSV `phase4_attribution.csv` carries both within-replicate bootstrap stability (`bootstrap_sign_stability_mean`, `bootstrap_top_k_frequency_mean`) and the cross-replicate `top_k_jaccard` / `sign_agreement`. `render_attribution_stability` already filters to `observed` but plots the cross-replicate lines under the title "Attribution stability".
- `scripts/run_study_shard.py` warns and proceeds when `--n-jobs` differs from the config; `motco_study_array.sbatch` forwards `STUDY_N_JOBS` when set.
- The shared zero-effect anchor is one `power_primary` cell (`trajectory_mode="none"`, `zero_effect_anchor=True`, `resolves_modes=[...]`); `build_operating_frame` expands it per mode and flags rows `from_shared_anchor`. The Type I *targets* read only `type_i_*` phases, so the anchor is never double-counted there; the gate's control observations list every `none`/translation cell once per statistic. Nothing in the outputs states the anchor's identity or that the modes share it.
- Fixture-based report tests require byte-identical re-reporting of the Phase 4, design-point, and ladder outputs.
- Costs measured 2026-09-08 at the design point: CV units median 57 s at 199 permutations; RRPP itself ≈ 0.2 s per 199 permutations in a rank-3 latent space; attribution was 0.45 % of Phase 4 compute at n = 300.

## Goals / Non-Goals

**Goals:**
- Every item-5 statement is either machine-enforced (runner refusal, driver table content) or machine-echoed (`report_contract.json`) from the committed config, so the findings report cites outputs, not memory.
- Zero behavior change for configs that declare no contract; existing fixtures re-report byte-identically.
- The Phase 5 profile is derived from the ladder's CV column so the paper-grade run is the same measurement at higher Monte Carlo precision, plus attribution.

**Non-Goals:**
- Automating `PROVENANCE.txt` (hand-written today; the template lists its fields).
- Renaming or removing the existing `phase4_*` output names for a Phase 5 run — they name the *mechanism* (gate, frames), not the phase.
- Any new statistic, estimator, or generator behavior.

## Decisions

**D1. The contract is a top-level `report_contract` block, and unknown top-level keys become errors.**
Alternatives: (a) fold the items into `acceptance` — wrong home, they are reporting/execution rules, not targets; (b) `metadata.notes` prose only — unenforceable, which is the status quo the item exists to fix. Fields:

| field | values | default | enforced by |
|---|---|---|---|
| `driver_component` | `observed` \| `pls_captured` \| `residual` | none (block optional) | `driver_report.csv`, attribution figure |
| `cross_replicate_driver_agreement` | `descriptive` | `descriptive` | driver table omits the columns; figure omits the lines; echo carries the statement. `claim` is not a legal value. |
| `n_jobs_override` | `forbid` \| `warn` | `warn` | `run_study_shard.py` |

The echo (`report/report_contract.json`) adds resolved facts: `n_jobs` from the records' evaluation params (asserted uniform), `zero_effect_anchor: {cell_id, resolves_modes, counted_as: 1}`, `driver_component`, and one plain-language statement per item. Unknown-top-level-key rejection is a one-line strictness change; the committed configs all use known keys (verified by the load-all test).

**D2. Driver output is a new file, not a filter on the existing CSV.**
`phase4_attribution.csv` stays complete (all components, cross-replicate columns) so the Phase 4 fixture is byte-identical and the raw evidence for "why not `pls_captured`" (precision 0.15) remains reproducible. `driver_report.csv` is what a paper table is built from: declared component only; columns `trajectory_mode, effect_size, transition_id, precision_mean, recall_mean, selected_count_mean, bootstrap_sign_stability_mean, bootstrap_top_k_frequency_mean, eligible/computed/failed_replicates`. Written only when a contract is declared. The figure `phase4_attribution_stability.png` becomes, under a contract, "Within-replicate bootstrap stability (<component> component)" with the two cross-replicate series removed; without a contract it is unchanged.

**D3. `forbid` refuses; it does not silently drop the flag.**
Alternative: ignore `--n-jobs` under `forbid`. Rejected — an operator who set `STUDY_N_JOBS` intended something, and silently running at 1 hides that the sbatch environment is wrong. Exit code 2 with both values and the signature consequence; `--n-jobs` *equal* to the config value is accepted (harmless). The sbatch template gains a comment; no logic change.

**D4. Profile = ladder CV column + paper-grade Monte Carlo + attribution + gate + contract.**
Derives from `phase5_latent_rank_ladder.json` (`metadata.derives_from`): identical `generator` and `evaluation.integration_params`; `design_grid` removed; `permutations` 999; `n_replicates` 500; new `base_seed` and family `phase5-primary` (a fresh matched-seed family, independent of both pilots). Choices inside it:
- *Effect axis 0/0.25/0.50/0.75/1.00.* Both pilots used four points to halve compute; the paper-grade curve wants the Phase 4 five-point axis, and at `p_dmp = 0.1`, ρ = 0 every point is under headroom (orientation saturates ≈ 1.69, translation ≈ 2.00). Cost: 19 cells (anchor + 16 power + `none` and translation Type I controls) × 500 = 9,500 units.
- *Attribution on nonzero orientation primary cells* (100 bootstraps, top-20, seed 0), as Phase 4. It is what the driver contract items govern, it is < 1 % of compute, and the paper's driver table needs it at the design point.
- *Gate enabled with Phase 4 roles*: `mandatory_power` magnitude/`delta`, orientation/`angle`, shape/`shape`; `mandatory_control` magnitude/`angle`, magnitude/`shape`; `descriptive` orientation/`delta`, orientation/`shape`, shape/`delta`, shape/`angle`; `control_modes` `none`, translation. The gate is the predeclared pass/fail for the paper.
- *Acceptance `specificity` mirrors the gate's mandatory controls* (translation × 3, magnitude/`angle`, magnitude/`shape`) and **drops** orientation/`shape`, shape/`delta`, shape/`angle` that `study.json` carried. Orientation → `shape` is predeclared projection-associated cross-talk (readiness item 2; ladder §6); a target that fails by construction is not a target. `type_i` `se_tolerance` 2.0; `power` floors 0.80 on the three diagonals.
- *No `surgery_censoring` key* (default `error`); `n_jobs` 1; `alpha` 0.05.
- `study.json` is not edited (historical record with `clamp`); the README marks it superseded by the new profile.

**D5. The findings-report template is a committed Markdown skeleton beside the config, not under `docs/reports/`.**
`docs/reports/` holds *dated findings*; mkdocs nav lists them explicitly. The template belongs with the profile it governs (`examples/trajectory_power_study/phase5_report_template.md`). Required sections: configuration and provenance (with the `PROVENANCE.txt` field list: config sha256, code revision, environment versions, unit timings, shard layout, error policy, `n_jobs`); unit accounting; gate decision; Type I (which cell each null claim reads; anchor is one measurement, `type_i_baseline` is a separate seed family); power per mode with eigengap terciles and `angle` null-width dispersion beside every orientation number; cross-talk (orientation → `shape` predeclared); drivers (declared component only, precision vs. truth, within-replicate bootstrap stability, explicit "no cross-replicate stability claim" with the reason); construction limitations (n-conditional orientation claim; orientation surgery's realized contrast not ρ-invariant; eigengap not ρ transfers to real data); reproduction without `--n-jobs`. A test asserts the template names each contract item so it cannot drift from the config.

**D6. Rehearsal, not run.**
One replicate of the profile with `permutations` reduced (a derived, uncommitted config under the scratch/results dir) exercises the gate + attribution + contract path at n = 1200 and measures the per-unit cost at 999 permutations for the README's Phase 5 cost line. The paper-grade run is a separate change so its report, provenance, and readiness/roadmap updates get their own review.

## Risks / Trade-offs

- [Unknown-top-level-key rejection breaks a user's local config] → Only committed configs are guaranteed; the error names the key. Accepted: silent ignore is what allowed contract drift.
- [Attribution cost at n = 1200 is unmeasured] → The rehearsal measures it; if bootstrap cost is material, lower `bootstrap_replicates` in the profile *before* the run, never after.
- [Effect 0.75 adds four cells (≈ 2,000 units, ≈ 35 core-hours)] → Accepted for the paper curve; the cluster ran 8,000 units in 20 minutes on 100 shards.
- [Dropping specificity targets could read as weakening acceptance] → They move to gate `descriptive` roles and stay reported in the specificity matrix; the change records the reason (predeclared cross-talk) in the config's metadata notes and the template.
- [The gate still writes `phase4_*` file names in a Phase 5 run] → Documented in the template and README; renaming would break fixture byte-identity for no scientific gain.

## Open Questions

None that change the specs or tasks. The number of SLURM shards and the results directory name are chosen at run time in the next change.
