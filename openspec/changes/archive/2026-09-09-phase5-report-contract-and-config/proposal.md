# phase5-report-contract-and-config

## Why

Phase 5 readiness items 1–4 are closed ([`docs/phase5-readiness.md`](../../../docs/phase5-readiness.md)); item 5 — "carry into the Phase 5 report contract" — is the last thing between the project and the paper-grade run, and the roadmap's first next change is to close it by committing the Phase 5 config. Its four items (present only the observed attribution component; make no cross-replicate driver-stability claim; run at the config's `n_jobs`; count the shared zero-effect anchor as one measurement) are all *evidenced* by the Phase 4 pilot but live only in prose: today the report renders all three attribution components side by side and plots cross-replicate Jaccard / sign agreement as "stability", the shard runner merely *warns* when `--n-jobs` overrides the config (and the sbatch template will forward it), and nothing in a run's outputs states that the anchor is one cell. The only config named "paper-grade", `examples/trajectory_power_study/study.json`, predates every decision since Phase 4: n = 300, `p_dmp = 0.2`, `n_jobs = -1`, `"surgery_censoring": "clamp"`, no matched seeds, no gate, and a specificity target on orientation → `shape` that the predeclared cross-talk (0.99–1.00 at CV rank, [ladder report](../../../docs/reports/latent-rank-ladder-2026-09-08.md) §6) would fail by construction. Running Phase 5 from it would re-open closed questions; running it from an ad-hoc edit would leave the contract unrecorded.

## What Changes

- **Declared report contract.** The study config accepts a top-level `report_contract` block: `driver_component` (which attribution component driver tables and figures present; Phase 5 declares `observed`), `cross_replicate_driver_agreement` (`descriptive` — kept in the raw CSV, absent from driver tables and figures, and labelled as not a stability claim; never claimable), and `n_jobs_override` (`forbid` | `warn`, default `warn` = today's behavior). The report echoes the resolved contract to `report/report_contract.json`, including the shared zero-effect anchor's cell id, the modes it resolves, and the `n_jobs` the records were produced with. The loader rejects unknown top-level keys so a misspelled contract fails loudly instead of being silently ignored.
- **Driver reporting honors the contract.** When a contract is declared the report writes `report/driver_report.csv` — one row per (mode, effect, transition) for the declared component only, with truth precision/recall, selected count, and *within-replicate* bootstrap stability — and the attribution figure plots within-replicate bootstrap stability only, retitled accordingly. `phase4_attribution.csv` keeps every component and the cross-replicate columns unchanged, so existing fixtures re-report byte-identically and configs without a contract see no change.
- **Worker-count lock.** With `n_jobs_override: forbid`, `scripts/run_study_shard.py` refuses a `--n-jobs` that differs from the config (non-zero exit, message naming both values and the signature consequence) instead of warning; the sbatch template documents that `STUDY_N_JOBS` will be refused by such a config.
- **Paper-grade profile** `examples/trajectory_power_study/phase5_power_study.json`: the chosen design point (ρ = 0, n = 1200, four stages, `p_dmp = 0.1`, default fail-loud censoring), pooled PLS on M-value methylation with the committed CV rank rule, all four modes, effects 0 / 0.25 / 0.50 / 0.75 / 1.00 (all under headroom at `p_dmp = 0.1`), 500 replicates × 999 permutations, `n_jobs = 1`, matched seeds with one shared anchor (new family), attribution on nonzero orientation cells (100 bootstraps, top-20), the Phase 4 gate enabled with the Phase 4 roles (orientation → `shape` and shape's off-diagonals *descriptive*), acceptance targets whose specificity list mirrors the gate's mandatory controls, and the report contract above. `study.json` stays as the historical record and is marked superseded.
- **Report template** `examples/trajectory_power_study/phase5_report_template.md`: the section skeleton the dated Phase 5 findings report must follow, encoding the contract items plus the two hand-offs from the design-point report (eigengap and `angle` null width beside every orientation result; the orientation surgery's non-ρ-invariant realized contrast stated as a construction limitation), the anchor statement, the `PROVENANCE.txt` field list, and reproduction commands without `--n-jobs`.
- **Cost rehearsal, not the study.** A one-replicate rehearsal of the profile (locally, reduced permutations) verifies the contract outputs and the gate/attribution path at n = 1200 and records the per-unit cost at 999 permutations for the READMEs. The 500-replicate run itself is the roadmap's next change.
- **Docs.** `docs/phase5-readiness.md` item 5 resolved; `docs/roadmap.md` "Not yet established" and "Next three changes" refreshed; study and examples READMEs (config table, outputs tree, Phase 5 section, `study.json` superseded); `CLAUDE.md` study bullet.

Out of scope: running the paper-grade grid; any change to the `delta`/`angle`/`shape` statistics, RRPP, the generator, or the surgery transforms; a design that holds the driver set fixed across replicates (the only way a stability claim could be made); automating `PROVENANCE.txt`; the Phase 6 real-data case study.

## Capabilities

### New Capabilities

(none)

### Modified Capabilities

- `trajectory-power-study`: the declarative configuration accepts a `report_contract` block and rejects unknown top-level keys; the report echoes the resolved contract and identifies the shared zero-effect anchor as one measurement; driver reporting presents only the declared component and labels cross-replicate agreement as descriptive; the shard runner enforces the configured worker count when the contract forbids overrides; the study provides a fixed Phase 5 paper-grade profile and a committed findings-report template that the Phase 5 report must follow.

## Impact

- `src/motco/simulations/study/config.py` — `ReportContract` dataclass on `StudyConfig`, `_build_report_contract`, unknown-top-level-key check, `dump_study_config` round-trip.
- `src/motco/simulations/study/phase4.py` / `report.py` — `build_driver_report` (declared component), `write_report_contract`, figure retitle/filter; both gated on a declared contract.
- `scripts/run_study_shard.py`, `scripts/motco_study_array.sbatch`, `scripts/motco_study.py` — override refusal; contract echo in `report`.
- `examples/trajectory_power_study/` — `phase5_power_study.json`, `phase5_report_template.md`, README; `study.json` marked superseded.
- Tests — contract parsing/validation/round-trip, unknown-key rejection, every committed config still loads and enumerates identically (cell-identity snapshot extended with the new profile), driver-report content and figure gating, byte-identical re-report of Phase 4 / design-point / ladder fixtures, runner refusal, profile test (cell count, headroom, seeds, no clamp, contract values, gate roles).
- Docs — readiness item 5, roadmap, READMEs, `CLAUDE.md`.
- Compatibility — no generator or evaluation-field changes; existing configs, shards, and committed results resume and re-report unchanged. Configs with stray top-level keys now fail to load (none of the committed ones do).
