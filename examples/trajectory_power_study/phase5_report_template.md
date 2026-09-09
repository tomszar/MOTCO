# Phase 5 paper-grade trajectory power study — <YYYY-MM-DD>

> **Template.** The dated Phase 5 findings report (`docs/reports/phase5-power-study-<date>.md`) must
> follow this skeleton: every section below is required, in this order, and every number it states must be
> readable out of a file under `results/phase5-<date>/report/`. The section prompts in *italics* say what
> each section must contain; replace them with the findings. The report contract this template encodes is
> the committed `report_contract` block of
> [`phase5_power_study.json`](phase5_power_study.json) — `driver_component` `observed`,
> `cross_replicate_driver_agreement` `descriptive`, `n_jobs_override` `forbid` — and the report must cite
> the run's `report/report_contract.json` echo for each item rather than restate it from memory.

**Run:** `results/phase5-<date>/` · **Config:** `examples/trajectory_power_study/phase5_power_study.json`
(sha256 from `PROVENANCE.txt`) · **Gate decision:** PROCEED / HOLD / INDETERMINATE (from
`report/phase4_gate_decision.json`).

## 1. Configuration and provenance

*State the design point and why it is fixed (ρ = 0, n = 1200, four stages, `p_dmp = 0.1`, pooled PLS on
M-value methylation, stage-supervised double-CV rank; each traced to the readiness item that chose it),
the Monte Carlo sizing (500 replicates × 999 permutations), the matched-seed family (`phase5-primary`) and
its independence from the two Phase 5 pilots, and that the configuration derives from the ladder's
cross-validated column (`metadata.derives_from`).*

Cite `PROVENANCE.txt`, which must carry at least these fields:

| field | content |
|---|---|
| `date_utc` | run date |
| `config` / `config_sha256` | path and sha256 of the committed config actually executed |
| `code_revision` | git revision (and, if any, a working-tree diff sha256 and the list of untracked files used) |
| `host` / `launch` | cluster, partition, and the exact `sbatch` line — with `STUDY_N_JOBS` **unset** |
| `blas_threads` | `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS` values |
| `units` | cells × replicates (19 × 500 = 9,500) |
| `versions` | python, motco, numpy, scikit-learn, scipy, uv |
| `job_id` / `outcome` | array job id; units present once, signature mismatches, failed records, censored surgeries |
| `wall` / `unit_timings` | wall clock; per-unit median/min/max seconds and core-hours (CV units, attribution units) |
| shard layout | `N_SHARDS` and how units map to shards (`--n-shards`) |
| `error_policy` | `record` (and how many failures it recorded) |
| `n_jobs` | the value the records carry, equal to `evaluation.n_jobs` = 1 — see `report/report_contract.json` |
| `merge` | the exact merge and report commands |

## 2. Unit and failure accounting

*Units expected vs. present once, parameter-signature mismatches, `status = failed` records and their
`diagnostic_error_type`, censored surgeries (`report/realized_surgery.csv`: `censored_fraction` must be 0 at
every cell), and attribution accounting on the orientation cells (`eligible` / `computed` / `failed`
replicates from `report/driver_report.csv`). A unit that is missing or failed is reported as such, never
dropped.*

## 3. Gate decision

*The Phase 4 gate's verdict (`report/phase4_gate_decision.json`) with every observation
(`report/phase4_gate.csv`) — mandatory power (magnitude/`delta`, orientation/`angle`, shape/`shape`, floor
0.80 at e = 1.00), mandatory control (magnitude/`angle`, magnitude/`shape`), descriptive (orientation/`delta`,
orientation/`shape`, shape/`delta`, shape/`angle`), Type I inflation on the control modes, completeness — and
any confirmation re-run the decision requires. The gate is the predeclared pass/fail; a HOLD is reported as a
HOLD.* Note that the gate's files keep their `phase4_*` names: they name the mechanism (gate, frames), not the
phase.

## 4. Type I error

*Per statistic and per control cell (`report/type_i_table.csv`, `report/acceptance_report.csv`). State which
cell each null claim reads: the `type_i_baseline` cells (`none` and translation — a separate seed family from
the power grid) carry the Type I acceptance target; the shared zero-effect anchor (`power_primary`,
`trajectory_mode = none`, cell id and `resolves_modes` in `report/report_contract.json`) is every mode's 0.00
power point and is **counted as one measurement**, never as four independent nulls. Translation at e > 0 is
reported as a negative control on all three statistics.*

## 5. Power per mode

*Power curves per mode × statistic (`report/power_curves.csv`, `report/power_curves.png`) over
0 / 0.25 / 0.50 / 0.75 / 1.00, with Monte Carlo SE. Beside **every orientation number** report the recorded
eigengap distribution (`report/config_spectrum.csv`; median and terciles at the anchor and at each effect) and
the `angle` null-width dispersion (`null_summary["angle"]["q95"]` median / IQR), and give orientation `angle`
power within eigengap terciles (`report/eigengap_stratified_power.csv`). Compare the diagonal at e = 1.00 with
the pilots (design-point 0.88, ladder CV column 0.85) at their pooled SE.*

## 6. Cross-talk

*The off-diagonal matrix (`report/specificity_matrix.csv`, `.png`) and the localization table
(`report/phase4_localization.csv`). State the **predeclared** orientation → `shape` cross-talk as such: it is
projection-associated cross-talk of the rank-3 stage-supervised latent space (readiness item 2; latent-rank
ladder §6, 0.99–1.00 at CV rank), not a finding about the constructions and not an acceptance target; it is a
descriptive gate role. Report shape's off-diagonals (`delta`, `angle`) as construction-present descriptives.
Magnitude's off-diagonals are mandatory controls and are reported against the α + 2·SE bound.*

## 7. Drivers

*Restricted to the declared component — `observed` — from `report/driver_report.csv`: per orientation
effect and transition, precision and recall against generator truth, mean selected count, and
**within-replicate** bootstrap stability (`bootstrap_sign_stability_mean`, `bootstrap_top_k_frequency_mean`;
figure `report/phase4_attribution_stability.png`, titled "Within-replicate bootstrap stability (observed
component)"). Then state explicitly:*

> **No cross-replicate driver-stability claim is made.** Cross-replicate top-k Jaccard and sign agreement
> (`top_k_jaccard`, `sign_agreement` in `report/phase4_attribution.csv`) are `descriptive` only: every
> replicate index draws a fresh differential-indicator set, so the true driver set genuinely differs across
> replicates and matched seeds pair cells *within* a replicate index, not across them. A stability claim needs
> a design that holds the driver set fixed, which this study is not.

*If the `pls_captured` component is mentioned at all, it is as the reason the observed component is the
declared one (Phase 4: precision 1.00 vs 0.15), citing `phase4_attribution.csv`.*

## 8. Construction limitations

*At minimum:*

- *The orientation power claim is **n-conditional**: it holds at n = 1200 (300 samples per group-stage cell)
  at the isotropic endpoint ρ = 0; along n at fixed ρ the eigengap is constant while the `angle` null width
  contracts, so the recorded eigengap distribution — not n — is the observable that carries the claim to real
  data.*
- *The orientation surgery's realized latent contrast is **not ρ-invariant**: the same nominal effect
  realizes a smaller contrast on a trending baseline (design-point pilot: median observed angle 28° at
  ρ = 0.5 vs 66° at ρ = 0), so orientation power at another continuity is a different construction, and
  baseline continuity must not be read as a lever that buys power.*
- *The eigengap, not ρ, transfers to real data; `delta`/`angle`/`shape` are measured within the constructed
  latent space, and the viz projection is display-only.*
- *Any deviation from the predeclared targets, with the revision it implies (method or claim — never the
  Monte Carlo size).*

## 9. Reproduction

Reproduction never passes a worker-count override: `n_jobs` is part of every cell's parameter signature and
the config's contract forbids it (the runner exits 2). Parallelize across shards.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN=results/phase5-<date>

# SLURM (partition/resource flags are cluster-specific); STUDY_N_JOBS must stay unset.
sbatch -p <partition> --cpus-per-task=1 --mem=2G --time=<limit> --array=0-99 \
    --export=ALL,STUDY_CONFIG=$(pwd)/examples/trajectory_power_study/phase5_power_study.json,STUDY_OUT=$(pwd)/$RUN,N_SHARDS=100 \
    scripts/motco_study_array.sbatch

# Locally, in K resumable shards (one process each):
for i in $(seq 0 $((K-1))); do
  uv run python scripts/run_study_shard.py \
      --config examples/trajectory_power_study/phase5_power_study.json \
      --out-dir $RUN --shard-index "$i" --n-shards K --error-policy record &
done; wait

uv run python scripts/motco_study.py merge  --out-dir $RUN
uv run python scripts/motco_study.py report \
    --config examples/trajectory_power_study/phase5_power_study.json --out-dir $RUN
```

The report writes `report/report_contract.json` and `report/driver_report.csv` beside the usual outputs and
the gate's `phase4_*` files; commit `report/` and `PROVENANCE.txt`, leave the shard and merged JSONL
gitignored.
