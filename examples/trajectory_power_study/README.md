# Trajectory power study

A reproducible, sharded study that characterizes the Type I error and power of the
MOTCO trajectory test (delta, angle, shape) under semi-synthetic InterSIM datasets.

## Workflow

```
study config (YAML/JSON)
    │
    ▼  enumerate_study → SimulationGrid (Type I + power cells)
    │
    ▼  shard runner: scripts/run_study_shard.py
    │     (one cluster array task per shard, writes shard_<i>.jsonl)
    │
    ▼  merge: python scripts/motco_study.py merge --out-dir <dir>
    │     (combines shards into merged.jsonl, dedup by (cell, replicate))
    │
    ▼  report: python scripts/motco_study.py report --config <cfg> --out-dir <dir>
          (per-statistic + combined-rule summaries → specificity matrix,
           power curves, Type I table; CSV + PNG; acceptance-target report)
```

## Local smoke run

```bash
# Generate a few shards locally (no cluster):
python scripts/run_study_shard.py \
    --config examples/trajectory_power_study/smoke.json \
    --out-dir /tmp/motco-smoke \
    --shard-index 0 --n-shards 4 --error-policy record
python scripts/run_study_shard.py \
    --config examples/trajectory_power_study/smoke.json \
    --out-dir /tmp/motco-smoke \
    --shard-index 1 --n-shards 4 --error-policy record
# (repeat for shards 2 and 3, or run with --n-shards 1 to do them all locally)

python scripts/motco_study.py merge --out-dir /tmp/motco-smoke
python scripts/motco_study.py report \
    --config examples/trajectory_power_study/smoke.json \
    --out-dir /tmp/motco-smoke
```

Outputs land under `/tmp/motco-smoke/report/`:

- `specificity_matrix.csv` / `.png` — mode × statistic rejection rates
- `power_curves.csv` / `.png` — per-statistic rejection rate vs effect size
- `type_i_table.csv` / `type_i.png` — null-cell per-statistic + combined-rule rates
- `acceptance_report.csv` / `.json` — pre-specified targets evaluated against
  observed Monte Carlo uncertainty (non-gating)

## SLURM cluster run

```bash
# Submit an array of size N_SHARDS:
sbatch \
    --array=0-63 \
    --export=ALL,STUDY_CONFIG=$(pwd)/examples/trajectory_power_study/smoke.json,STUDY_OUT=$(pwd)/results,N_SHARDS=64 \
    scripts/motco_study_array.sbatch

# After completion:
python scripts/motco_study.py merge  --out-dir results
python scripts/motco_study.py report --config examples/trajectory_power_study/smoke.json --out-dir results
```

Failed array tasks can be resubmitted with `--array=7,12,40` — the shard-resume
guard (parameter signature) skips already-completed replicates.

## Config quick reference

| Field             | Purpose                                                    |
|-------------------|------------------------------------------------------------|
| `intersim`        | Baseline InterSIM params (R generator)                     |
| `generator`       | Baseline semi-synthetic perturbation params                |
| `evaluation`      | Integration method, RRPP permutations, n_jobs              |
| `trajectory_modes`| Power-grid modes (e.g. `magnitude`, `orientation`, …)      |
| `effect_sizes`    | Power-grid effect sizes                                    |
| `axes`            | OFAT axes, namespaced `intersim.` / `generator.` / `evaluation.` |
| `design_grid`     | Crossed design points: `{"axes": {...}}`, every axis listing its baseline value (see below) |
| `n_replicates`    | Replicates per cell                                        |
| `base_seed`       | Deterministic seed root                                    |
| `alpha`           | Significance level for rejection rates                     |
| `acceptance`      | Pre-specified Type I, power, and specificity targets       |
| `acceptance.gate` | Phase 4 gate parameters (see below); omit for pre-Phase-4 configs |
| `attribution`     | Which cells get orientation-attribution diagnostics        |
| `matched_seeds`   | Opt-in matched generator seeds across primary cells        |
| `generator.surgery_censoring` | Pool-limited-surgery policy; leave at the `"error"` default (see below) |

`none` is always present as the Type I baseline (enforced by enumeration);
`translation` is added explicitly as a second negative control.

### `generator.surgery_censoring` — why every committed config sets `"clamp"`

`orientation`, `translation`, and `shape` with `shape_kind="relocate"` draw
their surgery from a finite pool of CpGs, so a large `group_effect_size` can
request more sites than the pool holds. The generator's default policy,
`"error"`, refuses to realize a partial surgery, and enumeration rejects any
cell whose requested effect exceeds the expected pool headroom before compute
is spent.

**Every config in this directory predates that policy** and was run under the
old silent clamping, so each one carries an explicit
`"surgery_censoring": "clamp"` to stay loadable and enumerable as the record of
what was actually run. At `p_dmp = 0.2` with four stages the axis saturates at
roughly `e ≈ 0.56` for `orientation` and `e ≈ 0.29` for `translation` — above
those, distinct requested effects produced near-identical realized datasets
(see [the geometry audit](../../docs/reports/geometry-audit-2026-09-01.md),
finding F2). Their top cells are therefore **not** independent power
measurements, and `realized_surgery.csv` in the study report flags the affected
pairs.

**A new config must not copy the flag.** Leave `surgery_censoring` at its
default and choose an effect axis that respects the headroom — lower the axis
top, lower `p_dmp`, or use fewer stages. Enumeration reports the saturating
effect for the offending cell, which is the number to design against.

## Phase 4 pilot

`phase4_pilot_100x199.json` is the committed Phase 4 medium pilot: pooled PLS on
M-value methylation, `n_samples=300`, four stages, 100 replicates per cell, 199
RRPP permutations, the four established modes, and effects `0.00`–`1.00` by
`0.25`. It supersedes `pilot_50x199.json`, which predates the corrected shape
estimator and realized-geometry diagnostics and is retained as historical
evidence only. `phase4_smoke.json` exercises the identical code paths at
development scale; its numbers carry no scientific meaning.

```bash
# Development-scale smoke over every Phase 4 path (PLS, matched seeds,
# attribution, geometry, gate report):
python scripts/run_study_shard.py \
    --config examples/trajectory_power_study/phase4_smoke.json \
    --out-dir /tmp/motco-phase4-smoke \
    --shard-index 0 --n-shards 1 --error-policy record
python scripts/motco_study.py merge  --out-dir /tmp/motco-phase4-smoke
python scripts/motco_study.py report \
    --config examples/trajectory_power_study/phase4_smoke.json \
    --out-dir /tmp/motco-phase4-smoke

# The medium pilot (1900 work units) on SLURM:
sbatch --array=0-63 \
    --export=ALL,STUDY_CONFIG=$(pwd)/examples/trajectory_power_study/phase4_pilot_100x199.json,STUDY_OUT=$(pwd)/results/phase4,N_SHARDS=64 \
    scripts/motco_study_array.sbatch
python scripts/motco_study.py merge  --out-dir results/phase4
python scripts/motco_study.py report \
    --config examples/trajectory_power_study/phase4_pilot_100x199.json \
    --out-dir results/phase4
```

Do **not** pass `--n-jobs` for a Phase 4 run. RRPP seeds one RNG stream per
worker, so the worker count changes the realized permutation draws, and `n_jobs`
is part of `evaluation_params` and therefore of each cell's parameter signature.
Overriding it makes completed replicates unresumable and the run irreproducible
from the committed config alone. Parallelize across shards instead; the sbatch
script only forwards `--n-jobs` when `STUDY_N_JOBS` is explicitly set.

Write the Phase 4 run to a **new** output directory. Its parameter signatures
include the bumped seed-derivation version and the diagnostic schema versions,
so July shards can neither be resumed into nor overwritten by it.

## Phase 5 design-point pilot

`phase5_design_point_pilot.json` is the committed Phase 5 design-point pilot
(readiness item 4). It keeps the Phase 4 integration and evaluation settings but
changes the baseline to `p_dmp = 0.1` — so the `0.25`–`1.00` effect axis is
realizable without censoring for every pool-limited mode at every design point,
including the ρ = 0 stress-test endpoint — and declares a **crossed design
grid**: baseline continuity ρ ∈ {0.0, 0.5, 0.8} × `n_samples` ∈ {300, 600,
1200}. Every design point gets its own zero-effect anchor plus `orientation`
(the estimand in question) and `translation` (the negative control that binds
the surgery headroom) at effects `0.25`, `0.50`, `1.00`; 100 replicates and 199
permutations per cell; 6,500 work units, with the `n = 1200` columns
dominating compute. `magnitude` and `shape` are left to the Phase 5 study at
the chosen design point — both reached power 1.00 at n = 300 in Phase 4.

A local rehearsal (2026-09-04; 4 replicates, 49 permutations, 14 single-thread
shards on a 16-core workstation) ran all 65 cells with zero failures and no
censored surgery. Measured per-unit cost on that machine, PLS fit + RRPP:
about 30 s + 0.5 s/permutation at n = 300, and 72 s + 2 s/permutation at
n = 1200 — so at 199 permutations expect roughly 135 s, 250 s, and 460 s per
unit for n = 300, 600, 1200, or ~500 core-hours for the full pilot on
comparable cores (Phase 4 ran ~3× faster per core on the cluster). Pin BLAS to
one thread per shard (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`): with the
default thread pool each shard spawns ~32 threads and 12 parallel shards drove
the load average past 100.

```bash
sbatch --array=0-63 \
    --export=ALL,STUDY_CONFIG=$(pwd)/examples/trajectory_power_study/phase5_design_point_pilot.json,STUDY_OUT=$(pwd)/results/phase5-design-point,N_SHARDS=64 \
    scripts/motco_study_array.sbatch
python scripts/motco_study.py merge  --out-dir results/phase5-design-point
python scripts/motco_study.py report \
    --config examples/trajectory_power_study/phase5_design_point_pilot.json \
    --out-dir results/phase5-design-point
```

The report adds, beside the usual outputs:

- `design_point_operating.csv` — one row per (design point, mode, effect,
  statistic): rejection rate ± MC SE, the recorded pooled-eigengap distribution,
  the `angle` null-width (`q95`) dispersion, and the selected-dimensionality
  distribution. The baseline column and each point's anchor (`none` at `0.0`)
  are included.
- `design_point_power.png` — orientation `angle` power at the top effect vs
  `n_samples`, one line per ρ, annotated with the median eigengap.
- `continuity_resolved_orientation.csv` — resolved on ρ **and** every other
  design coordinate, so rows never pool across `n_samples`.
- `design_point_decision.json` / `.csv` — the predeclared rule
  (`acceptance.design_point`) evaluated per column: `meets` when
  `rate − k·SE ≥ floor`, `marginal` when only the point estimate clears it,
  `fails` otherwise. The verdict names the first `meets` column in the declared
  preference order (`n_samples` ascending, then ρ ascending) or is
  `revise_claim`. It is advisory: it never feeds the Phase 4 gate or the
  acceptance targets.

### `design_grid` — crossed design points

`axes` varies one factor at a time off the baseline; `design_grid.axes` is
**crossed**. Every combination of the declared values is a *design point*, and
each non-baseline point enumerates the full power grid (one zero-effect anchor
when `0.0` is among the effect sizes, plus every mode × nonzero effect) with
the point's coordinates applied to the baseline generator/evaluation
parameters. Rules:

- every axis MUST list the baseline value — the baseline point is served by the
  primary cells (which are stamped with their coordinates) and never
  re-emitted;
- an axis is either crossed or OFAT, never both;
- design cells carry phase `power_design`, `design_point = {axis: value}`, and
  `varied_axis = "design_grid"`, so the primary power curves, specificity
  matrix, Type I table, gate, and acceptance targets read the baseline column
  only, exactly as they would without the grid;
- design cells join the primary matched-seed family, so columns are paired at
  the same replicate index; the duplicate-dataset guard and the surgery-headroom
  check apply to every design cell using its own parameters.

Configs without `design_grid` enumerate and report exactly as before.

### Matched seeds

`matched_seeds.enabled` makes every **primary** power cell draw its generator
seed from one shared `(seed family, replicate index)` key, so at a given
replicate index every mode and effect starts from the same generated reference
and requested-effect comparisons are paired. Negative-control, Type I, and OFAT
cells keep their own seed families and stay independent draws; design-grid
cells join the primary family (see above). Persistence keys remain
`(cell_id, replicate_index)`.

At `group_effect_size = 0` the generator returns group B's baseline unchanged
and consumes no extra randomness, so per-mode zero-effect cells inside one
family would be byte-identical datasets. With
`matched_seeds.shared_zero_effect_anchor`, enumeration therefore emits **one**
mode-agnostic zero-effect primary cell and every mode's power curve resolves its
`0.00` point from that shared anchor; the report flags those rows with
`from_shared_anchor` so the modes' nulls are not read as independent evidence.
Enumeration also asserts that no two primary cells would generate identical data
at the same replicate index.

Configs without `matched_seeds` keep the pre-Phase-4 per-cell seed derivation
unchanged.

### Attribution diagnostics

`attribution.enabled` turns on bounded orientation-attribution diagnostics for
the cells the selector names — in the Phase 4 pilot, every nonzero primary
`orientation` cell, with 100 frozen-model bootstrap replicates and `top_k=20`.
Eligibility is resolved **during enumeration**, so it enters the cell's
parameter signature and never depends on an observed p-value; conditioning on
significance would bias the reported stability.

Attribution requires `evaluation.integration_method: "pls"` — it conditions on
the fitted PLS estimator, which `concat` and `snf` do not produce. The exact
estimator and standardized joint matrix that produced the trajectory scores are
reused; no second fit or component selection happens.

Each eligible replicate persists a compact, versioned record: effective
settings, ordered transitions with observed / PLS-captured / residual path
lengths and retention, the top-k signed features per transition and component
(standardized **and** original units, with the unit basis labeled so M-value
methylation is not read as beta values), bootstrap sign and selection stability,
and precision/recall against generator truth. Fitted estimators, full
standardized matrices, bootstrap matrices, and unrestricted feature tables are
never persisted. A cell that was never selected records
`attribution_status: "not_requested"`; an eligible replicate whose attribution
fails records `"failed"` with a reason and still contributes its trajectory
measurement.

Generator truth is defined as the features whose group-stage differential *mean
change* differs between groups, including CpG→gene→protein propagated effects —
so a real downstream driver is not scored as a false positive.

### Phase 4 gate

`acceptance.gate` carries every threshold the gate applies; the study code
hard-codes none of its own, so a gate is re-specified by editing the config.

| Gate field | Meaning |
|---|---|
| `alpha` | Significance level for control checks |
| `control_se_tolerance` | `k` in the one-sided bound `alpha + k·sqrt(alpha(1-alpha)/n)` |
| `monotonicity_se_tolerance` | A downward power step is tolerated when it is at most this many combined MC SEs |
| `min_power_at_top` | Default power floor at the largest effect |
| `confirmation_se_threshold` | An exceedance smaller than this many MC SEs counts as *marginal* |
| `max_marginal_exceedances` | How many marginal control exceedances may be tolerated (default 1) |
| `control_modes` | Modes whose cells are Type I controls at every effect level |
| `rules` | `mandatory_power` / `mandatory_control` / `descriptive` mode-statistic pairs |
| `require_complete_records` | Whether incomplete records/diagnostics block `proceed` |

The mandatory rules are:

- **Type I inflation** — the `none` baseline and *every* `translation` effect
  level (translation is a location-only offset at any effect) must keep each
  available statistic at or below the one-sided bound.
- **Power** — each `mandatory_power` pair must reach its floor at the top effect
  and be non-decreasing within `monotonicity_se_tolerance` combined MC SEs.
- **Control** — each `mandatory_control` off-diagonal pair (magnitude's `angle`
  and `shape`) is checked against the same inflation bound at its largest effect.
- **Completeness** — every expected work unit resolved, every completed PLS
  record carrying selected-component and realized-geometry metadata, and every
  eligible orientation record carrying valid attribution diagnostics or a
  recorded failure.

`descriptive` pairs — orientation's and shape's off-diagonals — are reported
against realized geometry but never gate: Phase 2 established that both
constructions are genuinely mixed after biological propagation and joint
preprocessing, so demanding purity there would fail a correct estimator.

Gate multiplicity is predeclared. Each control cell contributes one one-sided
test per statistic, so at a true rate of exactly alpha each exceeds its bound
with probability around `0.023`. One marginal exceedance therefore must not
decide the phase: when exactly one control statistic exceeds its bound by less
than `confirmation_se_threshold` MC SEs and nothing else mandatory fails, the
report emits `indeterminate` and names the confirmation re-run. Two or more
exceedances, or any exceedance of at least one SE, is `hold`. Study execution
never aborts because a scientific gate fails.

### Phase 4 outputs

Alongside the existing specificity, Type I, and power outputs, `report` writes:

| Output | Contents |
|---|---|
| `phase4_operating.csv` | Rejection rate and MC SE by mode, effect, and statistic, with `from_shared_anchor` |
| `phase4_geometry.csv` / `.png` | Realized geometry by mode, effect, checkpoint, scope, statistic, and path length, with a `measurement_space` label and unavailable counts |
| `phase4_pls_selection.csv` / `phase4_selected_components.png` | Selected component counts, effective CV settings, AUROC, and missing-diagnostic counts |
| `phase4_attribution.csv` / `phase4_attribution_stability.png` | Availability, observed-versus-captured retention, cross-replicate top-k Jaccard and sign agreement, bootstrap stability, and truth recovery |
| `phase4_localization.csv` | First checkpoint at which each off-diagonal response becomes material |
| `phase4_gate.csv`, `phase4_gate_decision.json` | Every gate observation, and the `proceed` / `hold` / `indeterminate` decision |

Localization compares each checkpoint only against its **own** zero-effect null,
on a scale-free quantity (`delta` divided by that checkpoint's path length;
`angle` and `shape` already dimensionless). Raw distances are never compared
across the standardized feature space and the PLS latent space. Its labels —
construction-present, sampling/preprocessing-associated, projection-associated —
describe *where* a response first appears, not what caused it, and gate nothing.
