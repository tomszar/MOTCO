# phase6-small-n-operating-study

## Why

Phase 6 will apply MOTCO to SEA-AD middle temporal gyrus (MTG) astrocytes. The data are donor-level pseudobulks
of RNA and ATAC, with Sex as the group and three merged ADNC stages (Not AD + Low / Intermediate / High). That
gives 80 donors in six unbalanced cells: F 11/10/28, M 9/10/12. Every validated operating characteristic MOTCO
has, Type I or power, was measured with three omic blocks, four balanced stages, and n = 300–1200. The Phase 4
pilot already put orientation power at about 0.65 at n = 300. At n = 80 we know nothing. The Phase 5 exit review
lets Phase 6 start only if orientation is read against the cohort's own recorded eigengap. That reading is only
meaningful if we know what the test does at the cohort's own design. So this study must run before any SEA-AD
result is interpreted.

The study cannot be configured today. The generator has a single global `group_ratio`, so it cannot reproduce
an imbalance that varies by stage (55% / 50% / 70% female). The harness always integrates all three InterSIM
blocks, while the case study has two.

## What Changes

- **Explicit group × stage sample sizes in the generator.** An optional `group_stage_sizes` parameter takes
  one tuple of per-stage counts per group. When it is set, it fixes every cell's size exactly. It conflicts
  with `n_samples`, `stage_sample_prop` and `group_ratio`, and the generator rejects the combination. When it
  is absent, generation is byte-identical to today, so every committed signature and dataset is unchanged.
- **Evaluation-time block selection.** A new optional evaluation parameter,
  `integration_params.layers`, names the omic blocks that enter preprocessing, integration, realized-geometry
  diagnostics and attribution. The study then generates all three blocks and measures on the selected subset,
  methylation + expression. That subset is the two-block analogue of ATAC + RNA (design D2). When the key is
  absent, all three blocks are used and output is byte-identical. Because the key sits in the evaluation
  namespace, design-grid columns that differ only in `layers` share matched seeds.
- **A version-controlled small-n study profile.** It matches the SEA-AD design: 3 stages,
  `group_stage_sizes` from the ≥30-nuclei cohort, two blocks, pooled PLS with cross-validated rank, and the
  `joint` magnitude construction. A design grid adds the ≥50-nuclei cohort (n = 74: F 10/9/25, M 9/9/12) and a
  three-block column, crossed (four columns; design D4). The three-block column keeps the block-count effect
  separate from the n effect.
- **An effect-axis bracket pilot before the paper-grade run.** At n = 80 the Phase 5 effect axes do not
  transfer. Magnitude in particular saturated at e = 0.02 with n = 1200. The final effect grid is fixed from
  recorded pilot evidence, as the Phase 5 magnitude re-measurement did.
- **A dated findings report** following the Phase 5 template. It states per-statistic Type I and power at the
  SEA-AD design, the recorded eigengap distribution, and an explicit statement of which statistics are
  interpretable for the case study.

Out of scope:
- any SEA-AD preprocessing or real data in this repository, since that lives in the external case-study repo;
- a new generator calibrated to RNA/ATAC distributions;
- covariate support in the design matrix;
- any change to statistics, RRPP, PLS rank rule, or gate semantics.

## Capabilities

### New Capabilities

(none)

### Modified Capabilities

- `semisynthetic-trajectory-generator`: the requirement "Groups are assigned reproducibly within stages" gains
  explicit group × stage sizes as an alternative to `group_ratio`/`stage_sample_prop`, with conflict and
  minimum-size validation.
- `simulation-evaluation-harness`: the requirement "Harness supports initial integration methods" gains
  evaluation-time block selection, which applies to every integration method and to the downstream
  diagnostics. The default stays all three blocks.
- `trajectory-power-study`: adds the Phase 6 small-n study profiles (pilot and paper grade) and their dated
  findings report.

## Impact

- `src/motco/simulations/semisynthetic.py`: new param, sizing and validation; truth records the realized
  cell sizes.
- `src/motco/simulations/preprocessing.py`, `evaluation.py`, `diagnostics.py`,
  `attribution_diagnostics.py`: iterate over the selected layers instead of `OMIC_LAYERS`.
- `src/motco/simulations/study/config.py` and `enumerate.py`: JSON lists coerced to tuples for the new
  generator field. The design-grid axis needs to work with a tuple-valued field.
- `examples/trajectory_power_study/`: new `phase6_small_n_pilot.json`, `phase6_small_n_study.json` and its split sibling `phase6_small_n_magnitude.json`, plus a
  README entry.
- `tests/`: generator sizing, conflict validation, byte-identity when absent; layer-subset evaluation and
  default byte-identity; config round-trip.
- `docs/roadmap.md`, `CLAUDE.md`, `docs/api/simulations.md`, `src/motco/simulations/study/README.md`.
- Cluster time on `ing`: a pilot, then the paper-grade run. The cost is set after the pilot (D5). Small n makes
  each unit much cheaper than in Phase 5.
