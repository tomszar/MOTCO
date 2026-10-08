# Spec Delta

## ADDED Requirements

### Requirement: Study provides fixed Phase 5 magnitude re-measurement profiles

The study SHALL provide two version-controlled Phase 5 magnitude re-measurement configurations — a pilot and a paper-grade profile — that re-measure the magnitude mode under the `joint` magnitude kind at the Phase 5 paper-grade design point. Both MUST derive their generator parameters (other than `magnitude_kind`), integration parameters, base seed, and matched-seed family from the committed Phase 5 paper-grade profile, so that the shared zero-effect anchor and every group A baseline are byte-identical to the Phase 5 records at the same replicate index. Both MUST set `trajectory_modes` to `magnitude` only, `generator.magnitude_kind` to `joint`, attribution disabled, the default fail-loud surgery-censoring policy, and effect sizes consisting of `0.00`, `1.00`, and at least two further nonzero values strictly below `0.25` chosen from the committed analytic effect-axis bracket so the `delta` power curve is measured below saturation. The pilot MUST use 50 replicates per cell and 199 RRPP permutations with the Phase 4 gate disabled. The paper-grade profile MUST use 500 replicates per cell, 999 RRPP permutations, a worker count of one with overrides forbidden, the same report contract as the Phase 5 paper-grade profile, and the Phase 4 gate enabled with exactly three rules — mandatory power on magnitude's `delta`, mandatory control on magnitude's `angle` and `shape` — with `none` as the only control mode, and acceptance targets whose specificity entries are exactly the gate's mandatory controls.

#### Scenario: Pilot config is loaded

- **WHEN** the committed Phase 5 magnitude pilot configuration is loaded
- **THEN** it deterministically enumerates one shared zero-effect anchor and one nonzero power cell per nonzero effect, all at 50 replicates and 199 permutations under the joint magnitude kind, and requires no R runtime dependency

#### Scenario: Paper-grade re-measurement config is loaded

- **WHEN** the committed Phase 5 magnitude paper-grade configuration is loaded
- **THEN** it deterministically enumerates one shared zero-effect anchor and one nonzero power cell per nonzero effect, all at 500 replicates and 999 permutations under the joint magnitude kind, with the gate's rules naming only the magnitude mode and the control modes naming only `none`

#### Scenario: Anchor reproduces the Phase 5 anchor

- **WHEN** the re-measurement's zero-effect anchor cell is generated at any replicate index
- **THEN** its dataset is identical to the Phase 5 paper-grade anchor's dataset at the same replicate index

#### Scenario: Profiles do not touch the Phase 5 artefacts

- **WHEN** the two re-measurement profiles are added
- **THEN** the committed Phase 5 paper-grade configuration, its results directory, its report, the gate role definitions, the acceptance-target semantics, the report contract, and the report template are unchanged

### Requirement: Phase 5 magnitude re-measurement findings are versioned as an addendum

The study SHALL produce a dated magnitude re-measurement findings report, tied to the exact configuration, code revision, parameter signatures, software versions, record counts, failure counts, and reproduction commands, that follows the committed Phase 5 report template section for section with non-magnitude sections marked not applicable. It MUST record the reduced gate's decision, the `delta` power curve on the joint construction's native effect axis beside the realized joint `delta` per cell, the two control rates at every effect against the α + 2·SE bound, whether the anchor records reproduced the Phase 5 anchor, and the construction's biological framing (every tier of the methylation→expression→protein cascade scales together). It MUST either lift the Phase 5 exit review's withheld magnitude-specificity claim or report the claim as failed on the construction's own terms, and MUST explain any deviation from a predeclared target as a method or claim revision rather than a Monte Carlo sample-size question. The roadmap MUST record the outcome.

#### Scenario: Addendum report is committed

- **WHEN** the paper-grade magnitude re-measurement run and reporting complete
- **THEN** a versioned addendum records the gate verdict, the power curve with realized sizes, the control rates, the anchor reproduction check, limitations, and exact shard, merge, and report commands, and every claim traces to a committed CSV or JSON output

#### Scenario: Specificity claim is resolved, not deferred

- **WHEN** the addendum is written
- **THEN** it states explicitly whether the withheld magnitude-specificity claim is lifted, and if not, which control failed and what revision that implies
