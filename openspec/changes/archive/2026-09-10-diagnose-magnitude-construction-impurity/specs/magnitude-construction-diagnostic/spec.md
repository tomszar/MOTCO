## Purpose

Decide whether the magnitude trajectory surgery's off-target `angle` and `shape` response — the response
that holds the Phase 5 gate at HOLD — is caused by scaling one omic block's δ while leaving the others at
baseline, and whether a size-pure magnitude construction is realizable at all once each omic block is
standardized before integration. The finding tells the Phase 5 exit review whether to pursue a method
revision (correct the construction) or a claim revision (report the response as construction impurity).

## ADDED Requirements

### Requirement: Diagnostic run is committed and reproducible

MOTCO SHALL provide a committed diagnostic that produces the records needed to attribute the magnitude
mode's off-target response. It MUST run at the Phase 5 paper-grade design point — ρ = 0, n = 1200, four
stages, `p_dmp` = 0.1, pooled PLS on M-value methylation with the stage-supervised double-CV rank rule — at
the effect sizes where the mandatory controls fail, together with the shared zero-effect anchor as the null
reference. It MUST be executable on a workstation without a cluster or an R runtime, and MUST record the
code revision and configuration it ran under.

#### Scenario: Diagnostic configuration is versioned

- **WHEN** a contributor looks for the magnitude-construction diagnostic
- **THEN** its parameters — modes, effect sizes, replicate count, permutation count, and design point — are
  committed and named explicitly rather than passed ad hoc

#### Scenario: Diagnostic runs without cluster or R dependencies

- **WHEN** the diagnostic is executed on a workstation with no SLURM client and no R installation
- **THEN** it completes and writes its outputs, because it uses the numpy generator and workstation-scale
  replicate counts

### Requirement: Off-target response is attributed to omic blocks

The diagnostic SHALL decompose the magnitude mode's realized `angle` and `shape` response by omic block at
the pre-integration standardized checkpoint, reporting each block's contribution beside the pooled
response and the zero-effect anchor's value at the same checkpoint. It MUST report whether the response is
concentrated in the block whose δ is scaled or distributed across blocks, so block-asymmetric scaling is
confirmed or excluded as the mechanism rather than assumed.

#### Scenario: Block decomposition accompanies the pooled response

- **WHEN** the diagnostic reports the magnitude mode's off-target `angle` and `shape`
- **THEN** the pooled response, the per-block responses, and the anchor's value at the same checkpoint are
  reported together, each labelled with its measurement space

#### Scenario: Mechanism verdict is stated from the decomposition

- **WHEN** the block decomposition is complete at every diagnostic effect size
- **THEN** the diagnostic states whether the off-target response tracks the scaled block, and the
  statement cites the decomposition rather than the construction's source code

### Requirement: Orientation response is isolated from shape response

The diagnostic SHALL measure the magnitude mode in a shape-free configuration of two stages, where no
shape difference is definable after size and orientation are removed, so the orientation response is
observed without the shape confound that a four-stage trajectory introduces. It MUST report the two-stage
`angle` response beside the four-stage response.

#### Scenario: Two-stage isolation is reported

- **WHEN** the magnitude mode runs at two stages and at four stages under otherwise identical parameters
- **THEN** both `angle` responses are reported, and the two-stage result is identified as the shape-free
  measurement

### Requirement: A size-pure candidate construction is measured, not adopted

The diagnostic SHALL evaluate a candidate magnitude construction that scales every omic's δ together, and
report whether its realized `angle` and `shape` responses fall to the zero-effect anchor's level **after**
per-block standardization. The candidate MUST be available to the diagnostic only; it MUST NOT become a
selectable production trajectory mode, and the diagnostic MUST NOT change which construction the committed
study profiles use.

#### Scenario: Candidate is compared against the anchor after standardization

- **WHEN** the uniform-δ candidate is evaluated at the diagnostic effect sizes
- **THEN** its post-standardization `angle` and `shape` responses are reported against the anchor's values,
  establishing whether a size-pure construction survives per-block standardization

#### Scenario: Production mode selection is unchanged

- **WHEN** a committed study profile requests the magnitude mode after this diagnostic exists
- **THEN** it resolves to the same construction as before, and the candidate is not selectable through
  study configuration

### Requirement: Findings are versioned and hand the verdict to the exit review

The diagnostic's conclusions SHALL be recorded in a dated report that states the mechanism verdict, the
size-purity verdict, the shape-free orientation result, and the recalibrated localization classification
for both failing control pairs. Every number MUST cite a committed output path. The report MUST state which
revision each verdict implies for the exit review — method or claim — and MUST NOT itself adopt a revision.

#### Scenario: Report states both verdicts with citations

- **WHEN** the diagnostic completes
- **THEN** a dated report records the mechanism and size-purity verdicts, each number citing a committed
  file, and names the revision each implies without implementing it

#### Scenario: Deviation from the inferred mechanism is reported as found

- **WHEN** the block decomposition does not support block-asymmetric scaling as the mechanism
- **THEN** the report states that plainly and identifies what remains unexplained, rather than retaining
  the prior inference
