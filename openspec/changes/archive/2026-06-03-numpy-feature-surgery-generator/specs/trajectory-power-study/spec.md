## ADDED Requirements

### Requirement: Study runs on the numpy generator without an R runtime dependency
The trajectory power study SHALL generate every replicate through the numpy generator and cached reference data, so that study execution (including cluster shards) requires no `Rscript` or R `InterSIM` package.

#### Scenario: Shards run without R
- **WHEN** a study shard executes its `(cell, replicate)` units
- **THEN** each replicate is generated from the numpy generator and cached reference data, with no R invocation

#### Scenario: Negative-control modes are retained under the new semantics
- **WHEN** a study configuration is enumerated
- **THEN** the Type I cells still include `none` (no group effect) and `translation` (location-only group effect) as negative controls, now defined by the feature-surgery generator

### Requirement: Acceptance targets are re-specified for the new mode semantics
Because the trajectory modes are redefined, the study's pre-specified acceptance targets SHALL be reset to reflect the operating characteristics of the feature-surgery modes, and prior results SHALL be treated as superseded.

#### Scenario: Acceptance targets reference the new modes
- **WHEN** a study configuration's acceptance targets are evaluated
- **THEN** the per-statistic power and specificity targets correspond to the feature-surgery `magnitude`/`orientation`/`shape` modes

#### Scenario: Specificity demonstration is supported by indicator truth
- **WHEN** a replicate is summarized
- **THEN** the per-stage/group differential indicators emitted by the generator are available to confirm that the injected mode predominantly moves its matching statistic
