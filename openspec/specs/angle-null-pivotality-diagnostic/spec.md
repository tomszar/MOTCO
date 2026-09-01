# angle-null-pivotality-diagnostic Specification

## Purpose

Decide whether MOTCO's RRPP `angle` test behaves as a valid test under real orientation signal, by measuring whether each replicate's permutation null moves with its own observed statistic. The finding resolves the blocking Phase 5 gate: use the `angle` test as specified, replace it with a pivotal or studentized statistic, or revise its power target.

## Requirements

### Requirement: Diagnostic run is committed and reproducible

MOTCO SHALL provide a committed diagnostic profile that produces the records needed to test `angle` pivotality. The profile MUST run orientation at the top effect size where the shortfall was observed, together with a null control mode and a comparator mode whose statistic is known to behave correctly, so an association found under orientation can be contrasted against a calibrated case. It MUST use the same generator, integration, and evaluation contract as the Phase 4 pilot except for the replicate and permutation counts, MUST persist per-replicate null summaries, and MUST be executable through the existing study runner with documented commands.

#### Scenario: Profile is versioned alongside existing study configs

- **WHEN** a contributor looks for the diagnostic configuration
- **THEN** it is committed with the other study configurations and names its replicate count, permutation count, modes, and effect sizes explicitly

#### Scenario: Diagnostic run yields per-replicate null summaries

- **WHEN** the diagnostic profile is run to completion
- **THEN** every completed replicate record carries its observed statistics and the summary of its own permutation null

#### Scenario: Run is reproducible from documentation

- **WHEN** a contributor follows the documented reproduction commands
- **THEN** the diagnostic records regenerate from the committed configuration without manual parameter entry

### Requirement: Diagnostic measures association between observed statistic and its own null

The diagnostic SHALL quantify, across replicates within a cell, the association between each replicate's observed statistic and the location and spread of that replicate's own permutation null. For each cell and statistic it MUST report the association between the observed value and the null mean, the null standard deviation, and the alpha-level critical value, together with a measure of how strongly the null tracks the observed value. It MUST report these separately for the orientation cell, the null control, and the comparator, since a pivotal statistic is expected to show no such association.

#### Scenario: Association is reported per cell and statistic

- **WHEN** the diagnostic analysis runs over the merged diagnostic records
- **THEN** it reports, for each cell and statistic, the association between the observed statistic and its own null mean, null spread, and critical value

#### Scenario: Comparator distinguishes a pivotal case

- **WHEN** the analysis reports the orientation cell alongside the null control and the comparator mode
- **THEN** the same association measures are reported for all of them so a signal-specific association is distinguishable from one present everywhere

### Requirement: Diagnostic reproduces and explains the rejection inversion

The diagnostic SHALL report the observed-statistic distribution split by rejection outcome for each cell and statistic, so the Phase 4 inversion — non-rejecting replicates carrying a larger mean observed angle than rejecting ones — is either reproduced or shown absent in the diagnostic records. Where the inversion is reproduced, the report MUST state whether the measured null association accounts for it.

#### Scenario: Rejection split is reported

- **WHEN** the diagnostic analysis runs
- **THEN** it reports the mean observed statistic among rejecting and non-rejecting replicates for each cell and statistic

#### Scenario: Inversion is tied to the null association

- **WHEN** the inversion is present in the diagnostic records
- **THEN** the report states whether replicates with a larger observed statistic also carry a proportionally larger critical value

### Requirement: Diagnostic evaluates a standardized counterfactual

The diagnostic SHALL recompute the `angle` test with the observed statistic standardized against its own permutation null, and report the resulting rejection rate beside the rate from the test as specified, for the orientation cell and every control and comparator cell. This establishes whether a studentized statistic would recover power without inflating the null cells' rejection rates.

#### Scenario: Standardized rejection rates are reported beside the original

- **WHEN** the diagnostic analysis runs
- **THEN** it reports both the as-specified and the standardized rejection rate for each cell and statistic

#### Scenario: Counterfactual reports its control behavior

- **WHEN** the standardized test is evaluated
- **THEN** its rejection rate on the null control cells is reported so a power gain that comes from an inflated Type I rate is visible

### Requirement: Diagnostic issues a Phase 5 decision

The diagnostic SHALL conclude with a dated, versioned finding that states whether the `angle` permutation null is pivotal under signal and names exactly one consequence for Phase 5: the `angle` test proceeds as specified, is replaced by a pivotal or studentized statistic, or carries a revised power target. The finding MUST cite the code revision and configuration that produced it, and the project roadmap and Phase 5 readiness worklist MUST record the resolution.

#### Scenario: Finding names a single Phase 5 consequence

- **WHEN** the diagnostic report is complete
- **THEN** it states the pivotality verdict and exactly one of the three Phase 5 consequences

#### Scenario: Finding is traceable

- **WHEN** a reader opens the diagnostic report
- **THEN** it names the code revision, configuration, and record set behind every number it reports

#### Scenario: Blocking item is closed in the worklist

- **WHEN** the finding is published
- **THEN** the Phase 5 readiness worklist records item 1 as resolved with the decision, and the roadmap reflects it
