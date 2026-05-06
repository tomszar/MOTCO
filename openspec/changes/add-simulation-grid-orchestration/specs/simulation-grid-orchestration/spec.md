## ADDED Requirements

### Requirement: Orchestration defines simulation cells
MOTCO SHALL provide a simulation cell schema that captures all parameters needed to run repeated simulation evaluations.

#### Scenario: Cell contains required parameter groups
- **WHEN** a simulation cell is created
- **THEN** it contains cell identity, phase, InterSIM parameters, generator parameters, evaluation parameters, replicate count, and seed metadata

#### Scenario: Invalid replicate count is rejected
- **WHEN** a simulation cell has fewer than one replicate
- **THEN** the orchestrator raises a clear validation error

### Requirement: Orchestration enumerates Type I and power grids
MOTCO SHALL provide helpers for enumerating simulation cells from baseline parameters and parameter axes.

#### Scenario: Type I grid enumeration
- **WHEN** a caller requests a Type I error grid
- **THEN** generated cells use null trajectory settings unless explicitly overridden

#### Scenario: Power grid enumeration
- **WHEN** a caller requests a power grid
- **THEN** generated cells include non-null trajectory modes and effect sizes

#### Scenario: Cell identifiers are stable
- **WHEN** the same grid configuration is enumerated twice
- **THEN** generated cell identifiers are stable and deterministic

### Requirement: Orchestration runs replicates through evaluation harness
The orchestrator SHALL run each cell replicate by generating/evaluating data through the simulation evaluation harness.

#### Scenario: Replicate execution returns records
- **WHEN** a cell replicate completes
- **THEN** the orchestrator records cell metadata, replicate index, seeds, observed statistics, p-values, truth metadata, and runtime metadata

#### Scenario: Evaluation errors are recorded or surfaced
- **WHEN** a replicate evaluation fails
- **THEN** the orchestrator either records a failed replicate with error details or raises according to configured error policy

### Requirement: Orchestration persists and resumes results
The orchestrator SHALL persist per-replicate results and support resuming incomplete runs.

#### Scenario: Completed replicate is skipped on resume
- **WHEN** a result for a cell and replicate already exists with matching parameter signature
- **THEN** the orchestrator skips rerunning that replicate

#### Scenario: Parameter mismatch prevents unsafe resume
- **WHEN** an existing result has the same cell and replicate identifiers but a different parameter signature
- **THEN** the orchestrator raises a clear validation error or requires explicit overwrite

### Requirement: Orchestration summarizes rejection metrics
The orchestrator SHALL aggregate replicate-level p-values into rejection metrics for Type I error and power.

#### Scenario: Rejection rate summary
- **WHEN** replicate-level p-values are available for a cell
- **THEN** the summary includes completed replicate count, rejection count, rejection rate, and Monte Carlo standard error

#### Scenario: Alpha threshold is configurable
- **WHEN** a caller provides an alpha threshold
- **THEN** the summary uses that alpha threshold to compute rejection indicators

#### Scenario: Missing statistics are reported
- **WHEN** a statistic is unavailable for a cell
- **THEN** the summary reports the statistic as unavailable rather than treating it as non-significant
