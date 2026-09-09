## ADDED Requirements

### Requirement: Study configuration declares a report contract and rejects unknown top-level keys

The declarative configuration SHALL accept an optional `report_contract` block declaring which attribution component driver reports present (`observed`, `pls_captured`, or `residual`), that cross-replicate driver agreement is descriptive only, and whether a worker-count override at run time is forbidden or merely warned about. Loading a configuration MUST fail with a message naming the offending key when the root mapping contains a key the schema does not define or when the contract contains an unknown field or value. The contract MUST survive a dump-and-reload round trip. Configurations without the block MUST load and report exactly as before.

#### Scenario: Contract is loaded

- **WHEN** a configuration declares `report_contract` with `driver_component` `observed`, `cross_replicate_driver_agreement` `descriptive`, and `n_jobs_override` `forbid`
- **THEN** the loaded configuration exposes those values and writing it back and reloading it yields the same contract

#### Scenario: Unknown key is rejected

- **WHEN** a configuration contains an unknown root key, an unknown `report_contract` field, or an `n_jobs_override`/`driver_component` value outside the declared set
- **THEN** loading fails and the error names the key or value

#### Scenario: Absent contract is inert

- **WHEN** every committed configuration under the study examples directory is loaded and enumerated
- **THEN** each loads without error and its cell identities and parameter signatures are unchanged from the recorded snapshot

### Requirement: Report echoes the resolved contract and identifies the shared zero-effect anchor as one measurement

When a configuration declares a report contract, reporting SHALL write a machine-readable contract echo containing the declared driver component, the descriptive-only status of cross-replicate driver agreement, the worker count the merged records were produced with (which MUST be uniform across records), and the shared zero-effect anchor's cell identifier together with the trajectory modes whose zero-effect power point it resolves and the statement that it is counted as one measurement. Operating-characteristic rows derived from the anchor MUST remain flagged as anchor-derived.

#### Scenario: Contract echo is written

- **WHEN** reporting runs on merged records for a configuration with a report contract and a shared zero-effect anchor
- **THEN** the report directory contains a contract echo naming the driver component, the anchor cell, the modes it resolves, and the uniform worker count

#### Scenario: Non-uniform worker count is refused

- **WHEN** merged records carry more than one worker-count value
- **THEN** reporting fails naming the values found rather than echoing one of them

### Requirement: Driver reporting presents the declared component and makes no cross-replicate stability claim

When a configuration declares a report contract, reporting SHALL write a driver table restricted to the declared attribution component, per trajectory mode, effect size, and transition, carrying truth precision and recall, mean selected count, within-replicate bootstrap stability, and replicate accounting, and containing no cross-replicate agreement column. The attribution figure SHALL plot within-replicate bootstrap stability only and SHALL be titled as such. The complete per-component attribution table, including cross-replicate agreement, MUST continue to be written unchanged, and configurations without a contract MUST produce exactly the prior outputs.

#### Scenario: Driver table is restricted to the declared component

- **WHEN** reporting runs with a contract declaring the `observed` component on records that carry attribution diagnostics for all three components
- **THEN** the driver table contains rows for the `observed` component only and has no cross-replicate agreement column
- **AND** the full attribution table still contains every component and the cross-replicate columns

#### Scenario: Legacy outputs are byte-identical without a contract

- **WHEN** reporting re-runs on the committed Phase 4, design-point, and latent-rank fixtures, whose configurations declare no contract
- **THEN** every existing report file is byte-identical to the committed output and no driver table or contract echo is written

### Requirement: Shard runner enforces the configured worker count when the contract forbids overrides

When the configuration's report contract sets the worker-count override policy to forbid, the shard runner SHALL refuse a command-line worker count that differs from the configured value, exiting non-zero with a message naming both values and stating that the override changes the permutation draws and the cell parameter signature. A command-line value equal to the configured value SHALL be accepted. Under the warn policy, or with no contract, the runner SHALL keep its existing warn-and-proceed behavior. The SLURM array template MUST document that a forbidding configuration refuses the optional worker-count environment variable.

#### Scenario: Differing override is refused

- **WHEN** the shard runner is invoked with a worker count differing from a forbidding configuration
- **THEN** it exits non-zero before enumerating or running any unit and names both values

#### Scenario: Equal or absent override proceeds

- **WHEN** the shard runner is invoked without a worker count, or with one equal to the configured value, under a forbidding configuration
- **THEN** it runs the shard normally

### Requirement: Study provides a fixed Phase 5 paper-grade profile

The study SHALL provide a version-controlled Phase 5 paper-grade configuration using the numpy generator, pooled PLS integration with M-value methylation and cross-validated component selection, four stages, `n_samples = 1200`, baseline continuity `0.0`, `p_dmp = 0.1`, the default fail-loud surgery-censoring policy, trajectory modes `magnitude`, `orientation`, `shape`, and `translation`, effect sizes `0.00`, `0.25`, `0.50`, `0.75`, and `1.00`, 500 replicates per cell, 999 RRPP permutations, a worker count of one with overrides forbidden, matched seeds with one shared zero-effect anchor in a family distinct from both Phase 5 pilots, attribution enabled on nonzero-effect orientation primary cells, the Phase 4 gate enabled with mandatory power rules on the three diagonal statistics, mandatory control rules on magnitude's off-diagonals, and descriptive rules on the remaining off-diagonals, acceptance targets whose specificity entries are exactly the gate's mandatory controls, and a report contract declaring the `observed` driver component. The configuration MUST enumerate without any censored surgery and MUST derive its generator and integration parameters from the committed latent-rank ladder profile.

#### Scenario: Paper-grade config is loaded

- **WHEN** the committed Phase 5 paper-grade configuration is loaded
- **THEN** it deterministically enumerates the Type I controls, one shared zero-effect anchor, and sixteen nonzero power cells at 500 replicates and 999 permutations, every pool-limited cell fits its expected surgery headroom, and it requires no R runtime dependency

#### Scenario: Paper-grade config does not copy the historical clamp flag or the failing target

- **WHEN** the committed Phase 5 paper-grade configuration is inspected
- **THEN** it does not set `surgery_censoring` to `clamp` and its specificity targets contain no entry for orientation's `shape` statistic

### Requirement: Phase 5 findings follow a committed report template

The study SHALL commit a findings-report template beside the Phase 5 profile that the dated Phase 5 report MUST follow. The template SHALL require: configuration and provenance including the provenance file's field list; unit and failure accounting; the gate decision; a Type I section stating which cell each null claim reads and that the shared anchor is one measurement; per-mode power with the recorded eigengap distribution and `angle` null-width dispersion beside every orientation result; the predeclared orientation-to-`shape` cross-talk statement; a driver section limited to the declared component with within-replicate bootstrap stability and an explicit statement that no cross-replicate driver-stability claim is made and why; construction limitations including the n-conditional orientation claim and the non-ρ-invariant realized orientation contrast; and reproduction commands that pass no worker-count override. The readiness worklist MUST record item 5 as resolved by the committed contract, profile, and template.

#### Scenario: Template names every contract item

- **WHEN** the committed template is inspected
- **THEN** it contains a section for each required element above, and the contract items it names agree with the committed profile's report contract

#### Scenario: Readiness worklist records item 5

- **WHEN** the readiness worklist is inspected after this change
- **THEN** item 5 is marked resolved with references to the profile, the template, and the contract fields that enforce each of its four points
