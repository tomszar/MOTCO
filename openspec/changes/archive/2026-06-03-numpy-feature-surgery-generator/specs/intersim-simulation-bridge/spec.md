## ADDED Requirements

### Requirement: Bridge can export InterSIM reference data for the numpy generator
The InterSIM bridge SHALL provide a one-time export path that captures InterSIM's reference means, covariances, and cross-omic maps into a cached artifact consumed by the numpy generator, so that R is needed only to produce the cache and not for runtime generation.

#### Scenario: Reference export captures the required objects
- **WHEN** the export path is run with InterSIM available
- **THEN** it writes the reference means, covariances, cross-omic maps, and correlation constants needed to reproduce InterSIM's generative model

#### Scenario: Export records provenance
- **WHEN** the export completes
- **THEN** the cached artifact records the InterSIM version and export date

#### Scenario: Runtime generation does not invoke the bridge
- **WHEN** datasets are generated for evaluation, the grid, the study, or the showcase
- **THEN** generation uses the numpy generator and the cached reference data, without invoking the InterSIM bridge or R
