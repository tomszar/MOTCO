## ADDED Requirements

### Requirement: Diagnostic relates null width to the recorded configuration spectrum

The pivotality analysis SHALL report, per cell and statistic, the association between each replicate's recorded pooled configuration eigengap and the width of that replicate's own permutation null, beside the existing observed-versus-null associations and with the same uncertainty treatment. Run over records produced under the spectrum schema, the analysis MUST reproduce the geometry audit's finding: a negative and material association between eigengap and `angle` null width in the orientation cell, distinguishable from the associations in the control and comparator cells.

#### Scenario: Eigengap association is reported per cell and statistic

- **WHEN** the pivotality analysis runs over merged records carrying the spectrum block
- **THEN** it reports, for each cell and statistic, the association between the recorded pooled eigengap and the replicate's own null width with an uncertainty measure

#### Scenario: The audit association reproduces on new records

- **WHEN** the diagnostic profile is run under the spectrum schema and analyzed
- **THEN** the orientation cell shows a negative, material eigengap-to-`angle`-null-width association consistent with the audit's measurement

#### Scenario: Records without spectra are handled explicitly

- **WHEN** the analysis runs over records lacking the spectrum block
- **THEN** the eigengap association is reported as unavailable rather than computed from absent data
