# Spec Delta

## ADDED Requirements

### Requirement: Study configurations accept explicit group × stage sizes and block selection
The declarative study configuration SHALL accept the generator's explicit group × stage size table and the evaluation's layer selection. Each SHALL be accepted both in the baseline blocks and as an axis of a design grid. A size table given as nested JSON lists SHALL be normalized so that equal tables produce equal parameter signatures, cell identifiers, and matched seeds whatever their JSON spelling. Design-grid columns that differ only in the layer selection SHALL share generator parameters and matched seeds, as for every evaluation-namespace axis. A configuration that does not use either key SHALL enumerate exactly as before.

#### Scenario: Size table round-trips through a configuration
- **WHEN** a configuration sets `generator.group_stage_sizes` to `[[11, 10, 28], [9, 10, 12]]`
- **THEN** the configuration loads, enumerates, and generates datasets with those exact cell sizes, and reloading the same file yields identical cell identifiers and parameter signatures

#### Scenario: Size table as a design-grid axis
- **WHEN** a design grid declares `generator.group_stage_sizes` with two tables
- **THEN** enumeration emits one design point per table, each with its own anchored power grid

#### Scenario: Layer selection columns share data
- **WHEN** two design-grid columns differ only in `evaluation.integration_params.layers`
- **THEN** at every replicate index they evaluate the same generated dataset

#### Scenario: Committed configurations are unchanged
- **WHEN** any configuration committed before this change is enumerated
- **THEN** its cell identifiers, parameter signatures, and matched seeds are identical to those enumerated before this change

### Requirement: Phase 6 small-n study profiles
The study SHALL provide version-controlled Phase 6 small-n configurations, a pilot and a paper-grade profile, that reproduce the SEA-AD MTG astrocyte case-study design. Both SHALL use the numpy generator. Both SHALL use three stages and the explicit size table `((11, 10, 28), (9, 10, 12))` from the ≥30-nuclei cohort. Both SHALL use the layer selection methylation + expression as the two-block analogue of ATAC + RNA. Both SHALL use pooled PLS integration with M-value methylation and cross-validated component selection, the `joint` magnitude construction, the fail-loud surgery-censoring policy, matched seeds with a shared zero-effect anchor, and, across the profiles of the paper-grade study, the trajectory modes `magnitude`, `orientation`, `shape`, and `translation`. The paper-grade profile SHALL declare a design grid that crosses the ≥30-nuclei and ≥50-nuclei tables (`((10, 9, 25), (9, 9, 12))`) with the two-block and three-block layer selections. Its columns therefore include the baseline, the ≥50-nuclei table on two blocks, and the baseline table on all three blocks. The pilot SHALL measure the baseline column only, and SHALL exist to bracket the effect axes, which every column shares. The paper-grade profile SHALL fix its effect axis from recorded pilot evidence and SHALL record that evidence in its metadata. If the pilot shows that no single effect axis resolves the rise of every mode's target statistic, the paper-grade study SHALL be split into one profile per axis, as the Phase 5 magnitude re-measurement was. The split profiles SHALL share base seed, matched-seed family, generator, and evaluation parameters, so that their shared zero-effect anchors are byte-identical. The paper-grade profile SHALL run at least 500 replicates per cell and 999 permutations, and SHALL declare the Phase 5 report contract and the Phase 4 gate rules as advisory acceptance targets.

#### Scenario: Profiles load and enumerate without censoring
- **WHEN** the pilot or paper-grade profile is loaded and enumerated
- **THEN** it validates, every cell passes the configuration-time surgery-headroom check, and every cell's generator produces 80 samples (or 74 in the ≥50-nuclei column) in the declared group × stage sizes

#### Scenario: Effect axes are evidence-based
- **WHEN** the paper-grade profile is read
- **THEN** its metadata names the pilot result directory and states, per trajectory mode, why each enumerated effect was chosen

### Requirement: Phase 6 small-n findings report
The study SHALL produce a dated findings report for the paper-grade Phase 6 small-n run. The report SHALL follow the committed Phase 5 report template section for section, and SHALL be tied to the exact configuration, code revision, parameter signatures, software versions, record counts, failure counts, and reproduction commands. The report SHALL state, for each of `delta`, `angle`, and `shape`, the Type I rate and the power at each enumerated effect at the SEA-AD design, with Monte Carlo uncertainty. It SHALL state the recorded pooled-eigengap distribution of the zero-effect anchor and the orientation power stratified by eigengap tercile. It SHALL state the change in each operating characteristic from the baseline to the ≥50-nuclei column and to the three-block column. It SHALL conclude with an explicit per-statistic interpretability statement for the case study: whether a rejection, and whether a non-rejection, of that statistic may be reported as evidence at n = 80.

#### Scenario: Report states case-study interpretability per statistic
- **WHEN** the findings report is complete
- **THEN** it contains, for each of `delta`, `angle`, and `shape`, a statement of whether that statistic's result on the SEA-AD cohort is interpretable, grounded in the measured Type I and power at the SEA-AD design

#### Scenario: Report separates sample size from block count
- **WHEN** the findings report compares design columns
- **THEN** it attributes operating-characteristic differences to cohort size (baseline vs ≥50-nuclei column) and to block count (baseline vs three-block column) separately
