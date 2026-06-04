## MODIFIED Requirements

### Requirement: Generator realism is validated against InterSIM
The generator's output distributions SHALL be validated to match InterSIM's across a swept, replicate-based protocol — not a single degenerate point — as a guard against reimplementation drift and as paper-supportable evidence of fidelity. The validation SHALL run InterSIM and the numpy generator each multiple times per parameter cell over a grid of `delta` and `p.DMP`, and SHALL compare, per cell: per-omic marginal moments/quantiles, cluster separation (η²), differential-feature rates (the DMP→DEG→DEP coupling), covariance structure, and cross-omic coupling. Agreement SHALL be judged against InterSIM's own sampling distribution (the numpy statistic falling within InterSIM's documented central interval), so the criterion accounts for InterSIM's RNG variability. The InterSIM side SHALL be captured as committed fixtures so the validation runs without R, and SHALL be reproducible from a committed R script with recorded provenance.

#### Scenario: Fidelity holds across the parameter sweep
- **WHEN** the validation is run over the `delta` × `p.DMP` grid
- **THEN** for each cell and each compared statistic, the numpy generator's value falls within InterSIM's documented central interval for that statistic

#### Scenario: Effect injection and cross-omic coupling are validated at non-zero effect
- **WHEN** the validation is run at `delta > 0`
- **THEN** cluster separation (η²) and differential-feature rates (DMP→DEG→DEP) for the numpy generator agree with InterSIM's distribution, exercising the effect injection and the cross-omic coupling that a `delta = 0` check cannot

#### Scenario: Validation runs without R from committed fixtures
- **WHEN** the validation runs in CI with no `Rscript` available
- **THEN** it compares the numpy generator against the committed InterSIM summary fixtures and passes without invoking R

#### Scenario: Fixtures are reproducible with recorded provenance
- **WHEN** the InterSIM fixtures are regenerated
- **THEN** the committed R script reproduces them, and the fixtures record the InterSIM version, generation date, seeds, and the parameter grid

#### Scenario: A reproducible supplementary artifact is produced
- **WHEN** the supplementary artifact generator is run against the committed fixtures
- **THEN** it produces a paper-ready table and figure summarizing numpy-vs-InterSIM fidelity across the grid

#### Scenario: A qualitative visual supplement is produced
- **WHEN** the visual-supplement generator is run against regenerated InterSIM raw data (produced with InterSIM via the dev flake) with the numpy side generated live
- **THEN** it renders side-by-side InterSIM-vs-numpy figures — per-omic marginal densities, per-modality clustermap heatmaps (with sample/feature dendrograms and a cluster colour bar), per-modality PCA, per-feature mean/variance agreement scatter, and a cross-omic coupling correlation block
- **AND** the InterSIM raw matrices are not committed to the repository; the rendering code is exercised R-free in CI via a synthetic stand-in fixture
