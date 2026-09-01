## ADDED Requirements

### Requirement: Harness summarizes each permutation null

Whenever the harness runs RRPP, it SHALL return a compact per-statistic summary of the permutation null distribution alongside the observed statistics and p-values. The summary MUST locate the observed statistic against its own null: for each statistic with a null distribution it carries the number of retained permutation draws, the null mean, the null standard deviation, and null quantiles including the median and the alpha-level upper critical value. The summary MUST be JSON-safe scalars only, and non-finite draws MUST be excluded from the summary with the retained count reported.

The summary is separate from the existing opt-in retention of full null distributions: it is produced whenever permutations are run, whereas the full null vectors remain available only when the caller asks for them.

#### Scenario: Summary accompanies every RRPP evaluation

- **WHEN** a caller evaluates a dataset with permutation count greater than 0
- **THEN** the result carries a null summary for each statistic that has a null distribution
- **AND** each summary reports the retained draw count, mean, standard deviation, and quantiles for that statistic

#### Scenario: Summary is independent of full-distribution retention

- **WHEN** a caller runs RRPP without requesting retention of the full null distributions
- **THEN** the result still carries the compact null summary
- **AND** the full null vectors are absent

#### Scenario: No permutations means no summary

- **WHEN** a caller sets permutation count to 0
- **THEN** the result carries no null summary and the existing observed statistics are unchanged

#### Scenario: Summary does not alter the test

- **WHEN** the null summary is produced for an evaluation
- **THEN** the observed statistics, p-values, permutation draws, and every other field of the result are identical to those the same inputs produced before the summary existed
