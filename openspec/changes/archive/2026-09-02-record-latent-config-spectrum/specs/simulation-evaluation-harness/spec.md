## ADDED Requirements

### Requirement: Harness records the latent stage-mean configuration spectrum

On every evaluation, the harness SHALL compute and return a JSON-safe spectrum description of the centered stage-mean configuration in the evaluated latent space — for the pooled configuration and for each contrast group — derived from the same fitted LS-mean vectors the trajectory statistics use. Each description MUST carry the normalized eigenvalue spectrum and the relative eigengap (λ₁−λ₂)/Σλ as a named scalar. A configuration whose eigengap is undefined (zero total variance) MUST record that explicitly rather than a non-finite value, and a two-stage configuration's trivially saturated eigengap MUST be recorded as computed.

#### Scenario: Spectrum accompanies every evaluation

- **WHEN** a caller evaluates a dataset through the harness
- **THEN** the result carries a configuration-spectrum block with one entry for the pooled configuration and one per contrast group
- **AND** each entry reports the normalized spectrum and the relative eigengap as JSON-safe scalars

#### Scenario: Spectrum matches the measured configurations

- **WHEN** the spectrum block is computed for an evaluation
- **THEN** its values equal an independent singular value decomposition of the centered LS-mean configurations implied by the evaluation's fitted betas, contrast, and stage ordering

### Requirement: Harness summarizes the permutation eigengap distribution

Whenever the harness runs RRPP, it SHALL additionally summarize the pooled-configuration relative eigengap across the permutation draws — retained draw count, mean, standard deviation, and quantiles — so a replicate can locate its observed eigengap against its own permutation distribution. Full per-permutation spectra MUST NOT be retained in the result. The spectrum computation MUST consume no randomness: with identical inputs, seed, and worker count, the permutation draws, null distributions, observed statistics, and p-values MUST be identical to those produced before this capability existed.

#### Scenario: Permutation eigengap summary accompanies RRPP runs

- **WHEN** a caller evaluates a dataset with permutation count greater than 0
- **THEN** the result carries a summary of the pooled eigengap over the permutation draws with retained count, mean, standard deviation, and quantiles
- **AND** no per-permutation spectrum vectors are present in the result

#### Scenario: No permutations means no permutation summary

- **WHEN** a caller sets permutation count to 0
- **THEN** the result carries the observed configuration-spectrum block but no permutation eigengap summary

#### Scenario: Spectrum recording does not alter the test

- **WHEN** the same evaluation inputs are run with and without spectrum recording available
- **THEN** the observed statistics, p-values, permutation draws, and every pre-existing result field are identical
