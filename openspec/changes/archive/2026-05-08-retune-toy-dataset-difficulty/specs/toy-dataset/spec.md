## MODIFIED Requirements

### Requirement: Pre-generated toy dataset in repository
The repository SHALL include a pre-generated toy dataset at `examples/data/toy/` containing all files produced by `motco simulate` with the seed and parameters recorded in `truth.json`. The retuned canonical regeneration command is:

```
motco simulate \
  --seed 42 \
  --n-samples 90 \
  --trajectory-mode orientation \
  --effect-size 1.0 \
  --prop-affected-features 0.1 \
  --out-dir examples/data/toy/
```

*(Was: `--effect-size 2.0` and the absent `--prop-affected-features` flag.)*

This dataset SHALL be usable for tutorial purposes without R or InterSIM installed.

#### Scenario: Toy data is present after clone
- **WHEN** the repository is cloned
- **THEN** `examples/data/toy/` exists and contains at minimum: `methylation.csv`, `expression.csv`, `proteomics.csv`, `metadata.csv`, `model_full.csv`, `model_reduced.csv`, `ls_means.csv`, `contrast.json`, `truth.json`

#### Scenario: Dataset demonstrates orientation trajectory difference
- **WHEN** `motco de` is run on the toy data latent space
- **THEN** the angle statistic between groups A and B is substantially greater than 0° (orientation difference is detectable)

#### Scenario: Trajectory analysis is non-trivially uncertain
- **WHEN** `motco de` (RRPP) is run on the supervised PLS-DA latent space of the toy data with at least 199 permutations on seed 42
- **THEN** the empirical RRPP p-value for the angle statistic between groups A and B satisfies `0 < p_angle ≤ 0.1` (signal detectable, but not pinned at the floor of the permutation distribution as it was at the previous default of `effect_size = 2.0`), and the latent-space angle is at least 30° but well below the saturation level previously observed (~93°)

### Requirement: truth.json documents generation parameters
`examples/data/toy/truth.json` SHALL record the seed, trajectory mode, effect size, group labels, stage mapping, and affected features used to generate the dataset, so users can understand and reproduce it. Following the addition of the `--prop-affected-features` CLI flag, the `affected_features` lists in `truth.json` SHALL reflect the proportion specified at generation time.

#### Scenario: truth.json is valid JSON
- **WHEN** `examples/data/toy/truth.json` is parsed
- **THEN** it contains the keys: `seed`, `trajectory_mode`, `group_effect_size`, `group_labels`, `stage_mapping`, `affected_features`

#### Scenario: truth.json reflects the retuned defaults
- **WHEN** `examples/data/toy/truth.json` is parsed after this change lands
- **THEN** `group_effect_size == 1.0` and the `affected_features` list lengths reflect a `prop_affected_features` of approximately `0.1` (each list length within ±1 of `round(n_features_in_layer * 0.1)`)
