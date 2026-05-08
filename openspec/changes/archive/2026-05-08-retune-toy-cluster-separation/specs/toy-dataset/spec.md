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
  --cluster-mean-shift 0.10 \
  --out-dir examples/data/toy/
```

This dataset SHALL be usable for tutorial purposes without R or InterSIM installed.

#### Scenario: Toy data is present after clone
- **WHEN** the repository is cloned
- **THEN** `examples/data/toy/` exists and contains at minimum: `methylation.csv`, `expression.csv`, `proteomics.csv`, `metadata.csv`, `model_full.csv`, `model_reduced.csv`, `ls_means.csv`, `contrast.json`, `truth.json`

#### Scenario: Dataset demonstrates orientation trajectory difference
- **WHEN** `motco de` is run on the toy data latent space
- **THEN** the angle statistic between groups A and B is substantially greater than 0° (orientation difference is detectable)

#### Scenario: Stage classification is non-saturated
- **WHEN** `motco plsr` is run on the toy data with `y = stage` using moderate tutorial CV settings
- **THEN** the returned AUROC is less than `0.99` and `AUROC_std` is greater than `0.0`

#### Scenario: Trajectory analysis is non-trivially uncertain
- **WHEN** `motco de` (RRPP) is run on the supervised PLS-DA latent space of the toy data with at least 199 permutations on seed 42
- **THEN** the empirical RRPP p-value for the angle statistic between groups A and B satisfies `0 < p_angle <= 0.1` (signal detectable, but not pinned at the floor of the permutation distribution), and the latent-space angle is at least 30° but below saturation

### Requirement: truth.json documents generation parameters
`examples/data/toy/truth.json` SHALL record the seed, trajectory mode, effect size, group labels, stage mapping, affected features, and InterSIM generation metadata used to generate the dataset, so users can understand and reproduce it. Following the addition of the `--prop-affected-features` and `--cluster-mean-shift` CLI flags, the `affected_features` lists and InterSIM delta metadata in `truth.json` SHALL reflect the values specified at generation time.

#### Scenario: truth.json is valid JSON
- **WHEN** `examples/data/toy/truth.json` is parsed
- **THEN** it contains the keys: `seed`, `trajectory_mode`, `group_effect_size`, `group_labels`, `stage_mapping`, `affected_features`, `intersim_metadata`

#### Scenario: truth.json reflects the retuned defaults
- **WHEN** `examples/data/toy/truth.json` is parsed after this change lands
- **THEN** `group_effect_size == 1.0`, the `affected_features` list lengths reflect a `prop_affected_features` of approximately `0.1` (each list length within +/-1 of `round(n_features_in_layer * 0.1)`), and the InterSIM metadata records a cluster mean shift of approximately `0.10` for methylation, expression, and proteomics
