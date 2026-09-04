## MODIFIED Requirements

### Requirement: Pre-generated toy dataset in repository
The repository SHALL include a pre-generated toy dataset at `examples/data/toy/` whose generation seed and parameters are recorded in `truth.json`. The dataset is a **pinned fixture**: it was produced by the InterSIM-bridge generator that predates the numpy generator, so it SHALL NOT be described as reproducible by the current `motco simulate` CLI. The spec MUST NOT publish a regeneration command that the shipped CLI rejects.

The nearest current-CLI equivalent — a distinct dataset, not a reproduction — is:

```
motco simulate \
  --seed 42 \
  --n-samples 90 \
  --n-stages 3 \
  --trajectory-mode orientation \
  --effect-size 1.0 \
  --p-dmp 0.2 \
  --cluster-mean-shift 0.10 \
  --out-dir results/simulated/
```

This dataset SHALL be usable for tutorial purposes without R or InterSIM installed.

#### Scenario: Toy data is present after clone
- **WHEN** the repository is cloned
- **THEN** `examples/data/toy/` exists and contains at minimum: `methylation.csv`, `expression.csv`, `proteomics.csv`, `metadata.csv`, `model_full.csv`, `model_reduced.csv`, `ls_means.csv`, `contrast.json`, `truth.json`

#### Scenario: Documented commands are accepted by the shipped CLI
- **WHEN** any `motco simulate` command published in this spec or in `README.md` is run
- **THEN** the CLI accepts every flag it passes and exits successfully

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
`examples/data/toy/truth.json` SHALL record the seed, trajectory mode, effect size, group labels, stage mapping, affected features, and InterSIM generation metadata used to generate the dataset, so users can understand its provenance. Because the fixture predates the numpy generator, this metadata documents the historical generation run; it SHALL NOT be described in terms of CLI flags the shipped parser no longer accepts.

#### Scenario: truth.json is valid JSON
- **WHEN** `examples/data/toy/truth.json` is parsed
- **THEN** it contains the keys: `seed`, `trajectory_mode`, `group_effect_size`, `group_labels`, `stage_mapping`, `affected_features`, `intersim_metadata`

#### Scenario: truth.json reflects the fixture's recorded generation values
- **WHEN** `examples/data/toy/truth.json` is parsed
- **THEN** `group_effect_size == 1.0`, each `affected_features` list length is within +/-1 of `round(n_features_in_layer * 0.1)`, and the InterSIM metadata records a cluster mean shift of approximately `0.10` for methylation, expression, and proteomics
