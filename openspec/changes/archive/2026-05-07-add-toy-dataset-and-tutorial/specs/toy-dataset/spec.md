## ADDED Requirements

### Requirement: Pre-generated toy dataset in repository
The repository SHALL include a pre-generated toy dataset at `examples/data/toy/` containing all files produced by:
```
motco simulate --seed 42 --n-samples 90 --trajectory-mode orientation --effect-size 2.0 --out-dir examples/data/toy/
```
This dataset SHALL be usable for tutorial purposes without R or InterSIM installed.

#### Scenario: Toy data is present after clone
- **WHEN** the repository is cloned
- **THEN** `examples/data/toy/` exists and contains at minimum: `methylation.csv`, `expression.csv`, `proteomics.csv`, `metadata.csv`, `model_full.csv`, `model_reduced.csv`, `ls_means.csv`, `contrast.json`, `truth.json`

#### Scenario: Dataset demonstrates orientation trajectory difference
- **WHEN** `motco de` is run on the toy data latent space
- **THEN** the angle statistic between groups A and B is substantially greater than 0° (orientation difference is detectable)

### Requirement: truth.json documents generation parameters
`examples/data/toy/truth.json` SHALL record the seed, trajectory mode, effect size, group labels, stage mapping, and affected features used to generate the dataset, so users can understand and reproduce it.

#### Scenario: truth.json is valid JSON
- **WHEN** `examples/data/toy/truth.json` is parsed
- **THEN** it contains the keys: `seed`, `trajectory_mode`, `group_effect_size`, `group_labels`, `stage_mapping`, `affected_features`
