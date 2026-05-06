# Simulations

InterSIM bridge utilities and semi-synthetic trajectory generators for multi-omics simulation studies.

## R dependency

The bridge is optional and requires `Rscript` plus the R `InterSIM` package:

```r
install.packages(
  "InterSIM",
  repos = c("https://cran.r-universe.dev", "https://cloud.r-project.org")
)
```

Check availability before running a simulation:

```python
from motco.simulations import check_intersim_available

availability = check_intersim_available()
if not availability.available:
    print(availability.message)
```

## Example

```python
from motco.simulations import InterSIMParams, run_intersim

result = run_intersim(
    InterSIMParams(
        seed=1203,
        n_sample=100,
        cluster_sample_prop=(0.3, 0.3, 0.4),
        delta_methyl=1.0,
        delta_expr=1.0,
        delta_protein=1.0,
        p_dmp=0.1,
    )
)

methylation = result.methylation
expression = result.expression
proteomics = result.proteomics
clusters = result.clusters
```

## Semi-synthetic trajectory generation

The semi-synthetic trajectory generator converts an `InterSIMResult` into a MOTCO-ready dataset with:

- aligned methylation, gene expression, and proteomics matrices
- sample metadata containing `sample_id`, `group`, `stage`, and `cluster`
- truth metadata recording trajectory mode, affected features, stage mapping, and generator seed

The first generator uses an explicit **clusters-as-stages** assumption: sorted InterSIM cluster labels are mapped to ordered integer stages starting at 0. Original cluster labels are preserved in sample metadata.

```python
from motco.simulations import (
    InterSIMParams,
    SemiSyntheticTrajectoryParams,
    generate_semisynthetic_trajectory_from_intersim,
)

dataset = generate_semisynthetic_trajectory_from_intersim(
    InterSIMParams(seed=1203, n_sample=120, cluster_sample_prop=(0.3, 0.3, 0.4)),
    SemiSyntheticTrajectoryParams(
        seed=99,
        trajectory_mode="magnitude",
        group_effect_size=0.2,
        group_ratio=0.5,
        prop_affected_features=0.05,
    ),
)

sample_metadata = dataset.metadata
truth = dataset.truth
```

Supported trajectory modes:

| Mode | Injected group-specific pattern |
|------|---------------------------------|
| `none` | No group-specific shift; useful for Type I error scenarios |
| `translation` | Same affected-feature shift in every stage |
| `magnitude` | Stage-proportional shift along the affected-feature direction |
| `orientation` | Stage-proportional off-axis shift |
| `shape` | Non-monotone stage-specific shift; requires at least three stages |

## API

::: motco.simulations.InterSIMParams

::: motco.simulations.InterSIMResult

::: motco.simulations.InterSIMAvailability

::: motco.simulations.SemiSyntheticTrajectoryParams

::: motco.simulations.SemiSyntheticTrajectoryDataset

::: motco.simulations.check_intersim_available

::: motco.simulations.run_intersim

::: motco.simulations.generate_semisynthetic_trajectory

::: motco.simulations.generate_semisynthetic_trajectory_from_intersim
