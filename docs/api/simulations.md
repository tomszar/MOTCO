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

## Evaluation harness

The evaluation harness runs one `SemiSyntheticTrajectoryDataset` through MOTCO integration and trajectory testing. It is the per-replicate layer used before larger Type I error or power grids.

```python
from motco.simulations import (
    SimulationEvaluationParams,
    evaluate_semisynthetic_trajectory,
)

evaluation = evaluate_semisynthetic_trajectory(
    dataset,
    SimulationEvaluationParams(
        integration_method="concat",
        permutations=0,
    ),
)

observed_delta = evaluation.pair_statistics["delta"]
truth = evaluation.truth_metadata
```

Supported integration methods:

| Method | Behavior |
|--------|----------|
| `concat` | Column-binds methylation, expression, and proteomics matrices after deterministic per-feature standardization by default |
| `snf` | Builds per-omic affinity matrices, fuses them with SNF, and uses spectral embedding as the trajectory outcome matrix |

Set `permutations=0` for observed statistics only. When `permutations > 0`, the harness runs RRPP and computes upper-tail empirical p-values with plus-one correction:

```text
p = (1 + count(null >= observed)) / (1 + n_permutations)
```

The result includes observed `delta`, `angle`, and `shape` matrices, scalar two-group pair statistics, optional p-values, latent matrix metadata, generator truth metadata, runtime metadata, group/stage levels, and the trajectory contrast. Shape pair statistics and p-values are reported as unavailable for datasets with fewer than three stages.

## Grid orchestration

The grid orchestration layer enumerates parameter cells, runs local replicates through the evaluation harness, persists one JSONL row per replicate, resumes completed work, and summarizes rejection rates for Type I error or power studies.

```python
from pathlib import Path

from motco.simulations import (
    InterSIMParams,
    SemiSyntheticTrajectoryParams,
    SimulationEvaluationParams,
    SimulationRunConfig,
    enumerate_type_i_grid,
    run_simulation_grid,
    summarize_rejection_rates,
)

grid = enumerate_type_i_grid(
    baseline_intersim_params=InterSIMParams(seed=1, n_sample=60),
    baseline_generator_params=SemiSyntheticTrajectoryParams(seed=2),
    evaluation_params=SimulationEvaluationParams(integration_method="concat", permutations=99),
    axes={
        "intersim.n_sample": [60, 120],
        "generator.group_ratio": [0.5, 0.7],
    },
    n_replicates=3,
    base_seed=2026,
)

records = run_simulation_grid(
    grid,
    config=SimulationRunConfig(output_path=Path("simulation-results.jsonl")),
)
summaries = summarize_rejection_rates(records, alpha=0.05)
```

Each `SimulationCell` stores a stable `cell_id`, phase, `InterSIMParams`, `SemiSyntheticTrajectoryParams`, `SimulationEvaluationParams`, replicate count, base seed, and metadata such as the varied axis. Axis names use explicit namespaces: `intersim.<field>`, `generator.<field>`, or `evaluation.<field>`.

Initial persistence is JSON Lines. Each row records cell and replicate IDs, deterministic seeds, a parameter signature, status, p-values, pair statistics, truth metadata, runtime metadata, cell metadata, and optional error details. With `resume=True`, completed rows with matching parameter signatures are skipped. A matching cell/replicate with a different parameter signature raises unless `overwrite=True`.

`summarize_rejection_rates` groups completed replicate rows by cell and statistic, then reports available replicate count, rejection count, rejection rate, Monte Carlo standard error, and unavailable replicate count. Missing statistics, such as shape p-values for two-stage datasets, remain unavailable rather than being counted as non-significant.

## API

::: motco.simulations.InterSIMParams

::: motco.simulations.InterSIMResult

::: motco.simulations.InterSIMAvailability

::: motco.simulations.SemiSyntheticTrajectoryParams

::: motco.simulations.SemiSyntheticTrajectoryDataset

::: motco.simulations.SimulationEvaluationParams

::: motco.simulations.SimulationEvaluationResult

::: motco.simulations.SimulationTrajectoryDesign

::: motco.simulations.LatentIntegrationResult

::: motco.simulations.SimulationCell

::: motco.simulations.SimulationGrid

::: motco.simulations.SimulationReplicateResult

::: motco.simulations.SimulationRunConfig

::: motco.simulations.SimulationSummaryResult

::: motco.simulations.check_intersim_available

::: motco.simulations.run_intersim

::: motco.simulations.generate_semisynthetic_trajectory

::: motco.simulations.generate_semisynthetic_trajectory_from_intersim

::: motco.simulations.integrate_semisynthetic_dataset

::: motco.simulations.build_simulation_trajectory_design

::: motco.simulations.evaluate_semisynthetic_trajectory

::: motco.simulations.enumerate_type_i_grid

::: motco.simulations.enumerate_power_grid

::: motco.simulations.run_simulation_replicate

::: motco.simulations.run_simulation_grid

::: motco.simulations.read_replicate_results

::: motco.simulations.append_replicate_results

::: motco.simulations.summarize_rejection_rates
