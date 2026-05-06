## Context

The semi-synthetic generator produces MOTCO-ready omics matrices, sample metadata (`sample_id`, `group`, `stage`, `cluster`), and truth metadata. The next layer should answer: given one generated dataset, what does MOTCO estimate and test? This layer is deliberately smaller than a simulation study runner. It should be callable in notebooks and testable with small fixtures.

Current MOTCO building blocks:

- `motco.stats.snf`: affinity construction, SNF fusion, spectral embedding
- `motco.stats.design`: model matrix and LS means construction
- `motco.stats.trajectory`: observed trajectory difference estimation
- `motco.stats.permutation`: RRPP permutation procedure

## Goals / Non-Goals

**Goals:**

- Evaluate one `SemiSyntheticTrajectoryDataset`
- Produce a latent/outcome matrix suitable for trajectory analysis
- Construct design matrices, LS means, and contrast from sample metadata
- Estimate observed `deltas`, `angles`, and `shapes`
- Optionally run RRPP with configured permutations and compute p-values
- Return a structured result with observed statistics, p-values, timing, integration metadata, truth metadata, and evaluation parameters
- Keep deterministic behavior where the underlying integration/test path supports it

**Non-Goals:**

- Enumerate simulation grids
- Run many replicates or parallel jobs
- Write parquet/CSV result collections
- Produce power or Type I error summaries
- Add SLURM/local batch execution
- Add a public CLI command
- Implement MOFA or external integration methods unless a dependency already exists

## Decisions

### Treat evaluation as a single-dataset function

The core API should look conceptually like:

```python
evaluate_semisynthetic_trajectory(
    dataset: SemiSyntheticTrajectoryDataset,
    params: SimulationEvaluationParams,
) -> SimulationEvaluationResult
```

This keeps the harness composable. Later grid orchestration can call this repeatedly.

### Start with lightweight integration methods

The initial integration choices should be:

- `concat`: column-bind standardized omics matrices into one outcome matrix
- `snf`: build per-omic affinity matrices, fuse with SNF, and use spectral embedding as the outcome matrix

Rationale: both can be implemented with current MOTCO dependencies. PLS-DA is supervised by labels and may leak group/stage signal into the latent space; it should be considered later with a clear design. MOFA and PSN are out of scope unless their dependencies are added separately.

### Build design from generator metadata

The full model should use group, stage, and group x stage interaction via `get_model_matrix(..., full=True)`. The reduced model should use `full=False` for RRPP. LS means should use sorted group and stage levels through `build_ls_means`.

The contrast should be derived from LS-mean row order:

```text
group A stages: [0, 1, 2]
group B stages: [3, 4, 5]
contrast: [[0, 1, 2], [3, 4, 5]]
```

### Compute p-values from RRPP distributions

RRPP returns null distributions of full pairwise matrices. The harness should extract the same group-pair statistic from each permutation as the observed comparison and compute upper-tail empirical p-values using the standard +1 correction:

```text
p = (1 + count(null >= observed)) / (1 + n_permutations)
```

For `shape`, return `NaN` or omit the p-value when fewer than three stages make shape undefined upstream.

### Preserve truth and parameter metadata

The result should include:

- generator truth metadata from the dataset
- integration method and parameters
- RRPP permutation count and progress setting
- observed matrices or selected pairwise scalar statistics
- p-values
- runtime seconds

This allows the grid layer to aggregate results without rerunning the evaluation.

## Risks / Trade-offs

- **Integration choice affects power estimates** -> Record integration method and parameters explicitly; start with methods already supported by MOTCO.
- **SNF spectral embedding has constraints on `k` and sample count** -> Validate integration parameters against dataset size before running.
- **RRPP runtime can dominate** -> Support `permutations=0` for observed-only smoke tests and expose `n_jobs`/`progress` where available.
- **Shape statistics require at least three stages** -> Treat fewer-stage shape output consistently as unavailable rather than forcing a value.
- **High-dimensional concatenation scale imbalance** -> Standardize features by default in `concat` and record that choice.

## Migration Plan

This is additive. No existing MOTCO APIs change. The harness consumes the previously implemented generator and stats modules.
