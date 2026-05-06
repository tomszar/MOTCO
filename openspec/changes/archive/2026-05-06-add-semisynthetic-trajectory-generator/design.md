## Context

MOTCO now has a packaged InterSIM bridge that returns aligned methylation, gene expression, proteomics, sample IDs, clusters, and metadata. InterSIM's native output is cross-sectional cluster structure, not a MOTCO trajectory design. The simulation strategy needs a middle layer that turns those realistic multi-omics matrices into a known-truth trajectory dataset before any MOTCO integration or RRPP evaluation happens.

For the first generator, we will use the explicit clusters-as-stages assumption: InterSIM cluster labels define ordered trajectory stages. This is a pragmatic first step because it uses the clustering signal InterSIM already simulates while producing sample-level `stage` labels MOTCO can use in model matrices.

## Goals / Non-Goals

**Goals:**

- Convert an `InterSIMResult` into a MOTCO-ready semi-synthetic trajectory dataset
- Optionally invoke `run_intersim()` from generator parameters for one-call generation
- Map InterSIM clusters to ordered stages deterministically
- Assign samples to comparison groups reproducibly with configurable group ratio
- Inject group-specific trajectory effects for `none`, `translation`, `magnitude`, `orientation`, and `shape`
- Return aligned omics matrices, sample metadata, and truth metadata
- Keep the generator deterministic under explicit seeds
- Make the clusters-as-stages assumption explicit in metadata and documentation

**Non-Goals:**

- Run PLS, SNF, MOFA, or any other omics integration method
- Build MOTCO model matrices or call `estimate_difference` / `RRPP`
- Estimate p-values, power, Type I error, or simulation summaries
- Implement grid/batch orchestration or parquet result writers
- Support arbitrary InterSIM feature counts beyond what the bridge returns
- Solve the stronger "InterSIM as covariance engine" design in this change

## Decisions

### Represent generated data with a dedicated dataset dataclass

The generator should return a dataclass equivalent to:

```python
SemiSyntheticTrajectoryDataset(
    methylation=pd.DataFrame,
    expression=pd.DataFrame,
    proteomics=pd.DataFrame,
    metadata=pd.DataFrame,
    truth=dict,
)
```

`metadata` contains one row per sample with at least `sample_id`, `group`, `stage`, and `cluster`. The omics matrices remain row-aligned to `metadata`.

Rationale: this separates sample design from omics matrices and makes downstream conversion to MOTCO design matrices straightforward. It also avoids overloading `InterSIMResult`, whose responsibility is only R output normalization.

### Keep generation as a transformation layer

The core API should transform an existing `InterSIMResult`. A convenience API can call `run_intersim()` and then apply the same transformation.

Rationale: pure transformation tests can run without R, and downstream experiments can cache/reuse InterSIM outputs. The convenience path keeps interactive use ergonomic.

### Use clusters-as-stages with deterministic ordering

InterSIM cluster labels will be sorted by their string/numeric representation and mapped to integer stages `0..S-1`. The original cluster label is preserved in metadata.

Example:

```text
InterSIM cluster labels: 1, 2, 3
stage mapping:          1 -> 0, 2 -> 1, 3 -> 2
```

Rationale: MOTCO trajectory routines expect ordered levels. InterSIM clusters are not intrinsically longitudinal, so the assumption must be explicit and reproducible.

Alternative considered: infer ordering from omics centroids. Rejected for the first implementation because it would make the "truth" depend on a noisy estimate and could change across parameter settings.

### Assign groups within each stage

Group labels should be assigned within each stage according to `group_ratio`, using the generator seed. This preserves stage representation across groups as much as sample counts allow.

Rationale: assigning groups globally can accidentally leave a stage underrepresented in one group, which breaks or destabilizes trajectory comparisons. Within-stage assignment makes Type I and power scenarios cleaner.

### Inject trajectory effects after InterSIM simulation

InterSIM provides realistic background multi-omic structure. The generator will then add controlled group-specific shifts to selected features for group B by stage:

```text
none:        no group-specific shift
translation: constant shift across all stages
magnitude:   stage-proportional shift along the main direction
orientation: stage-proportional off-axis shift
shape:       non-monotone stage-specific off-axis shift
```

The implementation should apply shifts independently per omics matrix while recording affected features and effect vectors in truth metadata. `shape` requires at least three stages.

### Choose affected features deterministically

The generator should select affected features per omics layer using the generator seed and a configurable `prop_affected_features` or explicit feature lists. Explicit lists take precedence.

Rationale: explicit lists support reproducible tests and later benchmark designs; proportion-based selection supports simulation studies.

### Preserve Type I error scenario

When `trajectory_mode="none"` or `group_effect_size=0`, the generator must not introduce group-specific omics shifts. It should still assign group/stage labels and record truth metadata.

Rationale: Type I error evaluation requires null datasets that differ only by random group assignment, not injected signal.

## Risks / Trade-offs

- **Clusters are not true temporal stages** -> Record the assumption explicitly in truth metadata and docs; later work can replace the stage construction with an "InterSIM as covariance engine" generator.
- **Small stages may not support balanced group assignment** -> Validate minimum per-stage sample counts and raise clear errors when a stage cannot contain both groups.
- **Translation mode may not affect MOTCO magnitude/orientation/shape tests** -> Preserve it as a negative-control/centroid-shift scenario and document expected interpretation.
- **Effect scale may be hard to compare across omics** -> Start with additive shifts in feature space and record all affected features/effect vectors; calibration can be a later proposal.
- **Shape mode with fewer than three stages is undefined** -> Reject `shape` when fewer than three stages are available.

## Migration Plan

This is additive. The existing InterSIM bridge stays unchanged. The untracked `src/simulations` mockup can be used as conceptual reference, but the implementation should live under `src/motco/simulations/` so it is packaged and tested.
