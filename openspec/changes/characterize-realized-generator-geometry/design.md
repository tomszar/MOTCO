## Context

See `proposal.md` for motivation. The generator constructs group B by modifying methylation differential indicators, then derives expression and protein indicators through fixed incidence maps. It samples each group-stage cell from an omic-specific multivariate normal mean-shift model; methylation is sampled in M-value space and converted to B values for storage.

The evaluation harness currently preprocesses data independently inside `concat` and PLS integration. Both convert methylation back to M-values and standardize each omic feature over all samples, but the duplicated logic can drift. Only the post-integration trajectory statistics are persisted. Generator truth retains indicators in memory but drops them from serialized evaluation results.

## Goals / Non-Goals

**Goals:**

- Make the generator's exact group-stage population means available in integration units.
- Measure the same trajectory estimands at population-native, population-standardized, observed-standardized, and PLS-latent checkpoints.
- Separate per-omic cascade effects from joint preprocessing and PLS projection effects.
- Guarantee that diagnostic and production PLS preprocessing use the same fitted transformation.
- Persist compact, analysis-ready diagnostics without serializing high-dimensional indicator or mean matrices into every study row.

**Non-Goals:**

- Redesign the orientation or shape surgery before its realized behavior is measured.
- Require a perfectly diagonal mode-by-statistic response matrix.
- Compare raw Euclidean distances across native omic units or treat feature and latent distances as scale-equivalent.
- Add permutation tests at every diagnostic checkpoint; inferential testing remains attached to the production measurement space.
- Change the validated trajectory estimators or PLS fitting contract.

## Decisions

### 1. Represent the decomposition as four ordered checkpoints

Diagnostics use:

1. `population_native`: analytic group-stage means in per-omic integration units;
2. `population_standardized`: those means transformed by the replicate's fitted pooled block scalers;
3. `observed_standardized`: finite observations transformed by the same scalers; and
4. `pls_latent`: observed scores from the fitted production PLS representation.

`population_native` is per-omic only because concatenating M-values, expression, and protein abundances without scaling has no defensible joint metric. The two standardized checkpoints support per-omic and joint scopes; `pls_latent` is joint only.

This ordering localizes construction and cascade behavior at checkpoint 1, preprocessing behavior between checkpoints 1 and 2, sampling behavior between checkpoints 2 and 3 conditional on the fitted scaler, and projection behavior between checkpoints 3 and 4. The comparisons are interpretive rather than literal subtraction of nonlinear statistics.

Alternative considered: record only observed standardized and latent statistics. Rejected because repeated seeds can describe average sampling behavior but cannot identify whether a particular off-target response was present in the exact construction.

### 2. Derive analytic means from generator truth, not simulated averages

For each group, stage, and omic, population means are reconstructed from the same baseline, differential indicators, group-specific deltas, coupling parameters, and any constant shift used by generation. Methylation means remain in pre-`rev_logit` M-value space, matching the integration contract and avoiding logistic-normal expectation ambiguity.

The generator will expose the information needed for this reconstruction through a dedicated structured truth object or equivalent internal representation. Study persistence stores only derived scalar diagnostics and compact construction metadata, not the high-dimensional population arrays.

Alternative considered: approximate population means with a very large auxiliary sample. Rejected because it adds Monte Carlo error and runtime to quantities defined analytically by the generator.

### 3. Extract one fitted block-preprocessing artifact

One preprocessing path will:

- preserve canonical omic and feature order;
- convert methylation B values to clipped M-values;
- fit pooled per-feature means and population-standard-deviation scales on observed samples;
- replace scales below the existing tolerance with `1.0`; and
- transform observed matrices and arbitrary aligned population matrices.

`concat` and PLS will consume the transformed blocks produced by this path. This makes the observed joint diagnostic matrix exactly the PLS input rather than a separately reproduced approximation. The preprocessing artifact remains internal unless a later change promotes fitted analysis artifacts to a public API.

Alternative considered: calculate diagnostics through the existing `concat` integration while leaving PLS preprocessing separate. Rejected because equality would depend on duplicated implementations.

### 4. Use the production estimands with explicit path lengths

At each checkpoint, group-stage means are ordered using the production design's group and stage levels. For group `g` with ordered means `mu[g, s]`, path length is:

```text
L_g = sum_s ||mu[g, s + 1] - mu[g, s]||
```

The pairwise magnitude statistic is `abs(L_A - L_B)`. Orientation and shape use the same definitions and proper-rotation/reflection policy as the production trajectory routines. Diagnostics additionally retain `L_A` and `L_B`, because `delta` alone hides which group changed and whether both path lengths moved.

Undefined statistics use an explicit unavailable representation that serializes safely: shape for fewer than three stages, and angle when a path has zero length. They must not be silently converted to zero or counted as non-rejections.

Alternative considered: introduce new feature-space-specific geometry formulas. Rejected because checkpoint comparison is clearest when the estimand is held fixed and only the representation changes.

### 5. Persist a nested diagnostic schema and flatten only for summaries

Each evaluation result stores diagnostics under stable checkpoint and scope identifiers, with scalar path lengths and pairwise statistics at the leaves. Requested mode/effect and transform metadata remain linked at the result level. JSONL records preserve the nested structure; tabular study/report helpers may flatten it into names such as `population_standardized.joint.angle`.

This avoids a wide result dataclass with dozens of optional fields and makes applicability explicit. A schema/version tag will be included in parameter signatures or diagnostic metadata so resume logic cannot silently mix legacy and decomposed study rows.

Alternative considered: add one column per checkpoint/scope/statistic directly to the core result. Rejected because it is brittle as diagnostics evolve and obscures unavailable combinations.

### 6. Characterize before changing construction contracts

The Phase 2 study will use matched seeds over the roadmap's orientation and shape effect grid. Reports will show requested effect against realized statistics at all checkpoints, including per-omic cascade propagation. Acceptance focuses on monotonicity, preservation of intended controls, and an attributable explanation for off-diagonal behavior.

The study may recommend a later change to constrain orientation or introduce separate free-bend and magnitude-matched shape modes. Those changes are deliberately not bundled here because the decomposition should provide the evidence used to choose them.

## Risks / Trade-offs

- [Analytic truth reconstruction diverges from generation] → Centralize mean construction so sampling and diagnostics consume the same mean definition, and test analytic means against large-sample averages.
- [Observed-fitted scaling makes standardized population geometry replicate-dependent] → Document that checkpoint 2 represents exact means under the production scaler for that replicate; use the identical scaler at checkpoint 3 so their difference isolates finite-sample cell-mean error conditional on preprocessing.
- [High-dimensional diagnostics increase runtime or record size] → Compute and persist scalar geometry only; do not serialize population or transformed feature matrices in study rows.
- [PLS fitting currently returns scores without a reusable fitted transform] → Refactor the integration boundary carefully while enforcing score equivalence with regression tests.
- [Unavailable values break JSON or aggregation] → Use an explicit JSON-safe unavailable marker and make summary utilities exclude rather than zero-fill it.
- [A richer decomposition invites overinterpretation across scales] → Label every checkpoint with its measurement space and emphasize within-checkpoint trends and contrasts rather than raw cross-space distance equality.

## Migration Plan

1. Add analytic mean truth and shared preprocessing behind internal interfaces without changing existing result fields.
2. Add the nested diagnostic result and persistence schema with a version/signature update.
3. Update grid and reporting readers to accept new records; legacy rows remain readable but are marked as lacking decomposition diagnostics and cannot satisfy a resumed run with the new signature.
4. Run deterministic tests and a small matched-seed characterization before the medium pilot.

Rollback consists of disabling diagnostic production and reverting the study signature version; existing additive diagnostic fields do not alter the meaning of legacy result fields.
