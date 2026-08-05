# Realized Generator Geometry — Phase 2 Findings

## Decision

The Phase 2 gate is resolved at the construction level. MOTCO now records four geometry checkpoints for every generated replicate, which separates exact construction, pooled preprocessing, finite sampling, and PLS projection:

```text
population_native
        ↓ pooled observed-fitted block scaling
population_standardized
        ↓ finite-sample cell-mean error
observed_standardized
        ↓ fitted PLS projection
pls_latent
```

The generator modes are not required to form a perfectly diagonal mode-by-statistic matrix. Their revised contracts are:

- **Orientation** is a pure coordinate relocation in native methylation M-space. The biological CpG→gene→protein cascade makes it a joint multi-omic program relocation with expected magnitude and shape co-movement.
- **Shape/relocate** is a free biological bend. It changes the interior-stage program and may also change path length and principal orientation.
- **Shape/magnitude** is an interior-stage amplitude perturbation and is intrinsically mixed; retain it as a stress-test construction, not a shape-specific positive control.
- **Magnitude/all** is pure within native methylation but scales only one of three omic blocks, so joint standardization can introduce orientation and shape co-movement even though the established production test remains empirically specific.
- **Translation** remains the negative geometry control: its exact population delta, angle, and shape stay at numerical zero before and after standardization.

These interpretations replace the assumption that requested mode alone identifies the realized pre-integration geometry. All subsequent operating-characteristic studies must report the realized checkpoint diagnostics alongside rejection rates.

## Small matched-seed characterization

This implementation check used two matched seeds and three effects. It is sufficient to validate the decomposition and locate construction cross-talk, but it is not a power study and its noisy observed/PLS values must not be treated as stable operating characteristics.

Parameters:

- seeds: `0,1`
- effects: `0,0.5,1`
- `n_samples=80`, `n_stages=4`, `p_dmp=0.2`
- PLS CV: one repeat, two inner and outer folds, at most three components
- no permutations

### Orientation

Native methylation preserves equal path length and shape while rotating direction. At effects 0.5 and 1.0, its mean realized angles were 57.89° and 73.40°, with delta 0 and shape at numerical zero.

The cascade is the first source of cross-talk. At effect 0.5, native expression showed angle 64.99°, delta 1.12, and shape 0.066; native proteomics showed angle 59.32°, delta 1.36, and shape 0.037. Consequently, standardized joint population geometry was already mixed before sampling or PLS:

| Effect | Joint population angle | Joint population delta | Joint population shape |
|---:|---:|---:|---:|
| 0.00 | 0.00° | 0.00 | 0.000 |
| 0.50 | 67.41° | 1.43 | 0.029 |
| 1.00 | 85.35° | 1.78 | 0.052 |

The requested effect maps monotonically to the exact joint angle in this check. The nonzero joint delta and shape are construction/cascade behavior, not PLS leakage.

### Shape

Both constructions were mixed at the exact-population checkpoint.

For `shape_kind=relocate`, joint standardized population geometry increased as follows:

| Effect | Angle | Delta | Shape |
|---:|---:|---:|---:|
| 0.00 | 0.00° | 0.00 | 0.000 |
| 0.50 | 53.88° | 1.97 | 0.050 |
| 1.00 | 85.45° | 7.61 | 0.081 |

The relocation affects all three native omic layers through the cascade, confirming that it is a free multi-omic bend rather than an isolated Procrustes-shape intervention.

For `shape_kind=magnitude`, only native methylation changes because the binary cascade indicators remain unchanged. The interior-stage amplitude change nevertheless produces large angle and delta responses in that layer. Joint standardized population geometry was:

| Effect | Angle | Delta | Shape |
|---:|---:|---:|---:|
| 0.00 | 0.00° | 0.00 | 0.000 |
| 0.50 | 63.54° | 4.99 | 0.064 |
| 1.00 | 75.26° | 8.99 | 0.112 |

This construction is useful for sensitivity analysis but cannot support a claim of shape specificity.

### Controls and sampling

The exact `none` and translation checkpoints remained invariant. Observed standardized statistics were nonzero even for these controls because the two groups are independent finite samples. At `n_samples=80`, the PLS null angles were particularly unstable, so this run does not update the medium-pilot power conclusions.

Magnitude was pure in native methylation (effect 0.5/1.0 delta 31.97/63.94, angle and shape approximately zero) but became mixed in standardized joint population space (angle 13.56°/22.14°, shape 0.012/0.019). This occurs because methylation is scaled while expression and proteomics retain their baseline trajectories.

## Reproduction

Run from the repository root:

```bash
uv run python scripts/realized_geometry_characterization.py \
  --seeds 0,1 \
  --effects 0,0.5,1 \
  --n-samples 80 \
  --n-stages 4 \
  --pls-repeats 1 \
  --pls-cv1 2 \
  --pls-cv2 2 \
  --pls-max-components 3 \
  --out-dir /tmp/motco-phase2-small
```

The driver writes `config.json`, replicate-level JSONL, a long-form summary CSV, and a monotonicity CSV. For the medium pilot, use the roadmap sample size and effect grid with more matched seeds and the production PLS CV settings.

## Gate to Phase 3 and the medium pilot

Phase 3 orientation-driver attribution can proceed using the revised interpretation: it should explain the joint molecular-program direction captured by PLS, not claim a pure methylation-only rotation. Before the medium pilot, retain all four checkpoint diagnostics and predeclare interpretation against realized geometry rather than requested mode labels alone.
