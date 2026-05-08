# Acceptance sweep — `--cluster-mean-shift` × `y = stage` AUROC

Generated 2026-05-08 on seed 42, n_samples=90, default `--effect-size` and `--prop-affected-features`. PLS-DA: `cv1_splits=5, cv2_splits=5, n_repeats=5, max_components=8`. AUROC and AUROC_std are means across the 5 repeats of the CV table returned by `plsda_doubleCV` after the `fix-plsda-nested-cv-aggregation` change (each repeat is itself a mean across K=5 outer folds).

| `--cluster-mean-shift` | AUROC mean | AUROC_std mean |
|------------------------|-----------:|---------------:|
| (default — no flag)    | 1.0000     | 0.0000         |
| 0.10                   | **0.7919** | **0.0736**     |
| 0.25                   | 0.9992     | 0.0018         |
| 0.50                   | 1.0000     | 0.0000         |
| 0.75                   | 1.0000     | 0.0000         |
| 1.00                   | 1.0000     | 0.0000         |
| 1.50                   | 1.0000     | 0.0000         |
| 2.00                   | 1.0000     | 0.0000         |

## Acceptance against design D5

| Criterion | Result |
|---|---|
| Monotonicity: smaller `cluster_mean_shift` → lower `AUROC_mean` | ✓ (0.79 < 0.999 < 1.0 across 0.10 / 0.25 / 0.50) |
| At smallest tested, `AUROC_mean < 0.99` | ✓ (0.7919 at cms=0.10) |
| At smallest tested, `AUROC_std > 0.01`  | ✓ (0.0736 at cms=0.10) |

## Interpretation

InterSIM's internal default for the `delta.*` parameters places cluster centroids well above the noise floor on every omic, so even moderate fractions (e.g. `cms=0.5`) leave the centroids fully separable in the concatenated 660-feature space. AUROC saturation occurs once the per-feature shift exceeds approximately 0.25 on this 90-sample / 660-feature configuration; below that threshold, finite-sample variance becomes visible.

Users who want non-trivial `y = stage` classification difficulty for tutorials or experiments should pass `--cluster-mean-shift` in the range `[0.05, 0.20]`. Higher values (e.g. matching InterSIM's defaults) preserve the original easy-classification behaviour.
