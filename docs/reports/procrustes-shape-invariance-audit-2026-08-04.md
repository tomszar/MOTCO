# Procrustes Shape Invariance Audit

**Date:** 2026-08-04

> **Superseded on the reflection policy (2026-09-03).** The reflection rows below record the
> contract as it stood on this date. `unify-shape-reflection-policy` (resolving audit finding F3)
> replaced it: the Procrustes alignment now optimizes over the full orthogonal group, so
> reflections are aligned away at every ambient dimension and a mirror pair scores **zero**, not
> positive. Everything else in this audit — the translation, scale, proper-rotation, and interior-bend
> cases, and the legacy-GPA comparator — still holds.

## Summary

MOTCO now defines trajectory `shape` as direct pairwise Procrustes residual
configuration after removing translation, proper rigid rotation, and positive
uniform scale. Reflections are preserved as distinct shapes by default.

## Deterministic Cases

The audit uses a four-stage, non-collinear two-dimensional trajectory and
deterministic transforms:

| Case | Strict pairwise behavior | Legacy leave-one-out GPA diagnostic |
|---|---|---|
| Translation | Zero within tolerance | Matches strict reference |
| Uniform scale | Zero within tolerance | Matches strict reference |
| Proper rotation | Zero within tolerance | Diverges with a positive residual |
| Reflection | Positive by default | Recorded as legacy comparator |
| Interior bend | Positive | Recorded as legacy comparator |

## Interpretation

The production estimator is now the strict pairwise Procrustes calculation.
The previous leave-one-out GPA procedure remains represented in tests only as a
legacy comparator so the rotation-invariance divergence is visible. R
`pgpa`/`pPsup` comparison is optional in the test suite and skips when the R
environment or package functions are unavailable.

Older operating-characteristic results that depended on the legacy shape
implementation should be treated as superseded for shape-specific claims.
