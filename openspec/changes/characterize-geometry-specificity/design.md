## Context

The descriptive specificity matrix (from `numpy-feature-surgery-generator`,
recorded in that change's `specificity_results.md`) reads, at `effect_size=1.0`,
`n_stages=4`, 12 reps, 49 perms:

| mode               | delta | angle | shape |
|--------------------|-------|-------|-------|
| `none`             | 0.08  | 0.00  | 0.17  |
| `translation`      | 0.33  | 0.08  | 0.17  |
| `magnitude`        | 1.00  | 0.00  | 1.00  |
| `orientation`      | 0.17  | 0.42  | 1.00  |
| `shape` (relocate) | 0.83  | 0.50  | 1.00  |
| `shape` (magnitude)| 1.00  | 0.42  | 1.00  |

`delta` and `angle` are sensible (size for `magnitude`, direction for
`orientation`). The `shape` column is the problem: it fires for everything and
its null is ~0.17, not ~α.

## Goals / Non-Goals

**Goals:**
- Confirm `magnitude`/`orientation` specificity in a shape-free (2-stage) regime.
- Determine whether the `shape` saturation is a genuine MOTCO property or a
  calibration issue, and act accordingly.
- Add the endpoints-only `magnitude` variant as a probe.

**Non-Goals:**
- Not tuning the data to force a clean diagonal (specificity stays descriptive;
  realistic constructions are kept even when MOTCO cross-detects).
- Not re-running the paper-grade study here (cluster job, deferred).

## Decisions

- **2-stage isolation first.** With two stages the trajectory is a single step,
  so Procrustes shape is degenerate/undefined. Running `magnitude` and
  `orientation` at `n_stages=2` tests their target statistics with shape removed
  — the cleanest discriminator between "construction is fine, shape stat is the
  problem" and "construction leaks".
- **Split observed vs null for `shape`.** The RRPP rejection rate conflates the
  observed statistic and its permutation null. The probe will record, per
  replicate, the observed Procrustes distance and the spread/quantiles of its
  permutation null, to see whether the null is collapsing (anti-conservative) or
  the observed distance is genuinely extreme for any change.
- **Integration sensitivity.** Re-run the shape probe under raw (no
  standardize), concat-standardize, and SNF integration to see whether the
  per-feature standardization is what breaks Procrustes scale-invariance.
- **`magnitude_kind` is additive and backward-compatible.** Default stays
  "all stages" (current behavior); "endpoints" scales only stages `0` and
  `K-1`. This is a generator option, not a redefinition of `magnitude`.

## Risks / Trade-offs

- **The shape saturation may be genuine** (Procrustes on integrated,
  standardized multi-omic LS-means is simply very sensitive). Then there is no
  code fix — the deliverable is a documented finding and a caveat on the study's
  shape row. The investigation must be able to *conclude* "genuine", not just
  hunt for a bug.
- **Endpoints `magnitude` may not help** (the nonlinearity, not the stage
  profile, is the likely cause). Acceptable — it is a cheap, informative probe.
