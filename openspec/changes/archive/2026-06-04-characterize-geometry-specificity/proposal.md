## Why

The `numpy-feature-surgery-generator` change established the methylation-only
trajectory modes and a *descriptive* specificity matrix. Running it surfaced
open questions that need targeted follow-up before the paper-grade power study
is read as definitive:

- **The `shape` (Procrustes) statistic is near-saturated.** It rejects ≈1.0 for
  *every* geometry mode (`magnitude`, `orientation`, `shape`) and ~0.17 under
  the null. Procrustes shape is location- *and* scale-invariant, so a pure size
  change (`magnitude`) should not move it — yet it does. This is either a
  genuine property of MOTCO's shape test on integrated multi-omic data (a
  finding worth reporting) or an anti-conservative calibration (worth fixing).
- **Size/orientation are only checked in the 3-stage regime**, where shape can
  contaminate them. A 2-stage regime (where Procrustes shape is degenerate)
  isolates `magnitude`→`delta` and `orientation`→`angle` cleanly.
- **`magnitude` co-moves `shape`**, plausibly because scaling δ through
  methylation's `rev.logit` is nonlinear. An alternative construction that
  scales only the extreme stages is worth testing.

## What Changes

- Add a **2-stage validation pass** to the specificity characterization, to
  confirm `magnitude`→`delta` and `orientation`→`angle` when Procrustes shape is
  out of the picture.
- Add a **`magnitude` construction variant** (`magnitude_kind`) that scales only
  the extreme-stage methylation effects (endpoints), as a probe of whether
  localizing the scale reduces shape co-movement.
- **Investigate the `shape` statistic's saturation**: separate the observed
  Procrustes distance from its RRPP permutation null; test sensitivity to the
  integration choice (concat-standardize vs raw vs SNF); quantify the
  null-rejection rate vs α. Document the cause. If it is a calibration bug,
  recalibrate the statistic / its RRPP null; if it is genuine, record it as a
  MOTCO property to report in the study.

## Capabilities

### Modified Capabilities

- `semisynthetic-trajectory-generator`: add a `magnitude_kind` option selecting
  whether `magnitude` scales all stages (current default) or only the extreme
  stages.

## Impact

- `src/motco/simulations/semisynthetic.py` — `magnitude_kind` variant.
- `src/motco/simulations/specificity.py` — 2-stage characterization and a
  shape-null diagnostic (observed vs permutation distribution).
- Investigation only (conditional): `src/motco/stats/trajectory.py` /
  `permutation.py` / `evaluation.py` if the shape-statistic saturation turns out
  to be a calibration issue rather than a genuine property.
- Findings feed the paper-grade power-study run (deferred to the cluster); a
  saturated shape row would be reported with the calibration caveat resolved.
