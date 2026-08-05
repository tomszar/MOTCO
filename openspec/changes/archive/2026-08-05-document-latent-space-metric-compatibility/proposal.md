## Why

MOTCO's `delta`, `angle`, and Procrustes `shape` statistics use Euclidean
geometry in whatever outcome matrix they receive. Linear PCA/PLS coordinates
have an explicit relationship to the measured features, while SNF constructs a
nonlinear affinity graph and spectral embedding whose natural invariants are
neighbourhoods, connectivity, and diffusion. The software currently accepts
SNF coordinates without documenting that feature-space magnitude and direction
are not guaranteed to survive that transformation.

The latent-space cross-talk pilot also found no `delta`/`angle` sensitivity and
a saturated `shape` response under SNF. That empirical result needs to be kept
separate from the broader conclusion: the mismatch concerns Euclidean MOTCO
statistics applied to an SNF embedding, not the general value of SNF as an
integration method.

## What Changes

- Document the Euclidean measurement-space assumptions behind `delta`,
  `angle`, and `shape`.
- Explain how standardized concatenation, PCA/PLS, and SNF differ in their
  relationship to original feature-space geometry.
- Mark trajectory statistics calculated on SNF spectral coordinates as
  exploratory unless the intended geometry has been validated by simulation.
- Record candidate graph-native alternatives—diffusion/resistance distance,
  transition-profile divergence, neighbourhood measures, and normalized
  stage-distance-matrix comparison—as future methodology, not current API.
- Amend the pilot findings so “SNF plus Euclidean MOTCO metrics was unsuitable”
  is not misread as “SNF is unsuitable for multi-omics integration.”

## Capabilities

### New Capabilities

- `latent-space-metric-compatibility`: State which geometric interpretations
  are supported by linear and graph-spectral integration outputs and identify
  the validation boundary for SNF-derived results.

### Modified Capabilities

<!-- Documentation only; no runtime contract changes. -->

## Impact

- **Documentation:** `docs/api/snf.md` and `docs/api/sd.md`.
- **Study record:** clarification in the archived latent-space cross-talk
  findings.
- **Runtime behavior:** unchanged.
- **Future work:** a graph-native SNF trajectory test requires a separate
  design, implementation, permutation scheme, and validation study.
