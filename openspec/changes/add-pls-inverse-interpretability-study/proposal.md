## Why

MOTCO assumes that magnitude, orientation, and shape changes measured in a PLS latent space have interpretable consequences in the original features, but that reverse relationship has not been characterized. A small controlled study is needed to impose exact latent-space changes and measure the feature changes implied by the fitted linear PLS map.

## What Changes

- Add a simple Gaussian feature simulator with two initially identical groups and either two or three stages.
- Fit one frozen, two-component PLS-DA model to stage labels without cross-validation or production multi-omics machinery.
- Apply exact centroid-level interventions to Group B in PLS space: magnitude and orientation for two stages; magnitude, orientation, and shape for three stages.
- Reconstruct the implied feature changes additively so each sample's variation outside the retained PLS space is preserved.
- Report round-trip validity, intended versus recovered geometry, per-feature changes, loading alignment, and the metric induced by the PLS loadings.
- Provide a small reproducible driver and findings-ready tabular output; no hypothesis testing or power analysis is introduced.

## Capabilities

### New Capabilities

- `pls-inverse-interpretability-study`: Generate controlled null feature data, intervene on trajectory geometry in a fixed PLS latent space, reconstruct the implied feature changes, and summarize their geometry and loading structure.

### Modified Capabilities

<!-- None. This is an additive diagnostic study and does not change production PLS integration or trajectory statistics. -->

## Impact

- **New simulation code:** a self-contained module under `src/motco/simulations/` for data generation, latent interventions, additive reconstruction, and diagnostics.
- **New driver:** a script under `scripts/` that runs the two- and three-stage study and writes compact machine-readable and Markdown results.
- **Tests:** deterministic checks for identical baselines, exact latent interventions, residual preservation, round-trip reconstruction, and output schemas.
- **Dependencies:** existing NumPy, pandas, and scikit-learn only.
- **Production behavior:** unchanged; no changes to the semi-synthetic generator, PLS integration path, RRPP, CLI, or public statistical API.
