## 1. Fidelity protocol + statistic battery

- [x] 1.1 Define the parameter grid (`delta` × `p.DMP`, incl. `delta=0` anchor and ≥2 non-zero) and replicate counts `N` (InterSIM) / `M` (numpy)
- [x] 1.2 Implement the per-omic statistic battery (marginal moments/quantiles, cluster separation η², differential-feature rates, covariance relative-Frobenius) as a reusable function consuming a generated triple
- [x] 1.3 Implement the numpy side from the existing `generate_omics` + `bernoulli_indicators` + `derive_coupled_indicators` (validate the real generator, not a parallel one)

## 2. InterSIM fixtures (R, one-time) + provenance

- [x] 2.1 Add an R script that runs InterSIM `N` times per grid cell and writes the per-cell statistic distributions
- [x] 2.2 Commit the InterSIM summary fixtures (`.npz`/CSV) with provenance (InterSIM version, date, seeds, grid)
- [x] 2.3 Add a loader that reads the fixtures without R and errors clearly when absent

## 3. Replicate-distribution comparison + CI test

- [x] 3.1 Implement the comparison: per cell/statistic, test whether numpy's value falls within InterSIM's documented central interval
- [x] 3.2 Add an R-free CI test asserting fidelity across the grid (fast subset + `slow`-marked full grid); document the interval/tolerance choice
- [x] 3.3 Confirm effect injection and DMP→DEG→DEP coupling are exercised at `delta>0` (the gap the `delta=0` test left)

## 4. Paper supplement

- [x] 4.1 Add a script that renders a supplementary table + figure (numpy-vs-InterSIM fidelity across the grid) from the committed fixtures
- [x] 4.2 Document how to reproduce the fixtures and the supplement (R step + Python step)

## 5. Qualitative visual supplement

- [x] 5.1 Add an R script that emits the InterSIM raw data (density replicates + one multi-cluster replicate) and a Python builder/loader; the raw matrices are not committed (regenerated via `flake.nix`, gitignored `build/`)
- [x] 5.2 Implement the figures (side-by-side InterSIM vs numpy): per-omic densities, per-modality clustermap heatmaps (sample/feature dendrograms + cluster colour bar), per-modality PCA, per-feature mean/variance scatter, cross-omic coupling correlation block — numpy side generated live
- [x] 5.3 Add a thin CLI (`scripts/fidelity_visual.py`) and R-free smoke tests (synthetic stand-in fixture); document reproduction in `FIDELITY.md`

## 6. Wrap-up

- [x] 6.1 ruff + mypy + fast pytest green with no R on PATH
- [x] 6.2 Update `CLAUDE.md` / generator docs to point at the fidelity validation and supplement