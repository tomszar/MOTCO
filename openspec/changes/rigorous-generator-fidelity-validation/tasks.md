## 1. Fidelity protocol + statistic battery

- [ ] 1.1 Define the parameter grid (`delta` × `p.DMP`, incl. `delta=0` anchor and ≥2 non-zero) and replicate counts `N` (InterSIM) / `M` (numpy)
- [ ] 1.2 Implement the per-omic statistic battery (marginal moments/quantiles, cluster separation η², differential-feature rates, covariance relative-Frobenius) as a reusable function consuming a generated triple
- [ ] 1.3 Implement the numpy side from the existing `generate_omics` + `bernoulli_indicators` + `derive_coupled_indicators` (validate the real generator, not a parallel one)

## 2. InterSIM fixtures (R, one-time) + provenance

- [ ] 2.1 Add an R script that runs InterSIM `N` times per grid cell and writes the per-cell statistic distributions
- [ ] 2.2 Commit the InterSIM summary fixtures (`.npz`/CSV) with provenance (InterSIM version, date, seeds, grid)
- [ ] 2.3 Add a loader that reads the fixtures without R and errors clearly when absent

## 3. Replicate-distribution comparison + CI test

- [ ] 3.1 Implement the comparison: per cell/statistic, test whether numpy's value falls within InterSIM's documented central interval
- [ ] 3.2 Add an R-free CI test asserting fidelity across the grid (fast subset + `slow`-marked full grid); document the interval/tolerance choice
- [ ] 3.3 Confirm effect injection and DMP→DEG→DEP coupling are exercised at `delta>0` (the gap the `delta=0` test left)

## 4. Paper supplement

- [ ] 4.1 Add a script that renders a supplementary table + figure (numpy-vs-InterSIM fidelity across the grid) from the committed fixtures
- [ ] 4.2 Document how to reproduce the fixtures and the supplement (R step + Python step)

## 5. Wrap-up

- [ ] 5.1 ruff + mypy + fast pytest green with no R on PATH
- [ ] 5.2 Update `CLAUDE.md` / generator docs to point at the fidelity validation and supplement
