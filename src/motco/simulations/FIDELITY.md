# Generator fidelity validation (numpy vs InterSIM)

The numpy generator (`generator.py`) is a from-scratch reimplementation of
InterSIM's `μ = base + δ·v` model. Its fidelity is validated with a **swept,
replicate-based, R-free** protocol (`fidelity.py`), the rigorous successor to the
single `δ=0` realism fixture.

## What is validated

Across a `delta × p.DMP` grid (default `delta ∈ {0, 1, 2}`, `p.DMP ∈ {0.2, 0.5}`),
both InterSIM and the numpy generator are run many times per cell, and a
per-omic **statistic battery** is computed on each draw:

- marginal moments + quantiles (`mean`, `sd`, `q10/q50/q90`),
- cluster separation `eta2` (between/total variance, averaged over features),
- `diff_rate` — the differential-feature rate (the DMP→DEG→DEP coupling signal),
- `cov_frob` — Frobenius norm of the empirical covariance.

**Criterion:** for each statistic, the numpy *replicate mean* must lie within
InterSIM's `[q2.5, q97.5]` percentile interval over its replicates. Averaging
over numpy replicates removes numpy's Monte-Carlo noise so the test isolates
*systematic* disagreement; the percentile interval absorbs InterSIM's RNG
variability (the two cannot match bit-for-bit across RNGs).

`delta=0` is the degenerate anchor (no separation, no coupling); `delta>0`
exercises effect injection and the cross-omic coupling that `delta=0` cannot —
see `tests/test_generator_fidelity.py`.

## Runtime (CI) — no R

The InterSIM side is committed as `tests/data/intersim_fidelity_fixture.npz`
(per-cell statistic distributions + provenance). CI loads it with
`load_fidelity_fixture` and never invokes R:

```bash
pytest tests/test_generator_fidelity.py -m "not slow"   # fast subset
pytest tests/test_generator_fidelity.py                 # full slow-marked grid
```

## Regenerating the InterSIM fixture (one-time, needs R + InterSIM)

The dev `flake.nix` ships R + `rPackages.InterSIM`:

```bash
# 1. R step — run InterSIM N times per cell, write per-cell stat distributions
nix develop --command Rscript src/motco/simulations/fidelity_intersim.R \
    --output-dir /tmp/fidelity_export
#    optional: --deltas 0,1,2 --p-dmps 0.2,0.5 --n-sample 500 --n-intersim 30 --seed 20260604

# 2. Python step — pack the CSV export into the committed .npz fixture
python -c "from motco.simulations.fidelity import build_fidelity_fixture_from_export as b; \
    b('/tmp/fidelity_export', 'tests/data/intersim_fidelity_fixture.npz')"
```

The R battery in `fidelity_intersim.R` mirrors `fidelity._omic_statistics`
exactly (population moments, type-7 quantiles, same `eta2`/`diff_rate`/`cov_frob`
definitions), so the two sides are comparable by construction. Keep the grid in
the R script in sync with `fidelity.default_grid()`.

## Paper supplement (quantitative)

`scripts/fidelity_supplement.py` renders a supplementary table (CSV + Markdown)
and figure (numpy points vs InterSIM intervals across the grid) from the
committed fixture — no R:

```bash
python scripts/fidelity_supplement.py --out-dir build/fidelity_supplement
```

## Visual supplement (qualitative)

`fidelity_visual.py` + `scripts/fidelity_visual.py` render side-by-side
InterSIM-vs-numpy figures: per-omic **density** plots (a few replicates each),
per-modality clustermap-style **heatmaps** (free hierarchical clustering of
samples and features with dendrograms + a cluster colour bar) and **PCA**
(rendered at 4 clusters, one panel per modality per tool), per-feature
**moment-agreement scatter** (mean & variance, ours vs InterSIM with a y=x
line), and a cross-omic **coupling** correlation block.

Unlike the quantitative fixture, the InterSIM **raw matrices are not committed**
— they are large (~2 MB) and only needed to render this supplement. Reproducing
the visual supplement therefore requires InterSIM, which is provided by the dev
`flake.nix` (`R` + `rPackages.InterSIM`). The numpy side is always generated
live.

The numpy generator handles arbitrary cluster counts natively — each indicator
column is one cluster (InterSIM's count is `length(cluster.sample.prop)`); the
default uses 4 clusters to show the block structure beyond the 3-cluster grid.

Reproduce the visual supplement (needs InterSIM via the flake):

```bash
# 1. R step — raw InterSIM data (density replicates + one 4-cluster replicate)
nix develop --command Rscript src/motco/simulations/fidelity_visual_intersim.R \
    --output-dir build/fidelity_visual/export
#    optional: --n-sample 300 --n-cluster 4 --delta 2 --p-dmp 0.2 --n-rep-density 3

# 2. Python step — pack into a local (gitignored) .npz fixture under build/
python -c "from motco.simulations.fidelity_visual import build_visual_fixture_from_export as b; \
    b('build/fidelity_visual/export', 'build/fidelity_visual/intersim_visual_fixture.npz')"

# 3. Render the figures from that fixture (no R)
python scripts/fidelity_visual.py --out-dir build/fidelity_visual
```

The exact dataset is reproducible: the R script seeds each InterSIM draw
deterministically from `--seed` (default `20260604`), and the numpy side is
seeded from the fixture provenance — so re-running the three steps regenerates
the same matrices and figures.