## Why

The numpy omics generator is a from-scratch reimplementation of InterSIM's
generative model that lets MOTCO run large semi-synthetic studies without R. Its
fidelity to InterSIM is currently guarded only by a single `δ=0` fixture test —
which, because `δ=0` zeroes the cross-omic blend and injects no effect, never
validates the effect injection (`μ = base + δ·v`), the CpG→gene→protein
coupling, the covariance structure, or any non-degenerate regime. A one-shot
`δ=2` structural comparison showed strong agreement, but it was a single random
draw at one parameter point. For the reimplementation to stand as a
paper-supportable claim ("a numpy reimplementation that faithfully reproduces
InterSIM, enabling R-free large-scale studies"), the validation needs to be
rigorous, swept across parameters, replicate-based, and reproducible.

## What Changes

- Add a **paper-grade fidelity validation** comparing the numpy generator to
  InterSIM across a **sweep** of `delta` and `p.DMP` (not a single point), on:
  marginal moments/quantiles, cluster separation (η²), differential-feature
  rates (the DMP→DEG→DEP coupling), covariance structure, and cross-omic
  coupling.
- Use a **replicate-distribution comparison**: run InterSIM and the numpy
  generator each N times per cell, and test whether each numpy summary statistic
  falls within InterSIM's own sampling distribution (e.g. a central interval),
  so the check accounts for InterSIM's RNG variability rather than comparing
  single draws.
- Commit **InterSIM summary fixtures** (the per-cell statistic distributions) so
  the validation runs **R-free in CI**, alongside the reproducible R generation
  script and provenance (InterSIM version, date, seeds, parameter grid).
- Produce a **reproducible supplementary artifact** (table + figure) suitable for
  a paper's supplementary material, generated from the committed fixtures.
- Strengthen the existing realism requirement from the `δ=0` fixture to this
  swept, replicate-based validation.

## Capabilities

### New Capabilities
<!-- none: the work strengthens an existing capability and adds artifacts/tooling -->

### Modified Capabilities

- `numpy-omics-generator`: the "Generator realism is validated against InterSIM"
  requirement is upgraded from a single `δ=0` per-feature mean/variance fixture
  to a parameter-swept, replicate-distribution fidelity validation (marginals,
  cluster separation, differential-feature rates, covariance, cross-omic
  coupling) with committed R-free fixtures and a reproducible paper supplement.

## Impact

- New: a fidelity-validation module/script under `src/motco/simulations/` (or
  `scripts/`) computing the per-cell summary statistics and the
  numpy-within-InterSIM-distribution checks.
- New: committed InterSIM summary fixtures (`.npz`/CSV) + the R generation script
  + provenance; a supplementary table/figure generator.
- New/updated tests: an R-free CI test asserting numpy statistics fall within the
  committed InterSIM intervals across the parameter grid (slow-marked as needed).
- R is used only to (re)generate the fixtures, never in CI — consistent with the
  generator's no-R-at-runtime contract.
- No change to the generator's behavior; this is validation + evidence only.
