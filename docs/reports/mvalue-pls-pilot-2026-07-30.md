# M-value/PLS trajectory specificity pilot

**Run date:** 2026-07-29 to 2026-07-30  
**Code revision:** `7cffff1` (`main`)  
**Configuration:** `examples/trajectory_power_study/pilot_50x199.json`

## Purpose

This pilot tested the production PLS trajectory pipeline after correcting methylation integration to convert B values back to M values before standardisation and latent-space construction. The primary question was whether the previously observed magnitude-to-orientation cross-talk remained after that correction.

This is a medium-sized diagnostic run, not the paper-grade study. It used 50 replicates per cell, so a rejection rate near 0.05 has a Monte Carlo standard error of approximately 0.03.

## Run configuration

| Parameter | Value |
|---|---:|
| Integration | PLS, using M-value methylation |
| Samples per replicate | 300 |
| Stages | 4 |
| Replicates per cell | 50 |
| Permutations per test | 199 |
| Effect sizes | 0.00, 0.25, 0.50, 0.75, 1.00 |
| Trajectory modes | magnitude, orientation, shape, translation |
| Significance level | 0.05 |
| Total evaluations | 1,100 |
| Failed evaluations | 0 |

## Primary result

The M-value correction removed the targeted magnitude cross-talk. At the largest magnitude effect, the intended delta test had complete power, while the angle and shape rejection rates remained below the nominal 0.05 level.

| Injected mode | Delta | Angle | Shape |
|---|---:|---:|---:|
| Magnitude | **1.00** | 0.02 | 0.02 |
| Orientation | 0.72 | **0.76** | 1.00 |
| Shape | 1.00 | 0.84 | **1.00** |
| Translation | 0.02 | 0.04 | 0.08 |

Values are rejection rates at effect size 1.00, with 50 replicates per entry. Bold entries are the statistic intended to detect that mode. Translation is a negative control.

Magnitude specificity was stable across the entire effect-size sweep:

| Effect size | Delta | Angle | Shape |
|---:|---:|---:|---:|
| 0.00 | 0.04 | 0.02 | 0.08 |
| 0.25 | 1.00 | 0.02 | 0.02 |
| 0.50 | 1.00 | 0.00 | 0.02 |
| 0.75 | 1.00 | 0.00 | 0.00 |
| 1.00 | 1.00 | 0.02 | 0.02 |

## Type I error and controls

The ordinary null cell was consistent with the nominal 0.05 level:

| Baseline | Delta | Angle | Shape | Combined rule |
|---|---:|---:|---:|---:|
| No difference | 0.04 | 0.02 | 0.06 | 0.10 |
| Translation control | 0.06 | 0.02 | 0.10 | 0.14 |

All preregistered per-statistic Type I checks passed their two-Monte-Carlo-standard-error tolerance. The combined rule is descriptive here and was not among those individual Type I acceptance checks.

## Acceptance results

- Passed: all six Type I checks for the ordinary and translation baselines.
- Passed: magnitude/delta power and monotonicity; top-effect power was 1.00.
- Passed: shape/shape power and monotonicity; top-effect power was 1.00.
- Failed: orientation/angle power target and monotonicity; rates across increasing effects were 0.00, 0.60, 0.66, 0.56, and 0.76.
- Passed: all translation specificity checks and both magnitude off-diagonal checks.
- Failed: orientation-to-delta, orientation-to-shape, shape-to-delta, and shape-to-angle specificity checks.

## Interpretation

The pilot supports the rung-ladder conclusion that magnitude-to-angle leakage was caused by input representation rather than the trajectory estimator. Converting methylation to M-value space before PLS integration restores the expected separation: magnitude strongly affects delta without elevating angle or shape rejection.

The study does **not** establish general specificity of all three statistics. Orientation and shape injections still move multiple statistics. Those failures may reflect the definitions of the feature-space manipulations rather than a defect in the estimators: a global feature permutation can alter Procrustes shape as well as orientation, and relocating stage-specific feature support can change step lengths and angles as well as shape. This requires a separate geometry audit before interpreting those cells as estimator cross-talk.

The orientation curve is also non-monotone and misses the preregistered 0.80 top-effect power threshold (observed 0.76). With only 50 replicates, the standard error at 0.76 is approximately 0.06, but the strong off-diagonal rejection rates are too large to attribute to Monte Carlo noise.

## Decision and next work

The targeted magnitude finding is strong enough to retain M-value conversion as the production contract. Before launching the 500-replicate paper-grade grid, audit orientation and shape injections in the pre-integration generative frame and verify that each is a pure change in its intended geometric quantity. Then repeat this pilot with corrected manipulations. Running the full study now would make the remaining cross-talk estimates more precise without resolving their interpretation.

## Reproduction

From the repository root, run four resumable shards:

```bash
for i in 0 1 2 3; do
  uv run python scripts/run_study_shard.py \
    --config examples/trajectory_power_study/pilot_50x199.json \
    --out-dir results/mvalue-pls-pilot-50x199 \
    --shard-index "$i" --n-shards 4 --n-jobs 1 \
    --error-policy record
done
```

Then merge and report:

```bash
uv run python scripts/motco_study.py merge \
  --out-dir results/mvalue-pls-pilot-50x199

uv run python scripts/motco_study.py report \
  --config examples/trajectory_power_study/pilot_50x199.json \
  --out-dir results/mvalue-pls-pilot-50x199
```
