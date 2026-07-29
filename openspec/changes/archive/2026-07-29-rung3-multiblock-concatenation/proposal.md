## Why

Rungs 0–2 have progressively isolated and cleared individual factors. Rung 0 proved the linear floor is clean under a single-block PCA projector. Rung 1 showed the methylation `rev.logit` nonlinearity is fully inverted by M-value integration and does not introduce cross-talk. Rung 2 showed that the per-feature **standardization** step inside the production `concat` projector is also clean — it preserves orientation and introduces no magnitude→orientation cross-talk on a single block. The specificity study's cross-talk therefore does not originate in any of these single-block factors.

Rung 2's gate decision names **heterogeneous multi-omic concatenation** as the next single untested factor: the real `concat` pipeline z-scores each omic block *independently*, then concatenates them into a joint feature matrix before PCA. When the blocks differ in dimensionality or correlation structure, unequal block weight in the pooled covariance can tilt the top principal components away from the geometry-carrying block's axes, effectively rotating the combined representation and rotating injected magnitude into apparent orientation. Rung 3 adds **exactly that one factor** — multiple heterogeneous blocks — and nothing else, measuring whether block-size or block-correlation imbalance manufactures cross-talk on an otherwise clean linear problem.

## What Changes

- Add a Rung-3 **multi-block concatenation test bed** that reuses the Rung-0/1/2 geometry injection unchanged: a known 2-stage `none`/`magnitude`/`orientation` trajectory is injected into a **single anchor block** (the methylation block, in M-value space per Rung 1's gate, so no `rev.logit` distortion), then **one or two additional blocks** are drawn as independent Gaussian noise with a separately configurable dimensionality and noise scale. All blocks are per-block z-scored and concatenated before PCA — exactly the production `concat` transform — and the existing `stats/trajectory.py` estimators measure `delta`/`angle` on the joint representation. The estimator path (`get_model_matrix`/`build_ls_means`/`estimate_difference`, two-group contrast) is identical to Rungs 0–2.
- **Block configuration is the swept independent variable.** The test bed parameterises:
  - **Number of blocks** (1 / 2 / 3): the anchor block alone is the Rung-2 baseline; adding blocks introduces imbalance.
  - **Nuisance-block dimensionality** relative to the anchor (`p_nuisance / p_anchor`): sweeping from equal (1×) to severely overrepresented (10×) covers the realistic methylation-vs-expression imbalance in the production generator (450 k CpG sites vs ~20 k genes).
  - **Nuisance-block noise correlation** (`ρ_nuisance`): exchangeable-correlation structure (all pairs sharing a common ρ) covering uncorrelated (ρ = 0), mildly correlated (ρ = 0.3), and strongly correlated (ρ = 0.7) nuisance blocks; cross-block independence is maintained (blocks are drawn independently; no cross-omic coupling, deferred to Rung 4).
- **Stage = 2 only**, so Procrustes `shape` is degenerate and excluded — scope stays magnitude and orientation, identical to Rungs 0–2.
- **Cross-talk characterisation:** for each block configuration, quantify against the per-configuration `none` null and against the single-block PCA floor (a) absolute distortion of measured `delta`/`angle` vs the intended anchor-block geometry (magnitude target `signal_scale·(c−1)`, orientation target `θ`) and (b) cross-talk magnitude→`angle` and orientation→`delta`. Sweep over **nuisance-block dimensionality ratio** and **anchor effect size** to locate any onset and determine whether it scales with block imbalance.
- **Block-weight decomposition (secondary probe):** compute the fraction of total explained variance (in the joint PCA) attributable to the anchor block vs the nuisance block(s) as a function of the dimensionality ratio. If cross-talk tracks the variance-fraction tipping point, this is direct evidence that block-weight dominance, not some other property, is the mechanism.
- Add a committed **findings writeup** with the per-configuration distortion/cross-talk table, the block-weight decomposition, and the **gate decision for Rung 4**: if heterogeneous concatenation produces cross-talk, determine whether it explains the specificity-study result; if not, identify the next single untested factor (cross-omic coupling in the generator).
- **Out of scope (deferred to later proposals):** cross-omic coupling (correlated signal between blocks via the InterSIM incidence maps), real per-block dimensionalities from the generator reference data, the full `evaluation.py` end-to-end harness, SNF multi-block fusion (a separate probe given the single-block SNF verdict from Rung 2), PLS latent-space integration over multi-block inputs, and any power study or RRPP rejection rates.

## Capabilities

### New Capabilities
- `multiblock-geometry-recovery`: A Rung-3 test bed that injects the known two-stage trajectory geometry into a single anchor block (M-value space, no `rev.logit`), appends one or two independent nuisance blocks of configurable dimensionality and correlation structure, applies the production per-block z-score + concatenation + PCA transform, and measures how trajectory `delta`/`angle` recovery and magnitude↔orientation cross-talk depend on block-size imbalance and nuisance-block correlation.

### Modified Capabilities
<!-- None: this is a self-contained new test bed; it does not change requirements of the evaluation harness, generator, or trajectory estimators. -->

## Impact

- **New code:** a Rung-3 test-bed module `src/motco/simulations/multiblock_recovery.py`, unit tests `tests/test_multiblock_recovery.py`, and a driver script `scripts/multiblock_recovery_probe.py`.
- **New docs:** a findings writeup in the change folder (matching the Rung-0/1/2 pattern).
- **Reused, unchanged:** the geometry-injection helpers from `simulations/linear_recovery.py`, the `stats/trajectory.py` estimators, `stats/design.py` design builders, and the Rung-2 single-block baseline (serves as the 1-block reference arm).
- **Dependencies:** `numpy`, `scikit-learn` (PCA, StandardScaler), `pandas` — all already present.
- **No changes** to `evaluation.py`, `semisynthetic.py`, `generator.py`, `projector_recovery.py`, `methylation_recovery.py`, `linear_recovery.py`, `pls.py`, `snf.py`, or any existing spec.