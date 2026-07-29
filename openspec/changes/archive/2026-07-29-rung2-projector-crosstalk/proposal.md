## Why

Rung 0 (`rung0-gaussian-existence-proof/findings.md`) proved that under a pure-linear **PCA** projector the production estimators isolate magnitude (`delta`) and orientation (`angle`) with no cross-talk. Rung 1 (`rung1-methylation-nonlinearity/findings.md`) showed the methylation `rev.logit` nonlinearity is fully invertible by M-value integration and is therefore **not** a standing cross-talk source for a correctly-built pipeline — and, because `logit` acts coordinate-wise, it mooted the originally-planned heterogeneous-baseline rung. The specificity study's cross-talk (magnitude leaking into orientation) must therefore enter **downstream of the methylation representation**, in the integration/projection step itself. Rung 0 used a single, deliberately benign projector (mean-centered PCA, no standardization); the production trajectory pipeline does **not** use that projector — it standardizes-and-concatenates (`concat`) or builds an SNF spectral embedding. Rung 2 isolates **exactly one factor: the projector**, and asks whether swapping PCA for a standardizing, supervised, or graph-spectral projector manufactures cross-talk on an otherwise clean linear problem.

## What Changes

- Add a Rung-2 **projector test bed** that reuses the Rung-0/Rung-1 geometry injection unchanged (known 2-stage `none`/`magnitude`/`orientation` geometry in a clean feature space, methylation fixed in **M-value space** per Rung 1's gate decision — no `rev.logit` distortion), then measures `delta`/`angle` through a **selectable projector** instead of the single inline PCA. The estimator path (`get_model_matrix`/`build_ls_means`/`estimate_difference`, two-group contrast) is identical to Rung 0/1.
- **Projector is the swept independent variable.** Compare, on the same injected geometry:
  - **PCA** (mean-centered, inline) — the Rung-0/1 reference floor.
  - **Standardize** (per-feature z-score, then identity/PCA) — the transform inside the production `concat` integration (`evaluation.py:_concat_integration`); isolates the per-feature standardization that Rung 0/1 deliberately deferred.
  - **PLS-DA** latent space — the supervised production projector (`stats/pls.fit_plsda_transform`); a projector that *uses the group label*, the prime suspect for manufacturing group-aligned structure.
  - **SNF** spectral embedding — the graph-spectral production projector (`stats/snf`: `get_affinity_matrix` → `SNF` → `get_spectral`); a nonlinear, sample-similarity projector whose latent axes need not align with feature-space geometry.
- **Stage = 2 only**, so Procrustes `shape` is degenerate and excluded — scope stays magnitude and orientation, identical to Rung 0/1.
- **Cross-talk characterization per projector:** quantify, against the PCA floor and the per-projector `none` null, (a) absolute distortion of measured `delta`/`angle` vs the intended geometry (magnitude target `signal_scale·(c−1)`, orientation target `θ`) and (b) cross-talk (magnitude→`angle`, orientation→`delta`). Sweep over **effect size** and the projector's **latent dimensionality** (PCA/PLS components, SNF spectral components) to locate any onset.
- **Supervised-leakage probe (PLS-DA specific):** because PLS-DA conditions the projection on the group label, it can in principle align a *null* (`none`) trajectory with the group axis. The test bed reports the `none`-manipulation `delta`/`angle` under PLS-DA against the PCA floor to detect label-induced spurious geometry.
- Add a committed **findings writeup** with the per-projector distortion/cross-talk table and the **gate decision for Rung 3**: which projector(s), if any, are clean; whether the production `concat`/`snf` path leaks; and where the next rung should go (heterogeneous multi-omic concatenation, or the cross-omic coupling).
- **Out of scope (deferred to later proposals):** true **multi-omic** concatenation of heterogeneous-scale blocks (this rung is single-block, methylation-in-M-space, so "concatenation" reduces to standardization; heterogeneous concat is a candidate Rung 3), the cross-omic cascade and full InterSIM generator path, real per-CpG baselines, the `evaluation.py` end-to-end harness with RRPP rejection rates, and any purity metric or power study.

## Capabilities

### New Capabilities
- `projector-geometry-recovery`: A Rung-2 test bed that injects the known two-stage trajectory geometry into a clean linear (M-value) feature space and measures how trajectory `delta`/`angle` recovery and magnitude↔orientation cross-talk depend on the **projector** — comparing mean-centered PCA against per-feature standardization, supervised PLS-DA, and SNF spectral embedding — at two stages, against the PCA floor and a per-projector null.

### Modified Capabilities
<!-- None: this is a self-contained new test bed; it does not change requirements of the evaluation harness, generator, or trajectory estimators. -->

## Impact

- **New code:** a Rung-2 test-bed module under `src/motco/simulations/` (e.g. `projector_recovery.py`) plus unit tests under `tests/`, and a driver script under `scripts/`.
- **New docs:** a findings writeup in the change folder (matching the Rung-0/1 pattern).
- **Reused, unchanged:** the Rung-0/1 geometry-injection helpers (`simulations/linear_recovery.py`, `simulations/methylation_recovery.py`), the `stats/trajectory.py` estimators, `stats/pls.fit_plsda_transform`, and `stats/snf` (`get_affinity_matrix`/`SNF`/`get_spectral`).
- **Dependencies:** `numpy`, `scikit-learn`, `pandas` — all already present.
- **No changes** to `evaluation.py`, `semisynthetic.py`, `generator.py`, `pls.py`, `snf.py`, or any existing spec.
