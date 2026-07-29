## Why

Rungs 0–3 cleared every single-block and multi-block factor on the `concat` path. Rung 3 confirmed that per-block z-score + concatenate + PCA is clean under independent nuisance blocks at any dimensionality ratio up to 10×: the `magnitude` manipulation angle stays at the `none` null floor, `w_anchor` tracks the naive feature-fraction prediction, and the null floor rise with `dim_ratio` is a uniform k-noise dilution effect, not a directed rotation. The specificity-study cross-talk is therefore not explained by any structural property of the concatenation transform in isolation.

The one remaining untested factor is **cross-omic coupling**: in the production InterSIM generator (`generator.py`), methylation differential indicators drive expression indicators via the `incidence_cpg_gene` map (a gene is differential when any of its mapped CpGs is), and expression indicators drive protein indicators via `incidence_gene_protein`. This introduces *structured, direction-preserving signal* in the nuisance blocks that is derived from the anchor block's differential support. Crucially, the coupling is a linear map from CpG feature space to gene feature space — not from the anchor step vector to the nuisance step vector as an isometry, but as a general linear transformation whose operator structure can distort the effective inner-product geometry in the joint space.

The distortion mechanism is analytic: given anchor steps `a` (group A) and `b` (group B), and a coupling matrix M mapping anchor features to nuisance features, the joint steps are `[a; M@a]` and `[b; M@b]`. The cosine of the measured joint-space angle between them is:

```
cos(θ_joint) = (a·b + a·MᵀM b) / (‖[a; M@a]‖ · ‖[b; M@b]‖)
             = a · (I + MᵀM) b / (‖[a; M@a]‖²)    [if ‖a‖ = ‖b‖]
```

This equals the true angle `cos(θ_anchor) = a·b / ‖a‖²` only when `MᵀM = λI`, i.e. when M preserves the inner product up to scale. In general, M has unequal singular values — the coupling matrix stretches or shrinks different directions in anchor feature space by different amounts — and the joint-space angle is distorted accordingly. Rung 4 adds exactly this factor: a synthetic incidence matrix M coupling the anchor block's step to the nuisance block, and measures whether the resulting joint-space geometry distortion produces the magnitude→orientation cross-talk the specificity study found.

## What Changes

- Add a Rung-4 **cross-omic coupling test bed** that extends the Rung-3 multi-block setup by replacing the independent nuisance block with a **coupled nuisance block**: the nuisance block's per-group mean shift is `coupling_scale × M_norm @ anchor_step_g`, where M_norm is a column-normalised synthetic incidence matrix and `coupling_scale ∈ [0, 1]` interpolates between Rung-3 pure noise (`coupling_scale = 0`) and full coupling (`coupling_scale = 1`). The anchor block, per-block z-score + concatenate + PCA transform, and estimator path are identical to Rung 3.
- **Coupling matrix M is the single new structural variable.** M ∈ {0,1}^{p_nuis × p_anchor} is a random sparse binary matrix: each nuisance feature independently links to `nnz` anchor features drawn uniformly at random (i.e., expected non-zeros per column of M equal `nnz × p_nuis / p_anchor`). This mirrors the InterSIM incidence map structure (genes linked to a small number of CpGs). Three structural regimes are swept:
  - **Random sparse** (`nnz_per_nuis_feature` ≈ 3–5): the production-realistic case.
  - **Dense** (all anchor features couple to all nuisance features): the maximum-distortion upper bound; `MᵀM` is far from `λI`.
  - **Rank-1** (all nuisance features link to the same single anchor feature): the minimum-distortion case; `MᵀM` is a rank-1 matrix whose top eigenvector aligns with a single canonical direction, maximally misaligned with a random anchor step.
- **Coupling strength `coupling_scale`** ∈ {0, 0.25, 0.5, 0.75, 1.0} is the primary swept variable. `coupling_scale = 0` is the Rung-3 independent-nuisance floor. The headline run uses the random-sparse M structure and `dim_ratio = 1` (equal-size blocks).
- **Analytic geometry prediction (secondary probe):** For each (M, `coupling_scale`, seed) triple, compute the expected joint-space angle distortion from the formula above (`cos(θ_joint) = a·(I + γ·MᵀM)b / (...)`, where `γ = coupling_scale²`) and compare it to the measured angle. Agreement between the closed-form prediction and the empirical result isolates the coupling geometry as the mechanism and distinguishes it from any additional noise or PCA discretisation effects.
- **Magnitude cross-talk control:** For the `magnitude` manipulation, the nuisance step scales as `c × M_norm @ a` — same direction as group A, scaled by `c`. The joint-space direction of group B relative to group A is unchanged by the coupling (the coupling is direction-preserving for uniform scaling), so **no magnitude→angle cross-talk is predicted** regardless of M structure or coupling strength. This serves as a within-run positive control that the coupling mechanism specifically targets orientation, not magnitude.
- **Dim_ratio secondary sweep:** Repeat the coupling strength sweep at `dim_ratio ∈ {0.5, 1, 5}` to check whether the orientation distortion scales with the relative weight of the nuisance block in the joint PCA, consistent with the `(I + MᵀM)` formula.
- Add a committed **findings writeup** with the coupling-strength distortion curve, the analytic-vs-empirical angle comparison, and the **gate decision for Rung 5**: if coupling explains the specificity-study cross-talk magnitude, the rung ladder is complete; if it produces distortion but not at the observed magnitude, Rung 5 (full generator fidelity with real dimensionalities and covariance) is next.
- **Out of scope (deferred):** the real InterSIM incidence matrices (real dimensionalities; would require loading the reference cache and operating at CpG/gene scale — valid as a stretch goal); multiple coupled blocks (methylation→expression→protein cascade); the full `evaluation.py` end-to-end harness with RRPP rejection rates; any power study.

## Capabilities

### New Capabilities
- `coupling-geometry-recovery`: A Rung-4 test bed that adds a linear coupling from the anchor block's step vector to the nuisance block via a synthetic sparse incidence matrix, applies per-block z-score + concatenate + PCA (the production `concat` transform), and measures how trajectory `delta`/`angle` recovery and magnitude↔orientation cross-talk depend on coupling strength and incidence matrix structure. Includes a closed-form joint-space angle prediction as a mechanistic check.

### Modified Capabilities
<!-- None: this is a self-contained new test bed; it does not change requirements of the evaluation harness, generator, or trajectory estimators. -->

## Impact

- **New code:** `src/motco/simulations/coupling_recovery.py`, unit tests `tests/test_coupling_recovery.py`, driver script `scripts/coupling_recovery_probe.py`.
- **New docs:** a findings writeup in the change folder (matching the Rung-0/1/2/3 pattern).
- **Reused, unchanged:** the geometry-injection helpers from `simulations/linear_recovery.py`, the multi-block generation infrastructure from `simulations/multiblock_recovery.py` (anchor block generation and `build_joint_matrix` are reused directly), the `stats/trajectory.py` estimators, and `stats/design.py` design builders.
- **Dependencies:** `numpy`, `scikit-learn` (PCA), `scipy.sparse` (for efficient sparse incidence matrix construction) — all already present.
- **No changes** to `evaluation.py`, `semisynthetic.py`, `generator.py`, `multiblock_recovery.py`, `projector_recovery.py`, `methylation_recovery.py`, `linear_recovery.py`, or any existing spec.