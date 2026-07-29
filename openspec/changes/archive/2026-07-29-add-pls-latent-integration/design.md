# Design — PLS latent integration

## Context

The trajectory pipeline is **features → latent (molecular) space → trajectory measurement**. `integrate_semisynthetic_dataset` returns a `LatentIntegrationResult` whose `matrix` is the space `estimate_difference`/`RRPP` operate in. SNF already builds a genuine latent space (`get_affinity_matrix → SNF → get_spectral`). `concat` does not — it standardizes and column-binds, leaving the data in rescaled feature space. PLS, the second intended latent method, was deferred over a label-leakage concern that Rung 2 has now resolved.

## The PLS latent space

PLS-DA finds the subspace of X that maximizes covariance with a label Y. We use it as a **molecular-space constructor**:

1. **Build X** — standardize each omic block per feature and concatenate (same front end as `concat`). PLS also standardizes internally (`PLSRegression(scale=True)`), so this double-handles scale harmlessly; we keep the explicit standardize for parity and transparency.
2. **Label Y = the stage label** (`params.stage_col`), not group. This is the load-bearing decision; see below.
3. **Select latent dimensionality by double-CV** — run `plsda_doubleCV(X, y=stage, ...)` and take the modal `LV` across repeats (parsimony tie-break, via `_modal_int_with_parsimony`). This yields a *stable, non-overfitted* dimensionality: the space generalizes rather than fitting the training sample.
4. **Project** — `fit_plsda_transform(X, y=stage, n_components=selected_LV)` returns the X-scores; these are the latent coordinates (`pls_0..pls_{LV-1}`), wrapped in a `LatentIntegrationResult`.

## Key decisions

### Condition on stage, not group
Rung 2 settled this empirically. A **group**-conditioned PLS projection inflates and destabilizes the orientation null at low component counts (19.9° ± 40.6 at k=2) — it invents group-aligned direction where there is none. A **stage**-conditioned projection has a clean, stable null (3.2° ± 2.1). It is also methodologically sound: the latent space is built from the stage axis that *both* groups share, and the group A-vs-B trajectory difference is then estimated within that space — the null hypothesis (no group difference) is never used to construct the space. This is exactly the "clear design" the original deferral asked for.

### Dimensionality by double-CV, not a fixed k
A fixed `n_components` (e.g. 2, the viz default) compresses the geometry (Rung 2: 45° orientation → 11° at k=2). More importantly, because the latent space *is* the measurement substrate, a hand-fixed dimensionality risks an over- or under-fitted molecular space. The double nested CV selects the dimensionality that generalizes across folds (parsimony-biased), securing a stable space. The selection criterion optimizes class (stage) separation, which is acceptable: we are choosing *how many* stable molecular axes exist, then measuring trajectories within them — not tuning the trajectory statistics themselves.

### Multi-stage conditioning
Trajectories have ≥ 2 stages, so stage-conditioning is generally **multiclass** PLS-DA. `plsda_doubleCV` one-hot encodes Y and already supports `n_classes ≥ 2`, so multi-stage is handled by the existing machinery. Tests must cover the multi-stage case explicitly.

### Performance and configurability
`plsda_doubleCV` defaults (`n_repeats=30`, `cv2_splits=8`, `cv1_splits=7`) make it the most expensive integration path by far — prohibitive inside large power grids at defaults. The harness therefore exposes the CV knobs through `integration_params` (`n_repeats`, `cv1_splits`, `cv2_splits`, `max_components`, `random_state`, `n_jobs`), with `progress=False` forced. `max_components` is bounded to the feature count. Grid runs can dial these down; the default per-replicate call should use a modest, documented setting rather than the classifier-study defaults.

## Alternatives considered

- **Fixed configurable `n_components`** (mirror SNF's `spectral_components`): simpler and deterministic, but does not guarantee a non-overfitted space and reintroduces the k-compression of Rung 2. Rejected in favor of double-CV per the measurement-substrate argument.
- **Reuse a `plsda_doubleCV` `models[]` entry** instead of refitting: the per-repeat models are already full-data refits, but selecting the modal LV across repeats and refitting once via `fit_plsda_transform` is clearer and deterministic. Chosen.
- **Editing `concat` into a latent space** (standardize → PCA): rejected — it muddies the SNF/PLS design intent and re-baselines the rung ladder. `concat` stays a baseline.

## Risks

- **Cost**: double-CV per replicate. Mitigated by configurable CV knobs; documented as the trade-off.
- **Determinism**: `plsda_doubleCV` is seeded (`random_state`); the selected LV and the final transform are deterministic given fixed params. Tests assert reproducibility.
- **Small samples / degenerate folds**: stratified CV needs enough samples per stage. The harness should surface a clear error (reuse `SimulationEvaluationError`) when CV is infeasible rather than failing opaquely.
