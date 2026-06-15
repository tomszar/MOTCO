## Why

MOTCO's intended trajectory architecture is **features → latent (molecular) space → trajectory measurement**: the integration step constructs the low-dimensional molecular space, and trajectory geometry (`delta`/`angle`/`shape`) is estimated **within** that space, which is the measurement substrate — not a visualization artifact.

Two integration methods were designed to build that latent space: **SNF** (graph-spectral embedding) and **PLS** (the transform of X into the subspace that maximizes covariance with the label). Only SNF was implemented. PLS was explicitly **deferred** in the original harness design (`2026-05-06-add-simulation-evaluation-harness/design.md`): *"PLS-DA is supervised by labels and may leak group/stage signal into the latent space; it should be considered later with a clear design."* In its place, the harness shipped `concat` — described in that same proposal as *"a simple concatenated feature matrix **baseline**"* — which does **not** construct a latent space at all: it standardizes and column-binds the omic blocks, so trajectories are measured in (rescaled) **feature space**.

The deferred design is now realizable. The Rung-2 projector study (`rung2-projector-crosstalk/findings.md`) supplies the "clear design" PLS was waiting for: a **stage-conditioned** PLS projection has a clean, stable null for group differences (orientation null 3.2° ± 2.1), whereas the **group-conditioned** variant is the one that leaks into orientation. Conditioning on the stage label is therefore both safe and correct — the latent space is built from the trajectory axis shared by both groups, and the group A-vs-B difference is then measured within it.

## What Changes

- Add **`pls`** as a third integration method in the simulation evaluation harness, implementing the production PLS latent space: standardize-and-concatenate the omic blocks, fit PLS-DA conditioned on the **stage** label, and return the X-score matrix as the molecular latent space in which `estimate_difference`/`RRPP` operate.
- **Latent dimensionality is selected by the existing double nested cross-validation** (`stats/pls.plsda_doubleCV`), not fixed by hand. The selected number of latent variables is the modal `LV` across repeats (parsimony tie-break). Rationale: the latent space is the measurement substrate, so its dimensionality must be **stable and non-overfitted** — double-CV secures a molecular space that generalizes rather than one tuned to the training sample.
- **Reclassify `concat` explicitly as a baseline/diagnostic**, not a designed latent space, everywhere it is documented. `concat` remains available (it is the dependency-light reference path the rung ladder measures against) but is no longer presented as a production molecular space.
- **Record the latent-space architecture in three high-visibility places** so the concat-vs-latent confusion does not recur: `CLAUDE.md` (always-loaded project instructions), the `simulation-evaluation-harness` spec, and the `evaluation.py` module/function docstrings. Each states: integration constructs the molecular latent space; SNF and PLS are the production latent methods; `concat` is a baseline; the latent space is the measurement substrate; and the viz down-projection (`plot_trajectory_from_*`) is display-only and distinct from measurement.

## Capabilities

### New Capabilities
<!-- None: this extends an existing capability rather than adding a new one. -->

### Modified Capabilities
- `simulation-evaluation-harness`: add the `pls` integration method (double-CV-selected, stage-conditioned PLS latent space) and clarify that integration constructs the molecular latent space — the measurement substrate — of which SNF and PLS are the production methods and `concat` is a baseline.

## Impact

- **Modified code:** `src/motco/simulations/evaluation.py` — new `_pls_integration`, dispatch in `integrate_semisynthetic_dataset`, `pls` added to `IntegrationMethod` and the `_validate_evaluation_params` allow-set, module/function docstrings.
- **Reused, unchanged:** `stats/pls.plsda_doubleCV`, `fit_plsda_transform`, `fit_plsda_model` (consumed, not modified); `stats/trajectory`, `stats/permutation` estimator path.
- **New docs:** latent-space architecture block in `CLAUDE.md`; PLS scenario + concat-baseline clarification in the harness spec.
- **New tests:** `tests/` coverage for `_pls_integration` (shape/columns/determinism, multi-stage conditioning, selected-LV metadata, validation error path).
- **Dependencies:** none new (`scikit-learn` PLS already present).
- **Performance note:** double-CV per replicate is expensive; the harness exposes the CV knobs (`n_repeats`, `cv1_splits`, `cv2_splits`, `max_components`, `n_jobs`) via `integration_params` so grid runs can trade rigor for speed. Default `progress=False`.
- **Out of scope:** changing `concat` or `snf` behavior; re-running the specificity study or the rung ladder through the new PLS space (a follow-up); wiring PLS integration into the CLI or grid study config.
