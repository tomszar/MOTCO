## Context

The semi-synthetic generator (`simulations/semisynthetic.py`) injects a group trajectory difference whose geometry is set by `trajectory_mode` (per-stage coefficient profile × per-feature sign pattern): `none` (null), `translation` (constant offset), `magnitude` (size/`delta`), `orientation` (`angle`), `shape` (localized bend). The trajectory test (`stats/trajectory.py`) estimates exactly these axes (`delta`/`angle`/`shape`), but nothing renders the modes visually.

The viz layer (`viz.py`) already separates a core renderer (`plot_trajectories`, projector-agnostic) from a PCA-fitting convenience wrapper (`plot_trajectory_from_data`). The PLS layer (`stats/pls.py`) exposes `fit_plsda_transform`, which fits a `PLSRegression` but returns only `x_scores_`. Integration of the three omics into one outcome matrix is provided by `integrate_semisynthetic_dataset` (concat / SNF).

## Goals / Non-Goals

**Goals:**
- Make each `trajectory_mode` visually concrete in one comparison figure.
- Reuse the existing renderer and integration code; add the smallest new surface needed.
- Keep the demo runnable headless and testable without R.

**Non-Goals:**
- Not part of, and not coupled to, the power study (`simulations/study/`).
- No statistical inference (no RRPP, no p-values) — this is illustrative geometry only.
- No new generator or new integration method; InterSIM's fixed feature dimensions are accepted as-is.

## Decisions

- **`fit_plsda_model` returns the fitted projector.** `plot_trajectories` needs an object with `.transform()` to project LS-mean vectors; `fit_plsda_transform` returns only scores. Rather than re-fit inside viz, factor out `fit_plsda_model(X, y, n_components) -> PLSRegression` and make `fit_plsda_transform` a thin wrapper (`fit_plsda_model(...).x_scores_`). Keeps a single fitting code path.
  - *Alternative considered:* fit `PLSRegression` ad hoc inside `viz.py`. Rejected — duplicates the one-hot encoding/scaling contract already owned by `pls.py`.

- **PLS-DA on stage, 2 components, no cross-validation.** The response is the stage/level factor (one-hot), so latent axes separate stages (the trajectory ordering) rather than maximizing total variance. Two components is exactly the plotting plane; skipping CV is the deliberate speed shortcut appropriate for single illustrative examples.

- **Per-scenario projector.** Each panel fits its own PLS on that scenario's data. This mirrors how the real pipeline runs each dataset and maximizes per-panel stage separation. Consequence: axes differ between panels — panels compare *shape/direction*, not absolute position. Implemented via `component_label="PLS"` so axes read `PLS1/PLS2`.
  - *Alternative considered:* one shared projector fit on the null and reused everywhere (identical axes). Rejected as the default because off-baseline scenarios project less cleanly; left as an easy future option.

- **Shared InterSIM baseline + shared seed across modes.** `generate_showcase_datasets` runs InterSIM once and injects every mode with the same seed, so stage-as-cluster and within-stage group assignment are identical across panels; only the injected effect differs. `none` always gets `group_effect_size=0` regardless of the requested effect size.

- **Module + thin script packaging.** Orchestration lives in importable/testable `simulations/showcase.py`; `scripts/trajectory_showcase.py` is a thin CLI that saves a figure. Mirrors the existing `study/` + `scripts/` pattern rather than adding a 5th production `motco` subcommand for a demo.

## Risks / Trade-offs

- **InterSIM feature count is fixed (~650 total) and not parameterizable** → documented as a known constraint; `n_sample` and stage count remain controllable, which is sufficient for illustration. If thousands of features are ever required, a separate generator or feature-padding change would be needed.
- **Per-scenario axes are not comparable in absolute terms** → panel titles name the geometric axis each mode exercises; documentation instructs comparing shape/direction, not position.
- **Runtime needs R + InterSIM** → tests stub `InterSIMResult` so CI exercises the generation/integration/projection/plotting path without R; the R-dependent end-to-end path is the same one the InterSIM bridge tests skip when R is absent.

## Migration Plan

Additive only — new functions and files, no breaking changes to existing signatures (`fit_plsda_transform` keeps its return type; `plot_trajectories` gains a defaulted `component_label`). No rollback concerns.
