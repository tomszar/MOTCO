## Context

MOTCO estimates trajectory geometry (size, orientation, Procrustes shape) from LS-mean vectors in a high-dimensional latent space. All output is currently numeric — symmetric matrices of deltas, angles, and shape distances, plus scalar pair statistics. There is no way to see what the trajectories actually look like.

The typical use case is 2 groups × 2–3 stages. The outcome matrix `Y` comes from either a concatenated omics integration or SNF spectral embedding, and can have many features. The LS-mean vectors (one per group × stage cell) are computed by `get_observed_vectors()` and live in that same feature space.

## Goals / Non-Goals

**Goals:**
- Project LS-mean trajectory paths into 2D via PCA and plot them as connected, directed paths
- Optional sample scatter overlay behind the trajectories (opt-in, default off)
- Two-layer API: a core plotting function and a convenience wrapper that handles PCA fitting
- The wrapper returns the fitted PCA so callers can reuse the same projection
- Axis labels include PCA explained variance percentages

**Non-Goals:**
- 3D projections
- Interactive / plotly figures
- Overlaying derived statistics (angle, delta, shape annotations) — deferred
- CLI subcommand for visualization
- Support for more than 2D projection in this iteration

## Decisions

### PCA fitted on the full outcome matrix Y, not on LS-mean vectors alone

Fitting PCA on all samples gives a projection that reflects the true variance structure of the data. LS-mean vectors are then projected into that same space, so the trajectories are shown in context of the actual data cloud. Fitting PCA on only 4–6 LS-mean points would give a trajectory-maximizing projection but hide where the data actually lives, which is less useful for exploration.

**Alternative considered**: Offer both modes via a flag. Rejected — adds surface area without clear benefit at this stage.

### Two-layer API with projector returned from wrapper

```
plot_trajectories(observed_vectors, projector, ...)  →  (fig, ax)
plot_trajectory_from_data(Y, metadata, ...)          →  (fig, ax, pca)
```

The core function is pure visualization: it projects the pre-computed LS-mean vectors through the supplied projector and draws the plot. The wrapper fits PCA on `Y`, calls the core, and returns the fitted PCA so the caller can reuse the same coordinate system (e.g., projecting additional samples, comparing two runs side-by-side without axis drift).

**Alternative considered**: Single-function API that always fits PCA internally. Rejected — refitting on every aesthetic tweak is wasteful and prevents consistent axes across figures.

### Arrow direction at segment midpoints for stage ordering

Each connecting segment between consecutive LS-mean points gets a directional arrow placed at the midpoint, pointing toward the later stage. This encodes temporal order without adding a separate legend entry for stages. The first point (stage_0) is annotated with the stage label to provide an anchor.

**Alternative considered**: Marker shape (○ △ □) per stage. Rejected — doubles the legend (color for group, shape for stage) and becomes ambiguous with 3+ stages.

**Alternative considered**: Marker size gradient. Rejected — small early-stage markers can disappear in scatter, especially with opt-in sample overlay.

### `src/motco/viz.py` as a standalone module

Visualization is a separate concern from statistics. Keeping it in its own top-level module avoids coupling it to any specific stats submodule and makes it easy to find and extend.

### `matplotlib` as explicit dependency

`matplotlib` is added to `pyproject.toml` as a runtime dependency. It is the only new dependency; `scikit-learn` (for PCA) is already present.

## Risks / Trade-offs

- **Low explained variance on PC1+PC2** — PCA on high-dimensional omics-derived data may capture little variance in 2 components, making the projection uninformative. Mitigation: always show explained variance percentages on axis labels so the caller can judge how much to trust the 2D view.

- **Arrow invisibility on short segments** — if two consecutive LS-mean points are very close in PC space, the midpoint arrow can be too small to see. Mitigation: use a fixed arrow head size relative to the axis scale rather than relative to the segment length.

- **Core function requires pre-fitted projector** — callers who want a quick one-liner must use the wrapper. This is intentional but worth documenting clearly.
