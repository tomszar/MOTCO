"""Qualitative side-by-side fidelity figures: InterSIM vs the numpy generator.

Companion to :mod:`motco.simulations.fidelity` (which provides the *quantitative*
within-interval validation). This module renders the visual supplement:

- **density** — per-omic marginal value distributions, a few replicates each;
- **heatmaps** — one per modality per tool, samples ordered by cluster, showing
  the cluster block structure (rendered at a non-trivial cluster count);
- **PCA** — first two PCs per modality per tool, coloured by cluster;
- **moment scatter** — per-feature mean and variance, ours vs InterSIM (y=x);
- **coupling** — cross-omic correlation block structure side-by-side (the
  visual counterpart of the DMP->DEG->DEP coupling).

Unlike the *quantitative* fidelity fixture, the InterSIM raw matrices are **not
committed** (they are large and only needed to render the supplement). Regenerate
them locally with InterSIM (available via ``flake.nix``) using
``fidelity_visual_intersim.R`` + :func:`build_visual_fixture_from_export`; the
numpy side is then generated live. See ``simulations/FIDELITY.md``. Entry point:
:func:`run_fidelity_visual`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from motco.simulations.fidelity import FidelityError
from motco.simulations.generator import (
    OMIC_LAYERS,
    bernoulli_indicators,
    derive_coupled_indicators,
    generate_omics,
)
from motco.simulations.reference import IntersimReference, load_reference

#: Filename of the committed InterSIM visual fixture (under ``tests/data``).
VISUAL_FIXTURE_NAME = "intersim_visual_fixture.npz"

#: Tool labels / colours used consistently across panels.
_INTERSIM = "InterSIM"
_NUMPY = "numpy (ours)"
_TOOL_COLOR = {_INTERSIM: "#1f77b4", _NUMPY: "#ff7f0e"}

#: Features per omic shown in the cross-omic coupling correlation block.
_COUPLING_FEATURES_PER_OMIC = 50


@dataclass(frozen=True)
class VisualData:
    """Raw per-tool data backing the qualitative figures.

    ``density[omic]`` is ``(n_rep, subsample)`` of marginal values; ``matrices``
    holds one full ``(n_sample, n_feat)`` replicate per omic; ``cluster_ids`` is
    the per-sample cluster label aligned to those matrices.
    """

    density: dict[str, np.ndarray]
    matrices: dict[str, np.ndarray]
    cluster_ids: np.ndarray


@dataclass(frozen=True)
class VisualFixture:
    """Committed InterSIM raw data + provenance for the visual supplement."""

    data: VisualData
    provenance: dict[str, str]

    @property
    def n_sample(self) -> int:
        return int(self.provenance["n_sample"])

    @property
    def n_cluster(self) -> int:
        return int(self.provenance["n_cluster"])

    @property
    def delta(self) -> float:
        return float(self.provenance["delta"])

    @property
    def p_dmp(self) -> float:
        return float(self.provenance["p_dmp"])

    @property
    def seed(self) -> int:
        return int(self.provenance["base_seed"])


# --------------------------------------------------------------------------- #
# Fixture I/O
# --------------------------------------------------------------------------- #


def load_visual_fixture(path: str | Path) -> VisualFixture:
    """Load the committed InterSIM visual fixture without invoking R."""

    fixture_path = Path(path)
    if not fixture_path.exists():
        raise FidelityError(
            f"InterSIM visual fixture not found at {fixture_path}. Regenerate it in "
            "R (one-time): `nix develop --command Rscript "
            "src/motco/simulations/fidelity_visual_intersim.R --output-dir <dir>` then "
            "`build_visual_fixture_from_export(<dir>, <fixture_path>)`."
        )
    with np.load(fixture_path, allow_pickle=False) as data:
        provenance = json.loads(str(data["provenance"]))
        density = {omic: data[f"density__{omic}"].astype(float) for omic in OMIC_LAYERS}
        matrices = {omic: data[f"matrix__{omic}"].astype(float) for omic in OMIC_LAYERS}
        cluster_ids = data["cluster_ids"].astype(int)
    return VisualFixture(
        data=VisualData(density=density, matrices=matrices, cluster_ids=cluster_ids),
        provenance=provenance,
    )


def build_visual_fixture_from_export(
    export_dir: str | Path, fixture_path: str | Path
) -> Path:
    """Pack the CSV export from ``fidelity_visual_intersim.R`` into the ``.npz``."""

    export = Path(export_dir)
    out = Path(fixture_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, Any] = {}
    for omic in OMIC_LAYERS:
        arrays[f"density__{omic}"] = np.loadtxt(
            export / f"density_{omic}.csv", delimiter=",", ndmin=2
        )
        arrays[f"matrix__{omic}"] = np.loadtxt(
            export / f"matrix_{omic}.csv", delimiter=",", ndmin=2
        )
    arrays["cluster_ids"] = np.loadtxt(
        export / "cluster_ids.csv", delimiter=",", skiprows=1, ndmin=1
    ).astype(int)

    provenance: dict[str, str] = {}
    for line in (export / "provenance.txt").read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            provenance[key.strip()] = value.strip()
    arrays["provenance"] = json.dumps(provenance)

    np.savez_compressed(out, **arrays)
    return out


# --------------------------------------------------------------------------- #
# Matched numpy side (generated live, no R)
# --------------------------------------------------------------------------- #


def _cluster_sizes(n_sample: int, n_cluster: int) -> list[int]:
    """Equal split with InterSIM's rounding rule (remainder on the last cluster)."""

    each = round(n_sample / n_cluster)
    head = [each] * (n_cluster - 1)
    return [*head, n_sample - sum(head)]


def _generate_one(
    sizes: list[int], delta: float, p_dmp: float, reference: IntersimReference,
    rng: np.random.Generator,
):
    ind_m = bernoulli_indicators(rng, reference.n_cpg, len(sizes), p_dmp)
    ind_e, ind_p = derive_coupled_indicators(ind_m, reference)
    return generate_omics(
        cell_sizes=sizes,
        indicators_methyl=ind_m,
        indicators_expr=ind_e,
        indicators_protein=ind_p,
        delta_methyl=delta,
        delta_expr=delta,
        delta_protein=delta,
        rng=rng,
        reference=reference,
    )


def generate_numpy_visual_data(
    fixture: VisualFixture,
    *,
    reference: IntersimReference | None = None,
    rng: np.random.Generator | None = None,
) -> VisualData:
    """Generate matched numpy data for the same protocol as the fixture.

    Mirrors the fixture's ``n_sample`` / ``n_cluster`` / ``delta`` / ``p_dmp`` and
    its density replicate count + subsample size, so the figures compare like for
    like. The numpy cluster labels are sorted blocks ``0..n_cluster-1``.
    """

    ref = reference if reference is not None else load_reference()
    gen = rng if rng is not None else np.random.default_rng(fixture.seed)
    sizes = _cluster_sizes(fixture.n_sample, fixture.n_cluster)
    n_rep, subsample = fixture.data.density[OMIC_LAYERS[0]].shape

    density: dict[str, list[np.ndarray]] = {omic: [] for omic in OMIC_LAYERS}
    for _ in range(n_rep):
        generated = _generate_one(sizes, fixture.delta, fixture.p_dmp, ref, gen)
        for omic, matrix in zip(
            OMIC_LAYERS,
            (generated.methylation, generated.expression, generated.proteomics),
            strict=True,
        ):
            flat = matrix.reshape(-1)
            idx = gen.choice(flat.size, size=min(subsample, flat.size), replace=False)
            density[omic].append(flat[idx])

    full = _generate_one(sizes, fixture.delta, fixture.p_dmp, ref, gen)
    matrices = {
        OMIC_LAYERS[0]: full.methylation,
        OMIC_LAYERS[1]: full.expression,
        OMIC_LAYERS[2]: full.proteomics,
    }
    return VisualData(
        density={omic: np.vstack(reps) for omic, reps in density.items()},
        matrices=matrices,
        cluster_ids=full.cell_ids.astype(int),
    )


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def _hist_density(values: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts, edges = np.histogram(values, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts


def build_density_figure(intersim: VisualData, numpy_data: VisualData) -> Figure:
    """Per-omic marginal densities, a translucent line per replicate per tool."""

    fig, axes = plt.subplots(1, len(OMIC_LAYERS), figsize=(5 * len(OMIC_LAYERS), 4))
    for ax, omic in zip(axes, OMIC_LAYERS, strict=True):
        pooled = np.concatenate(
            [intersim.density[omic].reshape(-1), numpy_data.density[omic].reshape(-1)]
        )
        bins = np.linspace(pooled.min(), pooled.max(), 80)
        for tool, vd in ((_INTERSIM, intersim), (_NUMPY, numpy_data)):
            color = _TOOL_COLOR[tool]
            for r, rep_vals in enumerate(vd.density[omic]):
                centers, dens = _hist_density(rep_vals, bins)
                ax.plot(centers, dens, color=color, alpha=0.5, lw=1.2,
                        label=tool if r == 0 else None)
        ax.set_title(omic)
        ax.set_xlabel("value")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
    fig.suptitle("Marginal value densities (replicates overlaid): InterSIM vs numpy")
    fig.tight_layout()
    return fig


def _zscore_rows(matrix: np.ndarray) -> np.ndarray:
    mu = matrix.mean(axis=0, keepdims=True)
    sd = matrix.std(axis=0, keepdims=True)
    return (matrix - mu) / np.where(sd > 0, sd, 1.0)


def _cluster_color_strip(cluster_ids: np.ndarray) -> np.ndarray:
    """RGBA strip (1 x n x 4) colouring samples by their cluster label."""

    cmap = plt.get_cmap("tab10")
    return np.array([cmap((int(c) - 1) % 10) for c in cluster_ids])[None, :, :]


def _clustered_panel(
    fig: Figure, spec, matrix: np.ndarray, cluster_ids: np.ndarray, title: str
) -> None:
    """Draw one clustermap-style panel into ``spec`` (a SubplotSpec).

    Free hierarchical clustering (Ward on z-scored features) reorders both
    samples (dendrogram on top) and features (dendrogram on the left); a
    categorical cluster colour bar under the sample dendrogram shows whether the
    free ordering recovers the cluster assignment. Axes are laid out so the
    dendrogram leaf positions line up with the heatmap cells.
    """

    from matplotlib.gridspec import GridSpecFromSubplotSpec
    from scipy.cluster.hierarchy import dendrogram, linkage

    z = _zscore_rows(matrix)  # (n_sample, n_feat), each feature standardised
    samp_link = linkage(z, method="ward")
    feat_link = linkage(z.T, method="ward")

    inner = GridSpecFromSubplotSpec(
        3, 2, subplot_spec=spec,
        width_ratios=[0.16, 1.0], height_ratios=[0.16, 0.05, 1.0],
        wspace=0.02, hspace=0.02,
    )
    ax_top = fig.add_subplot(inner[0, 1])
    ax_bar = fig.add_subplot(inner[1, 1])
    ax_left = fig.add_subplot(inner[2, 0])
    ax_heat = fig.add_subplot(inner[2, 1])

    d_top = dendrogram(samp_link, ax=ax_top, no_labels=True, color_threshold=0,
                       above_threshold_color="#555555")
    d_left = dendrogram(feat_link, ax=ax_left, orientation="left", no_labels=True,
                        color_threshold=0, above_threshold_color="#555555")
    leaves_s, leaves_f = d_top["leaves"], d_left["leaves"]
    assert leaves_s is not None and leaves_f is not None
    samp_order = np.asarray(leaves_s, dtype=int)
    feat_order = np.asarray(leaves_f, dtype=int)
    ax_top.set_axis_off()
    ax_left.set_axis_off()

    # origin="lower": heatmap row 0 (feat_order[0]) sits at the bottom, matching
    # the left dendrogram's first leaf; columns follow the top dendrogram leaves.
    img = z[np.ix_(samp_order, feat_order)].T
    ax_heat.imshow(img, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3,
                   interpolation="nearest", origin="lower")
    ax_heat.set_xticks([])
    ax_heat.set_yticks([])
    ax_heat.set_xlabel("samples")
    ax_heat.set_ylabel("features")

    ax_bar.imshow(_cluster_color_strip(cluster_ids[samp_order]), aspect="auto")
    ax_bar.set_xticks([])
    ax_bar.set_yticks([])
    ax_bar.set_ylabel("cluster", rotation=0, ha="right", va="center", fontsize=7)

    ax_top.set_title(title, fontsize=10)


def build_heatmap_figure(intersim: VisualData, numpy_data: VisualData) -> Figure:
    """Clustermap-style heatmaps, one per modality per tool (InterSIM vs ours).

    Each panel has a sample dendrogram (top), a cluster colour bar, a feature
    dendrogram (left), and the z-scored heatmap. Clustering is free and per
    panel — fair to both tools — so InterSIM's hidden differential features and
    ours' known ones organise from the data alone.
    """

    fig = plt.figure(figsize=(12, 4.6 * len(OMIC_LAYERS)))
    outer = fig.add_gridspec(len(OMIC_LAYERS), 2, wspace=0.12, hspace=0.18)
    for row, omic in enumerate(OMIC_LAYERS):
        for col, (tool, vd) in enumerate(
            ((_INTERSIM, intersim), (_NUMPY, numpy_data))
        ):
            _clustered_panel(
                fig, outer[row, col], vd.matrices[omic], vd.cluster_ids,
                f"{tool} — {omic}",
            )
    fig.suptitle(
        "Hierarchically clustered heatmaps (z-scored; free clustering, "
        "cluster colour bar): InterSIM vs numpy"
    )
    return fig


def _pca_2d(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def build_pca_figure(intersim: VisualData, numpy_data: VisualData) -> Figure:
    """First two PCs per modality per tool, coloured by cluster."""

    fig, axes = plt.subplots(
        len(OMIC_LAYERS), 2, figsize=(10, 3.6 * len(OMIC_LAYERS)), squeeze=False
    )
    cmap = plt.get_cmap("tab10")
    for row, omic in enumerate(OMIC_LAYERS):
        for col, (tool, vd) in enumerate(
            ((_INTERSIM, intersim), (_NUMPY, numpy_data))
        ):
            ax = axes[row][col]
            pcs = _pca_2d(vd.matrices[omic])
            for c in np.unique(vd.cluster_ids):
                mask = vd.cluster_ids == c
                ax.scatter(pcs[mask, 0], pcs[mask, 1], s=10, alpha=0.7,
                           color=cmap(int(c) % 10), label=f"cluster {c}")
            ax.set_title(f"{tool} — {omic}")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            if row == 0 and col == 0:
                ax.legend(fontsize=7, markerscale=1.2)
    fig.suptitle("Per-modality PCA (first two PCs), coloured by cluster")
    fig.tight_layout()
    return fig


def build_moment_scatter_figure(intersim: VisualData, numpy_data: VisualData) -> Figure:
    """Per-feature mean and variance agreement (ours vs InterSIM, y=x)."""

    fig, axes = plt.subplots(2, len(OMIC_LAYERS), figsize=(4 * len(OMIC_LAYERS), 8))
    for col, omic in enumerate(OMIC_LAYERS):
        i_mat = intersim.matrices[omic]
        n_mat = numpy_data.matrices[omic]
        for row, (label, stat) in enumerate(
            (("per-feature mean", lambda m: m.mean(0)),
             ("per-feature variance", lambda m: m.var(0)))
        ):
            ax = axes[row][col]
            x = stat(i_mat)
            y = stat(n_mat)
            ax.scatter(x, y, s=8, alpha=0.5, color="#444444")
            lo = float(min(x.min(), y.min()))
            hi = float(max(x.max(), y.max()))
            ax.plot([lo, hi], [lo, hi], color="crimson", lw=1.0)
            ax.set_title(f"{omic} — {label}")
            ax.set_xlabel("InterSIM")
            ax.set_ylabel("numpy (ours)")
    fig.suptitle("Per-feature moment agreement: numpy vs InterSIM (line = y=x)")
    fig.tight_layout()
    return fig


def _top_variance_block(matrices: dict[str, np.ndarray], k: int) -> tuple[np.ndarray, list[int]]:
    """Concatenate the top-``k``-variance features per omic; return matrix + sizes."""

    blocks = []
    sizes = []
    for omic in OMIC_LAYERS:
        mat = matrices[omic]
        keep = min(k, mat.shape[1])
        top = np.argsort(mat.var(0))[::-1][:keep]
        blocks.append(mat[:, top])
        sizes.append(keep)
    return np.hstack(blocks), sizes


def build_coupling_figure(intersim: VisualData, numpy_data: VisualData) -> Figure:
    """Cross-omic correlation block structure side-by-side (coupling evidence)."""

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, (tool, vd) in zip(
        axes, ((_INTERSIM, intersim), (_NUMPY, numpy_data)), strict=True
    ):
        block, sizes = _top_variance_block(vd.matrices, _COUPLING_FEATURES_PER_OMIC)
        corr = np.corrcoef(block, rowvar=False)
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        cuts = np.cumsum(sizes)[:-1]
        for c in cuts:
            ax.axhline(c - 0.5, color="black", lw=0.8)
            ax.axvline(c - 0.5, color="black", lw=0.8)
        centers = np.cumsum([0, *sizes])[:-1] + np.array(sizes) / 2
        ax.set_xticks(centers)
        ax.set_yticks(centers)
        ax.set_xticklabels(OMIC_LAYERS, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(OMIC_LAYERS, fontsize=8)
        ax.set_title(f"{tool}: feature correlation (top-variance per omic)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Within- and cross-omic correlation structure: InterSIM vs numpy")
    fig.tight_layout()
    return fig


_FIGURE_BUILDERS = {
    "density": build_density_figure,
    "heatmaps": build_heatmap_figure,
    "pca": build_pca_figure,
    "moment_scatter": build_moment_scatter_figure,
    "coupling": build_coupling_figure,
}


def run_fidelity_visual(
    fixture_path: str | Path,
    out_dir: str | Path,
    *,
    reference: IntersimReference | None = None,
    dpi: int = 150,
) -> dict[str, Path]:
    """Render every qualitative fidelity figure from the committed fixture (no R).

    Returns a mapping of figure name -> saved PNG path.
    """

    fixture = load_visual_fixture(fixture_path)
    numpy_data = generate_numpy_visual_data(fixture, reference=reference)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}
    for name, builder in _FIGURE_BUILDERS.items():
        fig = builder(fixture.data, numpy_data)
        path = out / f"fidelity_{name}.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        saved[name] = path
    return saved
