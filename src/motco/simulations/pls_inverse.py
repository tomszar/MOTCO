"""Controlled inverse-PLS trajectory interpretability study.

This module asks what feature-space perturbation is implied by an exact
trajectory intervention in a fixed two-component PLS-DA score space.  It is a
small linear diagnostic, not a production multi-omics simulation or an
inferential power study.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression

from motco.stats.pls import fit_plsda_model
from motco.stats.trajectory import estimate_difference

Intervention = Literal["magnitude", "orientation", "shape"]


class PLSInverseStudyError(ValueError):
    """Raised when an inverse-PLS study configuration is invalid."""


@dataclass(frozen=True)
class PLSInverseStudyParams:
    """Parameters for one controlled inverse-PLS dataset."""

    seed: int = 0
    n_stages: int = 3
    n_features: int = 50
    n_samples_per_stage: int = 30
    noise_scale: float = 0.5
    signal_scale: float = 4.0
    magnitude_scale: float = 2.0
    orientation_degrees: float = 45.0
    shape_displacement_fraction: float = 0.5
    n_components: int = 2


@dataclass(frozen=True)
class PLSInverseDataset:
    """Paired feature data with exactly identical groups."""

    X: pd.DataFrame
    metadata: pd.DataFrame
    stage_means: np.ndarray


@dataclass(frozen=True)
class PLSInverseFit:
    """Frozen PLS-DA model and its training scores."""

    model: PLSRegression
    scores: np.ndarray
    loadings: np.ndarray


@dataclass(frozen=True)
class GeometrySummary:
    """Pairwise trajectory differences between Groups A and B."""

    delta: float
    angle: float
    shape: float


@dataclass(frozen=True)
class PLSInverseCellResult:
    """Outputs for one stage-count/intervention cell."""

    params: PLSInverseStudyParams
    intervention: Intervention
    latent_geometry: GeometrySummary
    feature_geometry: GeometrySummary
    roundtrip_max_error: float
    residual_max_error: float
    feature_table: pd.DataFrame
    induced_metric: np.ndarray
    induced_eigenvalues: np.ndarray
    induced_condition_number: float
    nonnegligible_feature_count: int
    loading_change_correlation: float
    top_10pct_change_fraction: float
    change_participation_ratio: float

    def summary_record(self) -> dict[str, float | int | str]:
        """Return a flat, machine-readable summary row."""

        return {
            "seed": self.params.seed,
            "n_stages": self.params.n_stages,
            "n_features": self.params.n_features,
            "n_samples_per_stage": self.params.n_samples_per_stage,
            "noise_scale": self.params.noise_scale,
            "signal_scale": self.params.signal_scale,
            "intervention": self.intervention,
            "latent_delta": self.latent_geometry.delta,
            "latent_angle": self.latent_geometry.angle,
            "latent_shape": self.latent_geometry.shape,
            "feature_delta": self.feature_geometry.delta,
            "feature_angle": self.feature_geometry.angle,
            "feature_shape": self.feature_geometry.shape,
            "roundtrip_max_error": self.roundtrip_max_error,
            "residual_max_error": self.residual_max_error,
            "induced_condition_number": self.induced_condition_number,
            "nonnegligible_feature_count": self.nonnegligible_feature_count,
            "loading_change_correlation": self.loading_change_correlation,
            "top_10pct_change_fraction": self.top_10pct_change_fraction,
            "change_participation_ratio": self.change_participation_ratio,
            "metric_eigenvalue_min": float(self.induced_eigenvalues[0]),
            "metric_eigenvalue_max": float(self.induced_eigenvalues[-1]),
        }


def _validate_params(params: PLSInverseStudyParams) -> None:
    if params.n_stages not in {2, 3}:
        raise PLSInverseStudyError("n_stages must be 2 or 3")
    if params.n_components != 2:
        raise PLSInverseStudyError("n_components must be exactly 2 for the intervention plane")
    if params.n_features < 2:
        raise PLSInverseStudyError("n_features must be at least 2")
    if params.n_samples_per_stage < 2:
        raise PLSInverseStudyError("n_samples_per_stage must be at least 2")
    if params.noise_scale <= 0 or params.signal_scale <= 0:
        raise PLSInverseStudyError("noise_scale and signal_scale must be positive")
    if params.magnitude_scale <= 0:
        raise PLSInverseStudyError("magnitude_scale must be positive")
    if params.shape_displacement_fraction < 0:
        raise PLSInverseStudyError("shape_displacement_fraction must be non-negative")


def generate_paired_dataset(params: PLSInverseStudyParams) -> PLSInverseDataset:
    """Generate Gaussian stage samples and duplicate them exactly across groups."""

    _validate_params(params)
    rng = np.random.default_rng(params.seed)
    basis, _ = np.linalg.qr(rng.normal(size=(params.n_features, 2)))
    direction_1, direction_2 = basis[:, 0], basis[:, 1]
    positions: tuple[tuple[float, float], ...]
    if params.n_stages == 2:
        positions = ((-0.5, 0.0), (0.5, 0.0))
    else:
        positions = ((-0.5, 0.0), (0.0, 0.35), (0.5, 0.0))
    means = np.vstack(
        [params.signal_scale * (x * direction_1 + y * direction_2) for x, y in positions]
    )
    base = np.vstack(
        [
            rng.normal(loc=means[stage], scale=params.noise_scale, size=(params.n_samples_per_stage, params.n_features))
            for stage in range(params.n_stages)
        ]
    )
    stages = np.repeat(np.arange(params.n_stages), params.n_samples_per_stage)
    paired_ids = np.arange(base.shape[0])
    X = np.vstack([base, base.copy()])
    metadata = pd.DataFrame(
        {
            "group": np.repeat(["A", "B"], base.shape[0]),
            "stage": np.tile(stages, 2),
            "paired_id": np.tile(paired_ids, 2),
        }
    )
    columns = [f"feature_{i:03d}" for i in range(params.n_features)]
    return PLSInverseDataset(X=pd.DataFrame(X, columns=columns), metadata=metadata, stage_means=means)


def fit_inverse_pls(dataset: PLSInverseDataset, *, n_components: int = 2) -> PLSInverseFit:
    """Fit the single frozen PLS-DA model used by an inverse study."""

    if n_components != 2:
        raise PLSInverseStudyError("the inverse study requires exactly two PLS components")
    model = fit_plsda_model(dataset.X, dataset.metadata["stage"], n_components=n_components)
    scores = np.asarray(model.transform(dataset.X.to_numpy(dtype=float)), dtype=float)
    return PLSInverseFit(model=model, scores=scores, loadings=np.asarray(model.x_loadings_, dtype=float).copy())


def stage_centroids(values: np.ndarray, metadata: pd.DataFrame, group: str) -> np.ndarray:
    """Return ordered stage centroids for one group."""

    group_mask = metadata["group"].to_numpy() == group
    stages = metadata["stage"].to_numpy(dtype=int)
    return np.vstack([values[group_mask & (stages == stage)].mean(axis=0) for stage in sorted(np.unique(stages))])


def _target_centroids(
    centroids: np.ndarray,
    intervention: Intervention,
    params: PLSInverseStudyParams,
) -> np.ndarray:
    center = centroids.mean(axis=0)
    if intervention == "magnitude":
        return center + params.magnitude_scale * (centroids - center)
    if intervention == "orientation":
        theta = np.deg2rad(params.orientation_degrees)
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        return center + (centroids - center) @ rotation.T
    if intervention == "shape":
        if centroids.shape[0] != 3:
            raise PLSInverseStudyError("shape intervention requires exactly three stages")
        endpoint = centroids[2] - centroids[0]
        endpoint_norm = float(np.linalg.norm(endpoint))
        if endpoint_norm <= 1e-12:
            raise PLSInverseStudyError("shape intervention requires distinct endpoint centroids")
        perpendicular = np.array([-endpoint[1], endpoint[0]]) / endpoint_norm
        target = centroids.copy()
        target[1] += params.shape_displacement_fraction * endpoint_norm * perpendicular
        return target
    raise PLSInverseStudyError(f"unknown intervention: {intervention!r}")


def intervene_scores(
    scores: np.ndarray,
    metadata: pd.DataFrame,
    intervention: Intervention,
    params: PLSInverseStudyParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Move Group B stage centroids and preserve every within-stage residual."""

    if scores.shape[1] != 2:
        raise PLSInverseStudyError("scores must have exactly two columns")
    original = stage_centroids(scores, metadata, "B")
    target = _target_centroids(original, intervention, params)
    modified = scores.copy()
    groups = metadata["group"].to_numpy()
    stages = metadata["stage"].to_numpy(dtype=int)
    for stage in range(params.n_stages):
        mask = (groups == "B") & (stages == stage)
        modified[mask] += target[stage] - original[stage]
    return modified, target


def additive_reconstruction(
    X: np.ndarray,
    original_scores: np.ndarray,
    modified_scores: np.ndarray,
    metadata: pd.DataFrame,
    model: PLSRegression,
) -> tuple[np.ndarray, float, float]:
    """Add the PLS-represented score change while preserving feature residuals."""

    groups = metadata["group"].to_numpy()
    b_mask = groups == "B"
    X_star = np.asarray(X, dtype=float).copy()
    original_inverse = np.asarray(model.inverse_transform(original_scores[b_mask]), dtype=float)
    modified_inverse = np.asarray(model.inverse_transform(modified_scores[b_mask]), dtype=float)
    X_star[b_mask] += modified_inverse - original_inverse
    recovered_displacement = np.asarray(model.transform(X_star[b_mask])) - original_scores[b_mask]
    requested_displacement = modified_scores[b_mask] - original_scores[b_mask]
    roundtrip_error = float(np.max(np.abs(recovered_displacement - requested_displacement)))
    original_residual = np.asarray(X, dtype=float)[b_mask] - original_inverse
    modified_residual = X_star[b_mask] - modified_inverse
    residual_error = float(np.max(np.abs(modified_residual - original_residual)))
    return X_star, roundtrip_error, residual_error


def trajectory_geometry(values: np.ndarray, metadata: pd.DataFrame) -> GeometrySummary:
    """Compute MOTCO trajectory differences directly from group-stage centroids."""

    centroids_a = stage_centroids(values, metadata, "A")
    centroids_b = stage_centroids(values, metadata, "B")
    vectors = np.vstack([centroids_a, centroids_b])
    n_stages = centroids_a.shape[0]
    identity = np.eye(2 * n_stages)
    contrast = [list(range(n_stages)), list(range(n_stages, 2 * n_stages))]
    deltas, angles, shapes = estimate_difference(vectors, identity, identity, contrast)
    shape = float(shapes[0, 1]) if n_stages >= 3 else float("nan")
    return GeometrySummary(delta=float(deltas[0, 1]), angle=float(angles[0, 1]), shape=shape)


def induced_metric(loadings: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``P.T @ P``, its eigenvalues, and spectral condition number."""

    P = np.asarray(loadings, dtype=float)
    if P.ndim != 2 or P.shape[1] != 2:
        raise PLSInverseStudyError("loadings must have shape (n_features, 2)")
    metric = P.T @ P
    eigenvalues = np.linalg.eigvalsh(metric)
    condition = float(np.inf if eigenvalues[0] <= 1e-15 else eigenvalues[-1] / eigenvalues[0])
    return metric, eigenvalues, condition


def feature_change_table(
    X: np.ndarray,
    X_star: np.ndarray,
    metadata: pd.DataFrame,
    loadings: np.ndarray,
    feature_names: list[str],
    intervention: Intervention,
    n_stages: int,
) -> tuple[pd.DataFrame, int, float, float, float]:
    """Build tidy feature-change rows and aggregate interpretability diagnostics."""

    b_mask = metadata["group"].to_numpy() == "B"
    delta = X_star[b_mask] - X[b_mask]
    signed_mean = delta.mean(axis=0)
    mean_absolute = np.abs(delta).mean(axis=0)
    loading_magnitude = np.linalg.norm(loadings, axis=1)
    threshold = max(float(mean_absolute.max()) * 1e-10, 1e-12)
    nonnegligible = int(np.sum(mean_absolute > threshold))
    if np.std(mean_absolute) <= 1e-15 or np.std(loading_magnitude) <= 1e-15:
        correlation = float("nan")
    else:
        correlation = float(np.corrcoef(mean_absolute, loading_magnitude)[0, 1])
    table = pd.DataFrame(
        {
            "n_stages": n_stages,
            "intervention": intervention,
            "feature": feature_names,
            "signed_mean_change": signed_mean,
            "mean_absolute_change": mean_absolute,
            "loading_lv1": loadings[:, 0],
            "loading_lv2": loadings[:, 1],
            "loading_magnitude": loading_magnitude,
        }
    )
    table["absolute_change_rank"] = table["mean_absolute_change"].rank(method="first", ascending=False).astype(int)
    total_change = float(mean_absolute.sum())
    top_k = max(1, int(np.ceil(0.1 * len(mean_absolute))))
    top_fraction = float(np.sort(mean_absolute)[-top_k:].sum() / total_change) if total_change > 0 else 0.0
    squared_sum = float(np.square(mean_absolute).sum())
    participation = float(total_change**2 / squared_sum) if squared_sum > 0 else 0.0
    return (
        table.sort_values("absolute_change_rank").reset_index(drop=True),
        nonnegligible,
        correlation,
        top_fraction,
        participation,
    )


def run_inverse_cell(params: PLSInverseStudyParams, intervention: Intervention) -> PLSInverseCellResult:
    """Run one complete latent-intervention-to-feature-reconstruction cell."""

    dataset = generate_paired_dataset(params)
    fit = fit_inverse_pls(dataset, n_components=params.n_components)
    modified_scores, _ = intervene_scores(fit.scores, dataset.metadata, intervention, params)
    X = dataset.X.to_numpy(dtype=float)
    X_star, roundtrip_error, residual_error = additive_reconstruction(
        X, fit.scores, modified_scores, dataset.metadata, fit.model
    )
    latent_geometry = trajectory_geometry(modified_scores, dataset.metadata)
    feature_geometry = trajectory_geometry(X_star, dataset.metadata)
    table, nonnegligible, correlation, top_fraction, participation = feature_change_table(
        X,
        X_star,
        dataset.metadata,
        fit.loadings,
        dataset.X.columns.astype(str).tolist(),
        intervention,
        params.n_stages,
    )
    metric, eigenvalues, condition = induced_metric(fit.loadings)
    return PLSInverseCellResult(
        params=params,
        intervention=intervention,
        latent_geometry=latent_geometry,
        feature_geometry=feature_geometry,
        roundtrip_max_error=roundtrip_error,
        residual_max_error=residual_error,
        feature_table=table,
        induced_metric=metric,
        induced_eigenvalues=eigenvalues,
        induced_condition_number=condition,
        nonnegligible_feature_count=nonnegligible,
        loading_change_correlation=correlation,
        top_10pct_change_fraction=top_fraction,
        change_participation_ratio=participation,
    )


def run_inverse_study(base_params: PLSInverseStudyParams | None = None) -> list[PLSInverseCellResult]:
    """Run the required two-stage and three-stage study cells."""

    base = base_params or PLSInverseStudyParams()
    results: list[PLSInverseCellResult] = []
    for n_stages, interventions in ((2, ("magnitude", "orientation")), (3, ("magnitude", "orientation", "shape"))):
        params = PLSInverseStudyParams(**{**asdict(base), "n_stages": n_stages})
        for intervention in interventions:
            results.append(run_inverse_cell(params, intervention))  # type: ignore[arg-type]
    return results


def results_frames(results: list[PLSInverseCellResult]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return cell-level and feature-level machine-readable tables."""

    cells = pd.DataFrame([result.summary_record() for result in results])
    features = pd.concat([result.feature_table for result in results], ignore_index=True)
    return cells, features


def render_markdown(results: list[PLSInverseCellResult]) -> str:
    """Render a compact findings-ready Markdown report."""

    cells, _ = results_frames(results)
    lines = [
        "# Inverse PLS trajectory interpretability study",
        "",
        "A deterministic linear diagnostic: Groups A and B are exact paired duplicates before a "
        "centroid-level intervention in a frozen two-component PLS-DA space.",
        "",
        "## Geometry comparison",
        "",
        "| stages | intervention | latent delta | latent angle | latent shape | feature delta | "
        "feature angle | feature shape | round-trip max | metric condition |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cells.to_dict(orient="records"):
        def fmt(value: object) -> str:
            number = float(value)  # type: ignore[arg-type]
            return "—" if not np.isfinite(number) else f"{number:.6g}"

        lines.append(
            f"| {row['n_stages']} | {row['intervention']} | {fmt(row['latent_delta'])} | "
            f"{fmt(row['latent_angle'])} | {fmt(row['latent_shape'])} | {fmt(row['feature_delta'])} | "
            f"{fmt(row['feature_angle'])} | {fmt(row['feature_shape'])} | "
            f"{fmt(row['roundtrip_max_error'])} | {fmt(row['induced_condition_number'])} |"
        )
    first = results[0].params
    lines.extend(
        [
            "",
            "## Parameters",
            "",
            f"- Seed: `{first.seed}`",
            f"- Features: `{first.n_features}`",
            f"- Samples per stage per group: `{first.n_samples_per_stage}`",
            f"- Noise / signal scale: `{first.noise_scale}` / `{first.signal_scale}`",
            f"- Magnitude factor: `{first.magnitude_scale}`",
            f"- Orientation rotation: `{first.orientation_degrees}°`",
            f"- Shape displacement fraction: `{first.shape_displacement_fraction}`",
            "",
            "## Induced metric and feature concentration",
            "",
            "| stages | intervention | metric eigenvalues | top 10% change fraction | participation ratio | "
            "loading/change correlation |",
            "|---:|---|---|---:|---:|---:|",
        ]
    )
    for result in results:
        eigenvalues = ", ".join(f"{value:.6g}" for value in result.induced_eigenvalues)
        lines.append(
            f"| {result.params.n_stages} | {result.intervention} | {eigenvalues} | "
            f"{result.top_10pct_change_fraction:.6g} | {result.change_participation_ratio:.6g} | "
            f"{result.loading_change_correlation:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Leading implied feature changes",
            "",
            "| stages | intervention | feature | mean absolute change | loading magnitude |",
            "|---:|---|---|---:|---:|",
        ]
    )
    for result in results:
        for row in result.feature_table.head(5).to_dict(orient="records"):
            lines.append(
                f"| {result.params.n_stages} | {result.intervention} | {row['feature']} | "
                f"{float(row['mean_absolute_change']):.6g} | {float(row['loading_magnitude']):.6g} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The reconstruction is the additive feature perturbation implied by this fitted low-rank PLS "
            "model. It preserves the original residual, but it is neither a unique inverse nor a "
            "biologically causal intervention.",
        ]
    )
    return "\n".join(lines) + "\n"
