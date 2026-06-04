"""Paper-grade fidelity validation of the numpy generator against InterSIM.

The numpy generator (:mod:`motco.simulations.generator`) is a from-scratch
reimplementation of InterSIM's ``mu = base + delta * v`` model. The single
``delta=0`` fixture test only exercises the degenerate baseline. This module
provides the rigorous, swept, replicate-based validation:

1. A **statistic battery** (:func:`compute_statistics`) that summarises one
   generated omics triple with RNG-robust, structural statistics: per-omic
   marginal moments/quantiles, cluster separation (eta-squared),
   differential-feature rate (the DMP->DEG->DEP coupling signal), and the
   Frobenius norm of the empirical covariance. The *same* formulas are
   implemented in ``fidelity_intersim.R`` so the two sides are comparable by
   construction.

2. A **parameter grid** (:func:`default_grid`) over ``delta`` x ``p.DMP`` with a
   ``delta=0`` anchor and >=2 non-zero effect sizes, plus replicate counts
   ``n_intersim`` / ``n_numpy``.

3. A **numpy cell runner** (:func:`run_numpy_cell`) that draws ``n_numpy``
   replicates from the *real* generator (``bernoulli_indicators`` +
   ``derive_coupled_indicators`` + ``generate_omics``) and returns the
   per-statistic distribution.

4. A **replicate-distribution criterion** (:func:`compare_cell`): each numpy
   statistic passes when its replicate mean falls inside InterSIM's own central
   interval (default ``[q2.5, q97.5]``) for that statistic, so the check accounts
   for InterSIM's RNG variability rather than comparing single draws.

The InterSIM side is captured as a committed fixture
(``tests/data/intersim_fidelity_fixture.npz``); :func:`load_fidelity_fixture`
reads it with no R dependency, and :func:`build_fidelity_fixture_from_export`
packs the CSV export produced by ``fidelity_intersim.R``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from motco.simulations.generator import (
    OMIC_LAYERS,
    bernoulli_indicators,
    derive_coupled_indicators,
    generate_omics,
)
from motco.simulations.reference import IntersimReference, load_reference

#: Per-omic statistic names produced by :func:`compute_statistics`.
STATISTIC_SUFFIXES = ("mean", "sd", "q10", "q50", "q90", "eta2", "diff_rate", "cov_frob")

#: A feature counts as "differential" when its standardised cluster-mean range
#: (range of per-cluster means divided by the feature's pooled SD) exceeds this.
#: Scale-free so it is comparable across omics and robust to RNG differences.
DIFF_RATE_THRESHOLD = 1.0

#: Default central-interval bounds (percentiles) for the InterSIM distribution.
INTERVAL_LOWER_PCT = 2.5
INTERVAL_UPPER_PCT = 97.5


class FidelityError(ValueError):
    """Raised for invalid fidelity-validation inputs or a missing fixture."""


@dataclass(frozen=True)
class FidelityGrid:
    """The swept validation protocol: a ``delta`` x ``p_dmp`` grid + replicates.

    ``cluster_prop`` are the per-cluster sample proportions (must sum to 1, with
    >1 entry), matching InterSIM's ``cluster.sample.prop``. ``n_intersim`` and
    ``n_numpy`` are the replicate counts per cell for the InterSIM and numpy
    sides. ``seed`` seeds the numpy side.
    """

    deltas: tuple[float, ...]
    p_dmps: tuple[float, ...]
    n_sample: int
    cluster_prop: tuple[float, ...]
    n_intersim: int
    n_numpy: int
    seed: int
    diff_threshold: float = DIFF_RATE_THRESHOLD

    @property
    def cells(self) -> list[tuple[float, float]]:
        """All ``(delta, p_dmp)`` cells in row-major (delta-major) order."""

        return [(d, p) for d in self.deltas for p in self.p_dmps]

    def cluster_sizes(self) -> list[int]:
        """Per-cluster sample counts, matching InterSIM's rounding rule."""

        props = self.cluster_prop
        head = [round(p * self.n_sample) for p in props[:-1]]
        return [*head, self.n_sample - sum(head)]


def default_grid() -> FidelityGrid:
    """The committed default protocol (kept small enough to regenerate in R).

    ``delta=0`` is the degenerate anchor (matches the existing realism fixture's
    regime); ``delta in {1, 2}`` exercise effect injection and the cross-omic
    coupling. ``p_dmp in {0.2, 0.5}`` vary the differential-feature density.
    """

    return FidelityGrid(
        deltas=(0.0, 1.0, 2.0),
        p_dmps=(0.2, 0.5),
        n_sample=500,
        cluster_prop=(0.3, 0.3, 0.4),
        n_intersim=30,
        n_numpy=30,
        seed=20260604,
    )


def _omic_statistics(
    matrix: np.ndarray, cluster_ids: np.ndarray, threshold: float
) -> dict[str, float]:
    """The per-omic statistic battery for one ``(n_sample, n_feat)`` matrix.

    Mirrors ``omic_statistics`` in ``fidelity_intersim.R`` exactly. All moments
    use the population convention (divide by ``n``); quantiles use linear
    interpolation (numpy default == R ``type=7``).
    """

    n = matrix.shape[0]
    flat = matrix.reshape(-1)
    grand = matrix.mean(axis=0)  # per-feature grand mean
    centered = matrix - grand
    sst = np.einsum("ij,ij->j", centered, centered)  # per-feature total SS

    labels = np.unique(cluster_ids)
    cluster_means = np.empty((labels.size, matrix.shape[1]))
    ssb = np.zeros(matrix.shape[1])
    for k, label in enumerate(labels):
        rows = matrix[cluster_ids == label]
        cm = rows.mean(axis=0)
        cluster_means[k] = cm
        ssb += rows.shape[0] * (cm - grand) ** 2

    with np.errstate(invalid="ignore", divide="ignore"):
        eta2_feat = np.where(sst > 0, ssb / sst, np.nan)
    eta2 = float(np.nanmean(eta2_feat))

    feat_sd = np.sqrt(sst / n)
    with np.errstate(invalid="ignore", divide="ignore"):
        std_range = np.where(
            feat_sd > 0, (cluster_means.max(0) - cluster_means.min(0)) / feat_sd, 0.0
        )
    diff_rate = float((std_range > threshold).mean())

    cov = (centered.T @ centered) / n
    cov_frob = float(np.sqrt(np.sum(cov**2)))

    q10, q50, q90 = (float(q) for q in np.quantile(flat, (0.1, 0.5, 0.9)))
    return {
        "mean": float(flat.mean()),
        "sd": float(flat.std()),
        "q10": q10,
        "q50": q50,
        "q90": q90,
        "eta2": eta2,
        "diff_rate": diff_rate,
        "cov_frob": cov_frob,
    }


def compute_statistics(
    methylation: np.ndarray,
    expression: np.ndarray,
    proteomics: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    threshold: float = DIFF_RATE_THRESHOLD,
) -> dict[str, float]:
    """RNG-robust structural summary of one generated omics triple.

    Returns a flat ``{"<omic>_<stat>": value}`` dict over :data:`OMIC_LAYERS`
    and :data:`STATISTIC_SUFFIXES`. The same statistics are computed on the
    InterSIM side, so the dicts are directly comparable.
    """

    out: dict[str, float] = {}
    for omic, matrix in zip(
        OMIC_LAYERS, (methylation, expression, proteomics), strict=True
    ):
        for suffix, value in _omic_statistics(matrix, cluster_ids, threshold).items():
            out[f"{omic}_{suffix}"] = value
    return out


def statistic_names() -> list[str]:
    """The full ordered list of ``<omic>_<stat>`` statistic keys."""

    return [f"{omic}_{suffix}" for omic in OMIC_LAYERS for suffix in STATISTIC_SUFFIXES]


def run_numpy_cell(
    delta: float,
    p_dmp: float,
    grid: FidelityGrid,
    *,
    reference: IntersimReference,
    rng: np.random.Generator,
) -> dict[str, list[float]]:
    """Draw ``grid.n_numpy`` replicates from the real generator for one cell.

    Each replicate draws methylation indicators ``~ Bernoulli(p_dmp)`` per
    cluster, derives the coupled expression/protein indicators, generates the
    triple at ``delta``, and computes the statistic battery. Returns
    ``{statistic: [value_per_replicate]}``.
    """

    n_cluster = len(grid.cluster_prop)
    sizes = grid.cluster_sizes()
    accumulator: dict[str, list[float]] = {name: [] for name in statistic_names()}
    for _ in range(grid.n_numpy):
        ind_m = bernoulli_indicators(rng, reference.n_cpg, n_cluster, p_dmp)
        ind_e, ind_p = derive_coupled_indicators(ind_m, reference)
        generated = generate_omics(
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
        stats = compute_statistics(
            generated.methylation,
            generated.expression,
            generated.proteomics,
            generated.cell_ids,
            threshold=grid.diff_threshold,
        )
        for name, value in stats.items():
            accumulator[name].append(value)
    return accumulator


# --------------------------------------------------------------------------- #
# Committed InterSIM fixture: I/O
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FidelityFixture:
    """Committed InterSIM summary fixture: per-cell statistic distributions.

    ``cells`` are ``(delta, p_dmp)`` pairs (row order matches ``distributions``).
    ``distributions[i][stat]`` is the array of ``n_intersim`` InterSIM replicate
    values for statistic ``stat`` in cell ``i``. ``provenance`` records the
    InterSIM/R version, date, seeds, and the grid.
    """

    cells: list[tuple[float, float]]
    distributions: list[dict[str, np.ndarray]]
    provenance: dict[str, str]
    grid: FidelityGrid = field(repr=False)

    def cell_index(self, delta: float, p_dmp: float) -> int:
        for i, (d, p) in enumerate(self.cells):
            if np.isclose(d, delta) and np.isclose(p, p_dmp):
                return i
        raise FidelityError(f"No fixture cell for (delta={delta}, p_dmp={p_dmp})")


#: Filename of the committed InterSIM fidelity fixture (under ``tests/data``).
FIDELITY_FIXTURE_NAME = "intersim_fidelity_fixture.npz"


def load_fidelity_fixture(path: str | Path) -> FidelityFixture:
    """Load the committed InterSIM fidelity fixture without invoking R.

    Raises :class:`FidelityError` with regeneration instructions when absent.
    """

    fixture_path = Path(path)
    if not fixture_path.exists():
        raise FidelityError(
            f"InterSIM fidelity fixture not found at {fixture_path}. Regenerate it "
            "in R (one-time): `nix develop --command Rscript "
            "src/motco/simulations/fidelity_intersim.R --output-dir <dir>` then "
            "`build_fidelity_fixture_from_export(<dir>, <fixture_path>)`."
        )

    with np.load(fixture_path, allow_pickle=False) as data:
        provenance = json.loads(str(data["provenance"]))
        cell_delta = data["cell_delta"]
        cell_pdmp = data["cell_pdmp"]
        names = [str(n) for n in data["stat_names"]]
        cells = [(float(d), float(p)) for d, p in zip(cell_delta, cell_pdmp, strict=True)]
        distributions: list[dict[str, np.ndarray]] = []
        for i in range(len(cells)):
            distributions.append(
                {name: data[f"stat__{name}"][i].astype(float) for name in names}
            )

    grid = FidelityGrid(
        deltas=tuple(sorted({d for d, _ in cells})),
        p_dmps=tuple(sorted({p for _, p in cells})),
        n_sample=int(provenance["n_sample"]),
        cluster_prop=tuple(float(x) for x in provenance["cluster_prop"].split(",")),
        n_intersim=int(provenance["n_intersim"]),
        n_numpy=int(provenance.get("n_numpy", provenance["n_intersim"])),
        seed=int(provenance.get("numpy_seed", 0)),
        diff_threshold=float(provenance.get("diff_threshold", DIFF_RATE_THRESHOLD)),
    )
    return FidelityFixture(
        cells=cells, distributions=distributions, provenance=provenance, grid=grid
    )


def build_fidelity_fixture_from_export(
    export_dir: str | Path, fixture_path: str | Path
) -> Path:
    """Pack the CSV export from ``fidelity_intersim.R`` into the ``.npz`` fixture.

    Reads ``stats.csv`` (one row per ``(cell, replicate)`` with the statistic
    columns) and ``provenance.txt``; writes per-statistic ``(n_cells,
    n_intersim)`` arrays plus provenance. Mirrors
    :func:`motco.simulations.reference.build_cache_from_export`.
    """

    export = Path(export_dir)
    out = Path(fixture_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows, header = _read_csv(export / "stats.csv")
    col = {name: i for i, name in enumerate(header)}
    names = statistic_names()
    for required in ("delta", "p_dmp", "replicate", *names):
        if required not in col:
            raise FidelityError(f"stats.csv missing column {required!r}")

    # Group rows by cell in first-seen order.
    cell_order: list[tuple[float, float]] = []
    per_cell: dict[tuple[float, float], list[list[str]]] = {}
    for row in rows:
        key = (float(row[col["delta"]]), float(row[col["p_dmp"]]))
        if key not in per_cell:
            per_cell[key] = []
            cell_order.append(key)
        per_cell[key].append(row)

    n_cells = len(cell_order)
    n_rep = len(per_cell[cell_order[0]])
    cell_delta = np.array([d for d, _ in cell_order], dtype=float)
    cell_pdmp = np.array([p for _, p in cell_order], dtype=float)
    stat_arrays: dict[str, np.ndarray] = {}
    for name in names:
        arr = np.empty((n_cells, n_rep), dtype=float)
        for c, key in enumerate(cell_order):
            cell_rows = per_cell[key]
            if len(cell_rows) != n_rep:
                raise FidelityError("All cells must have the same replicate count")
            arr[c] = [float(r[col[name]]) for r in cell_rows]
        stat_arrays[f"stat__{name}"] = arr

    provenance = _read_provenance(export / "provenance.txt")

    arrays: dict[str, Any] = {
        "cell_delta": cell_delta,
        "cell_pdmp": cell_pdmp,
        "stat_names": np.array(names, dtype=str),
        "provenance": json.dumps(provenance),
        **stat_arrays,
    }
    np.savez_compressed(out, **arrays)
    return out


def _read_csv(path: Path) -> tuple[list[list[str]], list[str]]:
    lines = path.read_text().splitlines()
    header = [c.strip().strip('"') for c in lines[0].split(",")]
    rows = [[c.strip().strip('"') for c in line.split(",")] for line in lines[1:] if line.strip()]
    return rows, header


def _read_provenance(path: Path) -> dict[str, str]:
    provenance: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        provenance[key.strip()] = value.strip()
    return provenance


# --------------------------------------------------------------------------- #
# Replicate-distribution comparison
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StatisticComparison:
    """One numpy-vs-InterSIM comparison for a single statistic in one cell."""

    statistic: str
    numpy_mean: float
    intersim_mean: float
    interval_low: float
    interval_high: float
    passed: bool


def central_interval(
    values: np.ndarray, lower: float = INTERVAL_LOWER_PCT, upper: float = INTERVAL_UPPER_PCT
) -> tuple[float, float]:
    """The ``[lower, upper]`` percentile interval of ``values``."""

    lo, hi = np.percentile(values, (lower, upper))
    return float(lo), float(hi)


def compare_cell(
    numpy_dist: dict[str, list[float]],
    intersim_dist: dict[str, np.ndarray],
    *,
    lower: float = INTERVAL_LOWER_PCT,
    upper: float = INTERVAL_UPPER_PCT,
) -> dict[str, StatisticComparison]:
    """Compare numpy vs InterSIM per statistic via the central-interval criterion.

    For each statistic, the numpy *replicate mean* passes when it lies within
    InterSIM's ``[lower, upper]`` percentile interval. Averaging over numpy
    replicates removes numpy's own Monte-Carlo noise so the test isolates
    systematic disagreement.
    """

    results: dict[str, StatisticComparison] = {}
    for name, samples in intersim_dist.items():
        intersim_values = np.asarray(samples, dtype=float)
        lo, hi = central_interval(intersim_values, lower, upper)
        numpy_mean = float(np.mean(numpy_dist[name]))
        results[name] = StatisticComparison(
            statistic=name,
            numpy_mean=numpy_mean,
            intersim_mean=float(intersim_values.mean()),
            interval_low=lo,
            interval_high=hi,
            passed=lo <= numpy_mean <= hi,
        )
    return results


def validate_grid(
    fixture: FidelityFixture,
    *,
    reference: IntersimReference | None = None,
    cells: list[tuple[float, float]] | None = None,
    lower: float = INTERVAL_LOWER_PCT,
    upper: float = INTERVAL_UPPER_PCT,
) -> dict[tuple[float, float], dict[str, StatisticComparison]]:
    """Run the numpy side and compare against the fixture across cells.

    ``cells`` defaults to every fixture cell; pass a subset for a fast check.
    Returns ``{(delta, p_dmp): {statistic: StatisticComparison}}``.
    """

    ref = reference if reference is not None else load_reference()
    grid = fixture.grid
    rng = np.random.default_rng(grid.seed)
    target_cells = cells if cells is not None else fixture.cells
    out: dict[tuple[float, float], dict[str, StatisticComparison]] = {}
    for delta, p_dmp in target_cells:
        idx = fixture.cell_index(delta, p_dmp)
        numpy_dist = run_numpy_cell(delta, p_dmp, grid, reference=ref, rng=rng)
        out[(delta, p_dmp)] = compare_cell(
            numpy_dist, fixture.distributions[idx], lower=lower, upper=upper
        )
    return out
