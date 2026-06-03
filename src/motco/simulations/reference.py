"""Cached InterSIM reference data for the numpy generator.

InterSIM's generative model is fully determined by a set of package-level
reference objects (per-omic means and covariances, cross-omic incidence maps,
and per-feature correlation vectors). ``export_reference.R`` reads those objects
once in R and writes them to a plain-CSV export directory;
:func:`build_cache_from_export` packs that directory into the committed
``intersim_reference.npz``. At runtime the numpy generator calls
:func:`load_reference`, which reads the ``.npz`` with no R dependency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np

#: Filename of the committed reference cache shipped inside the package.
REFERENCE_CACHE_NAME = "intersim_reference.npz"

#: Package subdirectory (relative to ``motco.simulations``) holding the cache.
REFERENCE_DATA_PACKAGE = "motco.simulations.data"


class ReferenceCacheError(RuntimeError):
    """Base exception for reference-cache failures."""


class ReferenceCacheMissingError(ReferenceCacheError):
    """Raised when the cached reference artifact cannot be found."""


@dataclass(frozen=True)
class IntersimReference:
    """InterSIM reference data needed to reproduce its generative model.

    All arrays are aligned to the canonical feature orders given by
    ``cpg_names`` / ``gene_names`` / ``protein_names``. The incidence matrices
    encode InterSIM's cross-omic derivation: ``incidence_cpg_gene[i, j] == 1``
    iff CpG ``i`` maps to gene ``j`` (so a gene is differential when any of its
    CpGs is), and ``incidence_gene_protein[g, p] == 1`` iff protein ``p``'s
    mapped gene is ``g``.
    """

    mean_M: np.ndarray
    mean_expr: np.ndarray
    mean_protein: np.ndarray
    cov_M: np.ndarray
    cov_expr: np.ndarray
    cov_protein: np.ndarray
    methyl_gene_level_mean: np.ndarray
    mean_expr_with_mapped_protein: np.ndarray
    rho_methyl_expr: np.ndarray
    rho_expr_protein: np.ndarray
    incidence_cpg_gene: np.ndarray
    incidence_gene_protein: np.ndarray
    cpg_names: np.ndarray
    gene_names: np.ndarray
    protein_names: np.ndarray
    provenance: dict[str, str]

    @property
    def n_cpg(self) -> int:
        return int(self.mean_M.shape[0])

    @property
    def n_gene(self) -> int:
        return int(self.mean_expr.shape[0])

    @property
    def n_protein(self) -> int:
        return int(self.mean_protein.shape[0])


def _default_cache_path() -> Path:
    return Path(str(resources.files(REFERENCE_DATA_PACKAGE).joinpath(REFERENCE_CACHE_NAME)))


def load_reference(path: str | Path | None = None) -> IntersimReference:
    """Load the cached InterSIM reference data without invoking R.

    Raises :class:`ReferenceCacheMissingError` with regeneration instructions
    when the cache is absent.
    """

    cache_path = Path(path) if path is not None else _default_cache_path()
    if not cache_path.exists():
        raise ReferenceCacheMissingError(
            f"InterSIM reference cache not found at {cache_path}. "
            "Regenerate it by exporting from R (one-time, requires the InterSIM "
            "package): run `export_reference.R --output-dir <dir>` and then "
            "`build_cache_from_export(<dir>, <cache_path>)`."
        )

    with np.load(cache_path, allow_pickle=False) as data:
        provenance = json.loads(str(data["provenance"]))
        return IntersimReference(
            mean_M=data["mean_M"].astype(float),
            mean_expr=data["mean_expr"].astype(float),
            mean_protein=data["mean_protein"].astype(float),
            cov_M=data["cov_M"].astype(float),
            cov_expr=data["cov_expr"].astype(float),
            cov_protein=data["cov_protein"].astype(float),
            methyl_gene_level_mean=data["methyl_gene_level_mean"].astype(float),
            mean_expr_with_mapped_protein=data["mean_expr_with_mapped_protein"].astype(float),
            rho_methyl_expr=data["rho_methyl_expr"].astype(float),
            rho_expr_protein=data["rho_expr_protein"].astype(float),
            incidence_cpg_gene=data["incidence_cpg_gene"].astype(float),
            incidence_gene_protein=data["incidence_gene_protein"].astype(float),
            cpg_names=data["cpg_names"].astype(str),
            gene_names=data["gene_names"].astype(str),
            protein_names=data["protein_names"].astype(str),
            provenance=provenance,
        )


def _read_named(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a ``name,value`` CSV into ``(names, values)`` arrays."""

    names: list[str] = []
    values: list[float] = []
    with path.open() as handle:
        header = handle.readline()
        if "name" not in header:
            raise ReferenceCacheError(f"Expected a 'name,value' header in {path.name}")
        for line in handle:
            line = line.strip()
            if not line:
                continue
            name, value = line.rsplit(",", 1)
            names.append(name.strip('"'))
            values.append(float(value))
    return np.asarray(names, dtype=str), np.asarray(values, dtype=float)


def _read_vector(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=1)


def _read_matrix(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", ndmin=2)


def _read_provenance(path: Path) -> dict[str, str]:
    provenance: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        provenance[key.strip()] = value.strip()
    return provenance


def build_cache_from_export(export_dir: str | Path, cache_path: str | Path | None = None) -> Path:
    """Pack a CSV export directory (from ``export_reference.R``) into the ``.npz`` cache."""

    export = Path(export_dir)
    out = Path(cache_path) if cache_path is not None else _default_cache_path()
    out.parent.mkdir(parents=True, exist_ok=True)

    cpg_names, mean_M = _read_named(export / "mean_M.csv")
    gene_names, mean_expr = _read_named(export / "mean_expr.csv")
    protein_names, mean_protein = _read_named(export / "mean_protein.csv")
    _, methyl_gene_level_mean = _read_named(export / "methyl_gene_level_mean.csv")
    _, mean_expr_with_mapped_protein = _read_named(export / "mean_expr_with_mapped_protein.csv")

    provenance = _read_provenance(export / "provenance.txt")

    np.savez_compressed(
        out,
        mean_M=mean_M,
        mean_expr=mean_expr,
        mean_protein=mean_protein,
        cov_M=_read_matrix(export / "cov_M.csv"),
        cov_expr=_read_matrix(export / "cov_expr.csv"),
        cov_protein=_read_matrix(export / "cov_protein.csv"),
        methyl_gene_level_mean=methyl_gene_level_mean,
        mean_expr_with_mapped_protein=mean_expr_with_mapped_protein,
        rho_methyl_expr=_read_vector(export / "rho_methyl_expr.csv"),
        rho_expr_protein=_read_vector(export / "rho_expr_protein.csv"),
        incidence_cpg_gene=_read_matrix(export / "incidence_cpg_gene.csv"),
        incidence_gene_protein=_read_matrix(export / "incidence_gene_protein.csv"),
        cpg_names=cpg_names,
        gene_names=gene_names,
        protein_names=protein_names,
        provenance=json.dumps(provenance),
    )
    return out
