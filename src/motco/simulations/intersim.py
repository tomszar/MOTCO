"""Bridge from Python to the R InterSIM package.

InterSIM is an optional R dependency. Install it in R before calling
``run_intersim``:

```
install.packages("InterSIM", repos = c("https://cran.r-universe.dev", "https://cloud.r-project.org"))
```
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from shutil import which
from typing import Any

import pandas as pd


class InterSIMError(RuntimeError):
    """Base exception for InterSIM bridge failures."""


class InterSIMDependencyError(InterSIMError):
    """Raised when Rscript or the R InterSIM package is unavailable."""


class InterSIMRuntimeError(InterSIMError):
    """Raised when the R InterSIM subprocess fails."""


class InterSIMMalformedOutputError(InterSIMError):
    """Raised when the R helper output cannot be normalized."""


@dataclass(frozen=True)
class InterSIMAvailability:
    """Availability result for the external R InterSIM dependency."""

    available: bool
    message: str
    rscript_path: str | None = None


@dataclass(frozen=True)
class InterSIMParams:
    """Parameters for the InterSIM R package.

    Parameters left as ``None`` are omitted from the R call, allowing InterSIM
    defaults to apply. The covariance parameters currently support InterSIM's
    native ``"indep"`` string option or ``None``.
    """

    seed: int
    n_sample: int | None = None
    cluster_sample_prop: tuple[float, ...] | None = None
    delta_methyl: float | None = None
    delta_expr: float | None = None
    delta_protein: float | None = None
    p_dmp: float | None = None
    p_deg: float | None = None
    p_dep: float | None = None
    sigma_methyl: str | None = None
    sigma_expr: str | None = None
    sigma_protein: str | None = None
    cor_methyl_expr: float | None = None
    cor_expr_protein: float | None = None


@dataclass(frozen=True)
class InterSIMResult:
    """Normalized InterSIM simulation output."""

    methylation: pd.DataFrame
    expression: pd.DataFrame
    proteomics: pd.DataFrame
    sample_ids: pd.Index
    clusters: pd.Series
    metadata: dict[str, Any] = field(default_factory=dict)


_R_ARG_NAMES = {
    "n_sample": "n-sample",
    "cluster_sample_prop": "cluster-sample-prop",
    "delta_methyl": "delta-methyl",
    "delta_expr": "delta-expr",
    "delta_protein": "delta-protein",
    "p_dmp": "p-dmp",
    "p_deg": "p-deg",
    "p_dep": "p-dep",
    "sigma_methyl": "sigma-methyl",
    "sigma_expr": "sigma-expr",
    "sigma_protein": "sigma-protein",
    "cor_methyl_expr": "cor-methyl-expr",
    "cor_expr_protein": "cor-expr-protein",
}


def check_intersim_available(rscript: str = "Rscript") -> InterSIMAvailability:
    """Check whether Rscript and the R InterSIM package are available."""

    rscript_path = which(rscript)
    if rscript_path is None:
        return InterSIMAvailability(
            available=False,
            message=f"Rscript dependency is missing or not on PATH: {rscript}",
        )

    cmd = [
        rscript_path,
        "-e",
        'quit(status = if (requireNamespace("InterSIM", quietly = TRUE)) 0 else 1)',
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        details = _format_process_output(proc)
        return InterSIMAvailability(
            available=False,
            message="R package InterSIM is not installed or cannot be loaded." + details,
            rscript_path=rscript_path,
        )

    return InterSIMAvailability(
        available=True,
        message="Rscript and R package InterSIM are available.",
        rscript_path=rscript_path,
    )


def run_intersim(
    params: InterSIMParams,
    *,
    rscript: str = "Rscript",
    check_dependency: bool = True,
) -> InterSIMResult:
    """Invoke R InterSIM and return aligned omics matrices plus cluster metadata."""

    if check_dependency:
        availability = check_intersim_available(rscript)
        if not availability.available:
            raise InterSIMDependencyError(availability.message)
        rscript_cmd = availability.rscript_path or rscript
    else:
        rscript_cmd = which(rscript) or rscript

    with tempfile.TemporaryDirectory(prefix="motco-intersim-") as tmp:
        output_dir = Path(tmp)
        proc = subprocess.run(
            _build_rscript_command(params, output_dir=output_dir, rscript=rscript_cmd),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise InterSIMRuntimeError(
                f"InterSIM R process failed with exit code {proc.returncode}."
                f"{_format_process_output(proc)}"
            )
        return _load_result(output_dir, params=params)


_R_INT_MIN = -(2**31)
_R_INT_MAX = 2**31 - 1


def _build_rscript_command(params: InterSIMParams, *, output_dir: Path, rscript: str = "Rscript") -> list[str]:
    if not (_R_INT_MIN <= params.seed <= _R_INT_MAX):
        raise InterSIMError(
            f"InterSIM seed {params.seed} is outside R's signed-32-bit range "
            f"[{_R_INT_MIN}, {_R_INT_MAX}]; R's set.seed would coerce it to NA."
        )
    helper = resources.files("motco.simulations").joinpath("run_intersim.R")
    cmd = [
        rscript,
        str(helper),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(params.seed),
    ]
    for field_name, r_arg_name in _R_ARG_NAMES.items():
        value = getattr(params, field_name)
        if value is None:
            continue
        cmd.extend([f"--{r_arg_name}", _serialize_r_arg(value)])
    return cmd


def _serialize_r_arg(value: object) -> str:
    if isinstance(value, tuple):
        return ",".join(str(v) for v in value)
    return str(value)


def _load_result(output_dir: Path, *, params: InterSIMParams) -> InterSIMResult:
    methylation = _read_matrix(output_dir / "methylation.csv")
    expression = _read_matrix(output_dir / "expression.csv")
    proteomics = _read_matrix(output_dir / "proteomics.csv")
    clusters_df = _read_clusters(output_dir / "clusters.csv")

    sample_ids = methylation.index
    _validate_alignment(sample_ids, expression, proteomics, clusters_df)

    clusters = clusters_df["cluster"].copy()
    clusters.name = "cluster"
    clusters.index = sample_ids
    metadata = _metadata_from_params(params)
    metadata["n_samples"] = len(sample_ids)
    metadata["n_features"] = {
        "methylation": methylation.shape[1],
        "expression": expression.shape[1],
        "proteomics": proteomics.shape[1],
    }
    return InterSIMResult(
        methylation=methylation,
        expression=expression,
        proteomics=proteomics,
        sample_ids=sample_ids,
        clusters=clusters,
        metadata=metadata,
    )


def _read_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise InterSIMMalformedOutputError(f"Expected InterSIM output file is missing: {path.name}")
    try:
        df = pd.read_csv(path, index_col=0)
    except Exception as exc:  # pragma: no cover - pandas exception type varies
        raise InterSIMMalformedOutputError(f"Could not read InterSIM output file {path.name}: {exc}") from exc
    if df.empty:
        raise InterSIMMalformedOutputError(f"InterSIM output file is empty: {path.name}")
    return df


def _read_clusters(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise InterSIMMalformedOutputError(f"Expected InterSIM output file is missing: {path.name}")
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - pandas exception type varies
        raise InterSIMMalformedOutputError(f"Could not read InterSIM output file {path.name}: {exc}") from exc

    if "sample_id" not in df.columns or "cluster" not in df.columns:
        raise InterSIMMalformedOutputError("clusters.csv must contain sample_id and cluster columns.")
    if df.empty:
        raise InterSIMMalformedOutputError("clusters.csv is empty.")
    return df


def _validate_alignment(
    sample_ids: pd.Index,
    expression: pd.DataFrame,
    proteomics: pd.DataFrame,
    clusters_df: pd.DataFrame,
) -> None:
    expected = sample_ids.astype(str).tolist()
    observed = {
        "expression": expression.index.astype(str).tolist(),
        "proteomics": proteomics.index.astype(str).tolist(),
        "clusters": clusters_df["sample_id"].astype(str).tolist(),
    }
    for name, ids in observed.items():
        if ids != expected:
            raise InterSIMMalformedOutputError(
                f"InterSIM sample ID alignment failed for {name}: rows do not match methylation order."
            )


def _metadata_from_params(params: InterSIMParams) -> dict[str, Any]:
    metadata: dict[str, Any] = {"seed": params.seed}
    for field_name in _R_ARG_NAMES:
        value = getattr(params, field_name)
        if value is not None:
            metadata[field_name] = value
    return metadata


def _format_process_output(proc: subprocess.CompletedProcess[str]) -> str:
    parts = []
    if proc.stdout.strip():
        parts.append(f" stdout: {proc.stdout.strip()}")
    if proc.stderr.strip():
        parts.append(f" stderr: {proc.stderr.strip()}")
    return "".join(parts)
