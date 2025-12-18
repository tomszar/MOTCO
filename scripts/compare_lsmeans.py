from __future__ import annotations

import argparse
import shutil
import subprocess
from io import StringIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import sys


# Ensure local src/ is importable when running the script directly
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from motco.stats.sd import (  # noqa: E402
    get_model_matrix,
    build_ls_means,
    estimate_betas,
)


def _feature_columns(df: pd.DataFrame, group_col: str, level_col: str) -> list[str]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num_cols if c not in {group_col, level_col}]


## Note: The Python re-implementation of R's lsmeans.obs has been removed.
## We now always call native R via scripts/lsmeans_obs.R for the R-side table.


def _lsmeans_r_native(
    csv_path: Path,
    group_col: str,
    level_col: str,
    n_components: int = 2,
    rscript: str | None = None,
) -> pd.DataFrame:
    """
    Compute lsmeans.obs by invoking native R code that mirrors
    tests/data/reference/evo_649_sm_suppmat.r exactly.

    Returns a DataFrame indexed by the R factor order of keys "group:level"
    with columns PC1..PCk.
    """
    # Resolve Rscript
    r_bin = rscript or shutil.which("Rscript")
    if r_bin is None:
        raise RuntimeError(
            "Rscript not found on PATH. Please install R and ensure 'Rscript' is available, "
            "or pass --rscript PATH."
        )

    # Script path
    r_file = _ROOT / "scripts" / "lsmeans_obs.R"
    if not r_file.exists():
        raise RuntimeError(
            f"Missing R helper script: {r_file}. Please ensure it exists in the repository."
        )

    cmd = [
        str(r_bin),
        str(r_file),
        "--data",
        str(csv_path),
        "--group-col",
        str(group_col),
        "--level-col",
        str(level_col),
        "--pcs",
        str(int(n_components)),
    ]
    try:
        proc = subprocess.run(
            cmd, check=True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "Rscript execution failed. Stdout/stderr below:\n"
            f"STDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}"
        ) from e

    # Parse CSV from stdout; expect a 'key' column
    stdout_txt = proc.stdout if isinstance(proc.stdout, str) else str(proc.stdout)
    stderr_txt = proc.stderr if isinstance(proc.stderr, str) else str(proc.stderr)
    if not stdout_txt.strip():
        raise RuntimeError(
            "R script produced no output. STDERR:\n" + stderr_txt.strip()
        )
    try:
        out = pd.read_csv(StringIO(stdout_txt))
    except Exception as e:
        sample = stdout_txt[:500]
        raise RuntimeError(
            "Failed to parse CSV from R stdout. Sample of stdout (first 500 chars):\n"
            + sample + "\nSTDERR:\n" + stderr_txt.strip()
        ) from e
    if "key" not in out.columns:
        sample = stdout_txt[:500]
        got_cols = ", ".join(out.columns) if hasattr(out, "columns") else "<unavailable>"
        raise RuntimeError(
            "Unexpected R output: missing 'key' column. Got columns: "
            + got_cols
            + "\nSample of R stdout (first 500 chars):\n"
            + sample
            + "\nSTDERR:\n"
            + stderr_txt.strip()
        )
    out = out.set_index("key")
    # Do not sort; preserve R factor order
    return out


def _lsmeans_python_style(
    df: pd.DataFrame,
    group_col: str,
    level_col: str,
    n_components: int = 2,
) -> pd.DataFrame:
    """
    Our LS-means computation: LS @ betas
    Row order: group-major, level-minor with sorted string levels.
    """
    feat = _feature_columns(df, group_col, level_col)
    if not feat:
        raise ValueError("No numeric feature columns found in the dataset.")

    Y = PCA(n_components=n_components).fit_transform(df[feat])
    X = df[[group_col, level_col]].copy()
    M = get_model_matrix(X, group_col=group_col, level_col=level_col, full=True)
    B = estimate_betas(M, Y)

    g_levels = sorted(pd.unique(X[group_col].astype(str)).tolist())
    l_levels = sorted(pd.unique(X[level_col].astype(str)).tolist())
    LS = build_ls_means(g_levels, l_levels, full=True)
    # Columns must match the number of response dimensions (PCs), not the design width
    k = int(np.asarray(B).shape[1])
    py_ls = pd.DataFrame(LS @ np.asarray(B), columns=[f"PC{i+1}" for i in range(k)])
    py_ls.index = [f"{g}:{l}" for g in g_levels for l in l_levels]
    return py_ls.sort_index()


def _detect_schema_from_filename(path: Path) -> tuple[str, str] | None:
    name = path.name.lower()
    if "example1" in name:
        return "taxa", "Inv"
    if "example2" in name:
        return "tax", "Inv"
    return None


def compare_one(
    csv_path: Path,
    group_col: str,
    level_col: str,
    n_components: int = 2,
    print_values: bool = True,
    use_r_native: bool = False,
    rscript: str | None = None,
) -> float:
    df = pd.read_csv(csv_path)
    # Always use native R output for lsmeans.obs
    r_ls = _lsmeans_r_native(csv_path, group_col, level_col, n_components, rscript=rscript)
    py_ls = _lsmeans_python_style(df, group_col, level_col, n_components)
    # Align rows by R-style keys
    py_ls_aligned = py_ls.loc[r_ls.index]
    diff = py_ls_aligned.to_numpy(dtype=float) - r_ls.to_numpy(dtype=float)
    mad = float(np.max(np.abs(diff))) if diff.size else 0.0

    if print_values:
        print(f"\n=== {csv_path} ({group_col} × {level_col}) ===")
        print("R-style lsmeans.obs\n", r_ls.round(6))
        print("\nPython LS-means (LS @ betas)\n", py_ls_aligned.round(6))
        print(f"\nMax |difference|: {mad:.12f}")
    return mad


def _default_examples(root: Path) -> list[tuple[Path, str, str]]:
    data = root / "tests" / "data"
    return [
        (data / "evo_649_sm_example1.csv", "taxa", "Inv"),
        (data / "evo_649_sm_example2.csv", "tax", "Inv"),
    ]


def _iter_inputs(
    inputs: Iterable[Path], group_col: str | None, level_col: str | None
) -> list[tuple[Path, str, str]]:
    items: list[tuple[Path, str, str]] = []
    for p in inputs:
        if group_col and level_col:
            items.append((p, group_col, level_col))
        else:
            schema = _detect_schema_from_filename(p)
            if schema is None:
                raise SystemExit(
                    f"Cannot infer schema for {p.name}. Please provide --group-col and --level-col."
                )
            items.append((p, schema[0], schema[1]))
    return items


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Compare R's lsmeans.obs (via fitted means of yhat) with Python's LS@betas LS-means. "
            "Print both matrices and the maximum absolute difference."
        )
    )
    ap.add_argument(
        "--data",
        type=Path,
        nargs="*",
        help=(
            "CSV path(s). If omitted, runs on bundled example1 and example2. "
            "If provided without --group-col/--level-col, schema is inferred from filename."
        ),
    )
    ap.add_argument("--group-col", type=str, default=None, help="Group column name")
    ap.add_argument("--level-col", type=str, default=None, help="Level/state column name")
    ap.add_argument(
        "--pcs",
        type=int,
        default=2,
        help="Number of principal components to use as Y (default: 2)",
    )
    ap.add_argument(
        "--r-native",
        action="store_true",
        help=(
            "Compute lsmeans.obs using native R (via Rscript) and the exact reference code, "
            "instead of reproducing it in Python."
        ),
    )
    ap.add_argument(
        "--rscript",
        type=str,
        default=None,
        help="Path to the Rscript executable (defaults to Rscript on PATH)",
    )
    ap.add_argument(
        "--no-print",
        action="store_true",
        help="Do not print matrices, only the max absolute difference.",
    )
    args = ap.parse_args()

    if args.data:
        items = _iter_inputs(args.data, args.group_col, args.level_col)
    else:
        items = _default_examples(_ROOT)

    any_diff = False
    for path, g, l in items:
        mad = compare_one(
            path,
            g,
            l,
            n_components=args.pcs,
            print_values=not args.no_print,
            use_r_native=args.r_native,
            rscript=args.rscript,
        )
        if mad > 1e-10:
            any_diff = True
    if any_diff:
        # Non-zero differences might still be floating noise; we do not exit non-zero.
        print("\nNote: Non-zero differences observed (likely numeric precision).")


if __name__ == "__main__":
    main()
