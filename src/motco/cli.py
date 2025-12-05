import argparse
import json
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd

from motco.stats.pls import plsda_doubleCV
from motco.stats.snf import SNF, get_affinity_matrix, get_spectral
from motco.stats.sd import RRPP, estimate_difference


def _read_csv(path: Union[str, Path]) -> pd.DataFrame:
    return pd.read_csv(path, header=0)


def _save_csv(arr: Union[np.ndarray, pd.DataFrame], path: Union[str, Path]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if isinstance(arr, pd.DataFrame):
        arr.to_csv(path, index=False)
    else:
        pd.DataFrame(arr).to_csv(path, index=False)


def _save_json(obj, path: Union[str, Path]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)


def cmd_plsr(args: argparse.Namespace) -> None:
    if bool(args.data) == bool(args.x):
        raise SystemExit("Provide either --data with --label-col OR --x and --y CSV files.")

    if args.data:
        df = _read_csv(args.data)
        if args.label_col not in df.columns:
            raise SystemExit(f"Label column '{args.label_col}' not found in data.")
        y = df[args.label_col]
        X = df.drop(columns=[args.label_col])
    else:
        X = _read_csv(args.x)
        y_df = _read_csv(args.y)
        # If y is single-column, treat it as Series
        if y_df.shape[1] == 1:
            y = y_df.iloc[:, 0]
        else:
            y = y_df

    res = plsda_doubleCV(
        X=X,
        y=y,
        cv1_splits=args.cv1_splits,
        cv2_splits=args.cv2_splits,
        n_repeats=args.n_repeats,
        max_components=args.max_components,
        random_state=args.random_state,
    )
    table = res["table"]
    if args.out_table:
        _save_csv(table, args.out_table)
    else:
        # Print to stdout if no output path
        pd.set_option("display.max_columns", None)
        print(table)


def cmd_snf(args: argparse.Namespace) -> None:
    if not args.input or len(args.input) < 2:
        raise SystemExit("Provide at least two --input CSV files (same samples, same order).")
    datasets: List[np.ndarray] = []
    for p in args.input:
        df = _read_csv(p)
        datasets.append(df.values)
    Ws = get_affinity_matrix(datasets, K=args.K, eps=args.eps)
    fused = SNF(Ws, k=args.k, t=args.t)
    if args.out_fused:
        _save_csv(fused, args.out_fused)
    if args.out_embedding:
        emb = get_spectral(fused)
        _save_csv(emb, args.out_embedding)
    if not (args.out_fused or args.out_embedding):
        print(pd.DataFrame(fused))


def cmd_de(args: argparse.Namespace) -> None:
    Y = _read_csv(args.Y).values
    LS = _read_csv(args.ls_means).values
    with open(args.contrast, "r", encoding="utf-8") as fh:
        contrast = json.load(fh)

    if args.rrpp_permutations and args.rrpp_permutations > 0:
        if not args.model_full or not args.model_reduced:
            raise SystemExit("For RRPP, provide --model-full and --model-reduced CSVs.")
        Xf = _read_csv(args.model_full).values
        Xr = _read_csv(args.model_reduced).values
        deltas, angles, shapes = RRPP(
            Y=Y,
            model_full=Xf,
            model_reduced=Xr,
            LS_means=LS,
            contrast=contrast,
            permutations=args.rrpp_permutations,
        )
        # RRPP returns lists per permutation; convert recursively
        out = {
            "deltas": [[list(np.asarray(row)) for row in np.asarray(d)] for d in deltas],
            "angles": [[list(np.asarray(row)) for row in np.asarray(a)] for a in angles],
            "shapes": [list(np.asarray(s)) for s in shapes],
        }
    else:
        if not args.model_matrix:
            raise SystemExit("Provide --model-matrix CSV for estimate_difference.")
        X = _read_csv(args.model_matrix).values
        deltas, angles, shapes = estimate_difference(Y=Y, model_matrix=X, LS_means=LS, contrast=contrast)
        out = {
            "deltas": np.asarray(deltas).tolist(),
            "angles": np.asarray(angles).tolist(),
            "shapes": np.asarray(shapes).tolist(),
        }

    if args.out_json:
        _save_json(out, args.out_json)
    else:
        print(json.dumps(out, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="motco", description="MOTCO CLI: PLSR, SNF, and group differences")
    sub = p.add_subparsers(dest="command", required=True)

    # PLSR
    p_plsr = sub.add_parser("plsr", help="Run PLS-DA with double cross-validation")
    p_plsr.add_argument("--data", type=str, help="CSV with predictors and label column")
    p_plsr.add_argument("--label-col", type=str, help="Label column name when using --data")
    p_plsr.add_argument("--x", type=str, help="CSV with predictors (features)")
    p_plsr.add_argument("--y", type=str, help="CSV with labels/outcomes")
    p_plsr.add_argument("--cv1-splits", type=int, default=7)
    p_plsr.add_argument("--cv2-splits", type=int, default=8)
    p_plsr.add_argument("--n-repeats", type=int, default=30)
    p_plsr.add_argument("--max-components", type=int, default=50)
    p_plsr.add_argument("--random-state", type=int, default=1203)
    p_plsr.add_argument("--out-table", type=str, help="Path to save the best models table (CSV)")
    p_plsr.set_defaults(func=cmd_plsr)

    # SNF
    p_snf = sub.add_parser("snf", help="Similarity Network Fusion")
    p_snf.add_argument("--input", type=str, action="append", help="Input CSV (repeat for multiple omics)")
    p_snf.add_argument("--K", type=int, default=20, help="K for affinity construction")
    p_snf.add_argument("--eps", type=float, default=0.5, help="Epsilon for affinity construction")
    p_snf.add_argument("--k", type=int, default=20, help="k for sparse kernel in SNF")
    p_snf.add_argument("--t", type=int, default=20, help="Number of SNF iterations")
    p_snf.add_argument("--out-fused", type=str, help="Path to save fused matrix (CSV)")
    p_snf.add_argument("--out-embedding", type=str, help="Path to save spectral embedding (CSV)")
    p_snf.set_defaults(func=cmd_snf)

    # Differential Effects
    p_de = sub.add_parser("de", help="Group differences on trajectories (estimate or RRPP)")
    p_de.add_argument("--Y", type=str, required=True, help="Outcome matrix CSV (latent space coordinates)")
    p_de.add_argument("--ls-means", type=str, required=True, help="Least-squares means CSV")
    p_de.add_argument("--contrast", type=str, required=True, help="JSON file with groups (list of index lists)")
    p_de.add_argument("--model-matrix", type=str, help="Model matrix CSV (with intercept) for estimate_difference")
    p_de.add_argument("--model-full", type=str, help="Full model matrix CSV (with intercept) for RRPP")
    p_de.add_argument("--model-reduced", type=str, help="Reduced model matrix CSV (with intercept) for RRPP")
    p_de.add_argument("--rrpp-permutations", type=int, default=0, help="Number of permutations for RRPP")
    p_de.add_argument("--out-json", type=str, help="Output JSON file")
    p_de.set_defaults(func=cmd_de)

    return p


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
