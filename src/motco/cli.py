import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd

from motco import __version__
from motco.stats import RRPP, estimate_betas, estimate_difference
from motco.stats.pls import calculate_vips, fit_plsda_transform, plsda_doubleCV
from motco.stats.snf import SNF, get_affinity_matrix, get_spectral


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


def _standardize_and_concat(input_paths: List[str]) -> pd.DataFrame:
    frames = []
    for path in input_paths:
        df = _read_csv(path)
        values = df.to_numpy(dtype=float)
        mean = values.mean(axis=0, keepdims=True)
        std = values.std(axis=0, keepdims=True)
        std[std == 0.0] = 1.0
        values = (values - mean) / std
        frames.append(pd.DataFrame(values, columns=df.columns))
    return pd.concat(frames, axis=1).reset_index(drop=True)


def cmd_plsr(args: argparse.Namespace) -> None:
    has_input = bool(args.input)
    has_single = bool(args.data) or bool(args.x)

    if has_input and has_single:
        raise SystemExit("--input cannot be combined with --data or --x/--y.")

    if has_input:
        if not args.metadata:
            raise SystemExit("--input requires --metadata.")
        if not args.label_col:
            raise SystemExit("--input requires --label-col.")
        X = _standardize_and_concat(args.input)
        meta = _read_csv(args.metadata)
        if args.label_col not in meta.columns:
            raise SystemExit(f"Label column '{args.label_col}' not found in metadata.")
        y = meta[args.label_col].reset_index(drop=True)
    elif has_single:
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
            if y_df.shape[1] == 1:
                y = y_df.iloc[:, 0]
            else:
                y = y_df
    else:
        raise SystemExit("Provide either --input files with --metadata/--label-col OR --data/--x/--y.")

    try:
        res = plsda_doubleCV(
            X=X,
            y=y,
            cv1_splits=args.cv1_splits,
            cv2_splits=args.cv2_splits,
            n_repeats=args.n_repeats,
            max_components=args.max_components,
            random_state=args.random_state,
        )
    except ValueError as e:
        raise SystemExit(f"Error: {e}") from None

    table = res["table"]
    if args.out_table:
        _save_csv(table, args.out_table)
    else:
        pd.set_option("display.max_columns", None)
        print(table)

    if args.out_vips:
        vips_data = {}
        for rep_idx, model in enumerate(res["models"], start=1):
            vips_data[f"rep_{rep_idx}"] = calculate_vips(model)
        vips_df = pd.DataFrame(vips_data)
        _save_csv(vips_df, args.out_vips)

    if args.out_scores:
        if args.n_components is not None:
            n_comp = args.n_components
        else:
            lv_series = res["table"].iloc[:, 1]
            n_comp = int(lv_series.mode()[0])
        scores = fit_plsda_transform(np.asarray(X, dtype=float), y, n_comp)
        score_cols = [f"lv_{i + 1}" for i in range(scores.shape[1])]
        scores_df = pd.DataFrame(scores, columns=score_cols)
        _save_csv(scores_df, args.out_scores)


def cmd_snf(args: argparse.Namespace) -> None:
    if not args.input or len(args.input) < 2:
        raise SystemExit("Provide at least two --input CSV files (same samples, same order).")
    datasets: List[np.ndarray] = []
    for p in args.input:
        df = _read_csv(p)
        datasets.append(df.values)
    try:
        Ws = get_affinity_matrix(datasets, K=args.K, eps=args.eps)
        fused = SNF(Ws, k=args.k, t=args.t)
    except ValueError as e:
        raise SystemExit(f"Error: {e}") from None
    if args.out_fused:
        _save_csv(fused, args.out_fused)
    if args.out_embedding:
        emb = get_spectral(fused, n_components=args.spectral_components)
        _save_csv(emb, args.out_embedding)
    if not (args.out_fused or args.out_embedding):
        print(pd.DataFrame(fused))


def cmd_de(args: argparse.Namespace) -> None:
    Y = _read_csv(args.Y).values
    LS = _read_csv(args.ls_means).values
    with open(args.contrast, "r", encoding="utf-8") as fh:
        contrast = json.load(fh)

    try:
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
                "shapes": [[list(np.asarray(row)) for row in np.asarray(s)] for s in shapes],
            }
            if args.out_observed:
                betas = estimate_betas(Xf, Y)
                observed = np.asarray(LS, dtype=float) @ np.asarray(betas, dtype=float)
                _save_csv(observed, args.out_observed)
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
            if args.out_observed:
                betas = estimate_betas(X, Y)
                observed = np.asarray(LS, dtype=float) @ np.asarray(betas, dtype=float)
                _save_csv(observed, args.out_observed)
    except ValueError as e:
        raise SystemExit(f"Error: {e}") from None

    if args.out_json:
        _save_json(out, args.out_json)
    else:
        print(json.dumps(out, indent=2))


def cmd_simulate(args: argparse.Namespace) -> None:
    from motco.simulations.evaluation import build_simulation_trajectory_design
    from motco.simulations.intersim import InterSIMParams, check_intersim_available
    from motco.simulations.semisynthetic import (
        SemiSyntheticTrajectoryParams,
        generate_semisynthetic_trajectory_from_intersim,
    )

    if not 0 <= args.prop_affected_features <= 1:
        raise SystemExit(
            "Error: --prop-affected-features must be between 0 and 1 "
            f"(got {args.prop_affected_features})."
        )
    for flag_name, value in (
        ("--delta-methyl", args.delta_methyl),
        ("--delta-expr", args.delta_expr),
        ("--delta-protein", args.delta_protein),
        ("--cluster-mean-shift", args.cluster_mean_shift),
    ):
        if value is not None and value < 0:
            raise SystemExit(f"Error: {flag_name} must be non-negative (got {value}).")
    availability = check_intersim_available()
    if not availability.available:
        raise SystemExit(
            f"motco simulate requires R and the InterSIM package.\n"
            f"{availability.message}\n"
            "Install InterSIM in R with:\n"
            '  install.packages("InterSIM", repos = c('
            '"https://cran.r-universe.dev", "https://cloud.r-project.org"))'
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    delta_methyl = args.delta_methyl if args.delta_methyl is not None else args.cluster_mean_shift
    delta_expr = args.delta_expr if args.delta_expr is not None else args.cluster_mean_shift
    delta_protein = args.delta_protein if args.delta_protein is not None else args.cluster_mean_shift
    intersim_params = InterSIMParams(
        seed=args.seed,
        n_sample=args.n_samples,
        delta_methyl=delta_methyl,
        delta_expr=delta_expr,
        delta_protein=delta_protein,
    )
    traj_params = SemiSyntheticTrajectoryParams(
        seed=args.seed,
        trajectory_mode=args.trajectory_mode,
        group_effect_size=args.effect_size,
        prop_affected_features=args.prop_affected_features,
    )

    try:
        dataset = generate_semisynthetic_trajectory_from_intersim(
            intersim_params, traj_params
        )
    except ValueError as e:
        raise SystemExit(f"Simulation error: {e}") from None

    design = build_simulation_trajectory_design(
        dataset.metadata, group_col="group", stage_col="stage"
    )

    # Omics matrices — feature columns only (row order matches metadata.csv)
    for layer in ("methylation", "expression", "proteomics"):
        df = getattr(dataset, layer).copy()
        df.to_csv(out_dir / f"{layer}.csv", index=False)

    # Metadata
    dataset.metadata[["sample_id", "group", "stage", "cluster"]].to_csv(
        out_dir / "metadata.csv", index=False
    )

    # Design matrices — numeric CSVs with auto column headers, no row index
    pd.DataFrame(design.model_full).to_csv(out_dir / "model_full.csv", index=False)
    pd.DataFrame(design.model_reduced).to_csv(out_dir / "model_reduced.csv", index=False)
    pd.DataFrame(design.ls_means).to_csv(out_dir / "ls_means.csv", index=False)

    # JSON outputs
    _save_json(design.contrast, out_dir / "contrast.json")
    _save_json(dataset.truth, out_dir / "truth.json")

    print(f"Simulation complete. Files written to: {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="motco", description="MOTCO CLI: PLSR, SNF, and group differences")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--verbose", action="store_true", help="Enable debug logging to stderr")
    sub = p.add_subparsers(dest="command", required=True)

    # PLSR
    p_plsr = sub.add_parser("plsr", help="Run PLS-DA with double cross-validation")
    p_plsr.add_argument("--input", type=str, action="append", dest="input",
                        help="Omics layer CSV (repeat for multiple layers; standardized and concatenated)")
    p_plsr.add_argument("--metadata", type=str, default=None,
                        help="Metadata CSV with label column (required when using --input)")
    p_plsr.add_argument("--data", type=str, help="CSV with predictors and label column")
    p_plsr.add_argument("--label-col", type=str, help="Label column name (used with --data or --input/--metadata)")
    p_plsr.add_argument("--x", type=str, help="CSV with predictors (features)")
    p_plsr.add_argument("--y", type=str, help="CSV with labels/outcomes")
    p_plsr.add_argument("--cv1-splits", type=int, default=7)
    p_plsr.add_argument("--cv2-splits", type=int, default=8)
    p_plsr.add_argument("--n-repeats", type=int, default=30)
    p_plsr.add_argument("--max-components", type=int, default=50)
    p_plsr.add_argument("--random-state", type=int, default=1203)
    p_plsr.add_argument("--out-table", type=str, help="Path to save the best models table (CSV)")
    p_plsr.add_argument("--out-vips", type=str, default=None, help="Path to save VIP scores per feature (CSV)")
    p_plsr.add_argument("--out-scores", type=str, default=None,
                        help="Path to save latent space scores from final model (CSV)")
    p_plsr.add_argument("--n-components", type=int, default=None,
                        help="Number of latent variables for final score model (default: modal LV from CV)")
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
    p_snf.add_argument("--spectral-components", type=int, default=10,
                       help="Number of spectral embedding components (default: 10)")
    p_snf.set_defaults(func=cmd_snf)

    # Simulate
    p_sim = sub.add_parser("simulate", help="Generate a semi-synthetic multi-omics toy dataset (requires R + InterSIM)")
    p_sim.add_argument("--seed", type=int, required=True, help="Random seed for reproducibility")
    p_sim.add_argument("--out-dir", type=str, required=True, help="Directory to write output files")
    p_sim.add_argument("--n-samples", type=int, default=90, help="Total number of samples (default: 90)")
    p_sim.add_argument("--trajectory-mode", type=str, default="orientation",
                       choices=["none", "translation", "magnitude", "orientation", "shape"],
                       help="Group trajectory difference mode (default: orientation)")
    p_sim.add_argument("--effect-size", type=float, default=1.0,
                       help="Group effect size injected into the simulation (default: 1.0)")
    p_sim.add_argument("--prop-affected-features", type=float, default=0.1,
                       help="Proportion of features per omic layer carrying the injected group effect (default: 0.1)")
    p_sim.add_argument("--cluster-mean-shift", type=float, default=None,
                       help="InterSIM cluster mean shift applied to unspecified per-omic delta flags")
    p_sim.add_argument("--delta-methyl", type=float, default=None,
                       help="InterSIM methylation cluster mean shift; overrides --cluster-mean-shift for methylation")
    p_sim.add_argument("--delta-expr", type=float, default=None,
                       help="InterSIM expression cluster mean shift; overrides --cluster-mean-shift for expression")
    p_sim.add_argument("--delta-protein", type=float, default=None,
                       help="InterSIM proteomics cluster mean shift; overrides --cluster-mean-shift for proteomics")
    p_sim.set_defaults(func=cmd_simulate)

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
    p_de.add_argument("--out-observed", type=str, default=None, help="Save predicted LS-mean vectors as CSV")
    p_de.set_defaults(func=cmd_de)

    return p


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)
    args.func(args)


if __name__ == "__main__":
    main()
