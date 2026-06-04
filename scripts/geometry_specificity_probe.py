#!/usr/bin/env python3
"""Geometry-specificity probe (numpy generator, no R).

Three diagnostics behind the ``characterize-geometry-specificity`` change:

1. **2-stage isolation** — the shape-free regime (``n_stages=2``, Procrustes
   ``shape`` degenerate) for ``magnitude``/``orientation`` and the negative
   controls, to confirm ``magnitude``→``delta`` and ``orientation``→``angle``
   without shape contamination.
2. **Shape-null diagnostic** — splits the saturated ``shape`` rejection into the
   observed Procrustes distance vs its RRPP permutation null, under raw /
   concat-standardize / SNF integration, to see whether per-feature
   standardization is what collapses the null (anti-conservative).
3. **Magnitude variant** — ``magnitude_kind='all'`` vs ``'extremes'`` at
   ``n_stages=4``, to see whether confining the scale to the endpoints reduces
   the shape co-movement.

Runs serial (``n_jobs=1``): per-replicate cost is dominated by Procrustes GPA on
the integrated matrix and the multiprocessing pool spawn overhead dwarfs the
work for the small permutation counts used here.

Example:

    python scripts/geometry_specificity_probe.py \
        --reps 12 --perms 49 --n-samples 160 --out /tmp/geometry_specificity.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from motco.simulations.reference import load_reference
from motco.simulations.specificity import (
    SHAPE_FREE_MODES,
    characterize_two_stage,
    evaluate_mode_specificity,
    evaluate_shape_null,
)

#: Integration configs for the shape-null sweep: (label, method, standardize).
_INTEGRATIONS: tuple[tuple[str, str, bool], ...] = (
    ("concat-standardize", "concat", True),
    ("raw-concat", "concat", False),
    ("snf", "snf", True),
)

#: Modes carried through the shape-null sweep (geometry movers + null control).
_SHAPE_MODES: tuple[str, ...] = ("none", "magnitude", "orientation", "shape")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Geometry-specificity probe (no R).")
    p.add_argument("--reps", type=int, default=12, help="Replicates per cell (default: 12)")
    p.add_argument("--perms", type=int, default=49, help="RRPP permutations (default: 49)")
    p.add_argument("--n-samples", type=int, default=160, help="Samples per dataset (default: 160)")
    p.add_argument("--effect-size", type=float, default=1.0, help="Group effect size (default: 1.0)")
    p.add_argument("--p-dmp", type=float, default=0.2, help="Per-stage methylation DMP prob (default: 0.2)")
    p.add_argument("--base-seed", type=int, default=0, help="Base seed (default: 0)")
    p.add_argument("--out", type=str, default=None, help="Write the markdown report here (default: stdout)")
    return p


def _two_stage_table(reports: dict, reps: int) -> list[str]:
    lines = [
        "## 2-stage isolation (shape degenerate)",
        "",
        "| mode | delta | angle | group-in-stage |",
        "|------|-------|-------|----------------|",
    ]
    for mode in SHAPE_FREE_MODES:
        r = reports[mode]
        lines.append(
            f"| `{mode}` | {r.rejection_rates['delta']:.2f} | "
            f"{r.rejection_rates['angle']:.2f} | {r.group_in_stage_fraction:.2f} |"
        )
    lines.append("")
    lines.append(f"_{reps} reps, `n_stages=2`; `shape` omitted (a single step has no shape)._")
    lines.append("")
    return lines


def _shape_null_table(rows: list) -> list[str]:
    lines = [
        "## Shape-null diagnostic (observed Procrustes vs RRPP null)",
        "",
        "| integration | mode | observed | null q2.5 | null median | null q97.5 | null sd | reject |",
        "|-------------|------|----------|-----------|-------------|------------|---------|--------|",
    ]
    for d in rows:
        label = d.integration_method if d.integration_method == "snf" else (
            "concat-std" if d.standardize else "raw-concat"
        )
        lines.append(
            f"| {label} | `{d.mode}` | {d.observed_mean:.3f} | {d.null_q025_mean:.3f} | "
            f"{d.null_median_mean:.3f} | {d.null_q975_mean:.3f} | {d.null_spread_mean:.3f} | "
            f"{d.rejection_rate:.2f} |"
        )
    lines.append("")
    return lines


def _magnitude_variant_table(variants: dict, reps: int) -> list[str]:
    lines = [
        "## Magnitude variant: all-stages vs endpoints",
        "",
        "| magnitude_kind | delta | angle | shape |",
        "|----------------|-------|-------|-------|",
    ]
    for kind in ("all", "extremes"):
        r = variants[kind]
        lines.append(
            f"| `{kind}` | {r.rejection_rates['delta']:.2f} | "
            f"{r.rejection_rates['angle']:.2f} | {r.rejection_rates['shape']:.2f} |"
        )
    lines.append("")
    lines.append(f"_{reps} reps, `n_stages=4`._")
    lines.append("")
    return lines


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ref = load_reference()
    common = dict(
        n_replicates=args.reps,
        n_samples=args.n_samples,
        effect_size=args.effect_size,
        p_dmp=args.p_dmp,
        permutations=args.perms,
        n_jobs=1,
        base_seed=args.base_seed,
        reference=ref,
    )

    out = [
        "# Geometry-specificity probe results",
        "",
        f"Run: {args.reps} reps, {args.perms} perms, `n_samples={args.n_samples}`, "
        f"`effect_size={args.effect_size}`, `p_dmp={args.p_dmp}`, serial (`n_jobs=1`).",
        "",
    ]

    print("[1/3] 2-stage isolation...", file=sys.stderr)
    two_stage = characterize_two_stage(modes=SHAPE_FREE_MODES, **common)
    out += _two_stage_table(two_stage, args.reps)

    print("[2/3] shape-null diagnostic...", file=sys.stderr)
    shape_rows = []
    for label, method, standardize in _INTEGRATIONS:
        for mode in _SHAPE_MODES:
            print(f"      {label} / {mode}", file=sys.stderr)
            shape_rows.append(
                evaluate_shape_null(
                    mode,  # type: ignore[arg-type]
                    integration_method=method,
                    standardize=standardize,
                    n_stages=4,
                    **common,
                )
            )
    out += _shape_null_table(shape_rows)

    print("[3/3] magnitude variant...", file=sys.stderr)
    variants = {
        kind: evaluate_mode_specificity(
            "magnitude",
            n_stages=4,
            magnitude_kind=kind,
            **common,
        )
        for kind in ("all", "extremes")
    }
    out += _magnitude_variant_table(variants, args.reps)

    report = "\n".join(out)
    if args.out:
        Path(args.out).write_text(report)
        print(f"Wrote {args.out}", file=sys.stderr)
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
