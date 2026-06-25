#!/usr/bin/env python3
"""Rung 3 — trajectory cross-talk through the production latent spaces.

Runs the dominant-specificity per-statistic RRPP rejection-rate study through
each latent space — the ``concat`` baseline, ``snf``, and the production ``pls``
space — on identical seeds, effect size, stages, and permutation count, so the
only varied factor is the latent space. For each mode (``none``, ``magnitude``,
``orientation``, ``shape``) it reports the per-statistic (``delta``/``angle``/
``shape``) rejection rate and the group-in-stage fraction per latent space.

A *specific* construction rejects predominantly on its target statistic
(``magnitude``→``delta``, ``orientation``→``angle``, ``shape``→``shape``);
off-target rejections are cross-talk. Read ``pls``/``snf`` relative to the
``concat`` baseline on the *same generator* — not against the Rung-2 test-bed
numbers (the production path also carries generator coupling + ``rev.logit``).

PLS integration runs ``plsda_doubleCV`` once per replicate; keep ``--reps``
modest and the CV knobs small. Serial (``n_jobs=1``) for the small perm counts.

Example:

    python scripts/latent_space_crosstalk_probe.py \
        --reps 10 --perms 99 --n-samples 180 --n-stages 4 \
        --out /tmp/rung3_latent_space_crosstalk.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from motco.simulations.reference import load_reference
from motco.simulations.specificity import TARGET_STATISTIC, evaluate_mode_specificity

#: Modes carried through every latent space (geometry movers + null control).
_MODES: tuple[str, ...] = ("none", "magnitude", "orientation", "shape")

#: Latent spaces compared: (label, integration_method).
_LATENT_SPACES: tuple[tuple[str, str], ...] = (
    ("concat (baseline)", "concat"),
    ("snf", "snf"),
    ("pls", "pls"),
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Rung 3: cross-talk through the production latent spaces.")
    p.add_argument("--reps", type=int, default=10, help="Replicates per cell (default: 10)")
    p.add_argument("--perms", type=int, default=99, help="RRPP permutations (default: 99)")
    p.add_argument("--n-samples", type=int, default=180, help="Samples per dataset (default: 180)")
    p.add_argument("--n-stages", type=int, default=4, help="Trajectory stages (default: 4)")
    p.add_argument("--effect-size", type=float, default=1.0, help="Group effect size (default: 1.0)")
    p.add_argument("--p-dmp", type=float, default=0.2, help="Per-stage methylation DMP prob (default: 0.2)")
    p.add_argument("--base-seed", type=int, default=0, help="Base seed (default: 0)")
    p.add_argument("--n-jobs", type=int, default=1, help="RRPP parallel workers; -1 = all CPUs (default: 1)")
    # PLS double-CV knobs (kept small; double-CV runs once per replicate).
    p.add_argument("--pls-repeats", type=int, default=3, help="PLS CV n_repeats (default: 3)")
    p.add_argument("--pls-cv2", type=int, default=4, help="PLS outer CV splits (default: 4)")
    p.add_argument("--pls-cv1", type=int, default=3, help="PLS inner CV splits (default: 3)")
    p.add_argument("--pls-max-components", type=int, default=15, help="PLS max LV candidates (default: 15)")
    p.add_argument("--out", type=str, default=None, help="Write the markdown report here (default: stdout)")
    return p


def _integration_params(label: str, method: str, args: argparse.Namespace) -> dict[str, object]:
    if method == "pls":
        return {
            "n_repeats": args.pls_repeats,
            "cv2_splits": args.pls_cv2,
            "cv1_splits": args.pls_cv1,
            "max_components": args.pls_max_components,
        }
    return {}


def _table(reports: dict[str, dict[str, object]]) -> list[str]:
    lines = [
        "## Per-statistic rejection rates by latent space",
        "",
        "Target statistic per mode: `magnitude`→`delta`, `orientation`→`angle`, "
        "`shape`→`shape`. Off-target rejection = cross-talk. `*` marks the target cell.",
        "",
        "| latent space | mode | delta | angle | shape | group-in-stage |",
        "|--------------|------|-------|-------|-------|----------------|",
    ]
    for label, _ in _LATENT_SPACES:
        for mode in _MODES:
            r = reports[label][mode]
            target = TARGET_STATISTIC.get(mode)
            cells = {}
            for stat in ("delta", "angle", "shape"):
                value = r.rejection_rates[stat]  # type: ignore[attr-defined]
                text = "—" if value != value else f"{value:.2f}"  # nan check
                cells[stat] = f"**{text}**" if stat == target else text
            lines.append(
                f"| {label} | `{mode}` | {cells['delta']} | {cells['angle']} | "
                f"{cells['shape']} | {r.group_in_stage_fraction:.2f} |"  # type: ignore[attr-defined]
            )
    lines.append("")
    return lines


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ref = load_reference()
    common = dict(
        n_replicates=args.reps,
        n_samples=args.n_samples,
        n_stages=args.n_stages,
        effect_size=args.effect_size,
        p_dmp=args.p_dmp,
        permutations=args.perms,
        n_jobs=args.n_jobs,
        base_seed=args.base_seed,
        reference=ref,
    )

    reports: dict[str, dict[str, object]] = {}
    for label, method in _LATENT_SPACES:
        reports[label] = {}
        int_params = _integration_params(label, method, args)
        for mode in _MODES:
            print(f"[{label}] {mode}...", file=sys.stderr)
            reports[label][mode] = evaluate_mode_specificity(
                mode,  # type: ignore[arg-type]
                integration_method=method,
                integration_params=int_params,
                **common,
            )

    out = [
        "# Rung 3 — cross-talk through the production latent spaces",
        "",
        f"Run: {args.reps} reps, {args.perms} perms, `n_samples={args.n_samples}`, "
        f"`n_stages={args.n_stages}`, `effect_size={args.effect_size}`, `p_dmp={args.p_dmp}`, "
        f"serial (`n_jobs=1`). PLS CV: repeats={args.pls_repeats}, cv2={args.pls_cv2}, "
        f"cv1={args.pls_cv1}, max_components={args.pls_max_components}.",
        "",
    ]
    out += _table(reports)

    report = "\n".join(out)
    if args.out:
        Path(args.out).write_text(report)
        print(f"Wrote {args.out}", file=sys.stderr)
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
