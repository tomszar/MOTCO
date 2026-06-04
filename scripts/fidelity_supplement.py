#!/usr/bin/env python3
"""Render the numpy-vs-InterSIM fidelity supplement (table + figure).

Reads the committed InterSIM fixture (``tests/data/intersim_fidelity_fixture.npz``),
runs the numpy generator across the same grid, and writes a paper-ready
supplementary table (CSV + Markdown) and figure summarising fidelity. No R is
needed -- the InterSIM side comes from the committed fixture.

Example:

    python scripts/fidelity_supplement.py --out-dir build/fidelity_supplement
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from motco.simulations.fidelity import (
    FIDELITY_FIXTURE_NAME,
    load_fidelity_fixture,
    validate_grid,
)
from motco.simulations.generator import OMIC_LAYERS

DEFAULT_FIXTURE = Path("tests/data") / FIDELITY_FIXTURE_NAME
HIGHLIGHT_STATS = ("eta2", "diff_rate")


def build_table(results) -> list[dict[str, object]]:
    """Flatten the comparison results into table rows."""

    rows: list[dict[str, object]] = []
    for (delta, p_dmp), comparisons in results.items():
        for name, c in comparisons.items():
            rows.append(
                {
                    "delta": delta,
                    "p_dmp": p_dmp,
                    "statistic": name,
                    "intersim_mean": round(c.intersim_mean, 6),
                    "intersim_interval_low": round(c.interval_low, 6),
                    "intersim_interval_high": round(c.interval_high, 6),
                    "numpy_mean": round(c.numpy_mean, 6),
                    "within_interval": c.passed,
                }
            )
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    header = list(rows[0].keys())
    lines = [",".join(header)]
    lines.extend(",".join(str(r[h]) for h in header) for r in rows)
    path.write_text("\n".join(lines) + "\n")


def write_markdown(rows: list[dict[str, object]], provenance: dict, path: Path) -> None:
    n_pass = sum(1 for r in rows if r["within_interval"])
    header = list(rows[0].keys())
    lines = [
        "# Generator fidelity: numpy vs InterSIM",
        "",
        f"InterSIM {provenance.get('intersim_version', '?')} "
        f"(R {provenance.get('r_version', '?')}), generated "
        f"{provenance.get('generation_date', '?')}; "
        f"n_sample={provenance.get('n_sample', '?')}, "
        f"n_intersim={provenance.get('n_intersim', '?')} replicates/cell. "
        "Pass = numpy replicate mean within InterSIM's [q2.5, q97.5] interval.",
        "",
        f"**{n_pass}/{len(rows)} statistics within interval.**",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(str(r[h]) for h in header) + " |" for r in rows)
    path.write_text("\n".join(lines) + "\n")


def build_figure(results, deltas, p_dmps, path: Path) -> None:
    """Highlight-statistic figure: InterSIM intervals (bands) vs numpy points."""

    fig, axes = plt.subplots(
        len(OMIC_LAYERS), len(HIGHLIGHT_STATS),
        figsize=(4 * len(HIGHLIGHT_STATS), 3 * len(OMIC_LAYERS)),
        squeeze=False,
    )
    colors = plt.cm.viridis(np.linspace(0.15, 0.8, len(p_dmps)))
    for r, omic in enumerate(OMIC_LAYERS):
        for col, suffix in enumerate(HIGHLIGHT_STATS):
            ax = axes[r][col]
            stat = f"{omic}_{suffix}"
            for pi, p_dmp in enumerate(p_dmps):
                lo = [results[(d, p_dmp)][stat].interval_low for d in deltas]
                hi = [results[(d, p_dmp)][stat].interval_high for d in deltas]
                inter = [results[(d, p_dmp)][stat].intersim_mean for d in deltas]
                npm = [results[(d, p_dmp)][stat].numpy_mean for d in deltas]
                ax.fill_between(deltas, lo, hi, alpha=0.2, color=colors[pi])
                ax.plot(deltas, inter, "-", color=colors[pi], label=f"InterSIM p={p_dmp}")
                ax.plot(deltas, npm, "o", color=colors[pi], mec="black",
                        label=f"numpy p={p_dmp}")
            ax.set_title(stat)
            ax.set_xlabel("delta")
            if r == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("numpy generator within InterSIM's sampling interval")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fidelity_supplement", description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE,
                        help=f"InterSIM fidelity fixture (default: {DEFAULT_FIXTURE}).")
    parser.add_argument("--out-dir", type=Path, default=Path("build/fidelity_supplement"),
                        help="Output directory for the table + figure.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fixture = load_fidelity_fixture(args.fixture)
    results = validate_grid(fixture)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_table(results)
    write_csv(rows, args.out_dir / "fidelity_table.csv")
    write_markdown(rows, fixture.provenance, args.out_dir / "fidelity_table.md")
    build_figure(results, list(fixture.grid.deltas), list(fixture.grid.p_dmps),
                 args.out_dir / "fidelity_figure.png")

    n_pass = sum(1 for r in rows if r["within_interval"])
    print(f"Wrote supplement to {args.out_dir} ({n_pass}/{len(rows)} within interval)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
