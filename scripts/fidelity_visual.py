#!/usr/bin/env python3
"""Render the qualitative numpy-vs-InterSIM fidelity figures (no R).

Reads the committed InterSIM visual fixture
(``tests/data/intersim_visual_fixture.npz``), generates matched numpy data, and
writes the density / heatmap / PCA / moment-scatter / coupling comparison PNGs.

Example:

    python scripts/fidelity_visual.py --out-dir build/fidelity_visual
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from motco.simulations.fidelity_visual import VISUAL_FIXTURE_NAME, run_fidelity_visual

# The visual fixture holds raw InterSIM matrices and is NOT committed (it needs
# InterSIM, available via flake.nix). Regenerate it locally — see FIDELITY.md —
# under build/ (gitignored) by default.
DEFAULT_FIXTURE = Path("build/fidelity_visual") / VISUAL_FIXTURE_NAME


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fidelity_visual", description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE,
                        help=f"InterSIM visual fixture (default: {DEFAULT_FIXTURE}).")
    parser.add_argument("--out-dir", type=Path, default=Path("build/fidelity_visual"),
                        help="Output directory for the figures.")
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI (default: 150).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    saved = run_fidelity_visual(args.fixture, args.out_dir, dpi=args.dpi)
    print(f"Wrote {len(saved)} figures to {args.out_dir}:")
    for name, path in saved.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())