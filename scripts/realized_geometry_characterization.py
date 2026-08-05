#!/usr/bin/env python3
"""Run the matched-seed Phase 2 realized-geometry characterization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from motco.simulations.realized_geometry_study import (
    Phase2CharacterizationConfig,
    monotonicity_summary,
    run_phase2_characterization,
    summarize_phase2_characterization,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated matched seeds")
    parser.add_argument("--effects", default="0,0.25,0.5,0.75,1", help="Comma-separated effect sizes")
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--n-stages", type=int, default=4)
    parser.add_argument("--p-dmp", type=float, default=0.2)
    parser.add_argument("--pls-repeats", type=int, default=3)
    parser.add_argument("--pls-cv1", type=int, default=3)
    parser.add_argument("--pls-cv2", type=int, default=4)
    parser.add_argument("--pls-max-components", type=int, default=15)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = Phase2CharacterizationConfig(
        effect_sizes=tuple(float(value) for value in args.effects.split(",")),
        seeds=tuple(int(value) for value in args.seeds.split(",")),
        n_samples=args.n_samples,
        n_stages=args.n_stages,
        p_dmp=args.p_dmp,
        integration_params={
            "n_repeats": args.pls_repeats,
            "cv1_splits": args.pls_cv1,
            "cv2_splits": args.pls_cv2,
            "max_components": args.pls_max_components,
        },
    )
    rows = run_phase2_characterization(config)
    summary = summarize_phase2_characterization(rows)
    monotonicity = monotonicity_summary(summary)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "replicates.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    monotonicity.to_csv(args.out_dir / "monotonicity.csv", index=False)
    (args.out_dir / "config.json").write_text(
        json.dumps(
            {
                "effect_sizes": config.effect_sizes,
                "seeds": config.seeds,
                "n_samples": config.n_samples,
                "n_stages": config.n_stages,
                "p_dmp": config.p_dmp,
                "integration_params": dict(config.integration_params),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
