#!/usr/bin/env python3
"""Compare Phase 4 against the corrected-orientation re-run.

Answers tasks 4.3-4.5 of `openspec/changes/fix-orientation-sign-anchor`:

- 4.3 observed-angle distribution per cell, and specifically whether the
  150-180 degree mass has left the null cells;
- 4.4 rejection rates per cell and statistic, with Monte Carlo standard errors;
- 4.5 the realized-geometry orientation checkpoints (population and latent).

Usage:
    python scripts/orientation_signfix_analysis.py \
        --baseline results/phase4-2026-08-27/merged.jsonl \
        --corrected results/orientation-signfix-2026-08-28/merged.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

STATISTICS = ("delta", "angle", "shape")
NULL_MODES = ("none", "translation")
ANGLE_BANDS = ((0, 30), (30, 60), (60, 90), (90, 135), (135, 150), (150, 180.0001))


def load(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def cell_key(record: dict[str, Any]) -> tuple[str, str, float]:
    meta = record.get("cell_metadata") or {}
    mode = meta.get("trajectory_mode") or "none"
    effect = meta.get("effect_size")
    return (record["phase"], mode, 0.0 if effect is None else float(effect))


def group_cells(records: list[dict[str, Any]]) -> dict[tuple[str, str, float], list[dict]]:
    cells: dict[tuple[str, str, float], list[dict]] = defaultdict(list)
    for record in records:
        cells[cell_key(record)].append(record)
    return cells


def rejection_rate(records: list[dict], statistic: str, alpha: float) -> tuple[float, float, int]:
    values = [
        r["p_values"][statistic]
        for r in records
        if r.get("status") == "completed" and (r.get("p_values") or {}).get(statistic) is not None
    ]
    if not values:
        return (float("nan"), float("nan"), 0)
    n = len(values)
    # Strict `<`, matching the study's own convention
    # (`motco.simulations.grid._is_rejection`, `study/summary.py`).
    rate = sum(1 for v in values if v < alpha) / n
    return (rate, math.sqrt(max(rate * (1 - rate), 0.0) / n), n)


def observed_angles(records: list[dict]) -> list[float]:
    return [
        r["pair_statistics"]["angle"]
        for r in records
        if r.get("status") == "completed" and (r.get("pair_statistics") or {}).get("angle") is not None
    ]


def band_counts(angles: list[float]) -> list[int]:
    return [sum(1 for a in angles if lo <= a < hi) for lo, hi in ANGLE_BANDS]


def checkpoint_angle(record: dict, checkpoint: str, scope: str = "joint") -> float | None:
    geometry = record.get("realized_geometry") or {}
    node = ((geometry.get("checkpoints") or {}).get(checkpoint) or {}).get(scope) or {}
    if not (node.get("availability") or {}).get("angle"):
        return None
    return node.get("angle")


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def failed(records: list[dict]) -> int:
    return sum(1 for r in records if r.get("status") != "completed")


def report(baseline: Path, corrected: Path, alpha: float) -> None:
    runs = {"baseline (Phase 4)": group_cells(load(baseline)),
            "corrected": group_cells(load(corrected))}
    shared = sorted(set(runs["baseline (Phase 4)"]) & set(runs["corrected"]), key=str)

    print("=" * 96)
    print("COMPLETENESS")
    print("=" * 96)
    for label, cells in runs.items():
        total = sum(len(v) for v in cells.values())
        bad = sum(failed(v) for v in cells.values())
        print(f"{label:22s} {len(cells):3d} cells  {total:5d} replicates  {bad:3d} failed")

    print()
    print("=" * 96)
    print("4.3  OBSERVED-ANGLE DISTRIBUTION  (bands: <30 | 30-60 | 60-90 | 90-135 | 135-150 | 150-180)")
    print("=" * 96)
    bands = " ".join(f"{f'{lo}-{hi:g}':>8s}" for lo, hi in ANGLE_BANDS)
    print(f"{'cell':40s} {'run':22s} " + bands + f" {'max':>8s}")
    null_totals = {label: [0] * len(ANGLE_BANDS) for label in runs}
    null_max = {label: 0.0 for label in runs}
    for key in shared:
        name = f"{key[0]}/{key[1]}/{key[2]:g}"
        for label, cells in runs.items():
            angles = observed_angles(cells[key])
            counts = band_counts(angles)
            hi = max(angles) if angles else float("nan")
            print(f"{name:40s} {label:22s} " + " ".join(f"{c:8d}" for c in counts) + f" {hi:8.1f}")
            if key[1] in NULL_MODES:
                null_totals[label] = [a + b for a, b in zip(null_totals[label], counts, strict=True)]
                null_max[label] = max(null_max[label], hi if angles else 0.0)
        print()

    print("-" * 96)
    print("NULL CELLS POOLED (`none` + `translation`) — the artifact check")
    for label in runs:
        total = sum(null_totals[label])
        print(f"{'':40s} {label:22s} " + " ".join(f"{c:8d}" for c in null_totals[label])
              + f" {null_max[label]:8.1f}   (n={total})")

    print()
    print("=" * 96)
    print(f"4.4  REJECTION RATES  (alpha={alpha})   rate [+/- 1 MC SE]")
    print("=" * 96)
    print(f"{'cell':40s} {'statistic':10s} {'baseline':>18s} {'corrected':>18s}")
    for key in shared:
        name = f"{key[0]}/{key[1]}/{key[2]:g}"
        for statistic in STATISTICS:
            b_rate, b_se, _ = rejection_rate(runs["baseline (Phase 4)"][key], statistic, alpha)
            c_rate, c_se, _ = rejection_rate(runs["corrected"][key], statistic, alpha)
            print(f"{name:40s} {statistic:10s} {b_rate:8.3f} +/-{b_se:5.3f} {c_rate:8.3f} +/-{c_se:5.3f}")
        print()

    print("=" * 96)
    print("4.5  REALIZED-GEOMETRY ORIENTATION CHECKPOINTS (joint scope, mean degrees)")
    print("=" * 96)
    print(f"{'cell':40s} {'checkpoint':26s} {'baseline':>10s} {'corrected':>10s}")
    for key in shared:
        name = f"{key[0]}/{key[1]}/{key[2]:g}"
        for checkpoint in ("population_standardized", "observed_standardized", "pls_latent"):
            vals = {}
            for label, cells in runs.items():
                angles = [a for a in (checkpoint_angle(r, checkpoint) for r in cells[key]) if a is not None]
                vals[label] = mean(angles)
            print(f"{name:40s} {checkpoint:26s} "
                  f"{vals['baseline (Phase 4)']:10.1f} {vals['corrected']:10.1f}")
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--corrected", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args(argv)
    report(args.baseline, args.corrected, args.alpha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
