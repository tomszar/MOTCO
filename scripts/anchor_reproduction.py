#!/usr/bin/env python3
"""Compare a study's shared zero-effect anchor against a reference run's anchor.

Two configs that share ``base_seed``, the primary matched-seed family, and the
generator parameters (other than parameters no mode reads at zero effect, such
as ``magnitude_kind``) generate the same anchor dataset at every replicate index
and evaluate it with the same RRPP draws. Their anchor records therefore
reproduce each other — the observed ``delta``/``angle``/``shape``, the RRPP
p-values, the selected PLS rank, and the generator seed — even though their
``parameter_signature`` differs (the dataclass and the resolved-mode set do).

This script joins the two anchors on ``replicate_index`` and writes one row per
replicate with both sides' values and their absolute differences, plus a summary
line on stdout. Statistics are compared at a relative tolerance (BLAS last-bit
differences between hosts are expected); seeds must match exactly, and p-values
must match exactly when both runs used the same permutation count (the RRPP
draws are then identical). When the counts differ (a 199-permutation pilot
against a 999-permutation reference) the p-values are reported but excluded
from the match verdict, since they cannot agree beyond Monte Carlo resolution.

Example:

    python scripts/anchor_reproduction.py \\
        --candidate results/phase5-magnitude-2026-10-08/merged.jsonl \\
        --reference results/phase5-2026-09-10/merged.jsonl \\
        --out results/phase5-magnitude-2026-10-08/report/anchor_reproduction.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

STATISTICS = ("delta", "angle", "shape")


def _anchor_rows(path: Path, cell_id: str | None) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if cell_id is not None:
                if record.get("cell_id") != cell_id:
                    continue
            elif not (record.get("cell_metadata") or {}).get("zero_effect_anchor"):
                continue
            row = {
                "cell_id": record["cell_id"],
                "replicate_index": int(record["replicate_index"]),
                "status": record.get("status"),
                "generator_seed": record.get("generator_seed"),
                "selected_lv": (record.get("integration_metadata") or {}).get("selected_lv"),
                "permutations": (record.get("runtime_metadata") or {}).get("permutations"),
            }
            for statistic in STATISTICS:
                row[f"observed_{statistic}"] = (record.get("pair_statistics") or {}).get(statistic)
                row[f"p_{statistic}"] = (record.get("p_values") or {}).get(statistic)
            rows.append(row)
    if not rows:
        raise SystemExit(f"No anchor records found in {path}")
    frame = pd.DataFrame(rows)
    if frame["cell_id"].nunique() != 1:
        raise SystemExit(f"{path}: anchor records span several cells: {sorted(frame['cell_id'].unique())}")
    return frame.sort_values("replicate_index").reset_index(drop=True)


def compare(candidate: pd.DataFrame, reference: pd.DataFrame, rtol: float) -> pd.DataFrame:
    merged = candidate.merge(
        reference, on="replicate_index", how="inner", suffixes=("_candidate", "_reference")
    )
    merged["seed_match"] = merged["generator_seed_candidate"] == merged["generator_seed_reference"]
    merged["lv_match"] = merged["selected_lv_candidate"] == merged["selected_lv_reference"]
    for statistic in STATISTICS:
        c = merged[f"observed_{statistic}_candidate"].to_numpy(dtype=float)
        r = merged[f"observed_{statistic}_reference"].to_numpy(dtype=float)
        merged[f"abs_diff_{statistic}"] = np.abs(c - r)
        merged[f"{statistic}_match"] = np.isclose(c, r, rtol=rtol, atol=0.0)
        merged[f"p_{statistic}_match"] = (
            merged[f"p_{statistic}_candidate"] == merged[f"p_{statistic}_reference"]
        )
    merged["p_values_comparable"] = merged["permutations_candidate"] == merged["permutations_reference"]
    columns = ["seed_match", "lv_match"] + [f"{s}_match" for s in STATISTICS]
    merged["all_match"] = merged[columns].all(axis=1) & (
        ~merged["p_values_comparable"] | merged[[f"p_{s}_match" for s in STATISTICS]].all(axis=1)
    )
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", type=Path, required=True, help="Merged JSONL of the run under check.")
    parser.add_argument(
        "--reference", type=Path, required=True, help="Merged JSONL (or anchor extract) of the reference run."
    )
    parser.add_argument(
        "--candidate-cell", default=None, help="Anchor cell id in the candidate (default: the zero_effect_anchor cell)."
    )
    parser.add_argument(
        "--reference-cell", default=None, help="Anchor cell id in the reference (default: the zero_effect_anchor cell)."
    )
    parser.add_argument("--rtol", type=float, default=1e-6, help="Relative tolerance for the observed statistics.")
    parser.add_argument("--out", type=Path, required=True, help="CSV path for the per-replicate comparison.")
    args = parser.parse_args()

    candidate = _anchor_rows(args.candidate, args.candidate_cell)
    reference = _anchor_rows(args.reference, args.reference_cell)
    merged = compare(candidate, reference, args.rtol)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)

    n = len(merged)
    print(
        f"candidate {candidate['cell_id'].iat[0]} ({len(candidate)} records) vs "
        f"reference {reference['cell_id'].iat[0]} ({len(reference)} records); "
        f"{n} replicate indices in common"
    )
    print(
        f"  seeds identical: {int(merged['seed_match'].sum())}/{n}; "
        f"selected rank identical: {int(merged['lv_match'].sum())}/{n}"
    )
    for statistic in STATISTICS:
        within = int(merged[f"{statistic}_match"].sum())
        same_p = int(merged[f"p_{statistic}_match"].sum())
        print(
            f"  {statistic}: observed within rtol {args.rtol:g} on {within}/{n} "
            f"(max abs diff {merged[f'abs_diff_{statistic}'].max():.3g}); p-values identical on {same_p}/{n}"
        )
    if not bool(merged["p_values_comparable"].all()):
        candidate_perms = sorted(int(v) for v in merged["permutations_candidate"].unique())
        reference_perms = sorted(int(v) for v in merged["permutations_reference"].unique())
        print(
            f"  p-values not comparable (permutation counts {candidate_perms} vs {reference_perms}); "
            "excluded from the verdict"
        )
    print(f"  all comparable fields match: {int(merged['all_match'].sum())}/{n}")
    print(f"Wrote {args.out}")
    return 0 if bool(merged["all_match"].all()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
