from __future__ import annotations

import numpy as np
import pandas as pd

from motco.stats.sd import pair_difference


def _feature_columns(df: pd.DataFrame, group_col: str, level_col: str) -> list[str]:
    return [
        c for c in df.select_dtypes(include=[np.number]).columns if c not in {group_col, level_col}
    ]


def test_example1_expected_results_match(data_dir):
    # Fixed schema for example1
    csv_path = data_dir / "evo_649_sm_example1.csv"
    df = pd.read_csv(csv_path)
    group_col = "taxa"
    level_col = "Inv"
    feat_cols = _feature_columns(df, group_col, level_col)

    # Load ground-truth pairs and expected metrics
    gt_path = data_dir / "results_example1.csv"
    gt = pd.read_csv(gt_path)
    assert {"group 1", "group 2", "angle", "magnitude"}.issubset(gt.columns)

    # Use the canonical level order present in example1
    levels = ("I1", "I2")

    # Compare each listed pair to our computation
    for _, row in gt.iterrows():
        g1 = str(row["group 1"])  # e.g., "t1"
        g2 = str(row["group 2"])  # e.g., "t3"
        exp_angle = float(row["angle"])  # degrees
        exp_mag = float(row["magnitude"])  # delta magnitude difference

        angle, delta = pair_difference(
            df,
            group_col=group_col,
            level_col=level_col,
            groups=(g1, g2),
            levels=levels,
            feature_cols=feat_cols,
        )

        print(angle, exp_angle)
        print(delta, exp_mag)

        # Allow for minor numerical differences; CSV has limited precision
        assert np.isclose(angle, exp_angle, atol=1e-3), (
            f"Angle mismatch for {g1} vs {g2}: got {angle:.5f}, expected {exp_angle:.5f}"
        )
        assert np.isclose(delta, exp_mag, atol=1e-3), (
            f"Magnitude mismatch for {g1} vs {g2}: got {delta:.5f}, expected {exp_mag:.5f}"
        )
