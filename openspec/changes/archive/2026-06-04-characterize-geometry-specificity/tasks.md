## 1. 2-stage isolation (shape-free)

- [x] 1.1 Add a 2-stage characterization run to `specificity.py` (magnitude, orientation, none, translation at `n_stages=2`)
- [x] 1.2 Record whether `magnitude`→`delta` and `orientation`→`angle` are clean when Procrustes shape is degenerate; note the contrast with the 3-stage matrix

## 2. Shape-statistic investigation

- [x] 2.1 Instrument the `shape` path to capture, per replicate, the observed Procrustes distance and the permutation-null spread/quantiles (not just the rejection)
- [x] 2.2 Quantify the null-mode `shape` rejection vs α across more replicates (is the ~0.17 anti-conservative?)
- [x] 2.3 Re-run the shape probe under raw / concat-standardize / SNF integration to isolate the standardization's role in breaking scale-invariance
- [x] 2.4 Conclude: genuine MOTCO property vs calibration issue, with evidence
- [x] 2.5 If calibration: recalibrate the statistic / RRPP null (`stats/trajectory.py` / `permutation.py` / `evaluation.py`) and re-verify; if genuine: document the finding and the study caveat

## 3. Magnitude endpoints variant

- [x] 3.1 Add `magnitude_kind` (`all` default, `extremes`) to `SemiSyntheticTrajectoryParams` and the magnitude construction
- [x] 3.2 Record `magnitude_kind` in truth; expose via the `simulate` CLI
- [x] 3.3 Characterize whether `extremes` reduces shape co-movement relative to `all`

## 4. Wrap-up

- [x] 4.1 Update tests across the touched modules; ruff + mypy + fast pytest green with no R
- [x] 4.2 Record the consolidated findings (2-stage, shape calibration, magnitude variant) for the deferred paper-grade study; update docs if behavior changed
