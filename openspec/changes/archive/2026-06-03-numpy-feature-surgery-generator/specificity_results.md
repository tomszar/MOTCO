# Specificity characterization (task 5) — descriptive

This is a **descriptive** characterization of how MOTCO responds to each
realistic, methylation-only trajectory difference — not a pass/fail gate.
Cross-talk and non-detection are findings, not failures.

Instrumentation: `src/motco/simulations/specificity.py`.
Run: 12 replicates/mode, `n_samples=160`, `n_stages=4`, `effect_size=1.0`,
`permutations=49`, `n_jobs=1`, `alpha=0.05`.

RRPP rejection rates (fraction of replicates rejecting at α=0.05) under the
redesigned **methylation-only** constructions (gene/protein re-derived via the
CpG→gene→protein cascade):

| mode               | delta | angle | shape | group-in-stage |
|--------------------|-------|-------|-------|----------------|
| `none`             | 0.08  | 0.00  | 0.17  | 0.13           |
| `translation`      | 0.33  | 0.08  | 0.17  | 0.05           |
| `magnitude`        | 1.00  | 0.00  | 1.00  | 0.10           |
| `orientation`      | 0.17  | 0.42  | 1.00  | 0.07           |
| `shape` (relocate) | 0.83  | 0.50  | 1.00  | 0.21           |
| `shape` (magnitude)| 1.00  | 0.42  | 1.00  | 0.43           |

## What each mode does to MOTCO

- **`none`** — clean null on `delta`/`angle`; `shape` sits slightly above α
  (~0.17), see the shape-statistic note below.
- **`translation`** — now essentially location-only: `angle` and `shape` are at
  the null floor; `delta` is mildly elevated (0.33). The earlier total leak
  (1.0/0.08/1.0) was traced to **CpG→gene saturation**: the extra constant set
  `U`, derived via the OR-rule (a gene is differential if *any* mapped CpG is),
  flattened ~30 of A's stage-varying genes in B, deforming the derived
  trajectory. Fix: draw `U` from CpGs whose mapped gene is *absent* from the
  stage program (an independent baseline gene program). This is the realistic
  reading of "a set that changes A/B *besides* the stage program".
- **`magnitude`** — hits `delta` (1.0); also moves `shape` (1.0): scaling δ
  through methylation's `rev.logit` is nonlinear, so a "uniform" methylation
  scale is not a uniform scale in observed space → shape co-moves (expected).
- **`orientation`** — `angle` (0.42) leads `delta` (0.17): the consistent
  relocation does rotate the path. But `shape` (1.0) dominates — per-feature
  standardization means relocating to a CpG with a different SD is not an
  isometry in the space MOTCO measures, and the cascade adds further deformation.
- **`shape`** — hits `shape` (1.0) in both flavors; `magnitude` flavor also
  saturates `delta` (1.0, by construction), `relocate` leaks less into `delta`
  (0.83).

## Open question: the `shape` statistic is near-saturated

`shape` (Procrustes GPA) rejects ≈1.0 for **every** geometry mode (magnitude,
orientation, shape) and sits at ~0.17 even under the null. In this
integrated + per-feature-standardized setting, essentially any methylation
change deforms Procrustes shape, and the null is mildly anti-conservative.
Whether this is a genuine property of MOTCO's shape test on integrated
multi-omic data (a valuable finding) or a calibration issue (integration /
standardization / the RRPP shape null) is the main thing to resolve before
reading the power study's specificity matrix as definitive.

The other statistics behave sensibly: `delta` is specific to size changes
(magnitude, shape-magnitude), `angle` responds to orientation, and the negative
controls (`none`, `translation`) stay near α on `angle`/`shape`.
