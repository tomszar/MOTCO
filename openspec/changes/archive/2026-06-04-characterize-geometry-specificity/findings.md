# Geometry-specificity findings

Consolidated results for the `characterize-geometry-specificity` change. The
three probes are driven by `scripts/geometry_specificity_probe.py` over
`src/motco/simulations/specificity.py` (`characterize_two_stage`,
`evaluate_shape_null`, and `evaluate_mode_specificity(..., magnitude_kind=...)`).

Run: **12 replicates, 49 RRPP permutations, `n_samples=160`, `effect_size=1.0`,
`p_dmp=0.2`, serial (`n_jobs=1`)**. Reproduce with:

```bash
python scripts/geometry_specificity_probe.py \
    --reps 12 --perms 49 --n-samples 160 --out findings_probe.md
```

These remain **descriptive** characterizations — not pass/fail gates. The
cluster-run study produces the definitive matrix and power curves.

---

## 1. 2-stage isolation (shape-free) — tasks 1.1 / 1.2

With two stages the trajectory is a single step, so Procrustes `shape` is
degenerate and never counts. This isolates size/orientation with shape removed.

| mode          | delta | angle | group-in-stage |
|---------------|-------|-------|----------------|
| `none`        | 0.00  | 0.17  | 0.08           |
| `translation` | 0.00  | 0.00  | 0.03           |
| `magnitude`   | 1.00  | 1.00  | 0.04           |
| `orientation` | 0.17  | 1.00  | 0.04           |

**Reading.**

- **`orientation`→`angle` is clean** (angle 1.00, delta 0.17): a consistent
  relocation rotates the single-step direction without changing its size.
- **`translation`** is a clean location-only control (0.00 / 0.00).
- **`magnitude`→`delta` is *not* clean here**: it hits `delta` (1.00) but also
  saturates `angle` (1.00). Removing shape did **not** decontaminate magnitude —
  it leaks into orientation instead.

**Contrast with the 3/4-stage matrix.** At `n_stages=4` (prior change), magnitude
shows `angle=0.00`. The difference is in how orientation is measured: the
multi-stage `_estimate_orientation` takes the principal axis of 4 trajectory
points, and a uniform δ scale enlarges all points proportionally → the principal
direction is ~unchanged. With a single step, "orientation" *is* that step's
direction, and scaling the methylation effect bends it through the `rev.logit`
nonlinearity → `angle` moves. So the 2-stage regime exposes a directional effect
of the magnitude nonlinearity that the multi-stage orientation statistic absorbs.
Net: orientation specificity is confirmed shape-free; magnitude is size-specific
only once enough stages exist for the principal-axis orientation to average over
the per-step nonlinearity.

---

## 2. Shape-statistic investigation — tasks 2.1–2.5

`evaluate_shape_null` splits the saturated `shape` rejection into the observed
Procrustes distance vs the spread/quantiles of its RRPP permutation null, under
raw concat / concat-standardize / SNF integration.

| integration | mode          | observed | null q2.5 | null med | null q97.5 | null sd | reject |
|-------------|---------------|----------|-----------|----------|------------|---------|--------|
| concat-std  | `none`        | 0.310    | 0.259     | 0.295    | 0.367      | 0.030   | 0.08   |
| concat-std  | `magnitude`   | 0.545    | 0.254     | 0.288    | 0.348      | 0.026   | 1.00   |
| concat-std  | `orientation` | 1.245    | 0.303     | 0.353    | 0.421      | 0.032   | 1.00   |
| concat-std  | `shape`       | 0.838    | 0.264     | 0.306    | 0.368      | 0.028   | 1.00   |
| raw-concat  | `none`        | 0.227    | 0.206     | 0.226    | 0.249      | 0.012   | 0.17   |
| raw-concat  | `magnitude`   | 0.276    | 0.205     | 0.225    | 0.248      | 0.012   | 0.83   |
| raw-concat  | `orientation` | 1.217    | 0.299     | 0.341    | 0.410      | 0.031   | 1.00   |
| raw-concat  | `shape`       | 0.754    | 0.230     | 0.259    | 0.301      | 0.020   | 1.00   |
| snf         | `none`        | 0.401    | 0.295     | 0.391    | 0.497      | 0.055   | 0.25   |
| snf         | `magnitude`   | 1.358    | 0.279     | 0.370    | 0.481      | 0.056   | 1.00   |
| snf         | `orientation` | 1.414    | 0.263     | 0.367    | 0.476      | 0.057   | 1.00   |
| snf         | `shape`       | 0.941    | 0.290     | 0.388    | 0.483      | 0.052   | 1.00   |

### 2.2 Is the null anti-conservative?

Under the **null** (`none`), shape rejection is **integration-dependent and
modest**: concat-standardize (the production default) is essentially calibrated
at **0.08 ≈ α**; raw-concat is **0.17** and SNF **0.25** — mildly
anti-conservative. The null is **not collapsing**: in every cell the permutation
null has a healthy median (~0.23–0.39) and spread (sd ~0.01–0.06). The earlier
"~0.17 null" reported in the prior change is reproduced as the *raw/SNF* figure;
the concat-standardize path used by the pipeline is calibrated.

### 2.3 Integration sensitivity — is standardization the culprit?

**No.** Raw concat (no per-feature standardization) still **saturates** `shape`
for `orientation` (1.00) and `shape` (1.00), and largely for `magnitude` (0.83).
Removing standardization does not restore Procrustes scale-invariance — so
per-feature standardization is **not** the root cause of the saturation.

Standardization does *amplify* one specific leak: magnitude's observed distance
moves from borderline at raw (0.276 vs null q97.5 0.248 → reject 0.83) to clearly
extreme at concat-std (0.545 vs 0.348 → reject 1.00). So standardization sharpens
magnitude→shape co-movement, but the general geometry→shape saturation is present
without it.

### 2.4 Conclusion: genuine property, not a calibration bug

The shape saturation is a **genuine property** of MOTCO's Procrustes shape test
on integrated multi-omic LS-means, evidenced by:

1. **Observed distances are genuinely extreme.** For every geometry mode under
   all three integrations the observed group-vs-group Procrustes distance sits
   *far above* the RRPP null q97.5 (e.g. orientation 1.245 vs 0.421). The
   rejection is driven by the observed statistic, not by a degenerate null.
2. **The null is well-behaved**, not collapsed (median ~0.3, sd ~0.03).
3. **The production path is calibrated** under the null (concat-standardize
   0.08 ≈ α); the mild anti-conservativeness at raw/SNF (0.17/0.25) is
   integration-dependent and modest.

Interpretation: on integrated multi-omic trajectories, **Procrustes shape is a
sensitive but non-specific omnibus geometry detector** — any size, orientation,
or bend that deforms the multi-stage LS-mean configuration registers as a shape
difference. This is biologically reasonable (a real methylation cascade rarely
moves one geometric facet in isolation) and is the property to *report*, not fix.

### 2.5 Action: document, no recalibration

Because the saturation is genuine (not a collapsed null and not a standardization
artifact), **no change to `stats/trajectory.py` / `permutation.py` /
`evaluation.py` is warranted.** The study should carry the caveat that the
`shape` row is an omnibus geometry signal: read `delta` (size) and `angle`
(orientation) for the *specific* facets, and `shape` as "geometry changed at
all." If a strictly shape-specific test is ever needed, that is a new-statistic
proposal (e.g. size-and-orientation-residualized Procrustes), out of scope here.

---

## 3. Magnitude endpoints variant — tasks 3.1–3.3

`magnitude_kind='extremes'` scales group B's methylation indicators only at the
first and last stages (leaving δ and the interior stages untouched), vs the
default `'all'` (global δ scale). At `n_stages=4`:

| magnitude_kind | delta | angle | shape |
|----------------|-------|-------|-------|
| `all`          | 1.00  | 0.00  | 1.00  |
| `extremes`     | 1.00  | 0.33  | 1.00  |

**`extremes` does not reduce shape co-movement** (still 1.00) and mildly
*increases* `angle` (0.00 → 0.33). Confining the scale to the endpoints does not
help: the shape co-movement is driven by the `rev.logit` nonlinearity (scaling
the methylation effect is not an isometry in observed space), **independent of
which stages are scaled**. Localizing to the endpoints additionally bends the
principal-axis orientation a little (endpoints move more than the interior), so
it trades a touch of orientation leak for no shape benefit. This is the expected
negative result anticipated in `design.md` — an informative, cheap probe that
confirms the nonlinearity (not the stage profile) is the mechanism.

---

## 4. Latent-space / frame-choice caveats (exploratory follow-up)

Not part of the original task list — a discussion-driven probe of *how the
measurement frame interacts with the `rev.logit` nonlinearity*. Numbers below are
single-seed / few-rep illustrations, not the multi-rep matrix. **Open question,
not a settled methodology.**

### 4.1 Mechanism: why a "pure size" change rotates the trajectory
Methylation is injected as an additive shift in **M-value (logit) space**
(`μ = base + δ·v`), then squashed by `rev.logit` into β-space. A *uniform* δ
scale (`magnitude`) is therefore **not** a uniform scale in observed β-space: the
sigmoid's local gain differs per CpG (it depends on that CpG's baseline), so
doubling δ enlarges low-baseline (steep) CpGs more than high-baseline (saturated)
ones. The per-CpG re-proportioning **rotates** the observed step vector. Worked
10-CpG example: three sites scaled ×2 in M-space (ratio exactly 2.0) come out at
β-space step ratios 1.59 / 2.49 / 2.29 → size ×2.0 **and** an 11° rotation. The
condition for *no* rotation is all differential CpGs sharing one baseline; real
CpGs span the range, so a tilt is unavoidable. This is also why `magnitude`
co-moves `shape`, and why the endpoints-only `magnitude_kind` does not help.

### 4.2 PLS loadings split by baseline (methylation-only PLS-DA on stage)
In a 2-component PLS-DA(stage) latent space, for `magnitude`: **PLS1** (size) and
**PLS2** (the orientation/divergence axis) load on **different** CpGs (top-12
overlap ≈ 3–5), and **both top sets are 100% within the CpGs we changed** — the
unchanged CpGs are silent (aggregate weight 6–45× lower). The two subsets
separate by **baseline methylation**: PLS1 ≈ low-baseline/steep (β≈0.30), PLS2 ≈
high-baseline/saturated (β≈0.6–0.74), stable across seeds. So the *same*
manipulation splits across two latent axes purely because the changed CpGs sit at
different sigmoid operating points.

### 4.3 Per-condition refit is circular; null-fixed PLS is blind to relocation
Re-fitting a supervised PLS per condition is circular: because we don't constrain
the modified features' variance/correlation, the *latent basis itself* rotates
with the manipulation. Freezing the PLS on the **null** and projecting modified
data removes that confound **for `magnitude`** (it reads as clean size), but is
**blind to `orientation`**: orientation relocates the signal onto CpGs that lie
*outside* the null's stage subspace, so B's trajectory nearly collapses (|B|/|A|
≈ 0.02) and the effect is mis-attributed (shows up as `delta`/`shape`, not its
target `angle`).

### 4.4 Null-frozen standardization (full concat space) — not a clean fix either
Freezing per-feature mean/std on the null and applying to all modes in the full
(un-projected) concat space, 8 reps / 49 perms / 4 stages:

| frame | mode | delta | angle | shape |
|-------|------|-------|-------|-------|
| own (per-dataset) | `magnitude`   | 1.00 | 0.00 | 1.00 |
| own               | `orientation` | 0.12 | 0.25 | 1.00 |
| **null-frozen**   | `magnitude`   | 1.00 | 0.00 | 1.00 |
| **null-frozen**   | `orientation` | **1.00** | **0.00** | 1.00 |

- `magnitude` is **identical** in both frames → confirms it is genuinely
  size-specific on `angle` at ≥4 stages (the 2-stage rotation is a single-step
  artifact the multi-stage principal axis absorbs); its only cross-talk is the
  genuine `rev.logit` `shape` leak, which standardization does not create.
- Freezing **breaks** `orientation`: relocated CpGs were near-constant in the
  null, so their frozen null-`std` is tiny → dividing the now-active signal by it
  **explodes** B's trajectory length → spurious `delta`=1.0, `angle`=0.0.

**Conclusion (provisional):** there is **no single fixed per-feature frame** that
fairly measures both a same-feature manipulation (`magnitude`) and a
feature-relocating one (`orientation`/`shape`) — the standardization interacts
with *which* features carry signal. `magnitude` is validated as pure-size +
genuine-shape-leak; `orientation`/`shape` are best read **descriptively with
their own standardization** (the existing specificity protocol), not against a
frozen frame. These two probes were ad-hoc (not committed); they are
reproducible from the descriptions above. If the frame question is pursued, the
next step is a committed probe alongside `scripts/geometry_specificity_probe.py`.

---

## Consolidated takeaways for the deferred power study — task 4.2

- **`orientation`→`angle`** is specific and clean (confirmed shape-free at 2
  stages, leads at 4).
- **`magnitude`→`delta`** is size-specific at ≥3 stages; at 2 stages the
  nonlinearity also rotates the single step (`angle` co-moves). It co-moves
  `shape` at every stage count via the `rev.logit` nonlinearity — not fixable by
  localizing the scale (`extremes`).
- **`shape`** is a **genuine, sensitive, non-specific omnibus geometry detector**
  on integrated multi-omic trajectories — not a calibration bug. The production
  concat-standardize null is calibrated (0.08 ≈ α); raw/SNF are mildly
  anti-conservative (0.17 / 0.25).
- **No recalibration applied.** The deliverable is this documented finding plus
  the study caveat on the `shape` row.
