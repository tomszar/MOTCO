## Context

See `proposal.md` — Why, for the defect and the evidence. Three constraints shape the fix:

- The estimator is a port of Adams & Collyer's phenotypic trajectory analysis (`tests/data/reference/evo_649_sm_suppmat.r`). Its *estimator* — PC1 of the centered stage configuration, `:57` — is the published method and is not in question. Only the sign convention at `:64` is.
- The published convention is itself ad hoc. Its comment reads `#check startingpoint location`, and it anchors on the raw first-stage row in whatever coordinate frame the data arrives in. It is not a stated property of the method; it is an implementation patch.
- The committed R fixtures cannot discriminate between conventions in the direction that matters, and no candidate fix risks them. Measured directly (`max |angle − R|` over every reported pair):

  | fixture | centered (current) | raw (faithful) | net displacement |
  |---|---:|---:|---:|
  | `results_example2.csv` (5 levels) | 9.9e-12 | 9.9e-12 | 9.9e-12 |
  | `results_example1.csv` (2 levels) | **30.60** | 4.8e-13 | **30.60** |

  Example2 is reproduced exactly by all three. Example1 is *not* reproduced by the shipped code: pairs `t1/t3` and `t2/t3` sit on opposite sides of the PCA origin, so R's raw anchor reports 105.30/103.51 where the centered anchor reports the supplements 74.70/76.49. `tests/test_permutation.py` has always accepted `180 − exp_angle`, which is why this went unrecorded. Net displacement matches the shipped values exactly, so the change leaves both fixtures' reported outputs unchanged — it neither introduces nor removes the example1 deviation. That deviation is itself an instance of the origin-side dependence that rules the raw anchor out.

The two-stage case deserves care because it is where intuition is clearest and where the current code is already correct: with two stages the centered configuration is `[d/2, −d/2]`, so `PC1 · (stage₁ − centroid) = ±‖d‖/2`, never near zero. The defect is specific to three or more stages.

## Goals / Non-Goals

**Goals:**

- Make the orientation sign a property of the trajectory, not of the coordinate frame or of noise.
- Express the invariance contract as tests, since the existing fixtures are structurally incapable of detecting a sign flip.
- Establish what orientation power actually is once the artifact is removed.

**Non-Goals:**

- Changing what orientation *is*. PC1 stays. A move to net displacement or mean-normalized-step direction as the orientation summary itself — which would also make orientation order-aware and consistent with the path-length size statistic — is a separate design question raised below, not this change.
- Reconciling size and orientation semantics. `_estimate_size` is path length (order-aware); PC1 orientation is order-blind. Real, but out of scope.
- Re-running the whole Phase 4 grid. Only the cells whose conclusions this touches.

## Decisions

### Anchor on net displacement, not the raw first-stage row

```python
c1 = float(orientation @ (X_raw[-1, :] - X_raw[0, :]))
```

*Why:* it states directly what the reference's line is groping toward — orient PC1 along the direction of progression. It is intrinsic to the trajectory, so it is translation-invariant; it is well away from zero for any real progression; and it reduces to the two-stage transition direction exactly.

*Why not the faithful port (raw `X_raw[0, :]`):* it makes the sign depend on the trajectory's position relative to the origin, so a pure translation can reverse it. Verified across four regimes:

| regime | truth | centered (current) | raw (faithful) | net displacement |
|---|---|---|---|---|
| bent 4-stage, identical groups | 0° | **179.4°** | 0.6° | 0.6° |
| straight 2-stage, groups either side of origin | 0° | 0.0° | **180.0°** | 0.0° |
| translation control, group B shifted | 0° | 0.0° | **180.0°** | 0.0° |
| genuine 90° difference | 90° | 90.0° | 90.0° | 90.0° |

The faithful port trades one failure for two, and one of the two is MOTCO's own translation null control — which Phase 4 currently passes cleanly at every effect size. Adopting it would break a working control to fix a broken one.

*Residual degeneracy, accepted and documented:* net displacement vanishes for a closed trajectory whose last stage returns to its first. That is a trajectory with no net progression, for which no direction is defined; it is strictly rarer than the current failure mode, which triggers on any lateral bend. If it ever matters, the mean of normalized consecutive steps degrades more gracefully.

### PC1 is computed from the centered configuration, unchanged

Only the anchor moves. Keeping `X` centered for the SVD preserves the reference estimator exactly and keeps the existing fixtures passing.

### The invariance contract is tested directly, not through the fixtures

The fixtures are two datasets with benign geometry; they agreed with the broken code for the project's entire history. The new tests must construct the failure geometry on purpose: a bent configuration whose first stage projects near zero onto its own centered principal axis, repeated perturbation to catch sign instability, and translated/scaled/opposite-side pairs. The load-bearing assertion is the negative one — **no null configuration reports an angle approaching 180°** — because that is the observable that would have caught this.

### Re-run scope: orientation, shape, and translation cells only

Magnitude's `delta` conclusion does not depend on the orientation estimator, and the Type I controls that matter for this question live in `none` and `translation`. Re-running orientation (the shortfall), shape (whose `angle` column carries the same artifact, and whose non-rejecting replicates were 53% above 90°), and translation (the control that must stay clean under the new anchor) answers every question this change raises. Reuse the Phase 4 configuration and seeds so the comparison is like-for-like.

*What the re-run decides:* if orientation power clears 0.80, readiness item 1 closes and `diagnose-angle-null-pivotality` is unnecessary. If a shortfall survives, that change proceeds with a genuinely open question instead of a displaced one.

## Risks / Trade-offs

- **Published Phase 4 numbers change** → the report is committed evidence and must not be silently rewritten. Mark the affected orientation conclusions superseded, keep the original report intact as historical record, and publish the corrected numbers as a dated follow-up, exactly as the July pilot was retained when Phase 4 superseded it.
- **The realized-geometry checkpoints are contaminated too** → the population (81°) and latent (57°) orientation figures come from the same estimator, so they are suspect and must be recomputed rather than quoted. Any argument that rested on them — including "the construction survives integration" — needs restating on corrected numbers.
- **Angles above 90° are not per se wrong** → `results_example1.csv` contains a genuine 105.3°. Tests must not assert an upper bound on the angle in general; only null configurations get the near-180° prohibition.
- **A deliberate deviation from a published reference invites doubt** → mitigated by proving equality on both committed fixtures, and by documenting the deviation at the implementation and in the report rather than burying it.
- **The corrected estimator may not close the gap** → then the change still stands on its own as a bug fix with a tested invariance contract, and hands the pivotality diagnostic a clean starting point.
