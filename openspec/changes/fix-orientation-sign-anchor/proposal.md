## Why

MOTCO's trajectory `angle` statistic reports large spurious angles for trajectories that carry no orientation difference at all. In the Phase 4 pilot's null cells — `none` and `translation`, where both groups are constructed identically and the true orientation difference is zero — the observed-angle distribution is bimodal: 663 of 700 replicates fall under 30°, the 60–135° band is nearly empty (3 of 700), and a distinct secondary mass of 21 replicates sits at 150–180°, reaching 177.8°. That is the signature of a discrete sign flip mapping θ to 180−θ, not estimation noise.

The cause is a one-line deviation from the reference implementation this estimator is ported from. `tests/data/reference/evo_649_sm_suppmat.r:57` defines orientation as PC1 of the stage configuration, and `:64` resolves PC1's inherent sign ambiguity by anchoring on the **raw** first-stage row. MOTCO centers the configuration first (`src/motco/stats/trajectory.py:367`) and then anchors on the **centered** row (`:374`), computing `PC1 · (stage₁ − centroid)`. That quantity vanishes whenever a trajectory departs and returns laterally — precisely what a bend produces — leaving the sign to be decided by noise. On a bent four-stage trajectory the anchor projection is exactly 0.0, and under perturbations of sd 0.01 the sign flips in 105 of 200 draws; two identical copies of that trajectory are then measured at a median angle of 179.3°.

The consequences reach the Phase 4 conclusions. Because each RRPP permutation is equally subject to flipping, observed statistic and permutation null inflate together: Type I error is preserved — which is why all eighteen predeclared controls passed — while power is destroyed. It explains the orientation shortfall's every symptom: the 0.65 power against a 0.80 floor, the flat 0.59 → 0.65 curve (larger effects add bend, so more flips), and the inversion in which non-rejecting replicates carry a *larger* mean observed angle than rejecting ones (66.7° vs 52.5°, with 29% vs 17% above 90°). The reported orientation power is an artifact-suppressed underestimate.

The existing regression fixtures cannot detect this. `evo_649_sm_example1.csv` has two levels, the case where the anchor is provably stable; `example2` has five levels and does exercise the PC1 branch, but its trajectories are geometrically benign and every candidate convention reproduces the published R output exactly on it. (Example1 is a separate matter: MOTCO has never reproduced its `t1/t3` and `t2/t3` angles, reporting the supplements 74.70/76.49 against R's 105.30/103.51, because those pairs straddle the PCA origin and R anchors on the raw row. The regression test accepts `180 − exp_angle`, so this was never recorded. It is unchanged by this fix — see `design.md`.)

## What Changes

- **BREAKING (statistical output):** anchor the orientation sign on net displacement — `PC1 · (stage_last − stage_first)` — instead of on the centered first-stage row. Reported `angle` values change for trajectories of three or more stages whose geometry triggered the flip; some previously reported angles near 180° become their supplement near 0°.
- Keep the estimator itself unchanged: orientation remains PC1 of the centered stage configuration, exactly as in the reference. Only the sign convention changes, and the docstring must record that this is a deliberate, tested deviation from the supplement's ad-hoc `#check startingpoint location` line, together with why the raw-row anchor is unsuitable here.
- Add the invariance contract the current fixtures structurally cannot express: identical, translated, and uniformly scaled trajectories must report an angle of zero, and no null configuration may report an angle approaching 180°.
- Preserve reference fidelity: both committed R fixtures must continue to report exactly the values the shipped estimator reports today, which the candidate anchor has been verified to do (exact agreement with R on `example2`; identical to current behaviour on `example1`).
- Re-run the Phase 4 orientation, shape, and translation cells under the corrected estimator to measure what orientation power actually is, and record whether the 0.80 floor is met.
- Publish a dated finding, mark the affected Phase 4 orientation conclusions as superseded, and update the roadmap and the Phase 5 readiness worklist with the outcome.

## Capabilities

### New Capabilities

- `trajectory-orientation-invariance`: The production contract for MOTCO trajectory `angle` as a direction difference — what transformations it must ignore (translation, uniform scale, stage-configuration bending), the sign convention that makes PC1 a direction rather than an axis, and the null configurations that must never report a large angle.

### Modified Capabilities

None. The estimator's public interface, exported symbols, and call contract are unchanged; only the values it returns for affected geometries change.

## Impact

- Changes `_estimate_orientation` in `src/motco/stats/trajectory.py`; no signature, import-surface, or CLI change.
- Changes reported `angle` values for three-or-more-stage trajectories with bent geometry — including every `angle` figure in the Phase 4 report and the realized-geometry checkpoints, whose population (81°) and latent (57°) orientation numbers were computed with the same estimator and are therefore suspect.
- Supersedes the Phase 4 orientation power conclusion. The Type I, magnitude, and shape conclusions are unaffected in kind, but the shape cells' `angle` column carries the same artifact.
- Reframes `openspec/changes/diagnose-angle-null-pivotality` as contingent: the non-pivotality it hypothesized is a downstream consequence of this defect, and it should proceed only if a shortfall survives the corrected re-run.
- Existing R regression fixtures and their tests are unchanged and must keep passing.
