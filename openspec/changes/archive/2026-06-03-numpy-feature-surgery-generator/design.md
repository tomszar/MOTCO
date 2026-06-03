## Context

InterSIM's generative model (read directly from `body(InterSIM)`) is, per omic:

```
DMP[, i] ~ Bernoulli(p.DMP)            # independent indicator per cluster i
μ_i      = base + DMP[, i] · δ         # cluster mean = baseline + shift on differential features
X_i      ~ MVN(μ_i, Σ)                 # samples; methylation then passes through rev.logit
```

Differential genes/proteins are mapped from the differential CpGs (`CpG.gene.map.for.DEG`, `protein.gene.map.for.DEP`) unless `p.DEG`/`p.DEP` are set; expression/protein means blend a cross-omic correlation term (`rho.methyl.expr`, `rho.expr.protein`). The reference objects (`mean.M`, `cov.M`, `mean.expr`, `cov.expr`, `mean.protein`, `cov.protein`, `methyl.gene.level.mean`, `mean.expr.with.mapped.protein`, the two maps, the two rhos) are exposed as package data. The `InterSIM()` return drops all `DMP/DEG/DEP` truth.

Today MOTCO shells out to R per dataset and then injects a group effect along random features. The trajectory geometry, however, is fully determined by `{v_i}` (the indicators) and `δ`, so modes belong on the indicators.

## Goals / Non-Goals

**Goals:**
- Trajectory modes defined as exact operations on per-stage differential indicators, mapping cleanly onto MOTCO's location-invariant statistics (size/orientation/shape).
- Remove R from the runtime; generation reproducible and unit-testable in CI.
- Emit ground-truth indicators so "injected mode → matching statistic moves" is checkable.

**Non-Goals:**
- Not forcing a continuous baseline trajectory (non-straight paths are the target regime).
- Not changing the integration (`concat`/`SNF`) or the trajectory test itself.
- Not preserving the random-direction modes (they are superseded, not kept in parallel).

## Decisions

- **β-Py: numpy generator, R only for a one-time reference export.** Export the InterSIM reference objects to a cached `.npz` committed to the repo (with provenance: InterSIM version, export date), then replicate the ~30-line generative math in numpy (`MVN` sampling, `rev.logit` on methylation, the cross-omic blend and maps). Removes per-replicate subprocess cost (dominant in the power study) and makes generation testable without R.
  - *Alternative — β-R* (feed our indicators back into an extended R script): keeps R in the per-dataset loop, can't be tested in CI, slower for the study. Rejected.
  - *Alternative — α, post-hoc surgery on InterSIM output*: must recover indicators empirically through the `rev.logit` nonlinearity and edit arbitrary choices after the fact; can't give clean ground truth. Rejected.

- **Surgery touches methylation only; expression and protein cascade from it.** A central realism constraint (from the AD use case): we manipulate only the *original* features, never the latent space, and only the *methylation* values — letting them feed forward to gene expression and protein through the cached CpG→gene→protein incidence maps (the biological cascade). So group B's transform is defined solely on its per-stage **methylation** indicators, and group B's gene/protein indicators are always **re-derived** from group B's methylation. This keeps datasets close to real ones rather than tailored to be MOTCO-amenable — and it is precisely the discipline that fixes the earlier leaks (which came from independently editing all three omics).

- **Modes operate on methylation indicators; group A inherits a random baseline.** Group A keeps InterSIM-style independent per-stage methylation indicators `v_iᴬ`; group B is a deterministic transform of the *methylation* indicators (with gene/protein re-derived):
  - `translation` → group B keeps A's stage-changing sites and adds an extra set `U` of methylation sites (disjoint from the stage-changing sites) that are differential at **every** B stage and at none of A's. Constant across stages → a pure group offset → moves only the (untested) group main effect, not `delta`/`angle`/`shape`. `e` ↔ |U|.
  - `magnitude` → same methylation indicators, `δ_methyl_B = (1+e)·δ_methyl` (scale only methylation's magnitude; gene/protein follow the same sites at baseline `δ`).
  - `orientation` → relocate a fraction `e` of the stage-changing methylation sites to different CpGs, using a **single relocation applied identically to every stage**, so the per-stage on/off pattern is preserved on different feature axes → a rotation (`angle`), with size and shape preserved in the linear limit. The consistency-across-stages is exactly what distinguishes it from `shape`.
  - `shape` → perturb a **single interior stage** of B (≥3 stages): either relocate a fraction `e` of that stage's methylation sites (`relocate`) or scale that stage's methylation effect (`magnitude-bump`), bending one interior vertex. Both flavors are supported (configurable).
  - These are *realistic* constructions, not orthogonal ones. Cross-talk is **expected and reported**, not engineered away: the methylation `rev.logit` nonlinearity and MOTCO's per-feature standardization mean e.g. `magnitude` bends shape and `orientation` induces some shape. Whether (and how well) MOTCO separates them is an **open question the study characterizes**, not a property we tune the data to satisfy.

- **Methylation surgery lives in M-value (logit) space.** Indicator-driven `+δ` shifts are additive *before* `rev.logit`, applied to the pre-logit means; `rev.logit` is the last step. (The earlier attempt to add a `translation` offset in β-space then clip distorted geometry stage-dependently — replaced by the disjoint-`U` indicator construction above, which goes through the same generative pipeline cleanly.)

- **Ground-truth indicators are first-class output.** The generator returns, per stage and per group, the methylation (and derived gene/protein) differential indicators and the per-omic `δ`, recorded in `truth`, so the showcase/study can **characterize** how MOTCO responds (a descriptive specificity matrix, not a pass/fail gate).

## Risks / Trade-offs

- **Realism drift from reimplementation** → validate numpy output distributions (per-omic means, covariances, cluster separation, cross-omic correlation) against a held InterSIM run before committing; ship the comparison as a test/fixture.
- **Modes' specificity is an open question, not a gate** → the instrumentation (group-vs-stage projection, per-statistic movement) *characterizes* how MOTCO responds to each realistic difference; cross-talk and even non-detection are findings, not failures. We deliberately do **not** tune the constructions (e.g. via latent-space or standardization tricks) to force a clean diagonal — that would produce MOTCO-amenable but unrealistic data. The deliverable is the descriptive specificity matrix + power curves.
- **Cached reference data goes stale / provenance unclear** → store InterSIM version + export script + date alongside the `.npz`; the export is reproducible on demand.
- **Power-study invalidation** → the change explicitly includes a re-run and acceptance-target reset; downstream consumers (`evaluation`, `grid`, `study`, `showcase`, `simulate`) are adapted as wiring, not re-specified, where their behavior is unchanged.

## Migration Plan

1. Export reference data (R, one-time) → cached `.npz` + provenance.
2. Build + validate the numpy generator (realism fidelity vs InterSIM).
3. Implement indicator-surgery modes; prove dominant specificity with instrumentation.
4. Swap downstream consumers onto the numpy generator; make InterSIM generation tests R-free.
5. Re-run the power study; reset acceptance targets; update the study spec.

Rollback: the change is a wholesale replacement; the prior InterSIM-subprocess path remains in git history if a revert is needed, but the two paths are not maintained in parallel.
