## 1. Reference-data export (R, one-time)

- [x] 1.1 Add an R export script that captures InterSIM's reference objects (`mean.M`, `cov.M`, `mean.expr`, `cov.expr`, `mean.protein`, `cov.protein`, `methyl.gene.level.mean`, `mean.expr.with.mapped.protein`, `CpG.gene.map.for.DEG`, `protein.gene.map.for.DEP`, `rho.methyl.expr`, `rho.expr.protein`)
- [x] 1.2 Export to a cached artifact (`.npz` + index) committed to the repo, with provenance (InterSIM version, export date, script)
- [x] 1.3 Add a loader that reads the cache without R and a clear error when the cache is missing

## 2. numpy generative core

- [x] 2.1 Implement per-omic sampling `μ = base + δ·v` → `MVN(μ, Σ)`, with methylation `rev.logit` applied after the additive (M-value) shift
- [x] 2.2 Implement default cross-omic coupling (differential genes mapped from differential CpGs, proteins from genes) and the correlation blend
- [x] 2.3 Accept explicit per-stage/per-group differential indicators and per-omic `δ`; return matrices + indicator truth
- [x] 2.4 Seeded reproducibility (identical output for identical seed/params)

## 3. Realism validation vs InterSIM

- [x] 3.1 Compare numpy vs InterSIM output distributions (per-omic means, covariance, cluster separation, cross-omic correlation) under matched params
- [x] 3.2 Capture the comparison as a test/fixture with a documented tolerance

## 4. Feature-surgery trajectory modes

- [x] 4.1 Build group A's baseline per-stage indicators (random, not forced continuous)
- [x] 4.2 `magnitude` = same indicators, `δ_B = λ·δ` (uniform step scaling)
- [x] 4.3 `orientation` = single global feature permutation `v_iᴮ = π(v_iᴬ)` applied to every stage
- [x] 4.4 `shape` = altered consecutive-stage overlap (with the ≥3-stage guard)
- [x] 4.5 `none` = identical groups; `translation` = constant location offset
- [x] 4.6 Record per-stage/group indicators and per-omic `δ` in truth metadata
- [x] 4.7 Reproducible indicators and transform from seed/params

## 5. Characterize specificity (descriptive, not a gate)

- [x] 5.1 Generate datasets per mode and run the instrumentation (group-vs-stage projection, per-statistic movement) — `specificity.py`
- [x] 5.2 Characterize how MOTCO responds to each mode and record the cross-talk (descriptive specificity matrix; cross-talk and non-detection are findings, not failures) — results in `specificity_results.md`. Modes redesigned to methylation-only surgery (gene/protein re-derived) per the agreed semantics.

## 6. Swap downstream consumers (wiring)

- [x] 6.1 Replace the InterSIM-subprocess + random-direction surface with the numpy generator params in `semisynthetic.py`
- [x] 6.2 Adapt `evaluation.py`, `grid.py`, `study/*`, `showcase.py`, and `cli.py` (`simulate`) onto the new generator
- [x] 6.3 Make former InterSIM-generation tests R-free; keep the bridge only for the one-time reference export
- [x] 6.4 Update `tests/` across affected modules; ensure ruff + mypy + fast pytest pass with no R on PATH

## 7. Re-run the power study

- [x] 7.1 Update study config + `trajectory-power-study` spec wiring for the numpy generator
- [~] 7.2 Re-run the grid; reset acceptance targets to the feature-surgery modes' operating characteristics — **smoke validated** end-to-end with the numpy generator + new modes; paper-grade grid + final acceptance targets deferred to the cluster (run by the user)
- [~] 7.3 Regenerate the report (specificity matrix, Type I table, power curves) and record that prior results are superseded — smoke report regenerated; paper-grade report deferred to the cluster run

## 8. Docs

- [x] 8.1 Update `CLAUDE.md` and the study/showcase READMEs for the numpy generator and new mode semantics
