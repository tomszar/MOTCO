# Design

## Context

The case-study design was fixed on 2026-10-09 from the SEA-AD MTG metadata. The cohort is astrocytes with
RNA and ATAC donor pseudobulks. Sex is the group. ADNC is merged into three stages (Not AD + Low /
Intermediate / High). At ≥30 nuclei per donor per omic there are 80 donors: F 11/10/28, M 9/10/12. At ≥50
nuclei there are 74: F 10/9/25, M 9/9/12. The cross-release supertype labels agree exactly, and the
`Astro_6-SEAAD` fraction does not trend with stage.

Today the generator sizes cells as `n_samples × stage_sample_prop × group_ratio`, with one ratio for every
stage. Every downstream consumer iterates the fixed tuple `OMIC_LAYERS = (methylation, expression,
proteomics)`: preprocessing, evaluation, realized-geometry diagnostics and attribution. The study engine
already supports exactly one nested evaluation axis form, `evaluation.integration_params.<key>`, with matched
seeds shared across columns that differ only there. The InterSIM reference has 367 CpGs, 131 genes and 160
proteins.

## Goals / Non-Goals

**Goals:**
- Measure `delta`/`angle`/`shape` Type I and power at the case study's real sample size, imbalance, stage
  count and block count, using the existing validated generator, estimator and gate machinery.
- Keep every committed configuration, signature, matched seed and result byte-identical.

**Non-Goals:**
- Emulating RNA or ATAC distributions, sparsity, or feature counts. The simulation is a structural analogue,
  not a SEA-AD emulator.
- Deciding the real-data preprocessing (feature filtering, VST), which belongs to the external case-study
  repo.
- Covariate adjustment, which is a separate change if the case study needs it.

## Decisions

### D1 — Explicit group × stage size table, not a per-stage ratio

Add `group_stage_sizes: tuple[tuple[int, ...], tuple[int, ...]] | None = None` to
`SemiSyntheticTrajectoryParams`. Rows follow `group_labels`; columns follow stages. When it is set,
`_stage_sizes`/`_group_stage_sizes` return the table directly. `n_samples`, `stage_sample_prop` and
`group_ratio` must then sit at their defaults, and any non-default value is an error. A silent override would
let a config say two contradictory things.

*Alternative:* make `group_ratio` accept a per-stage tuple alongside `stage_sample_prop`. Rejected. With cells
of 9–12, rounding `n × prop × ratio` can miss the target cell by one, and a reader can't see the design from
the parameters. The table states the design directly.

Byte-identity when absent: `parameter_signature` hashes `_to_jsonable(generator_params)`, which serializes
**every** dataclass field, `None` included. A new field would therefore change every committed signature.
That is the break the 2026-09-04 continuity change accepted (its D5), but
`tests/test_magnitude_default_pins.py` now pins the ten historical configs' signatures, and Phase 5
results must stay resumable. So the signature payload omits a named set of fields while they hold their
absent value: `_SIGNATURE_OMIT_WHEN_NONE = {"group_stage_sizes"}`. Truth metadata likewise gains the
realized-sizes key only when the table is set.

*Alternative:* accept the break, following the continuity precedent. Rejected. It would fail the existing
pin test and strand resumable Phase 5 shards for a field those runs never used. On the evaluation side no
special case is needed: `integration_params` is a mapping, and an unset `layers` key is simply absent from
the hash.

### D2 — Two blocks by evaluation-time selection of methylation + expression

Generate all three InterSIM blocks as usual, then measure only methylation and expression.

Why this mapping: ATAC and methylation are both regulatory, epigenetic blocks upstream of expression, with
many features and per-feature effects that cascade into transcription. More decisively, every trajectory
surgery acts on **methylation** indicators. Keeping methylation keeps each mode's construction present in
the measured space. Its documented realized geometry (Phase 2) still applies, now in a two-block joint
scope.

*Alternatives:*
- **Keep three blocks.** Rejected as the primary design. The block count is part of the design we need to
  characterize, and a third standardized block changes the joint space's composition and PLS's
  covariance structure. It stays as a design-grid column, so the block-count effect is measured, not assumed.
- **Measure expression + proteomics.** Rejected. The surgeries would reach the measured space only through
  the cascade, so the study would measure attenuated, undocumented constructions.
- **A two-block generator calibrated to RNA/ATAC.** Out of scope. It would invalidate the fidelity
  validation and is a project in itself.

### D3 — Layer selection is an evaluation `integration_params` key

`integration_params.layers` holds a list of layer names. It is normalized to canonical `OMIC_LAYERS` order
and validated as a non-empty, duplicate-free, known set. A `selected_layers(params)` helper replaces direct
`OMIC_LAYERS` iteration in `fit_omics_preprocessor`/`transform_*`, `concatenate_blocks`, the `concat`/`snf`/
`pls` integrations, `calculate_realized_geometry` (per-block scopes for selected layers; joint scopes over
them, population checkpoints included) and attribution's per-layer feature indexing. The preprocessor
records the layers it was fitted on, so `transform` can't silently mix sets.

Putting it in `integration_params` reuses the existing nested evaluation axis. The three-block column then
shares generator parameters and matched seeds with the baseline: same data, different measurement. That
gives a paired comparison of block count.

*Alternative:* a top-level `SimulationEvaluationParams.layers` field. Rejected. It would need new axis
plumbing in `study/config.py`, and the nested-axis contract already provides the pairing.

Absent key → `OMIC_LAYERS`, and integration metadata does not gain a `layers` entry, so committed results
regenerate byte-identically. With the key present, metadata records the canonical list.

### D4 — Group mapping and cohort columns

The table's rows are `(F, M)`, so group A (baseline) is female (n = 49) and group B (transformed) is male
(n = 31). This mirrors the cohort. The statistics are symmetric in the two groups, but the surgery is applied
to B, so B's smaller cells add to the transformed group's estimation noise. That is the less favorable and
therefore conservative assignment. The design grid has three columns, varying one factor at a time:

| Column | `group_stage_sizes` | `layers` |
|---|---|---|
| baseline | ((11, 10, 28), (9, 10, 12)) | methylation, expression |
| ≥50 nuclei | ((10, 9, 25), (9, 9, 12)) | methylation, expression |
| three blocks | ((11, 10, 28), (9, 10, 12)) | all three |

A full cross (2 × 2) adds a fourth column that answers no question the case study asks. It is omitted.

### D5 — Pilot-first effect axis, then paper grade

The Phase 5 axis (0–1.0) does not transfer. The `delta` null widens as n falls, and the joint magnitude
construction saturated at e = 0.02 even at n = 1200. The pilot runs 100 replicates × 199 permutations over a
wide, log-spaced shared axis, `0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0`, on the baseline column only.
That shows where each mode's target statistic rises. The paper-grade axis (or one axis per profile, if the
spec's split rule triggers) is then chosen on the recorded pilot curves, following the Phase 5 magnitude
re-measurement's bracket → pilot → paper-grade precedent. The headroom check runs at config time, as always,
and its pool is independent of n.

Cost: per-unit time should fall sharply from Phase 5's 61.5 s, since PLS CV and RRPP at n = 80 are much
cheaper than at n = 1200. The pilot measures the real figure. A rough paper-grade envelope is ~3 columns ×
~35 cells × 500 replicates ≈ 50k units. At ~5–10 s per unit that is ~70–140 core-hours, the same order as
Phase 5. The final budget is set from the pilot's median unit time.

### D6 — Gate and acceptance are advisory; the deliverable is an interpretability statement

The Phase 4 gate rules and Type I/specificity targets are declared so that the report computes them. At
n = 80, a missed 0.80 power floor is an expected **finding**, not a defect to fix. The binding output is the
report's per-statistic interpretability statement (spec: "Phase 6 small-n findings report"). For example,
"`delta` rejections and non-rejections are both informative; `angle` non-rejection is uninformative at
n = 80". The report contract is copied from Phase 5 (observed driver component, `n_jobs_override: forbid`).

## Risks / Trade-offs

- **Feature-to-sample ratio differs from the real data.** The simulation has 498 features against 80
  samples; the SEA-AD blocks will have thousands of filtered genes and peaks. → The report states this as a
  limitation. Pooled PLS's CV-selected rank is driven by stage supervision, not p, and stays at
  `n_stages − 1` across n in every study so far. The case-study repo's feature filtering should be reported
  beside the result.
- **Distribution mismatch.** InterSIM blocks are Gaussian/M-value; RNA/ATAC pseudobulks after VST are only
  approximately so. → This is a structural analogue only. The report says so and does not present the
  operating characteristics as SEA-AD-specific.
- **CV feasibility at 9-sample cells.** `cv2_splits`/`cv1_splits` are clamped by the minimum stage count
  (20 here). → The harness already raises a clear error when CV is infeasible. Enumeration-time loading of
  both profiles is a task, so infeasibility surfaces before compute.
- **Orientation with 3 stages lives in a 2-D stage-mean configuration.** The eigengap is then a single
  ratio and may be poorly separated more often than with four stages. → The eigengap is recorded per
  replicate already. The report stratifies orientation power by tercile, per the binding condition.
- **Signature drift from the new fields.** → D1/D3 byte-identity tests on every committed config, plus a
  pre-change signature fixture.

## Migration Plan

Additive. No existing behavior changes. Rollback is reverting the change; committed results are unaffected
because absent keys reproduce prior signatures exactly.

## Open Questions

- Whether the paper-grade study needs the per-axis split (D5) is answered by the pilot and does not change
  the specs or the task list.
