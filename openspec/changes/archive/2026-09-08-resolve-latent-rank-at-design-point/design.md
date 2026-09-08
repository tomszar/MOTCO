# Design — resolve-latent-rank-at-design-point

## Context

See `proposal.md` — Why. The relevant current state:

- Axis names are split once on `.` into `(namespace, field)` (`_split_axis` in `grid.py`; `_validate_axis_namespace` and `axis_baseline_value` in `study/config.py`). `_apply_axis_value` does `dataclasses.replace(params, **{field: value})`, so an axis can only address a top-level dataclass field. `SimulationEvaluationParams.integration_params` is a plain mapping; the rank override `forced_components` lives inside it.
- The PLS evaluator already implements the override (`_pls_integration` in `evaluation.py`): when `forced_components` is set it skips the double CV, fits at that rank, and records `component_selection = "forced"` and `selected_lv = <rank>`; when the key is absent or `None` it runs the CV path unchanged. The feasible range is `[2, min(n_features, n_samples)]`; at n = 1200 over ~660 standardized features every ladder value is feasible.
- `parameter_signature` hashes `_to_jsonable(cell.evaluation_params)` in full, so anything placed in `integration_params` already changes the signature. `_require_distinct_primary_datasets` keys on `(family, generator identity, evaluation identity)` — the comment there anticipates "design points that differ only in an evaluation-namespace axis share a dataset on purpose".
- Design-point cells carry `metadata["design_point"] = {axis: value}` for every declared axis; `record_design_point` reads it back and `resolve_operating_by_design_point` already emits per-point rows with rate, MC SE, eigengap summaries, `angle` `q95` dispersion, and `median/min/max_selected_lv` (read from the integration metadata's `selected_lv`). `evaluate_design_point_decision` (targets.py) implements the item-4 "first confirmed column in preference order" rule.
- `_to_jsonable` round-trips `None` as JSON `null`, so `null` inside an axis value list is representable; the current validators never see it because no axis has needed it.
- Measured cost: 55.8 s per unit at n = 1200, 199 permutations, on the cluster's EPYC cores (design-point pilot `PROVENANCE.txt`); 6,500 units took 70 core-hours.

## Goals / Non-Goals

**Goals:**

- One config can hold the chosen design point fixed and vary only the retained PLS rank, with every rank column measuring byte-identical datasets, and the report can say which rank rule Phase 5 should commit to from recorded evidence and a predeclared rule.
- The nested-axis mechanism is the minimum that makes `forced_components` addressable; nothing else about `axes`/`design_grid` semantics moves.
- Every existing config, shard, and committed result set enumerates, resumes, and re-reports byte-identically.

**Non-Goals:**

- A general nested-path axis language (`a.b.c.d`, list indices, generator sub-structures). Exactly one nested form is supported: `evaluation.integration_params.<key>`.
- Making the rank decision authoritative. It is advisory like the design-point rule; the human decision is recorded in the readiness worklist.
- Changing how the CV path selects components, or adding any selection criterion that reads group labels.

## Decisions

### D1 — One nested form, resolved at the same three seams that handle top-level axes

`_split_axis` returns `(namespace, path)` where `path` is a tuple of one or two segments. A two-segment path is legal only as `("integration_params", key)` under `evaluation`. `_apply_axis_value` for that form builds a new mapping `{**evaluation.integration_params}` with the key set, or removed when the value is `None`, and `replace`s `integration_params`. `_get_axis_value` / `axis_baseline_value` return `mapping.get(key)` (→ `None` when absent). `_validate_axis_namespace` accepts the two-segment form and rejects any other multi-segment axis with an error naming the axis.

*Why:* the three helpers are the only places an axis name is interpreted; extending them keeps enumeration, metadata, seed families, headroom checks, and the design-point tables completely unaware of nesting — the design point already records `{axis: value}` verbatim, and the value already reaches the signature via `evaluation_params`.

*Alternative rejected:* promoting `forced_components` to a top-level `SimulationEvaluationParams` field. It would need a new field on the evaluation dataclass, a new signature key, and a translation layer into `integration_params` that the evaluator reads — three moving parts to avoid one dotted path — and it would change `_to_jsonable(evaluation_params)` for every existing cell (new field with default `None`), breaking signature stability for committed shards unless special-cased.

*Alternative rejected:* arbitrary dotted paths via a generic setter. Nothing needs it, and the failure mode (silently creating a key the evaluator never reads) is exactly the class of implicit contract the geometry audit spent P1–P3 removing.

### D2 — `null` is the baseline value of the rank axis, and "key absent" is its semantics

The rank axis lists `null` explicitly (`[null, 3, 4, 6, 9, 12]`). Baseline detection compares `_to_jsonable` values, so `None == None` satisfies the "baseline value present" rule. Applying `None` removes the key rather than storing `None`, so the baseline column's evaluation parameters are byte-identical to the config's baseline (no `forced_components: null` key appears), and therefore its cells' signatures equal what the same config would produce without the design grid — the baseline-column-is-the-primary-grid invariant (item-4 design D2) holds.

*Why store nothing rather than `None`:* `_pls_integration` treats absent and `None` identically, but the signature does not; storing `None` would silently change the primary cells' signatures relative to a design-grid-free config and break the "configuration without a design grid is unchanged" scenario in spirit.

### D3 — No enumeration changes; rely on the existing evaluation-identity guard, and assert it in tests

Rank columns differ only in evaluation identity, so `_require_distinct_primary_datasets` accepts them, `_generator_identity` is equal across columns, and the primary matched-seed family gives every column the same generator seed per replicate index. The ladder therefore reproduces the probe's matched-datasets design inside the study harness with zero enumeration code. The change adds tests that pin this (same seeds and generator params across columns; two columns with equal generator *and* evaluation identity still rejected) so a future refactor cannot silently un-pair the ladder.

### D4 — Operating table gains `component_selection`; the ladder figure is a new renderer, not a change to the design-point power figure

`resolve_operating_by_design_point` adds one column, `component_selection`, taken from the integration metadata (`cv`/`forced`, or `mixed` if a row somehow has both — which would indicate a config error). `render_rank_ladder` draws, per mode (magnitude / orientation / shape / translation / `none` anchor), each statistic's rate vs rank with MC error bars; the CV column is plotted at its `median_selected_lv` with a distinct marker and a "CV" label. It is written only when the design grid's axes include the rank axis; the existing `render_design_point_power` (orientation `angle` vs `n_samples`, one line per ρ) is left untouched and simply finds a single-point n axis on the ladder study (it should degrade to a single marker or skip when `generator.n_samples` is not a declared axis — verify, don't assume).

*Why not generalize the design-point figure:* the item-4 figure encodes the (ρ, n) reading; a rank ladder needs all four modes × three statistics, which is a different plot. Two small renderers beat one configurable one.

### D4b — The production forced-rank guard admits *declared* rank columns only

*(Found during implementation.)* `report.assert_production_component_selection` refused every record with `component_selection = "forced"`, which would have made the ladder unreportable. The guard now passes a forced record only when its design point declares the rank axis (`evaluation.integration_params.forced_components`) with a value equal to the rank the integration recorded (`selected_lv`); any other forced record — no design point, no rank axis, or a mismatch — is still rejected with the original error. This keeps the guard's purpose (no hand-picked rank presented as the operating point) while recognizing that a predeclared design column is a measurement, not a diagnostic. Forced columns remain invisible to every baseline reader by the existing `varied_axis` marker, so the CV column is still the only production operating point the curves, gate, and targets read.

### D5 — `acceptance.rank_decision`: a comparative rule with the CV column as reference

Config block:

```json
"rank_decision": {
  "axis": "evaluation.integration_params.forced_components",
  "target": {"trajectory_mode": "orientation", "statistic": "angle"},
  "protected": [
    {"trajectory_mode": "magnitude", "statistic": "delta"},
    {"trajectory_mode": "shape", "statistic": "shape"}
  ],
  "type_i_bound": {"alpha": 0.05, "se_tolerance": 2.0},
  "gain_se_multiplier": 2.0,
  "loss_se_multiplier": 2.0
}
```

Evaluation (`evaluate_rank_decision` in `targets.py`), per forced column `k` against the reference column (`axis` value `null`), at the largest enumerated effect in each column:

- (a) gain: `rate_k − rate_cv > gain_se_multiplier · sqrt(SE_k² + SE_cv²)`;
- (b) anchor: for each statistic, anchor rate ≤ `alpha + se_tolerance · sqrt(alpha(1−alpha)/n)` (the same bound the Type I control target uses);
- (c) protected: for each protected pair, `rate_cv − rate_k ≤ loss_se_multiplier · sqrt(SE_k² + SE_cv²)`.

A column qualifies when all three hold. Verdict `adopt_fixed_rank` with the smallest qualifying rank, else `keep_cv`. Output `rank_decision.json` (verdict, chosen rank or `null`, per-column per-criterion booleans with the rates, SEs, and thresholds used) and `rank_decision.csv`.

*Why the pooled-SE difference test rather than reusing the item-4 floor rule:* item 3 is a comparison, not a threshold — the CV column already meets the 0.80 floor, so the question is whether a fixed rank is *better enough* to justify abandoning CV, and whether it costs anything. Because the columns are paired on identical datasets, the pooled independent SE is conservative, which is the right direction for a rule that would change the production configuration.

*Why smallest qualifying rank:* parsimony is predeclared so the choice is not made after seeing which rank looks best; every qualifying rank already beats CV on the target without loss, so the cheapest such space is the defensible commitment.

*Why protect magnitude and shape but not translation:* translation is a negative control; its Type I behavior is covered by the anchor criterion and by the existing Type I control target on the baseline column. Its `e > 0` cells are still reported in the ladder for the record.

*Alternative rejected:* generalizing `DesignPointDecisionRule` with a `mode: "floor" | "comparative"` switch. The two rules share no fields beyond the target pair; a second dataclass is clearer than a union.

### D6 — Ladder profile

`phase5_latent_rank_ladder.json`: `derives_from` the design-point pilot; generator identical except `n_samples: 1200` (baseline continuity `0.0` and `p_dmp: 0.1` already); evaluation identical (same CV knobs, `random_state`, `permutations: 199`, `n_jobs: 1`); `trajectory_modes` all four; effects `[0.0, 0.25, 0.5, 1.0]`; `design_grid.axes = {"evaluation.integration_params.forced_components": [null, 3, 4, 6, 9, 12]}`; `matched_seeds` on with the shared anchor and a new `primary_family` name; attribution off; gate off; `type_i` control target kept (baseline column); `rank_decision` as in D5; no `surgery_censoring` key.

Size: 6 columns × (1 anchor + 4 modes × 3 effects) = 78 power cells, plus the baseline Type I controls (the same two cells the design-point pilot carried) → ~8,000 units. Forced columns skip the double CV (5 repeats × 4 outer × 3 inner × up to 19 candidate fits per unit), so they are cheaper than the 55.8 s/unit measured for the CV path at n = 1200; budget ~120 core-hours on the cluster (~2 h wall on 64 shards), ~60 on a fast desktop across 22 shards. Same SLURM recipe as the design-point pilot; do not pass `--n-jobs`; pin BLAS to one thread.

*Why include forced 3:* the CV column's selected rank ranges 2–4 with median 3. Without a forced-3 column, any difference between CV and forced 4 confounds "one more component" with "no selection noise". Forced 3 also *is* a candidate group-blind rule ("fixed `n_stages − 1`"), so it must be measured to be adoptable.

*Why 12 as the top rung and not higher:* the probe's orientation→shape response was flat from 9 to 12, and the decision rule's parsimony clause means a rank above the plateau can only be chosen if lower rungs fail. Extending the ladder costs 13 cells per rung; 12 is where the probe stopped seeing change.

*Why all four modes at every rung:* criteria (b) and (c) need magnitude and shape at every rank, and the design-point report already requires their re-measurement at the chosen point under CV. Translation at every rung checks that a fixed rank does not turn a location offset into a detectable geometry difference.

### D7 — Records, signatures, resume

No generator or evaluation-field changes; no version bump. Forced columns have different `integration_params` and therefore different signatures from the CV column and from each other. Merge, sharding, and resume are untouched. Existing design-grid fixtures (`tests/test_study_design_grid.py`, `tests/test_study_design_point_report.py`) must re-report byte-identically.

### D8 — Reporting the answer

The findings report (`docs/reports/latent-rank-ladder-<run date>.md`) reads, per mode and statistic, the rate-vs-rank curve beside the recorded eigengap (which should be nearly constant across rank at fixed data — a useful sanity check) and the `angle` null `q95` dispersion per rank. It must separately answer: does orientation `angle` power move with rank at n = 1200; does the orientation→shape artifact decay as in the probe (now at the real design point and on the corrected estimator); do magnitude and shape hold power 1.00 at CV rank with `p_dmp = 0.1`; and is any anchor inflated at high rank. The committed rank rule is stated in group-blind terms ("stage-supervised double CV" or "fixed rank k, predeclared"), and `docs/phase5-readiness.md` item 3, `docs/roadmap.md` (Phase 5 planned baseline; "Not yet established"; "Next three changes" — items 1 and 2 there are done and the list should be refreshed), and `CLAUDE.md` are updated in the same change.

## Risks / Trade-offs

- [High fixed rank could inflate the anchor's Type I error on `angle` or `shape` at n = 1200 by admitting noise directions into the configuration.] → Criterion (b) vetoes such a rank; the ladder figure shows the anchor row explicitly so inflation is visible even for ranks the rule never considers.
- [`render_design_point_power` assumes `generator.n_samples` and `generator.baseline_continuity` are design axes and may error on a rank-only grid.] → Task 3.x verifies on a rank-only fixture; if it cannot degrade gracefully, it is written only when its axes are present.
- [A qualifying fixed rank would change the production configuration on the evidence of one design point.] → The rule's SE multipliers are predeclared at 2.0 (conservative pooled SE on paired data), the verdict is advisory, and the readiness write-up must state that the rule is committed for the Phase 5 design point and that real-data use of a fixed rank is a preregistered choice, not a tuned one.
- [Forced columns cost less than CV columns, so shard wall-times are uneven.] → Shards are partitioned by (cell, replicate); uneven shards only lengthen the tail. Submit CV-column shards first if the queue allows.
- [`forced_components` at rank ≥ 9 on some replicate could be infeasible if a dataset had fewer features than expected.] → The evaluator rejects rather than clamps; `--error-policy record` keeps the study running and the report counts failures per column. Expected feasibility is `min(660, 1200) ≥ 12` for every replicate.
- [The eigengap is a property of the *configuration in the latent space* and will differ across rank columns even on identical data.] → It is reported per column as recorded; the findings report must not read a rank-induced eigengap change as a baseline-geometry change.

## Migration Plan

Additive. Existing configs load unchanged (no nested axis → validators take the top-level path; no `rank_decision` → no new outputs). Existing merged result sets re-report identically. Rollback is removing the config block and the nested axis; no persisted-record format changes.

## Open Questions

- Findings-report filename date and the `results/phase5-latent-rank-<date>/` directory are fixed at run time.
