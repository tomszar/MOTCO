# Design — phase5-design-point-pilot

## Context

See `proposal.md` — Why. The relevant current state:

- `enumerate_power_grid` (`grid.py`) emits `power_primary` cells for modes × effects at the baseline and, for each `axes` entry, `power_ofat` cells that vary exactly one field off the baseline. There is no crossed enumeration.
- `_assign_seed_families` (`study/enumerate.py`) puts every `power_primary` cell in `matched_seeds.primary_family` and gives every other cell (Type I, controls, OFAT) a family of its own. `_require_distinct_primary_datasets` rejects two same-family primary cells with identical generator identity; that identity already includes `baseline_continuity` and `n_samples`.
- `_zero_effect_anchor_cell` emits one mode-agnostic `none`/`e = 0` cell per study, because at `e = 0` the transform is a no-op and per-mode zero cells inside one family would be byte-identical.
- `_require_surgery_headroom` reads each cell's own generator params, so it is already correct per column.
- `report.py`/`targets.py` build the primary curves, gate, and acceptance targets from cells with `varied_axis is None`; `spectrum.POWER_PHASES = ("power_primary", "power_ofat")` decides which records feed the eigengap-stratified and continuity-resolved tables. `resolve_orientation_by_continuity` buckets on `(ρ, mode, effect)` and pools every cell sharing those coordinates.
- Headroom at `p_dmp = 0.1`, four stages: orientation saturates at `e ≈ 1.69` (ρ = 0) rising to 6.48 (ρ = 0.9); translation at 2.00 rising to 4.82. The historical `0.25 … 1.00` axis is realizable everywhere. At `p_dmp = 0.2` translation binds at `e ≈ 0.29` for ρ = 0.
- Phase 4 cost 23.3 core-hours for 1,900 units at n = 300, 199 permutations (~44 s/unit).

## Goals / Non-Goals

**Goals:**

- A study config can declare a crossed grid of generator/evaluation axes and get one complete, matched, anchored power grid per design point, with headroom enforced per point, without touching generator or evaluation semantics.
- Every existing config enumerates and reports exactly as before (no `design_grid` → no new cells, no new files, identical continuity table).
- The design-point decision is read off recorded covariates (eigengap, `angle` null width, selected components) as well as rejection rates, and the rule that picks the design point is declared in the config, not in the reader's head.

**Non-Goals:**

- Generalizing `axes` into an arbitrary factorial design language. The design grid is a flat cross of named axes; nesting, exclusions, or per-mode grids are not supported.
- Design-point-specific acceptance gates for the paper-grade study. The decision output is advisory.
- Any change to how the primary column feeds the Phase 4 gate, power curves, or acceptance targets.
- Varying `forced_components` (item 3).

## Decisions

### D1 — A separate `design_grid` block and a new phase, not an extension of `axes`

`axes` keeps its OFAT meaning; `design_grid.axes` is crossed. Design-grid cells get phase `power_design` and metadata `design_point: {"<axis>": value, ...}` (all declared axes, including the baseline column's own values) alongside the usual `trajectory_mode`/`effect_size`.

*Why:* every reader that filters `varied_axis is None` (curves, gate, targets) must keep seeing only the baseline column; a distinct phase makes the new cells invisible to those readers by default and visible to the new tables by opt-in (`POWER_PHASES` gains `power_design`). Reusing `power_ofat` with multi-valued `varied_axis` would silently change what the OFAT report columns mean.

*Alternative rejected:* emitting the crossed cells as `power_primary` with extra metadata. Cheaper, but the Phase 4 gate and acceptance targets would then average over columns unless every reader learned to filter on `design_point`, which is the kind of implicit contract the audit spent P1–P3 removing.

### D2 — The baseline column is required and is the `power_primary` grid

`design_grid` MUST include the baseline value for every declared axis (validation error otherwise). The baseline column is *not* re-emitted as `power_design`; the existing `power_primary` cells and shared anchor are that column, and the design-point tables read them under the baseline coordinates.

*Why:* avoids duplicating the baseline datasets in one seed family (which `_require_distinct_primary_datasets` would correctly reject), keeps the primary curves as one of the design points so a pilot's baseline is directly comparable to Phase 4, and guarantees the acceptance/gate machinery has a column to read.

### D3 — One matched-seed family across the whole grid; one anchor per column

All `power_design` cells join `matched_seeds.primary_family`. Each non-baseline column gets its own `none`/`e = 0` anchor cell (phase `power_design`, `zero_effect_anchor: true`, `resolves_modes` as today).

*Why:* the ρ axis thresholds the same uniform block at ρ = 0 and ρ > 0 (byte-identity at ρ = 0 was a P4 acceptance criterion), so sharing the seed across ρ at fixed n pairs the comparison at the baseline indicator draw — a strictly better estimate of the ρ effect for free. Across n the streams diverge harmlessly. Distinct columns differ in generator identity, so the same-family duplicate guard still rejects only true duplicates. Per-column anchors are needed because the anchor's dataset depends on ρ and n.

*Alternative rejected:* a family per column (`primary_family:<coords>`). Cleaner to explain, but throws away the paired ρ comparison and offers nothing the duplicate guard does not already provide.

### D4 — Type I cells stay at the baseline; per-column anchors are the nulls elsewhere

`enumerate_type_i_grid` and `_negative_control_cells` are unchanged (baseline `none`, translation controls, OFAT nulls from `axes`). The `none`/`e = 0` anchor in each design column is the null measurement at that point and is reported as such in `design_point_operating.csv` (mode `none`, effect `0.0`).

*Why:* enumerating a Type I baseline per column would double the null cells for no information gain; the anchor is already an independent-of-mode null draw at that column.

### D5 — Reporting: a design-point table, and the continuity table keyed on the full point

- `design_point_operating.csv`: one row per (design point coordinates as columns, mode, effect, statistic). Carries `n_cells`, `n_replicates`, rejection rate + MC SE, pooled-eigengap mean/terciles, `angle` null `q95` median/IQR/SD, and selected-component median/min/max (from the recorded `selected_components`). Written only when at least one `power_design` record exists. Built in `spectrum.py` beside `resolve_orientation_by_continuity`, reusing its helpers.
- `resolve_orientation_by_continuity` bucket key becomes `(ρ, other design coordinates, mode, effect)`; the extra coordinates are emitted as columns only when present. Without a design grid there are no extra coordinates and the frame is byte-identical to today's.

*Why:* pooling the ρ = 0.5 rows across n = 300/600/1200 would report an average of three different operating points as one number — the same mistake as the censored-axis duplicates, one level up.

### D6 — Decision rule lives in `acceptance.design_point` and is advisory

Config block: `{"trajectory_mode": "orientation", "statistic": "angle", "min_power_at_top": 0.80, "confirmation_se_threshold": 1.0, "prefer": ["evaluation_or_generator axis names in preference order"]}`. Evaluation, per design point, takes the target statistic's rejection rate at the top *realized* effect (largest effect enumerated in that column, all uncensored by construction), marks the column `meets` if `rate − threshold·SE ≥ floor`, `marginal` if `rate ≥ floor` but unconfirmed, else `fails`. The chosen point is the first `meets` column in preference order (`generator.n_samples` ascending, then `generator.baseline_continuity` ascending for the pilot). No `meets` column → verdict `revise_claim` with the table of columns and their statuses. Output: `design_point_decision.json` (+ CSV of per-column statuses). It does not feed the Phase 4 gate.

*Why:* the readiness worklist says a failed floor means revising the method or the claim, not the Monte Carlo size; the report should say which of those it is in one place. Preference order is declared so "smallest defensible" is not decided after seeing the numbers: smaller n is cheaper for Phase 5 and for any real cohort; lower ρ is the more general claim.

### D7 — Pilot profile

`phase5_design_point_pilot.json`: Phase 4 generator/evaluation values except `p_dmp: 0.1` and no `surgery_censoring` key; `design_grid.axes = {"generator.baseline_continuity": [0.0, 0.5, 0.8], "generator.n_samples": [300, 600, 1200]}`; modes `orientation` and `translation`; effects `[0.0, 0.25, 0.5, 1.0]` (the `0.0` entry is what turns on the shared anchor); 100 replicates; 199 permutations; `n_jobs: 1`; attribution disabled; Phase 4 gate disabled; acceptance `type_i` control target kept; `design_point` block as in D6; `matched_seeds` on with the shared anchor.

Size: 9 columns × (1 anchor + 2 × 3) = 63 design/primary cells, plus baseline Type I and translation controls — 6,500 work units. PLS double CV scales roughly linearly in n, so columns weigh ~1 : 2 : 4 across n; expected cost was estimated at ~150 core-hours; the 2026-09-04 local rehearsal (4 reps, 49 perms, all 65 cells, zero failures) measured ~30 s + 0.5 s/perm per unit at n = 300 and ~72 s + 2 s/perm at n = 1200 on a workstation core, i.e. ~135/250/460 s per unit at 199 perms for n = 300/600/1200 — ~500 workstation-core-hours, or ~170 at Phase 4's cluster per-core speed (~2.7 wall-hours on 64 shards). Shards must pin BLAS to one thread. `magnitude` and `shape` are left out (decided 2026-09-04): they reached power 1.00 at n = 300 in Phase 4 and only gain from larger n, so their response along ρ is measured in the Phase 5 study at the chosen point rather than paid for here.

*Why p_dmp = 0.1 rather than keeping 0.2 and dropping translation:* the ρ = 0 column is the declared stress-test endpoint and the Phase 4 comparator; at 0.2 the only realizable shared axis under translation's headroom is `e ≤ 0.29`, which cannot be compared with anything Phase 4 measured. Changing `p_dmp` is a baseline change that touches every mode, which is why the pilot re-measures all four rather than orientation alone.

### D8 — Records and signatures

No generator or evaluation field changes, so `parameter_signature` is unchanged for existing cells; the new phase string enters `cell_id`/metadata only for new cells. No version bump. Merge and resume logic are untouched.

## Risks / Trade-offs

- [ρ = 0.8 columns may make orientation trivially easy while still being labelled "semi-synthetic".] → The report presents power beside the recorded eigengap distribution per column; the readiness write-up must state the claim as continuity-conditional and point to the eigengap as the real-data observable, not to ρ.
- [n = 1200 columns dominate compute; a failed shard there costs most.] → Resumable shards; partition is by (cell, replicate) so a rerun touches only missing units. Submit n = 1200 shards first if the queue allows.
- [The design-point decision could pick a column that meets the floor on `angle` while another statistic misbehaves there (e.g. Type I drift at large n).] → `design_point_operating.csv` includes the per-column anchor's Type I rates for all three statistics; the findings report must check them before adopting the column, and the decision JSON lists the anchor rates beside the verdict.
- [A shared seed family across columns could be misread as "same datasets".] → Column metadata carries the coordinates; the report's provenance section states what is paired (baseline uniform block) and what is not (post-transform sampling, n-dependent streams).
- [Selected components will saturate at `n_stages − 1 = 3` in every column, as in Phase 4.] → Recorded, not varied; the findings report hands the retained-rank question to item 3 explicitly rather than reading anything into a constant.

## Migration Plan

Additive. Existing configs load unchanged (`design_grid` absent → no design cells, no new report files). Existing merged result sets re-report identically. Rollback is removing the config block; no persisted-record format changes.

## Open Questions

- Report date/filename for the findings report is fixed at run time (`docs/reports/phase5-design-point-pilot-<run date>.md`).
