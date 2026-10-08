# Design

## Context

See proposal.md for motivation. The facts that shape the approach:

- The uniform-δ construction already exists as `_magnitude_uniform_probe` in `semisynthetic.py`, reachable
  only through the private `_probe_uniform_delta` keyword on `generate_semisynthetic_trajectory`, and the
  study loader refuses any kind outside `_MAGNITUDE_KINDS`. The diagnostic spec requires that refusal.
- Group B's transform returns before reading `magnitude_kind` whenever `e = 0`, and no non-magnitude mode
  reads it. Adding a value therefore cannot change the shared anchor or any other mode's data.
- `parameter_signature` hashes the generator dataclass with defaults resolved. Adding an enum value changes no
  existing hash; flipping the default would change the hash of every magnitude cell in every unpinned
  historical config, including the Phase 5 anchor.
- The study config has one `effect_sizes` list shared by all modes; acceptance blocks are validated against
  `trajectory_modes`, so a magnitude-only profile must carry a magnitude-only rule set.
- Realized `delta`/`angle`/`shape` are already recorded per replicate at the analytic population,
  standardized population, observed standardized, and PLS-latent checkpoints
  (`realized-geometry-diagnostics`), so the "realized joint `delta` covariate" needs no new instrumentation.
- Matched seeds derive each generator seed from (base seed, family, replicate index). Two configs sharing
  those produce the same group A baseline per replicate; the anchor is then byte-identical across configs.
- Phase 5 cost about 69 s per unit at 999 permutations and the design-point pilot about 56 s at 199; the
  double-CV PLS fit dominates, so a pilot saves time only through its replicate count.

## Goals / Non-Goals

**Goals:**

- One new production value, `joint`, selected everywhere through the public `magnitude_kind`.
- No change to any existing record's parameter signature, any committed Phase 5 artefact, or the default.
- An effect grid chosen from realized size, so the magnitude `delta` power curve is informative below
  saturation while e = 1.00 stress-tests the controls.
- A gate decision about magnitude alone, at Phase 5 precision, with the anchor's reproduction of the Phase 5
  anchor stated as a check.

**Non-Goals:**

- Making `extremes` or `shape_kind='magnitude'` joint-scaled (both remain methylation-only).
- Per-mode effect sizes in the study config.
- Coupling δ magnitudes through the cascade (InterSIM has no channel for it; it would be a generator-model
  change).
- Rescaling `joint`'s effect axis to the `all` construction's realized sizes.

## Decisions

### D1. `joint` is the native per-omic size ratio; the two axes are related by recorded realized size

The bracket from the diagnostic (population-standardized joint `delta`, four stages) shows the two
constructions differ by a near-constant factor: `all` 5.95 / 10.92 / 14.94 / 18.14 against `joint`
14.96 / 27.12 / 36.91 / 44.80 at e = 0.25 / 0.50 / 0.75 / 1.00, ratio 2.47–2.52. A rescaled grid
(~{0.10, 0.20, 0.30, 0.40}) would therefore land on the old curve's realized sizes, but the old curve is
saturated at 1.000 from its first point, so matching it carries no information and would tie a production
mode's units to an impure construction. Under `joint`, `e` has an exact meaning — group B's native-space
trajectory is `(1 + e)` times group A's in every block — and the realized joint `delta` at every checkpoint
is already recorded per replicate. The report relates the axes through that covariate.

Caveat the report must carry: the exact ratio holds in native units. After pooled per-block standardization
the pooled standard deviation includes the between-group spread, which grows with `e`, so the realized
standardized ratio is smaller than `1 + e` and sub-linear in `e`. The bracket CSV makes that visible.

*Alternative rejected:* rescaling (above); a per-mode `effect_sizes` feature (unneeded once the profile is
magnitude-only).

### D2. Sibling value; default stays `'all'`; the flip is a later change

The censoring-policy change flipped a default by pinning the old value into every historical config first
(all ten configs name `surgery_censoring` today), which preserves signatures because the dataclass value is
unchanged. The same recipe applies to `magnitude_kind`, so the flip is cheap — but it should follow the
re-measurement, not precede it. The exit review's instruction is to re-measure and then lift the claim *or
report it as failed on its own terms*; the production default must not already point at a construction whose
controls have not been measured at paper grade. The CLI, showcase, and every historical config therefore keep
`'all'` by omission. The follow-up change (if the re-measurement passes) pins `"all"` into the ten configs,
flips the dataclass default, rewrites the spec's default scenario, and updates the showcase description.

*Alternative rejected:* flipping now with pinning (wrong order of evidence and adoption); flipping without
pinning (breaks reproduction of every committed result).

### D3. Naming: `joint`

`magnitude_kind` conflates two axes — stage scope (`all` stages vs `extremes`) and omic scope (methylation
only vs all three). The value `all` is already documented as "a uniform δ scale", so `uniform` would be
actively misleading. `joint` matches the vocabulary the diagnostic tables and reports already use for the
three-block space ("joint `delta`"). A second field (`magnitude_scope`) would be cleaner in the abstract but
adds a config key for one cell and a second validation path; not worth it for one value.

### D4. Code shape: one enum value, one branch, no private flag

`_magnitude_uniform_probe` becomes the `joint` branch of `_magnitude_methyl` (its body is already correct:
`methyl_a.copy()`, all three δ scaled, truth notes). `joint` enters `_MAGNITUDE_KINDS` and the
`MagnitudeKind` literal; the loader's set membership check accepts it with no edit (its explanatory comment
is rewritten). `_probe_uniform_delta` is removed from `generate_semisynthetic_trajectory` and
`_transform_group_b`; `specificity.py`'s uniform-δ comparison passes `magnitude_kind` instead and labels its
rows `all` / `joint` (the committed diagnostic CSV keeps its historical `production` / `uniform_probe`
labels — it is a frozen output; only newly written rows use the new labels). Truth notes keep
`delta_methyl_scale`, `delta_expr_scale`, `delta_protein_scale` and drop `probe_only`. The CLI's
`--magnitude-kind` choices gain `joint`. The RNG call sequence is untouched (the branch consumes no
randomness), so `all` and `extremes` remain byte-identical at every seed.

### D5. The analytic bracket lives in the diagnostic script and writes to a new dated directory

`scripts/magnitude_construction_diagnostic.py` already has the population-geometry path (step 3) that
computes realized geometry with no sampling. A `--bracket` mode runs it for both kinds over a dense grid
(e.g. 0 to 1 in steps of 0.01, plus the Phase 5 grid points) and writes `effect_axis_bracket.csv` to
`results/magnitude-axis-bracket-<date>/`, leaving `results/magnitude-construction-2026-09-10/` frozen. The
grid rule for the profiles: include `0.00` and `1.00`; include at least two nonzero values below `0.25`
chosen so their realized joint `delta` straddles the realized size at which the `all` construction's curve is
already saturated (`all` at e = 0.25 realizes 5.95 and power 1.000, so the rise for `joint` lies below
e ≈ 0.10); the candidate grid is `{0.00, 0.02, 0.05, 0.10, 0.25, 1.00}`, confirmed or adjusted by the bracket
*before* the pilot config is committed. The paper-grade grid equals the pilot grid unless the pilot shows the
rise was misplaced, in which case the adjustment and its reason are recorded in the addendum.

### D6. Three rungs; magnitude-only paper grade; anchor re-run, translation not

- **Bracket** (minutes, workstation): picks the grid.
- **Pilot** `phase5_magnitude_pilot.json`, 50 × 199 (~350 units, ~6 core-hours): confirms the bracket
  against the real `delta` null, checks the controls sit at α before paper-grade spend, and surfaces any
  small-`e` interaction with CV rank selection (selected components are recorded per replicate).
- **Paper grade** `phase5_magnitude_remeasurement.json`, 500 × 999 (~3,000–3,500 units, ~60–70 core-hours,
  ~15 min wall on 100 shards): gate rules reduced to the three magnitude rules, `control_modes: ["none"]`,
  specificity targets exactly the two mandatory controls, same `report_contract`, attribution absent.

The anchor is re-run (500 units) because the gate requires its records present; it reproduces Phase 5's
anchor byte for byte (same base seed, family `phase5-primary`, same generator and evaluation params), and the
addendum states the check (record-level `parameter_signature` differs — the dataclass differs in
`magnitude_kind` and the phase cell set — so the check compares generated data and RRPP outputs, not
signatures). Translation, orientation, and shape are excluded: byte-identical to Phase 5 records, findings
already stand. A full 19-cell re-run would triple the compute to regenerate existing records and re-issue
findings under a second config name.

*Alternative rejected:* one config with CLI overrides for replicate/permutation counts (Phase 5 precedent is
one committed file per run, for provenance).

### D7. Reporting: an addendum, not a re-issue

`docs/reports/phase5-magnitude-remeasurement-<date>.md` follows `phase5_report_template.md` with §7 (drivers)
and the orientation/shape parts of §5–§6 marked not applicable. The Phase 5 report and exit review are not
edited; the roadmap's Phase 5 section, "Not yet established" magnitude bullet, and "Next three changes" are
updated to point at the addendum.

## Risks / Trade-offs

- [Bracket misplaces the rise; the pilot's curve is flat or all-floor] → the pilot exists for this; adjust
  the paper-grade grid and record why. The bracket is population geometry; the null width is the sampled
  quantity it cannot see, which is what the pilot adds.
- [At tiny `e` the realized size sits inside the Monte Carlo floor; power ≈ α looks like "no effect"] → that
  is the correct reading of a power curve's left end; the report states realized size beside each point.
- [Small `e` changes CV rank selection modally away from 3] → recorded per replicate
  (`phase4_pls_selection.csv`); reported, not gated.
- [The anchor does not reproduce Phase 5's anchor] → a reproducibility break that must be explained before
  the paper-grade run is read; the pilot (same seeds, first 50 replicates) catches it early.
- [A control fails under `joint` at paper grade] → report it as failed on the construction's own terms with
  the revision it implies; do not adjust the grid or the bound post hoc.
- [Removing `_probe_uniform_delta` breaks a caller] → only `specificity.py` and one test use it; both are in
  the touch list.
- [`joint` at e = 1.00 realizes 2.47× the production size; the controls face a harder test than Phase 5's] →
  intended; a pure construction's off-target rates do not depend on `e`.

## Migration Plan

No user-facing migration. Existing configs, records, and the showcase are unaffected. The private
`_probe_uniform_delta` keyword is removed in the same change that makes `magnitude_kind="joint"` available;
no external caller exists. Rollback is removing the enum value; no persisted artefact depends on it except
the new results directory.
