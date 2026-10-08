# adopt-joint-magnitude-construction

## Why

The Phase 5 paper-grade run ([report](../../../docs/reports/phase5-paper-grade-2026-09-10.md)) holds the
Phase 4 gate at **HOLD** on its two mandatory magnitude controls: the production magnitude surgery rejects
`angle` at 0.276 and `shape` at 0.742 at e = 1.00 against an α + 2·SE bound near 0.09, so the paper's
magnitude-specificity claim is withheld ([exit review](../../../docs/reports/phase5-exit-review-2026-09-10.md),
§4). The [magnitude-construction diagnostic](../../../docs/reports/magnitude-construction-diagnostic-2026-09-10.md)
established the mechanism and the fix: `magnitude_kind='all'` scales `delta_methyl` alone, which is exactly
size-pure *within* every omic block and impure only as the geometry of concatenating one block that grew
with two that did not; scaling all three per-omic δ together is exactly size-pure in the joint standardized
space (`angle` ~1e-6, `shape` ~1e-15, the anchor's floating-point floor). That candidate exists today only as
a diagnostic-only probe that the study loader refuses by design. The exit review names adopting it as the
first of the three consequent changes and the gating item for the paper's specificity section.

The biological framing supports the same construction on its own merits. InterSIM — and the numpy port the
fidelity battery validates against it — couples the three omics only through *which* features are
differential (the CpG→gene→protein incidence maps); the per-tier shift sizes `delta_methyl`, `delta_expr`,
`delta_protein` are independent, and the cross-omic correlation term uses reference constants, never the
generated methylation values. So the production construction describes a cascade whose methylation tier
strengthened while nothing downstream noticed. A construction where every tier scales together is the
natural meaning of "a magnitude difference", independent of the purity argument.

## What Changes

- **Promote the uniform-δ probe to a production `magnitude_kind` value named `joint`.** Under `joint`,
  group B's three per-omic δ are all scaled by `1 + e`, so group B's native-space trajectory is exactly
  `1 + e` times group A's in every omic block. The private `_probe_uniform_delta` keyword is retired; the
  diagnostic (`specificity.py`, `scripts/magnitude_construction_diagnostic.py`) selects the construction
  through `magnitude_kind` like any caller. Truth records `magnitude_kind: "joint"` and the three per-omic
  scale factors; the `probe_only` marker goes.
- **Sibling value, not a default flip.** `magnitude_kind` keeps its default `'all'`, no historical study
  config is edited, every existing parameter signature is unchanged, and the committed Phase 5 config,
  results and report are byte-identical. Flipping the default (with the pinning recipe used when the
  surgery-censoring default changed) is deferred to its own change once the re-measurement has passed.
- **Declare the effect axis natively.** For `joint`, `e` is the per-omic size ratio minus one, exactly — the
  cleanest effect-size definition in the study. It is *not* rescaled to match the production curve's realized
  sizes; the realized joint `delta` already recorded per replicate at every geometry checkpoint is the
  covariate that relates the two. The re-measurement's effect grid is chosen from an analytic
  population-geometry bracket (no sampling, no cluster) so the `delta` power curve shows a rise below
  saturation — production `delta` power is already 1.000 at e = 0.25, so the shared grid would draw a flat
  line of ones — while e = 1.00 remains in the grid as the stress point for the two controls.
- **Re-measure the magnitude controls on a three-rung ladder.** (1) The analytic bracket, as an extension of
  the diagnostic's population-geometry path over a dense `e` grid; (2) a committed 50 × 199 magnitude-only
  pilot at the Phase 5 design point on the chosen grid; (3) a committed 500 × 999 magnitude-only paper-grade
  profile deriving its generator and integration parameters from `phase5_power_study.json`, with a reduced
  Phase 4 gate — `magnitude`/`delta` mandatory power, `magnitude`/`angle` and `magnitude`/`shape` mandatory
  control, `control_modes = ["none"]` — the same report contract, and the shared zero-effect anchor re-run
  (it is byte-identical to Phase 5's anchor, which the report states as a reproducibility check). Translation,
  orientation and shape cells are not re-run: matched seeds make them byte-identical to the committed Phase 5
  records, and their findings already stand.
- **Report the re-measurement as a dated addendum to the Phase 5 findings report**, following the committed
  template section for section with the non-magnitude sections marked not applicable, and lift the withheld
  specificity claim or report it as failed on its own terms. The roadmap's magnitude bullet under "Not yet
  established", its "Next three changes", `docs/api/simulations.md`, and CLAUDE.md are updated to the
  post-run state. `motco simulate --magnitude-kind` accepts `joint`.

Out of scope: `magnitude_kind='extremes'` and `shape_kind='magnitude'` keep their methylation-only scaling
(neither is in the committed Phase 5 profile, and shape cross-talk is descriptive by design); any edit to the
Phase 5 config, gate roles, acceptance-target semantics, report contract or report template; a per-mode
`effect_sizes` feature in the study config (the magnitude-only profiles make it unnecessary); any change to
the `delta`/`angle`/`shape` statistics, RRPP, the PLS rank rule or the other surgeries; the Phase 6 case study.

## Capabilities

### New Capabilities

(none)

### Modified Capabilities

- `semisynthetic-trajectory-generator`: the magnitude-variant requirement gains the `joint` kind — all three
  per-omic δ scaled by `1 + e` — with its truth record; the existing "magnitude mode scales the methylation
  effect" scenario is qualified as the `'all'`/`'extremes'` behavior, and the default remains `'all'`.
- `magnitude-construction-diagnostic`: the requirement that the size-pure candidate "MUST NOT become a
  selectable production trajectory mode" is superseded — the candidate is now the production `joint` kind,
  the diagnostic selects it through `magnitude_kind`, and committed profiles that do not name a kind still
  resolve to `'all'`. A new requirement adds the analytic effect-axis bracket for `joint`.
- `trajectory-power-study`: two new fixed profiles — the Phase 5 magnitude re-measurement pilot and
  paper-grade configurations — with their reduced gate, and a requirement that the re-measurement findings
  are versioned as an addendum following the Phase 5 template.

## Impact

- `src/motco/simulations/semisynthetic.py` — `joint` in `_MAGNITUDE_KINDS` / `MagnitudeKind`; the probe body
  becomes the `joint` branch of `_magnitude_methyl`; `_probe_uniform_delta` removed from
  `generate_semisynthetic_trajectory` and `_transform_group_b`; module docstring.
- `src/motco/simulations/specificity.py` — the uniform-δ comparison selects by `magnitude_kind`; construction
  labels `production` / `joint`.
- `src/motco/simulations/study/config.py` — no behavioral edit (the loader reads the kind set from the
  generator); the comment explaining the deliberate absence is rewritten.
- `src/motco/cli.py` — `--magnitude-kind` choices gain `joint`.
- `scripts/magnitude_construction_diagnostic.py` — the analytic effect-axis bracket (dense `e` grid, realized
  joint `delta` at `population_standardized`, written to CSV).
- `tests/test_uniform_delta_probe.py` — the two "not selectable" tests invert; the size-purity test runs
  against the public value; a loader test accepts `joint` in a study config. `tests/test_semisynthetic*.py`
  gain the `joint` truth and scaling assertions; a config-loading test for each new profile.
- `examples/trajectory_power_study/phase5_magnitude_pilot.json` and `phase5_magnitude_remeasurement.json` —
  new committed profiles; README entry.
- `results/phase5-magnitude-<run date>/` — `report/` and `PROVENANCE.txt` committed; shards and merged JSONL
  gitignored. `results/magnitude-construction-2026-09-10/` gains the bracket CSV (or a dated sibling
  directory — design decides).
- `docs/reports/phase5-magnitude-remeasurement-<run date>.md` — the dated addendum.
- `docs/roadmap.md`, `docs/api/simulations.md`, `CLAUDE.md`, `examples/trajectory_power_study/README.md` —
  updated to the post-run state.
- Cluster — `/home1/tgonza/MOTCO` on `ing` fast-forwarded to the run revision; ≈ 6 core-hours for the pilot
  and ≈ 70 for the paper-grade run (about 69 s per unit at 999 permutations, measured in Phase 5).
- No change to parameter signatures of any existing record, to the headroom computation (magnitude is not
  pool-limited), or to the showcase (it uses the default kind).
