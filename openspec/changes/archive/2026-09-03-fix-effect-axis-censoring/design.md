# fix-effect-axis-censoring — Design

## Context

See `proposal.md` — Why. Mechanically, two clamp sites exist in
`src/motco/simulations/semisynthetic.py`:

- `_relocate_rows` (`k = min(round(e·|src_pool|), |dst_pool|)`) — used by `_orientation_methyl`
  (source: stage-active CpGs, destination: stage-inactive CpGs) and `_shape_methyl` with
  `shape_kind='relocate'` (source: one stage's active CpGs, destination: globally inactive CpGs).
- `_translation_methyl` (`n_extra = min(n_extra, len(candidates))`) — candidates are stage-inactive
  CpGs whose mapped gene is outside the stage program.

The pools are **random per replicate** (they depend on the baseline indicator draw), so whether the
clamp binds is a per-dataset event, not a pure function of the config. With `p_dmp = 0.2` and four
stages the expected stage-active fraction is `1 − (1 − 0.2)⁴ ≈ 0.59`, which is why orientation
saturates near e ≈ 0.69 in expectation.

Constraints that shape the design:

- The existing spec scenario "no two enumerated primary cells generate identical datasets at the
  same replicate index" (trajectory-power-study, Phase-4 profile requirement) is *already violated
  in effect* by the clamp; the current `_require_distinct_primary_datasets` guard compares
  parameters and cannot see it. This change makes that scenario actually enforceable.
- The committed Phase-4 profile (`examples/trajectory_power_study/phase4_pilot_100x199.json`) is
  spec-required to stay loadable and enumerable as the historical record, yet contains censored
  cells (orientation/translation at e ∈ {0.75, 1.00}).
- `truth_metadata` in replicate records is an open dict that already flows through JSONL
  persistence (`evaluation.py:280`), so no record-schema version bump is needed — unlike P1, which
  added a top-level record field.

## Goals / Non-Goals

**Goals:**

- No silent clamping anywhere; requested-vs-realized explicit in every record; censoring caught at
  config time for study runs.
- Byte-identical duplicate cells detectable from records alone (no regeneration).

**Non-Goals:**

- Choosing a new Phase-5 effect axis (renormalized fractions, lower `p_dmp`, within-active
  relocation, or a lower axis top). That is readiness item 4's design-point decision; this change
  only guarantees whichever axis is chosen is non-degenerate.
- Re-running Phase 4, changing any statistic/test, or touching P4's baseline-continuity axis.

## Decisions

### D1 — Policy parameter: `surgery_censoring: Literal["error", "clamp"]`, default `"error"`

New field on `SemiSyntheticTrajectoryParams`, threaded to both clamp sites. Default is fail-loud.

- *Why not default `"clamp"` (backward compatible)?* The silent clamp is the defect, and there is
  no compatibility to preserve: adding the field changes `parameter_signature` regardless, so old
  shards refuse to resume either way. A permissive default would reintroduce the audit's
  "indefensible option" as the path of least resistance.
- *Why keep `"clamp"` at all?* (a) The committed Phase-4 profile must stay enumerable as a
  historical record (see D4); (b) exploratory/showcase use where partial surgery is acceptable.
  Under `"clamp"` the realized behavior must reproduce the old generator replicate-for-replicate at
  the same seed (same RNG call sequence — the policy check must not consume RNG draws).
- *Why not a third `"renormalize"` policy (map e onto realized fractions)?* Pool sizes are random
  per replicate, so renormalizing makes the meaning of `e` depend on the baseline draw — the axis
  label would vary across replicates within one cell. If a renormalized axis is wanted for Phase 5,
  it should be chosen deliberately as config values that respect headroom, not silently per draw.

The error message names the mode, requested size, available pool, and the saturating effect implied
by this draw's pools (`e_sat = |dst_pool| / |src_pool|` for relocation).

### D2 — Truth metadata: add nominal size and censored flag beside existing keys

Existing keys (`orientation_relocated`, `translation_set_size`, `shape_relocated`) keep their names
and meaning — report code and the F2 audit analysis already read them. Added per pool-limited
surgery: the nominal size (e.g. `orientation_nominal`), and `censored: bool`
(realized < nominal). Under `"error"` a successful record always has `censored: false`; the flag
exists so `"clamp"` records are self-identifying downstream.

### D3 — Summary surfacing: a dedicated `summarize_realized_surgery`, not more fields on `SimulationSummaryResult`

`summarize_rejection_rates` emits one row per (cell, statistic); surgery is a per-cell property and
would be triplicated there. Instead, a cell-keyed summarizer (pattern: the existing
`summarize_realized_geometry`) returns per cell: mode, nominal size, mean/min/max realized size,
censored fraction. Report side: a new `realized_surgery.csv` in the study report, and the power-
curve frame gains the censored fraction so curves can be annotated. Modes with no pool-limited
surgery report the fields as absent (None), not zero.

### D4 — Enumeration headroom check: analytic expectation with a concentration guard band

New validation in `study/enumerate.py` beside `_require_distinct_primary_datasets`, applied when
the cell's `surgery_censoring` is `"error"` (clamp opt-in is exempt — the runtime records
censoring instead):

- Orientation: expected active fraction `a = 1 − (1 − p_dmp)^n_stages` over `n_cpg` CpGs;
  requested `k = e·a·n_cpg` must not exceed the expected pool `(1 − a)·n_cpg` minus a guard band of
  3 binomial standard deviations (`3·√(n_cpg·a·(1−a))`) — the pools are random, and a config that
  passes only in expectation would still fail loudly at runtime on unlucky draws. With
  n_cpg ≈ 367 the band is ~9% of the pool: conservative, cheap, and documented.
- Shape/relocate: same arithmetic with the per-stage source pool (`p_dmp·n_cpg`); with the interior
  stage's sources vs global-inactive destinations this never binds at sane parameters, but the
  check is generic rather than mode-cased.
- Translation: the fresh-pool expectation depends on the CpG→gene incidence, so compute it from the
  cached reference maps (`load_reference()` is already R-free and cheap) rather than approximating.

The error names the cell, requested effect, and the saturating effect. *Alternative considered:*
probe-generate one dataset per cell at enumeration time — rejected: tests one seed only, couples
enumeration to generation cost, and gives a noisier answer than the expectation + band.

### D5 — Duplicate-construction flag computed from records, no regeneration

Two matched-seed cells at the same replicate index generate byte-identical datasets **iff** they
share the generator seed and the realized surgery is identical — and since the RNG draw sequence
depends only on the seed and the realized `k`, "both censored to the same realized size" is exactly
that condition. The report therefore flags, per same-family cell pair (same mode, different
effect), the fraction of replicate indices where both records are censored with equal realized
size; pairs above 5% are marked as duplicated constructions in `realized_surgery.csv` and the
report narrative. This reproduces the audit's 80/100 finding from records alone.

### D6 — The committed Phase-4 profile opts into `"clamp"`

`phase4_pilot_100x199.json` (and the signfix/pivotality configs that reproduce historical runs)
gain an explicit `"surgery_censoring": "clamp"` in their generator block, keeping them loadable and
enumerable as historical records whose realized behavior matches what was actually run. Their
signatures change (new field) — but resume into those result sets is already impossible after any
generator-params change, and the committed reports remain the record. `examples/.../README.md`
documents why the flag is set there and that new configs must not copy it.

## Risks / Trade-offs

- [Guard band too conservative near the boundary] → The 3σ band forfeits ~9% of headroom on this
  reference (n_cpg ≈ 367). Acceptable: an axis that needs the last 9% of pool headroom is already
  measuring a near-saturated construction; the error message reports the saturating effect so the
  config author can choose deliberately.
- [Runtime error despite passing enumeration] → Possible in principle beyond 3σ; `error-policy
  record` in the shard runner captures it as a failed replicate rather than killing the shard, and
  the failure message is self-explanatory.
- [Existing tests exercising e ≥ 0.75 orientation/translation break] → Intended; update them to
  explicit `"clamp"` where they test clamping, or to headroom-respecting effects elsewhere.
- [RNG-sequence drift under `"clamp"`] → D1 requires replicate-for-replicate equality with the old
  generator at the same seed; add a pinned-output regression test against the current
  implementation before refactoring `_relocate_rows`.

## Open Questions

None — the Phase-5 axis choice (which values replace the censored cells) is explicitly deferred to
readiness item 4, which this change unblocks.
