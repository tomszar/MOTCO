# Design: add-baseline-continuity-axis

## Context

See `proposal.md` — Why. The pieces the design must fit around:

- `generator.bernoulli_indicators(rng, n_feat, n_cell, p)` currently draws one uniform block
  `rng.random((n_feat, n_cell))` and thresholds it at `p` (`generator.py:104`). It is used both by
  `semisynthetic._baseline_methyl` (stages as cells) and by `motco simulate` (clusters as cells).
- `semisynthetic._transform_group_b` operates on the *realized* baseline indicators (permutation,
  relocation, scaling), so the surgery machinery is agnostic to how the baseline was drawn.
- `expected_surgery_headroom` (`semisynthetic.py:698`) uses the independence union
  `a = 1 − (1−p_dmp)^n_stages` for the expected stage-active fraction; `study/enumerate.py` calls
  it to reject over-headroom cells at config time.
- `grid.parameter_signature` hashes `_to_jsonable(cell.generator_params)` — any new dataclass field
  changes every signature, and pre-change shards refuse resume by design (the
  `SHAPE_STATISTIC_VERSION` precedent).
- Study configs already support arbitrary `generator.<field>` axes
  (`study/config.py:_validate_axis_namespace`), so sweeping the new field needs no axis machinery.
- `config_spectrum` (P1) already persists the pooled/per-group eigengap per replicate;
  `report/eigengap_stratified_power.csv` is the existing stratified-report precedent to mirror.
- RRPP conditioning caution (audit S1/sequencing): the change must stay entirely inside the
  generator's baseline; nothing group-aware may enter integration.

## Goals / Non-Goals

**Goals:**
- One scalar knob, `baseline_continuity` (ρ), that moves the baseline from isotropic
  (ρ = 0, today's behavior, byte-identical) to strongly trending, with per-stage marginals held at
  `p_dmp` throughout.
- Analytic headroom that remains exact-in-expectation along the axis, so enumeration's fail-loud
  guarantee carries over.
- Reporting that ties power along the axis to the recorded eigengap, not just to the knob.

**Non-Goals:**
- No change to surgery transforms, statistics, RRPP, integration, or the viz layer.
- No continuity for the `motco simulate` cluster generator (clusters are unordered; continuity is a
  stage-order concept). `bernoulli_indicators`'s existing signature and call sites stay valid.
- Not running the Phase-5 design-point study; this change only makes it runnable.

## Decisions

### D1 — Mechanism: stationary first-order Markov chain, not cumulative programs

Per CpG, across stages in order: `x₁ ~ Bern(p)`; `P(x_t=1 | x_{t−1}=1) = p + ρ(1−p)`;
`P(x_t=1 | x_{t−1}=0) = p(1−ρ)`, with `p = p_dmp`.

- Stationarity keeps the Bernoulli(p) marginal at every stage, so per-stage indicator counts, δ
  semantics, the OR-cascade coupling, and every downstream consumer of "how many features are
  differential at stage t" are invariant along the axis. Comparisons across ρ isolate *geometry*.
- `corr(x_t, x_s) = ρ^|t−s|` gives `E‖μ_t − μ_s‖² ∝ 1 − ρ^|t−s|`, monotone in stage separation —
  a discretized-random-walk-like configuration with a dominant PC1 and a growing eigengap. That is
  exactly the "trajectory has a direction to differ in" regime.
- **Alternative rejected — cumulative/accreting programs** (stage t's set contains stage t−1's):
  marginal differential probability grows with stage, so per-stage counts, coupling density, and
  the headroom pools all become stage-dependent; the knob would confound geometry with abundance.
- **Alternative rejected — shared-core mixture** (a fixed core program plus per-stage innovations):
  equivalent to exchangeable correlation, which *shrinks* the configuration uniformly without
  ordering the stages — it raises overlap but not trend, so PC1 stays ill-conditioned.

### D2 — RNG layout: one uniform block, threshold per stage

Implement as: draw `U = rng.random((n_cpg, n_stages))` (the *same single block, same shape, same
position in the stream* as today), then set `x₁ = U₁ < p` and
`x_t = U_t < (stay if x_{t−1} else enter)`. At ρ = 0 both thresholds equal `p`, so every comparison
is literally today's comparison → byte-identical datasets at ρ = 0 with zero RNG-stream drift for
everything drawn after the indicators. This is the property that lets existing tests, fixtures, and
matched-seed reasoning survive untouched.

- Lives in `generator.py` as a new function (e.g. `markov_indicators(rng, n_feat, n_cell, p, rho)`)
  rather than a new parameter on `bernoulli_indicators`, keeping the simulate-command call site
  untouched; `_baseline_methyl` switches to it unconditionally (ρ defaults to 0).

### D3 — Parameter surface and validation

`SemiSyntheticTrajectoryParams.baseline_continuity: float = 0.0`, validated to `[0, 1)` in
`_validate_params` (ρ = 1 is a degenerate constant program; excluded). Recorded in the truth
`params` block like every other knob. Group B never draws indicators — it transforms A's — so the
axis composes with every trajectory mode with no per-mode code.

### D4 — Headroom: closed-form Markov union

Replace `active_fraction = 1 − (1−p)^n` with
`active_fraction = 1 − (1−p)·(1 − p(1−ρ))^(n−1)` (probability the chain never visits 1 in n steps:
start at 0 with prob `1−p`, then stay via `P(0→0) = 1 − p(1−ρ)` for n−1 transitions). Reduces to
the independence form at ρ = 0. Orientation's source pool (stage-active union) shrinks with ρ and
the relocation destination pool (its complement) grows — headroom *improves* along the axis for
orientation and shape-relocate. The translation pool goes through the same `active_fraction` into
the incidence-based computation unchanged. The guard-band variance formula keeps using the pooled
binomial approximation; a Monte Carlo test pins its adequacy at high ρ (see Risks).

### D5 — Signature break is accepted, not versioned around

The new dataclass field flows into `parameter_signature` automatically; pre-change shards refuse to
resume. This is the established policy for generation-affecting changes and needs no new version
key — the params hash *is* the version. Historical committed results keep their recorded values;
comparisons across the boundary must note the field's arrival.

### D6 — Reporting: continuity-resolved table from persisted records only

A new report output (sibling of `eigengap_stratified_power.csv`, e.g.
`continuity_resolved_orientation.csv`) emitted only when the merged set contains cells differing in
`generator_params.baseline_continuity`: per continuity value × mode × effect, the per-statistic
rejection rates, eigengap distribution summaries (median/terciles from `config_spectrum`), and
`null_summary["angle"]["q95"]` dispersion. Everything needed is already persisted per record
(P1 + the null summaries), so the report is computable from `merged.jsonl` alone — no regeneration.

## Risks / Trade-offs

- [Guard band mis-sized at high ρ] The headroom guard band uses a binomial SD for the pool size,
  but under the Markov baseline the union count has a different variance. → Keep the 3σ band,
  validate by Monte Carlo across ρ ∈ {0, 0.5, 0.9} in tests; if the realized censoring rate under
  `"error"` is nonzero at the analytic saturating effect, widen the band for ρ > 0 before merging.
- [Trend direction is feature-identity-random] The Markov baseline gives each replicate a trending
  configuration but a random *which-features* trend; the orientation surgery still permutes within
  the active union. The estimand becomes well-conditioned, which is the goal — but effect-size
  semantics along ρ should be read through realized-geometry diagnostics, not assumed constant. →
  The continuity-resolved report includes the population-geometry diagnostics already persisted.
- [Silent expectation drift downstream] Some tests or docs may implicitly assume independent stage
  programs (e.g. expected overlap counts). → ρ defaults to 0 with byte-identity, so only code that
  explicitly opts in sees the new regime; grep for `bernoulli_indicators` consumers during apply.
- [Resume break surprises a running study] Any in-flight shard set cannot resume across this
  change. → None are in flight (Phase 4 closed, Phase 5 not launched); note it in the change log
  anyway.

## Open Questions

- Which ρ values Phase 5 sweeps (e.g. {0, 0.3, 0.6, 0.9}) is a study-config decision for the
  design-point work, not part of this change.
