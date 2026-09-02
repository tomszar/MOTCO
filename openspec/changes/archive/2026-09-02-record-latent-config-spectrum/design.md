# Design — record-latent-config-spectrum

## 1. Which configurations, defined precisely

The spectrum is computed from the **fitted LS-mean vectors** (`obs_vect = LS_means @ betas`), not from raw
per-stage sample means:

- **Per-group configuration:** the k stage rows belonging to one contrast group (group-major, level-minor
  indexing, exactly as `contrast` enumerates them), column-centered.
- **Pooled configuration:** the per-stage average of the group rows (stage i averaged across groups),
  column-centered.

Rationale: the LS-mean rows are what the trajectory statistics measure, they are already in hand inside
`estimate_difference` (no extra solve), and — decisively — they exist **per permutation**, which raw sample
stage means would require recomputing. Under the study's balanced designs the pooled LS-mean configuration
coincides with pooled sample stage means, so the audit's evidence (computed from sample stage means) carries
over; a mirror test pins this equivalence on a balanced case.

Per configuration we persist the **normalized eigenvalue spectrum** (squared singular values of the centered
configuration divided by their sum — at most `min(k, d)` scalars) plus the **relative eigengap**
`(λ₁−λ₂)/Σλ` as a named scalar, so downstream consumers never re-derive it inconsistently.

Degenerate cases: with k = 2 stages the centered configuration has rank ≤ 1 and the relative eigengap is
identically 1 — recorded as such, documented as uninformative. A configuration with zero total variance
records an explicit null/NaN-free sentinel (eigengap absent) rather than dividing by zero.

## 2. How the spectrum reaches RRPP without touching the test

Preferred mechanism: `estimate_difference` gains an **opt-in keyword** (default off) under which it
additionally returns the spectra computed from the `obs_vect` it already fitted. `RRPP` gains a matching
opt-in flag that forwards it and returns the per-permutation pooled eigengap list alongside the three
existing distributions, in both the serial loop and `_RRPPWorker`.

Rejected alternative: computing spectra inside `RRPP` from `LS_means @ estimate_betas(...)` — that duplicates
the beta solve per permutation for no benefit.

Invariants, each pinned by a test:

- Default-off calls return exactly the pre-change tuple shapes — no caller outside the harness changes.
- The spectrum path consumes **no RNG**, so with equal seeds and `n_jobs` the permutation draws, null
  distributions, p-values, and observed statistics are byte-identical with the flag on or off.
- The per-permutation retention is a **summary only** (retained count, mean, sd, q05/q50/q95): full
  per-permutation spectra never enter the result or the record.

## 3. Signature policy: version bump, not silent additivity

The null-summary change was deliberately signature-neutral so pre-change shards stayed resumable. This
change takes the opposite policy, per the audit: the signature payload gains an explicit
`config_spectrum_version: 1` key (same pattern as `realized_geometry_version` and
`attribution_schema_version`).

Trade-off, made explicit: signature-neutral additivity would let an old shard resume and produce a merged
set where some records lack the spectrum — the exact silent-mix failure the Phase-5 stratified power table
cannot tolerate (a missing covariate is indistinguishable from a degenerate one at analysis time). The cost
is that pre-change shards refuse resume; the committed Phase-4 and pivotality **merged results and reports
are unaffected** (readers default the block to empty), only *resumption into* old shards is refused. No new
power runs are planned against old shards — the audit sequences this change before all of them.

## 4. Reporting and analysis surfaces

- **Study report:** orientation-mode power cells gain a rejection-rate table stratified by recorded pooled
  eigengap (tercile bins within the cell, with counts and Monte Carlo SEs), and the per-cell geometry
  summaries gain eigengap columns (pooled and per-group mean/quantiles). Stratification reads **recorded**
  values only — the report never regenerates datasets.
- **Pivotality analysis:** per cell and statistic, Spearman and log–log Pearson association between the
  recorded pooled eigengap and the replicate's own null q95, beside the existing observed-vs-null
  associations, with the same uncertainty treatment. The acceptance run re-executes the diagnostic-scale
  profile under the new schema and must reproduce the audit's association (negative and material for
  `angle` in the orientation cell; the audit measured Spearman −0.75 at n = 100 — the acceptance bound is
  directional and magnitude-aware, not a point match).

## 5. What this change refuses to do

- No change to any statistic or decision rule — the eigengap is a recorded covariate and reporting
  qualifier, not a filter, weight, or test modification.
- No group-aware use of the spectrum anywhere in integration or sizing: the audit's caution stands — any
  group-aware supervision voids the fixed-latent-space RRPP conditioning (S1). This change only observes.
- No new generator axes and no effect-axis policy: those are P4 and P2 respectively.
