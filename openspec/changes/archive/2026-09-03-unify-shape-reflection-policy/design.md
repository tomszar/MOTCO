# Design — unify-shape-reflection-policy

## Context

See proposal.md for motivation (audit finding F3). The relevant current state:

- `_proper_procrustes_distance` (`src/motco/stats/trajectory.py:596`) computes the SVD alignment and
  flips the last left singular vector when `det(R) < 0`. When configuration rank < ambient dimension,
  at least one singular value is zero, so the flip acts in the null space and costs nothing — the
  constraint is vacuous and the returned distance already equals the reflection-allowed one. The
  constraint only binds at full rank (in practice: the `pls_latent` checkpoint, 3-D rank-3).
- `_OPA` (`trajectory.py:607`) carries its own reflection correction but has no production callers.
- The test suite pins reflection-distinctness only in 2-D
  (`tests/test_trajectory.py:299`, `test_shape_distance_keeps_reflections_distinct_by_default`).
- `grid.parameter_signature` (`grid.py:405`) guards resume compatibility with explicit schema-version
  keys (`seed_derivation_version`, `realized_geometry_version`, `integration_metadata_version`,
  `attribution_schema_version`, `config_spectrum_version`). There is no key covering the statistic
  contract itself.
- `_pls_integration` (`evaluation.py:565`) always selects the component count by `plsda_doubleCV`;
  there is no way to force a rank, which the readiness item-2 probe needs.
- Audit evidence constraining the design: allowing reflections changed the latent shape distance in
  0/100 regenerated pilot replicates, so the policy switch is expected to be a numerical no-op on all
  existing study geometry — a property the implementation should verify, not assume.

## Goals / Non-Goals

**Goals:**

- One reflection policy — full orthogonal alignment — implemented in one place, provably identical
  across all checkpoints and ambient dimensions.
- Old study shards refuse to resume under the new statistic contract.
- The rank-limited-projection account of orientation→shape becomes measurable (forced-rank probe).

**Non-Goals:**

- No change to `delta`, `angle`, RRPP mechanics, or generator semantics.
- No re-run of Phase-4; committed report outputs stay as historical records.
- No production change to PLS component selection — `forced_components` is diagnostic-only and never
  set in study configs.
- Not deciding readiness item 2 here: the probe produces the evidence; the predeclaration decision is
  made in the Phase-5 report contract, outside this change.

## Decisions

### D1 — Allow reflections everywhere (drop the determinant branch), not document-the-regime

The alternative in the audit (keep the proper-rotation estimator, document the rank threshold,
annotate every cross-checkpoint comparison with its regime) preserves a constraint that is (a)
vacuous in every space the realized-geometry checkpoints use, (b) active only in the latent space,
where its measured effect on real replicates is zero, and (c) a permanent annotation burden on every
future report table. Full orthogonal alignment is also the standard identification in shape spaces
whose ambient dimension exceeds configuration rank (trajectories of k stages have rank ≤ k−1, and
k−1 < ambient dimension at every pre-integration checkpoint). Implementation: delete the
`det(R) < 0` branch so `R = U @ Vt` unconstrained — the estimator becomes plain orthogonal
Procrustes.

### D2 — Delete `_OPA` rather than align it

It has no callers outside its own definition. Keeping a second Procrustes routine "aligned to the
same policy" invites the exact divergence this change removes. If a rotation-returning helper is ever
needed, it can be rebuilt from the distance routine's SVD.

### D3 — Version the statistic contract in the parameter signature

Add `"shape_statistic_version": SHAPE_STATISTIC_VERSION` (new constant in `trajectory.py`, imported
by `grid.py`) to the `parameter_signature` payload, following the established pattern of the other
explicit version keys. Rationale for placing the constant in `trajectory.py`: the contract belongs to
the estimator, not the study machinery; the study merely refuses to mix contracts. Old shards then
fail signature comparison on resume with the existing mismatch message — no new error path needed.

### D4 — Verify the expected no-op empirically in tests, not just claim it

Beyond the mirror-pair contract tests, add a regression test that computes the old (proper) and new
(orthogonal) distances on random full-rank 3-D configurations and asserts they agree whenever the
optimal alignment is already proper — pinning that the only behavioral divergence is genuine
mirror-like pairs. The audit's 100/100 identity on real replicates is the field evidence; this test
pins the mechanism.

### D5 — `forced_components` as an `integration_params` key, mutually exclusive with CV knobs

`integration_params["forced_components"]` (absent by default). When present: validate against the
feasible range (2 ≤ r ≤ min(n_features, n_samples), same clamp bounds as `max_components` but
rejecting rather than clamping — a diagnostic override that silently clamps would defeat its
purpose), skip `plsda_doubleCV` entirely, fit `fit_plsda_model(X, y, n_components=r)`, and record
`{"forced_components": r, "component_selection": "forced"}` in the integration metadata (production
runs record `"component_selection": "cv"`). Rejecting the co-presence of CV knobs is unnecessary —
they are simply unused and unrecorded when forced. Because `evaluation_params` are already hashed
into the parameter signature, forced-rank runs can never resume into or contaminate production
shards. The RRPP conditioning caution from the audit does not apply: forcing a stage-supervised rank
is not group-aware supervision, and the probe is diagnostic, not a production test.

### D6 — Probe driver shape

`scripts/latent_rank_probe.py`, in the mold of `scripts/geometry_specificity_probe.py`: regenerate
the orientation-cell replicates (matched seeds), evaluate each at `forced_components` in
{3, 4, 6, 9, 12} (3 = the CV-selected production point, the rest spanning toward the joint program
rank), and report per-rank observed `shape`, its rejection against the replicate's own RRPP null,
and the pooled-configuration eigengap (already recorded via P1's `config_spectrum`). Output: one CSV
plus a short markdown summary under `results/`, following the existing probe conventions. The
predeclaration criterion the probe informs: the orientation→shape response decaying toward the
population value as rank grows supports the projection-artifact account.

### D7 — Documentation batch folds D9 in

Update in the same change: `_estimate_shape` / `_proper_procrustes_distance` docstrings (drop
"Mirror reflections are not aligned away by default"), README.md:270 shape row ("after removing
size and orientation differences, reflections included"), the audit D9 qualifier, and the item-2
entry in `docs/phase5-readiness.md` once the probe has run.

## Risks / Trade-offs

- [A genuine mirror-like group difference in a future full-rank latent space would become invisible
  to `shape`] → It re-appears in `angle` (an improper orthogonal difference is an orientation
  difference); the statistics stay jointly exhaustive. Documented in the spec's separation
  requirement.
- [Renaming requirements in the main spec churns downstream references] → The RENAMED deltas keep
  archive history coherent; grep for the old requirement names in docs is part of the tasks.
- [The forced-components knob could leak into production configs] → Metadata marks
  `component_selection: "forced"`; study enumeration does not need a guard because the parameter
  signature already isolates such cells, but the study report can assert `component_selection ==
  "cv"` cheaply — included in tasks as a one-line check.
- [Old shards silently kept] → They do not resume (D3), but stale merged JSONL on disk could be
  re-reported by hand. Mitigation: the probe/report README note; regenerable artifacts are
  gitignored already.

## Migration Plan

1. Land estimator + tests + signature version in one commit (atomic contract switch).
2. Land the `forced_components` override + probe script.
3. Run the probe; write results under `results/`; update `docs/phase5-readiness.md` item 2.
4. Docs batch (README, docstrings, spec sync via archive).

Rollback: revert the estimator commit; the signature version key reverts with it, so resume
compatibility is self-consistent in both directions.

## Open Questions

- The exact forced-rank ladder for the probe ({3, 4, 6, 9, 12} proposed) can be adjusted when the
  probe runs, based on the feasible range at the pilot design point — it does not affect specs or
  task structure.
