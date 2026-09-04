## Context

See `proposal.md` — Why. The tested `angle` lives in `stats/trajectory.py::_estimate_orientation` (PC1 of the centered stage configuration, signed by net displacement); attribution lives in `stats/attribution.py` and today decomposes only per-adjacent-transition unit-direction contrasts through a frozen pooled PLS model. Since the audit, P1 landed `configuration_spectrum`/`configuration_spectra` (eigengap machinery) in `trajectory.py`, and P4 made trending baselines — where PC1 is well-defined — the realistic operating regime. Attribution operates on feature-space group-stage means (observed, and PLS-reconstructed), not on the latent `obs_vect`; the new component applies the same orientation functional at attribution's checkpoint.

Constraints: the fitted PLS model stays frozen (existing contract); `OrientationAttributionResult` is consumed by `simulations/attribution_diagnostics.py` and `tests/test_attribution.py`; no study records, signatures, or persisted schemas are involved.

## Goals / Non-Goals

**Goals:**
- One orientation functional, shared with the tested statistic, applied per group to observed and reconstructed means.
- Additive result surface: existing fields, tables, and their schemas unchanged.
- Degeneracy is measured and flagged, never silently absorbed.

**Non-Goals:**
- No change to `estimate_difference`, RRPP, or any tested statistic or its null.
- No attribution in the latent space (the component explains the feature-space means the existing machinery already conditions on).
- No significance or resolvability *decision* — the eigengap qualifier annotates; the calling workflow decides.
- No group-aware refitting of any kind (the S1 caution from the audit).

## Decisions

**D1 — Reuse the production estimator, don't reimplement.** Expose the orientation functional from `stats/trajectory.py` (public wrapper around `_estimate_orientation`, or import of the private function within the package) and call it on each group's k×p mean configuration. Rationale: the estimator-consistency scenario is then true by construction, including the sign anchor and its documented deviations. Alternative — a local PC1 copy in `attribution.py` — rejected: two implementations of a convention that has already been fixed once (the 2026-08-28 sign-anchor change) is how F4-class drift happens.

**D2 — A dedicated `PrincipalOrientationAttribution` block, not a pseudo-transition.** The result gains a `principal_orientation` field (mirroring `TransitionAttribution`'s observed/PLS-captured/residual structure but keyed to the whole configuration), plus a dedicated frame in `attribution_frames`/`write_attribution_outputs`. Alternative — appending a synthetic entry with `transition_id="principal"` to `transitions` — rejected: it would break the "adjacent ordered stage transitions" invariant that downstream consumers and the existing spec scenarios iterate over. In *tabular* outputs (feature effects, bootstrap summaries, aggregates), the component is identified by a reserved identifier (e.g. `principal`) in the existing transition column so the frame schemas stay uniform; the reserved name is validated against stage-derived transition ids to prevent collision.

**D3 — Contrast algebra mirrors transitions.** Observed contrast = second ordered group's signed unit principal axis minus the first's; PLS-captured contrast = same functional on the reconstructed means; residual = observed − captured. Same group ordering, same feature-sign preservation, same original-units conversion path. Rationale: interpretation and table semantics carry over unchanged; k = 2 equivalence then follows from rank-1 geometry with no special-casing.

**D4 — Degeneracy gate via `configuration_spectrum`, threshold as a documented config parameter.** Report each contributing configuration's relative eigengap ((λ1−λ2)/Σλ) and flag the principal contrast degenerate when any falls below `eigengap_threshold` (config field, default informed by the audit's measurements: independent-baseline draws median ≈ 0.026, resolvable replicates ≈ 0.05+; default 0.05, overridable). The flagged contrast is still returned — callers stratify, they don't lose data. Alternative — hard `AttributionError` on degeneracy — rejected: degeneracy is a property of the data, not an input-contract violation, and the study needs the flagged rows.

**D5 — Closed trajectories reuse the zero-tolerance unavailability path.** Net displacement below `zero_tolerance` ⇒ that group's principal orientation (and every contrast needing it) is marked unavailable, exactly like a zero transition. This is the estimator's one documented degeneracy (no direction defined), distinct from the eigengap flag (direction defined but unstable).

**D6 — Bootstrap extends `_bootstrap`, sign anchored per replicate.** Each replicate recomputes group means, applies the shared functional (whose sign anchor is the replicate's own net displacement), and contributes to sign/rank/top-k stability for the principal component alongside transitions. Cost: one SVD of a k×p mean matrix per group per replicate — negligible next to the existing per-replicate work. An axis-flip under resampling is thereby impossible by convention rather than patched by post-hoc alignment to the point estimate; replicates whose net displacement crosses zero fall into the unavailable count, mirroring zero-transition replicates.

**D7 — Documentation alignment lands with the change, main-spec Purpose at sync time.** The `trajectory-orientation-invariance` purpose line ("a difference in direction of progression" → principal-axis divergence phrasing) is a main-spec Purpose edit made when the delta is synced/archived; the roadmap bullet (`docs/roadmap.md:133`) and module docstrings are edited in the implementation tasks. If P6 (`docs-sync-audit`) lands first and touches the same lines, reconcile there — the wording here wins for orientation estimand statements.

## Risks / Trade-offs

- [Flagged-but-present degenerate contrasts get over-interpreted downstream] → The degeneracy flag propagates into every table row for the principal component, and the interpretation metadata names the flag's meaning; `attribution_diagnostics` consumers see it in the frame schema, not only in prose.
- [Reserved `principal` identifier collides with a stage-derived transition id] → Validate at config time against the constructed transition ids and raise `AttributionError` on collision (stages named so that a transition id equals the reserved word are pathological but possible).
- [Default eigengap threshold is a judgment call] → It is a reported, overridable parameter recorded in `AttributionConfig`; changing it later relabels rows without changing any contrast value, so it is not load-bearing for reproducibility.
- [Feature-space vs latent-space checkpoint mismatch: the tested angle is measured in the latent space, the attribution functional runs on feature-space means] → This is the same checkpoint attribution already uses for transitions and is stated in the interpretation boundary; the PLS-captured component is precisely the bridge, and claiming more would require latent-space attribution (a non-goal).
- [Additive dataclass fields still break strict consumers (positional construction, exhaustive field checks)] → Grep for constructors of `OrientationAttributionResult` outside the module; `attribution_diagnostics.py` and tests are updated in-change.

## Open Questions

None — the deferrable unknowns (exact default threshold value, reserved identifier spelling) are recorded as overridable/validated parameters above.
