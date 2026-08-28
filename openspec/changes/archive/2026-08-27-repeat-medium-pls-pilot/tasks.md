## 1. Configuration and Study Selection

- [x] 1.1 Add typed, optional attribution-diagnostic settings to simulation evaluation parameters, including bootstrap count, seed, top-k, zero tolerance, and an explicit disabled default.
- [x] 1.2 Extend the study config schema with an attribution cell selector, a versioned matched-seed policy, and an acceptance block carrying every gate parameter (alpha, tolerance multipliers, minimum power, confirmation thresholds, mandatory versus descriptive mode/statistic pairs), including validation for supported modes, effects, PLS-only attribution, and complete gate parameters.
- [x] 1.3 Resolve the study-level attribution selector into per-cell evaluation settings during enumeration so eligibility and effective settings enter stable cell signatures.
- [x] 1.4 Add config and enumeration tests for disabled defaults, valid nonzero-orientation selection, invalid selectors, PLS-only enforcement, and deterministic signatures.

## 2. Single-Fit PLS Evaluation Boundary

- [x] 2.1 Refactor final PLS fitting so one pooled fitted estimator supplies both the production training-score matrix and any subsequent attribution projection/reconstruction.
- [x] 2.2 Keep the fitted estimator and standardized joint feature matrix ephemeral while exposing only JSON-safe integration metadata in the evaluation result.
- [x] 2.3 Add regression tests proving attribution-disabled scores, selected component counts, pair statistics, and p-values remain equivalent under identical inputs and seeds.
- [x] 2.4 Add tests proving attribution receives the exact estimator and feature matrix used for trajectory measurement and that requesting it for `concat` or `snf` fails clearly.

## 3. Compact Attribution Diagnostics

- [x] 3.1 Add an internal adapter that invokes the Phase 3 attribution API with the frozen model and the fitted preprocessor's per-feature scales as `original_scales`, then extracts transition identities, path lengths, observed/captured/residual metrics, and configured top-k signed feature records.
- [x] 3.2 Derive aligned generator-truth driver sets from changed group-stage differential patterns across methylation, expression, and proteomics, including propagated cascade effects.
- [x] 3.3 Compute per-transition/component top-k precision, recall, selected counts, observed-versus-captured retention, bootstrap sign stability, and top-k frequency from the full in-memory attribution result.
- [x] 3.4 Define a versioned JSON-safe attribution diagnostic record that includes effective settings, selected PLS components, feature-order signature, availability markers, runtimes, and bounded top-k payloads carrying standardized and original-unit effects with an explicit unit basis (M-value methylation).
- [x] 3.5 Add deterministic tests for multi-stage extraction, propagated truth recovery, unavailable transitions, top-k truncation, attribution-disabled behavior, and repeated-seed reproducibility.

## 4. Matched Seeds and Persistence

- [x] 4.1 Implement opt-in matched generator seeds by seed-family and replicate index while preserving unique cell/replicate persistence keys and existing seed behavior for configs without the policy.
- [x] 4.1a Emit one mode-agnostic zero-effect primary cell inside the matched-seed family, resolve every mode's `0.00` power point from that shared anchor, and give negative-control cells their own seed families.
- [x] 4.2 Bump `seed_derivation_version` for the matched-seed policy and add integration-metadata and attribution schema versions as explicit signature keys; confirm the existing `realized_geometry_version` key and the `evaluation_params` hash already cover realized geometry and attribution settings rather than duplicating them.
- [x] 4.3 Extend evaluation and grid replicate records with additive integration metadata, attribution status, compact diagnostics, and diagnostic failure details.
- [x] 4.4 Update JSONL serialization and readers to round-trip Phase 4 records and load legacy rows with empty additive fields.
- [x] 4.5 Add tests for matched seeds across mode/effect cells, deterministic seed differences across replicate indices, legacy behavior, signature mismatch on resume, diagnostic failures, JSON safety, and representative record size.
- [x] 4.6 Add an enumeration test proving no two primary cells generate identical datasets at the same replicate index and that each mode's zero-effect point resolves to the shared anchor.

## 5. Phase 4 Summaries and Gate Evaluation

- [x] 5.1 Add long-form realized-geometry summaries by mode, effect, checkpoint, scope, statistic, and path length, preserving unavailable values and measurement-space labels.
- [x] 5.2 Add PLS-selection summaries for selected component counts, effective CV settings, AUROC metadata, and missing-diagnostic counts.
- [x] 5.3 Add attribution summaries for availability, captured-versus-observed retention, cross-replicate top-k selection/sign agreement, bootstrap stability, and truth recovery by effect, transition, and component.
- [x] 5.4 Add localization output that identifies the first checkpoint of material off-diagonal geometry without comparing raw distances across feature and latent spaces.
- [x] 5.5 Implement the one-sided Type I inflation bound over the `none` baseline and every translation effect level, plus the two-combined-SE power monotonicity rule, reading all thresholds from the configuration's acceptance block and recording explicit observations with Monte Carlo standard errors for each decision.
- [x] 5.6 Implement Phase 4 gate aggregation that emits `proceed`, `hold`, or `indeterminate`, applies the single-marginal-exceedance confirmation rule (one control statistic short of one Monte Carlo standard error over its bound yields `indeterminate` with a named confirmation re-run; two exceedances or any exceedance of at least one standard error yields `hold`), and never emits `proceed` when mandatory records or diagnostics are incomplete.
- [x] 5.7 Add synthetic summary tests covering every gate outcome, the marginal and material control-exceedance paths, tolerated and material power reversals, mixed-construction off-diagonal reporting, unavailable metrics, and incomplete work units.

## 6. Reporting and Configuration Artifacts

- [x] 6.1 Extend report generation to write geometry, PLS-selection, attribution-stability, truth-recovery, diagnostic-completeness, localization, and gate-decision outputs as structured CSV/JSON.
- [x] 6.2 Add focused plots or tables for checkpoint trends, selected-component distributions, and attribution stability while retaining existing specificity, Type I, and power outputs.
- [x] 6.3 Commit a new Phase 4 config with 300 samples, four stages, 100 replicates, 199 permutations, the five effects and four modes, explicit production PLS CV settings, matched seeds with the shared zero-effect anchor, 100-by-top-20 orientation attribution diagnostics, and the full acceptance block of gate parameters.
- [x] 6.4 Add a tiny Phase 4 smoke config that exercises the same PLS, persistence, attribution, geometry, and gate-report paths at development-scale counts.
- [x] 6.5 Update the study README and API documentation with the new config fields, diagnostic semantics, matched-seed behavior, output inventory, gate rules, and commands without changing the historical July instructions.

## 7. Preflight Verification

- [x] 7.1 Run focused attribution, PLS, evaluation, grid, study config, sharding, merge, summary, target, and report tests and resolve regressions.
- [x] 7.2 Run the end-to-end Phase 4 smoke workflow through shard execution, merge, completeness validation, reporting, and gate output without R.
- [x] 7.3 Run the fast test suite, Ruff, mypy, and documentation build before launching medium-pilot compute.

## 8. Medium Pilot and Findings

- [x] 8.1 Record the clean code revision and software environment, then execute all Phase 4 shards into a new resumable output directory with failure recording enabled.
- [x] 8.2 Merge shards, verify expected work-unit counts and parameter signatures, resolve or rerun failed/missing units, and freeze the complete merged result.
- [x] 8.3 Generate all structured reports and figures, evaluate the predeclared gates, and audit representative replicate records for PLS, geometry, and attribution completeness.
- [x] 8.4 Write a dated findings report that states the Phase 5 `proceed`, `hold`, or `indeterminate` decision, interprets rejection rates against realized geometry, reports attribution stability and limitations, and includes exact reproduction commands.
- [x] 8.5 Update the scientific roadmap and relevant documentation with the Phase 4 outcome while retaining the July pilot as historical evidence.

## 9. Final Validation

- [x] 9.1 Re-run focused tests, Ruff, mypy, the fast test suite, and the documentation build after findings and roadmap updates.
- [x] 9.2 Run strict OpenSpec validation and confirm every Phase 4 requirement scenario is covered by a task, test, structured output, or findings section.
