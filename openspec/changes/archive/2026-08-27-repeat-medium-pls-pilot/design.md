## Context

See `proposal.md` for motivation. The current PLS evaluation selects a component count, fits a final model only to obtain scores, and returns JSON-safe integration metadata. The final estimator is then discarded. The grid record persists p-values, pair statistics, realized geometry, truth metadata, and runtime metadata, but does not persist the evaluation result's PLS integration metadata.

Phase 2 established that orientation and both shape constructions are mixed after biological propagation or joint preprocessing. The new pilot therefore cannot reuse the July report's assumption that every off-diagonal rejection is estimator cross-talk. Phase 3 provides the generic frozen-model attribution calculation, but study integration must keep its cost and result size bounded.

## Goals / Non-Goals

**Goals:**

- Reuse one final fitted PLS estimator for trajectory scores and attribution.
- Make the Phase 4 run fully determined by one committed config and matched-seed policy.
- Persist enough compact information to audit PLS selection, realized geometry, and attribution stability across replicates.
- Produce an explicit, uncertainty-aware decision on whether Phase 5 may proceed.

**Non-Goals:**

- Change the numpy generator, trajectory estimators, RRPP, PLS component-selection criterion, or Phase 3 attribution formulas.
- Refit preprocessing, select components, or refit PLS inside attribution bootstraps.
- Serialize fitted estimators, full standardized matrices, full feature-effect tables, or bootstrap matrices in JSONL.
- Add automatic pathway/module annotation or treat simulation attribution as causal evidence.
- Run or modify the 500-replicate, 999-permutation paper-grade study in this change.

## Decisions

### 1. Fit the final PLS estimator once and retain it only within evaluation

After double cross-validation selects the modal component count, fit one final pooled PLS estimator and use its `x_scores_` as the trajectory measurement matrix. Pass that same in-memory estimator and the exact standardized joint input to attribution when requested, then discard the estimator after the compact evaluation result is built.

This avoids the current second helper boundary where scores are returned without the model and guarantees that trajectory and attribution share one coordinate system. The estimator remains ephemeral because serializing sklearn objects into JSONL would be unsafe, version-fragile, and unnecessary. A score-equivalence regression test will protect evaluations that do not request attribution.

Alternative considered: refit a numerically equivalent model after trajectory evaluation. Rejected because equivalent hyperparameters do not prove the same fitted state and would violate the frozen shared-model contract.

### 2. Add an explicit study attribution selector and matched-seed policy

Extend the declarative study configuration with an optional attribution block and a seed-pairing policy. Enumeration resolves the selector into each cell's evaluation parameters, so diagnostic eligibility and effective bootstrap settings are part of the cell signature rather than inferred after execution.

The Phase 4 config selects all nonzero primary orientation cells, uses 100 bootstrap replicates and `top_k=20`, and runs attribution regardless of the observed angle p-value. Running every eligible replicate avoids significance-conditioned selection bias; reports may additionally stratify results by the global p-value.

For primary cells, derive the generator seed from the study base seed and replicate index through a versioned matched-seed key shared across mode/effect cells. Cell identity still determines persistence keys, while the shared seed makes requested-effect comparisons paired at the generated-reference level. Type I and OFAT cells receive explicit seed-family identifiers. The seed policy and version enter parameter signatures.

Effect `0.00` needs separate handling. The generator returns group B's baseline unchanged and consumes no additional randomness whenever the requested effect is zero, so under one matched-seed family the per-mode zero-effect cells would be byte-identical datasets: duplicated compute and four perfectly correlated null rows reported as if they were independent evidence. Phase 4 enumeration therefore emits one mode-agnostic zero-effect primary cell, and every mode's power curve resolves its `0.00` point from that shared anchor. The anchor stays inside the primary matched-seed family so each curve is paired with its own null, and enumeration asserts that no two primary cells generate identical data at the same replicate index.

Alternative considered: retain cell-specific seeds. Rejected because Phase 4 explicitly calls for matched seeds and cell-specific random variation weakens checkpoint comparisons across requested effects.

Alternative considered: keep one zero-effect cell per mode under separate seed families. They would be independent, but they would spend three extra cells of pilot compute re-measuring a construction the `none` control and the shared anchor already characterize.

### 3. Persist compact attribution records, not the full Phase 3 result

The in-memory Phase 3 result remains authoritative. A simulation adapter extracts only:

- effective configuration and schema version;
- transition identity, path lengths, and observed/PLS-captured/residual norms;
- observed-versus-captured cosine or equivalent retention summaries where defined;
- top 20 signed features per transition and component with effect, rank, sign stability, and bootstrap top-k frequency; and
- precision/recall and selected-count summaries against generator truth.

Feature effects are recorded in pooled standardized units and, wherever the fitted preprocessor exposes a positive per-feature scale, in original units as well: the adapter passes the `FittedOmicsPreprocessor` block scales as attribution's `original_scales` so `effect_original` is populated instead of null. The record labels the unit basis explicitly, including that methylation original units are M-values rather than beta values.

Generator truth is defined as the aligned feature set whose group-stage differential pattern differs after including CpG-to-gene-to-protein propagation. This prevents a real propagated expression or protein driver from being mislabeled as a false positive. The adapter computes truth comparisons while high-dimensional indicators remain in memory; those arrays are still excluded from persisted truth metadata.

Top-k truncation bounds JSONL growth while retaining enough identifiers to aggregate cross-replicate selection frequency and sign agreement. Feature-order hashes and counts make truncation and alignment auditable.

Alternative considered: serialize every feature effect. Rejected because record size scales with all features, transitions, components, and replicates while the Phase 4 questions only require stability and recovery summaries plus inspectable leading drivers.

### 4. Additive persistence uses explicit diagnostic versions

Add JSON-safe integration and attribution fields to replicate records with empty defaults for legacy reads.

The parameter signature already carries `seed_derivation_version` and `realized_geometry_version`, and it already hashes `evaluation_params`, so attribution settings enter it automatically once they are fields on the evaluation parameters. The remaining work is therefore narrow: bump `seed_derivation_version` when the matched-seed policy lands, add the integration-metadata and attribution schema versions as explicit keys, and confirm rather than re-add the versions that exist. Signature composition stays in one place so a diagnostic-schema change cannot silently resume into an incompatible shard.

Legacy records remain readable for historical reports, but their signatures cannot satisfy a resumed Phase 4 work unit. The new pilot writes to a distinct output directory and never appends to July shards.

### 5. Freeze the Phase 4 profile at 100 replicates

Commit a new config rather than editing `pilot_50x199.json`. It declares:

- numpy generator with `n_samples=300`, `n_stages=4`, `group_ratio=0.5`, and `p_dmp=0.2`;
- PLS integration with explicit production CV settings (`cv1_splits=3`, `cv2_splits=4`, `n_repeats=5`, `max_components=20`, `random_state=1203`);
- 199 RRPP permutations, effects `0.00` through `1.00` by `0.25`, and the four established modes;
- 100 replicates per cell and a versioned matched-seed root; and
- the bounded attribution selector from Decision 2.

One hundred replicates reduces Monte Carlo uncertainty relative to the July 50-replicate pilot while remaining far below Phase 5. Parallel worker counts may be overridden operationally because they do not alter statistical output; effective values and software versions are still recorded.

### 6. Separate construction interpretation from statistical gates

Reports flatten all realized-geometry checkpoints and join them to operating summaries by mode and effect. Off-diagonal rejection is classified as:

1. construction-present when the corresponding population checkpoint already moves;
2. sampling/preprocessing-associated when it first appears before PLS; or
3. projection-associated when it first appears at the PLS checkpoint.

These labels describe localization, not causality. Raw distances are never compared directly across standardized feature and latent spaces.

Gate thresholds are declared, not embedded. The Phase 4 configuration carries an acceptance block holding alpha, the tolerance multipliers, the minimum power at the top effect, the confirmation-rule thresholds, and which mode/statistic pairs are mandatory versus descriptive. Summary and report code implements the rules that consume those values and hard-codes no threshold of its own, so a gate can be re-specified by editing the committed config rather than the study package.

The Phase 4 gate uses the following mandatory rules:

- Type I inflation: the `none` baseline and every translation cell — the negative control at the top effect and each translation power-grid effect level, since translation is a location-only offset at any effect — are Type I controls. Each available statistic must be no greater than `alpha + 2 * sqrt(alpha * (1 - alpha) / n_available)`.
- Magnitude: `delta` reaches at least 0.80 at effect 1.00 and is non-decreasing within two combined Monte Carlo standard errors; magnitude `angle` and `shape` remain below the Type I inflation bound.
- Orientation and shape: the matching statistic reaches at least 0.80 at effect 1.00 and is non-decreasing within two combined Monte Carlo standard errors. Their off-diagonal rates are descriptive because the constructions are known to be mixed.
- Completeness: all expected work units are resolved; completed PLS records contain integration and realized-geometry metadata; eligible orientation records contain attribution diagnostics or a recorded diagnostic failure.

Gate multiplicity is predeclared rather than discovered after the run. Each control cell contributes one test per statistic, so the `none` baseline plus five translation effect levels is a family of eighteen one-sided tests, each exceeding its bound with probability roughly `0.023` when the true rate is exactly alpha. One marginal exceedance must therefore not decide the phase alone. When exactly one control statistic exceeds its bound by less than one Monte Carlo standard error and no other mandatory gate fails, the report emits `indeterminate`, names the cell and statistic, and specifies a confirmation re-run of that cell before any Phase 5 decision. Two or more exceedances, or any exceedance of at least one Monte Carlo standard error, is `hold`.

The shape gate is stated as at least `0.80` at effect `1.00` even though the roadmap's Phase 4 exit language only asks that shape match its validated invariance contract. Phase 1 established that the corrected pairwise Procrustes estimator returns a positive residual for a genuine interior bend, and Phase 2 documented the shape construction as a free bend that does move the configuration, so a mode that bends interior stages must be detectable for the statistic to be usable at all. The gate constrains only the diagonal response; shape's off-diagonal behavior stays descriptive, so this is a detectability requirement rather than the false purity Phase 2 ruled out. July's legacy-estimator shape power was `1.00`, which makes `0.80` a floor rather than a stretch target — and if the corrected estimator misses it, that is a scientific finding Phase 5 must not paper over.

For adjacent power points, a downward step is tolerated when its size is no greater than twice `sqrt(SE_i^2 + SE_j^2)`. Attribution stability and truth recovery are reported without a hard quality floor in this first integrated pilot; their completeness and reproducibility are mandatory. Study execution never aborts because a scientific gate fails. Reporting emits `proceed`, `hold`, or `indeterminate`. `indeterminate` covers both incomplete evidence and a single marginal control exceedance awaiting confirmation, and only `proceed` unlocks the Phase 5 recommendation.

Alternative considered: retain all old off-diagonal alpha checks. Rejected because Phase 2 directly demonstrated nonzero pre-integration off-diagonal geometry for orientation and shape.

### 7. Keep structured outputs primary and commit a traceable findings report

Extend report generation with geometry, PLS-selection, attribution-stability, truth-recovery, diagnostic-completeness, and gate-decision CSV/JSON outputs. A dated Markdown report cites those outputs, the exact config and revision, environment versions, shard counts, failures, and reproduction commands. The July report remains untouched and is labeled historical for the Phase 4 decision.

## Risks / Trade-offs

- [Attribution bootstrap cost expands the pilot] -> Restrict it to four nonzero orientation cells, keep the PLS model frozen, use 100 bootstraps, and record attribution runtime separately.
- [Top-k persistence hides lower-ranked instability] -> Record total feature counts and hashes, state the truncation explicitly, and keep aggregate recovery metrics computed from the full in-memory result.
- [Generator truth is ambiguous after biological propagation] -> Define truth from changed group-stage differential patterns across every omic layer and test the mapping on deterministic fixtures.
- [Matched seeds alter legacy seed behavior] -> Make pairing opt-in, version the derivation, include it in signatures, and use a new output directory.
- [Matched seeds make distinct cells generate identical data] -> Enumerate one shared zero-effect anchor, keep negative controls in their own seed families, and test that nonzero cells share a baseline within a replicate index while differing across replicate indices.
- [A single borderline control decides the phase] -> Predeclare the gate family size, apply the marginal-exceedance confirmation rule, and report every control observation with its Monte Carlo standard error.
- [One hundred replicates still leave Monte Carlo uncertainty] -> Report uncertainty on every rate and use uncertainty-tolerant monotonicity rather than exact ordering.
- [Persisted additive fields enlarge records] -> Store scalar and top-k summaries only and add round-trip and representative-size tests.
- [A fitted estimator could leak into serialization] -> Separate ephemeral integration artifacts from the JSON-safe evaluation payload and test serialization explicitly.

## Migration Plan

1. Add the optional configuration and additive result fields with legacy-reader defaults.
2. Refactor final PLS fitting to expose one ephemeral estimator and verify score equivalence with attribution disabled.
3. Add compact attribution extraction, truth recovery, signature versioning, and persistence tests.
4. Extend study summaries, gate evaluation, and structured reports; validate them on deterministic synthetic records.
5. Add the fixed Phase 4 config and run a tiny end-to-end smoke using the same diagnostic path.
6. Execute the resumable medium pilot into a new output directory, merge and validate completeness, generate reports, and commit the dated findings.
7. Update the roadmap with the Phase 4 gate decision. If rollback is required before the run, disable the optional attribution block and additive writers; legacy behavior remains the default. Existing July artifacts are never rewritten.
