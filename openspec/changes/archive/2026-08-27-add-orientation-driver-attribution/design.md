## Context

The proposal addresses the interpretation gap described in `proposal.md`. MOTCO already fits a pooled, stage-conditioned PLS representation in `_pls_integration`, applies a shared pooled preprocessing contract, and exposes fitted PLS transforms. The inverse-PLS study establishes that `inverse_transform` can express a retained score displacement in feature coordinates, but it is currently a controlled diagnostic rather than a production attribution API.

The attribution layer must work from a fixed feature matrix and fitted model so that observed group differences are compared in one coordinate system. It must also support the existing multi-stage trajectory convention and avoid treating pooled VIP as a group-difference driver measure.

## Goals / Non-Goals

**Goals:**

- Provide a reusable analysis API for two-group, multi-stage feature data and a frozen fitted PLS model.
- Compute adjacent group transitions, normalized directional contrasts, PLS-reconstructed contrasts, and observed-minus-reconstructed residuals.
- Preserve signed feature contributions and expose standardized effects plus optional original-unit effects.
- Quantify sampling stability by resampling within group-by-stage strata while conditioning on the fitted preprocessing and PLS model.
- Aggregate results over caller-supplied module or pathway labels without adding an annotation database or claiming causality.
- Return typed structured records and tabular views suitable for reports and later study integration.

**Non-Goals:**

- Selecting the PLS component count, fitting or comparing separate group-specific latent spaces, or changing the production PLS integration path.
- Computing global orientation p-values, replacing RRPP, or deciding whether an upstream orientation result is significant.
- Automatic pathway annotation, causal inference, or a claim that the PLS reconstruction is a unique inverse of the measured features.
- Adding SNF attribution or graph-native driver statistics.
- Refitting preprocessing or PLS inside bootstrap replicates; model-selection uncertainty is a separate study dimension.

## Decisions

### 1. Put the capability beside the existing trajectory and PLS statistics

Implement the core analysis in `src/motco/stats/attribution.py` and export its public types and entry points through `motco.stats`. This keeps the calculation reusable for simulated and real datasets, while the simulation harness can call it later without coupling attribution to InterSIM or the study runner.

Alternative considered: place the implementation in `simulations/`. Rejected because the required inputs are generic feature data, metadata, and a fitted model, and the output is an analysis result rather than a generator artifact.

### 2. Use explicit ordered group-stage means as the calculation boundary

Accept a feature DataFrame and metadata with exactly two groups and at least two stages. Compute arithmetic cell means by default. Also accept an optional precomputed group-by-stage mean table so callers can provide covariate-adjusted LS means from their own design model without duplicating that model inside attribution. The table must use the same feature order and complete group-stage design.

Order groups and stages deterministically using the existing sorted-label convention unless explicit levels are supplied. For adjacent stages `s` and `s+1`, calculate `d_g,s = mean(g,s+1) - mean(g,s)`, `u_g,s = d_g,s / ||d_g,s||`, and `q_s = u_group2,s - u_group1,s`. Keep `d` and `||d||` even when `u` and `q` are unavailable because a path is zero.

### 3. Define the PLS decomposition through score projection and inverse transformation

The caller supplies the already fitted PLS model used to construct the shared coordinate system. For each group-stage mean `mu`, calculate the model score `t = model.transform(mu)` and retained feature component `mu_pls = model.inverse_transform(t)`. Calculate the PLS-captured transition and normalized contrast from `mu_pls`; define the residual feature contrast as `q_observed - q_pls` at the normalized directional-contrast level. Keep the raw observed, captured, and residual transition vectors as well so the result does not imply that normalized contrasts add geometrically.

Alternative considered: use `x_loadings_` and hand-derived centering/scaling formulas. Rejected because estimator internals and fitted scaling are easy to misuse; the estimator's transform/inverse-transform contract is the authoritative map.

### 4. Treat original-unit reporting as a diagonal conversion of standardized effects

The main input is the pooled standardized matrix consumed by PLS. An optional positive scale per feature converts transition effects back to original units; centering cancels in differences, so `effect_original[j] = effect_standardized[j] * scale[j]`. Report both unit systems separately and never use original-unit effects to redefine PLS geometry. This matches the existing block-preprocessing contract while keeping the generic stats API independent of `FittedOmicsPreprocessor` internals.

### 5. Bootstrap rows while conditioning on the fitted representation

For each replicate, sample rows with replacement independently within every group-by-stage cell, recompute cell means, and run the same decomposition with the same model and feature scaling. Summarize sign stability as the proportion of valid replicates whose effect sign agrees with the modal nonzero sign, rank stability as the proportion of valid replicates in which a feature appears in a requested top-k set, and top-k selection frequency using absolute effect magnitude. Near-zero values use a configurable numerical tolerance and are excluded from sign denominators.

This estimates sampling stability conditional on the chosen coordinate system. Refitting PLS or preprocessing in every replicate would also measure model-selection instability, but would make driver ranks depend on changing axes and conflict with the shared-space interpretation.

### 6. Keep feature-group aggregation declarative

Accept a one-to-one feature-to-label mapping with a caller-provided label namespace such as `module` or `pathway`. Aggregate signed values by sum and absolute values by sum of absolute contributions, and carry feature counts and bootstrap selection summaries into the aggregate table. Record that labels were supplied by the caller. No correlation clustering or external annotation service is introduced in this change.

### 7. Return structured records and provide focused tabular views

Use frozen result/configuration records containing ordered levels, transition summaries, feature tables, aggregate tables, bootstrap summaries, and interpretation metadata. DataFrames are the primary machine-readable views; a later report or study runner can serialize them without parsing Markdown. Include model component count and feature order in metadata for reproducibility.

## Risks / Trade-offs

- [PLS reconstruction is lossy and non-unique] → Report captured and residual contrasts separately, preserve observed vectors, and document the non-causal interpretation boundary.
- [Correlated features can split or duplicate apparent driver signal] → Report signed and absolute effects, bootstrap rank/sign stability, and caller-supplied module/pathway aggregates.
- [Bootstrap stability is conditional on the fitted model] → Record that preprocessing/model fitting is frozen and expose the seed and replicate count; leave model-selection stability to a later sensitivity study.
- [Normalized directional contrasts do not add linearly] → Retain raw transition vectors and define the residual alongside, rather than as a geometric decomposition of, normalized contrasts.
- [Covariate-adjusted means may not have row-level bootstrap semantics] → Support them as an explicit precomputed mean input and document that the built-in stratified bootstrap applies to row-derived means unless a future resampling provider is added.
- [Original-unit scales may be supplied incorrectly] → Validate feature names, order, positivity, and finiteness before producing original-unit output.

## Migration Plan

This is additive. Add the new stats module, exports, tests, and documentation. Existing PLS integration, trajectory statistics, InterSIM bridge behavior, study persistence, and pooled VIP outputs remain unchanged. Removing the new module and its exports restores the previous public surface.
