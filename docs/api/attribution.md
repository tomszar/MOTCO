# Orientation Driver Attribution

Orientation attribution explains a two-group, multi-stage difference after an
upstream trajectory analysis has identified the shared PLS space of interest.
Pass the pooled, aligned feature matrix, matching group/stage metadata, and
the one fitted PLS estimator to `analyze_orientation_attribution`.

The analysis orders groups and stages deterministically unless explicit levels
are supplied. For every adjacent stage pair it reports each group's raw
transition and path length, the unit directional contrast, the same contrast
after PLS score projection and inverse transformation, and the signed residual
`observed - pls_captured`. Observed feature values remain in the result; the
reconstruction is a parallel, lossy view rather than a replacement.

## Units and stability

Effects are reported in the input coordinate units, normally pooled
standardized units. Positive per-feature scales can be supplied to add
original-unit columns. These scales only convert reported effects; they never
change PLS geometry.

Bootstrap resampling is independent within every group-by-stage cell and uses
the same fitted model and preprocessing contract in every replicate. The
result records the seed, requested count, valid count, sign stability, and
top-k selection frequency. Bootstrap is available for arithmetic row-derived
means; precomputed covariate-adjusted mean tables are accepted with stability
marked unavailable.

## Labels and interpretation

Feature-to-module or feature-to-pathway labels are optional and must be a
complete one-to-one mapping supplied by the caller. Aggregate tables preserve
the labels and mark their source as `caller-supplied`. Aggregation is not a
pathway annotation service and does not establish biological causality.

The result contains no p-values or significance decisions. Attribution should
be run after the calling workflow has made any global orientation inference
decision. It describes associations implied by one frozen shared PLS model;
the inverse reconstruction is not unique, and bootstrap stability is
conditional on that fitted representation.

## Functions

::: motco.stats.attribution.analyze_orientation_attribution

::: motco.stats.attribution.attribution_frames

::: motco.stats.attribution.write_attribution_outputs

