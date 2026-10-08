# Spec Delta

## MODIFIED Requirements

### Requirement: A size-pure candidate construction is measured, not adopted

The diagnostic SHALL evaluate the magnitude construction that scales every omic's δ together — the production `joint` magnitude kind — and report whether its realized `angle` and `shape` responses fall to the zero-effect anchor's level **after** per-block standardization. The diagnostic MUST select the construction through the generator's public `magnitude_kind` option, with no diagnostic-only entry point, and MUST NOT change which construction a committed study profile uses: a profile that does not name a magnitude kind resolves to the `all` construction exactly as before.

#### Scenario: Candidate is compared against the anchor after standardization

- **WHEN** the joint-δ construction is evaluated at the diagnostic effect sizes
- **THEN** its post-standardization `angle` and `shape` responses are reported against the anchor's values and beside the `all` construction's, establishing whether a size-pure construction survives per-block standardization

#### Scenario: Production mode selection is unchanged

- **WHEN** a committed study profile that does not set `generator.magnitude_kind` requests the magnitude mode
- **THEN** it resolves to the `all` construction, with datasets and parameter signatures identical to those it produced before the joint kind became selectable

#### Scenario: Diagnostic selects the construction publicly

- **WHEN** the diagnostic compares the `all` and `joint` constructions
- **THEN** both are generated through the same public generator parameters, differing only in `magnitude_kind`, and the comparison records are labelled by that value

## ADDED Requirements

### Requirement: Diagnostic brackets the joint construction's effect axis analytically

The diagnostic SHALL compute, for the `joint` magnitude construction at the Phase 5 paper-grade design point, the realized joint `delta` at the population-standardized checkpoint over a dense grid of effect sizes spanning at least `0` to `1`, beside the `all` construction's realized `delta` at the same effect sizes, without sampling, permutation, or latent-space fitting. It MUST write the bracket as a committed CSV naming the construction, effect size, realized joint `delta`, joint `angle`, and joint `shape`, so that an effect grid for a magnitude study can be chosen from the realized size rather than the nominal effect. The bracket MUST run on a workstation without a cluster or an R runtime.

#### Scenario: Bracket is computed over a dense effect grid

- **WHEN** the bracket is run at the Phase 5 design point
- **THEN** a CSV records, for both constructions and every grid effect size, the realized population-standardized joint `delta`, `angle`, and `shape`, and the joint construction's `angle` and `shape` sit at the anchor's floating-point floor at every effect

#### Scenario: Bracket relates the two effect axes

- **WHEN** the bracket CSV is read at a nominal effect present on both constructions
- **THEN** the ratio of the joint construction's realized `delta` to the `all` construction's is available per effect, so an effect on one axis can be mapped to the realized size it corresponds to on the other
