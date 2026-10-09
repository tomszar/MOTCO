# Spec Delta

## REMOVED Requirements

### Requirement: A size-pure candidate construction is measured, not adopted
**Reason**: The candidate has been adopted: it is the production `joint` kind and now the generator default. The requirement's guarantee that an unnamed kind resolves to `all` no longer holds.
**Migration**: Replaced by "Diagnostic compares the joint and all constructions explicitly" below. Committed profiles keep `all` through an explicit `generator.magnitude_kind`.

## ADDED Requirements

### Requirement: Diagnostic compares the joint and all constructions explicitly

The diagnostic SHALL evaluate the magnitude construction that scales every omic's δ together, which is the production `joint` magnitude kind and the generator's default, and report whether its realized `angle` and `shape` responses fall to the zero-effect anchor's level **after** per-block standardization. The diagnostic MUST select each construction it compares through the generator's public `magnitude_kind` option, naming the kind explicitly rather than relying on the default, with no diagnostic-only entry point. It MUST NOT change which construction a committed study profile uses: every committed profile names its magnitude kind explicitly, and a profile that names none resolves to the generator default, `joint`.

#### Scenario: Candidate is compared against the anchor after standardization

- **WHEN** the joint-δ construction is evaluated at the diagnostic effect sizes
- **THEN** its post-standardization `angle` and `shape` responses are reported against the anchor's values and beside the `all` construction's, establishing whether a size-pure construction survives per-block standardization

#### Scenario: Committed profile selection is preserved

- **WHEN** a committed study profile that predates the `joint` default requests the magnitude mode
- **THEN** it resolves to the `all` construction through its explicit `generator.magnitude_kind`, with datasets and parameter signatures identical to those it produced before the default changed

#### Scenario: Diagnostic selects the construction publicly

- **WHEN** the diagnostic compares the `all` and `joint` constructions
- **THEN** both are generated through the same public generator parameters, each naming its `magnitude_kind` explicitly, and the comparison records are labelled by that value
