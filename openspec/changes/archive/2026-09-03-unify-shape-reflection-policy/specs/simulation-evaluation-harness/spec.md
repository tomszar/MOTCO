## ADDED Requirements

### Requirement: PLS integration supports a diagnostic forced-component override

The harness SHALL accept an optional integration parameter that forces the number of PLS latent variables, bypassing cross-validated component selection, for rank diagnostics only. When the override is absent the behavior MUST be unchanged. When the override is set, the harness MUST skip the double nested cross-validation, fit the single pooled PLS estimator at the forced component count, and record in the integration metadata both the forced count and an explicit marker that component selection was overridden, so no downstream consumer can mistake a forced-rank run for a production evaluation. The override MUST be validated against the feasible component range and rejected with a clear error otherwise.

#### Scenario: Default behavior is unchanged

- **WHEN** a PLS evaluation runs without the forced-component parameter
- **THEN** the number of latent variables is selected by the double nested cross-validation exactly as before
- **AND** the integration metadata carries no override marker

#### Scenario: Forced component count is honored and recorded

- **WHEN** a PLS evaluation runs with the forced-component parameter set to a feasible value
- **THEN** the latent space has exactly that many components and no cross-validation is performed
- **AND** the integration metadata records the forced count and marks component selection as overridden

#### Scenario: Infeasible forced count is rejected

- **WHEN** the forced-component parameter exceeds the feasible range for the sample
- **THEN** the harness raises a clear validation error
