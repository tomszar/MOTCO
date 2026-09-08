## ADDED Requirements

### Requirement: Parameter axes address nested evaluation integration parameters

Parameter axes SHALL accept, in addition to top-level `generator.<field>` and `evaluation.<field>` names, a dotted path into the evaluation integration-parameter mapping of the form `evaluation.integration_params.<key>`. Applying such an axis value MUST set that key inside the cell's evaluation integration parameters, leaving every other key unchanged; a `null` axis value MUST remove the key so the evaluation runs as if it had never been set. Reading the baseline value of such an axis MUST return the mapping's current value, or `null` when the key is absent. Because the value lands in the cell's evaluation parameters, it MUST enter the cell's parameter signature, so two cells that differ only in a nested axis value have distinct signatures. Top-level axes MUST behave exactly as before, and any other nesting depth or namespace MUST be rejected with a clear error naming the axis.

#### Scenario: Nested axis value is applied into the integration parameters

- **WHEN** an axis `evaluation.integration_params.forced_components` is applied with value `6` to evaluation parameters whose integration parameters already hold cross-validation knobs
- **THEN** the resulting evaluation parameters carry `forced_components = 6` alongside the unchanged existing knobs

#### Scenario: Null removes the nested key

- **WHEN** the same axis is applied with value `null` to evaluation parameters that hold `forced_components`
- **THEN** the resulting integration parameters do not contain `forced_components`
- **AND** the baseline value read for that axis on parameters without the key is `null`

#### Scenario: Nested axis values are distinguished by the parameter signature

- **WHEN** two otherwise identical cells are built from axis values `null` and `6` for `evaluation.integration_params.forced_components`
- **THEN** their parameter signatures differ

#### Scenario: Unsupported nesting is rejected

- **WHEN** an axis names a nested path outside the evaluation integration-parameter mapping (for example `generator.something.key` or `evaluation.attribution.key`)
- **THEN** the orchestrator raises a clear error naming the axis
