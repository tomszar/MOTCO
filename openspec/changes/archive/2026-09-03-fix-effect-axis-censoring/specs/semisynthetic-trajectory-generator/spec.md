## ADDED Requirements

### Requirement: Pool-limited surgeries apply an explicit censoring policy
The generator SHALL apply an explicit, configurable censoring policy to every surgery whose size is
limited by a destination or candidate pool (`orientation` relocation, `translation` extra set,
`shape` with `shape_kind='relocate'`). The policy SHALL default to failing loudly: when the
requested surgery size exceeds the available pool, generation raises a validation error naming the
mode, the requested size, and the available pool size. An explicit opt-in policy value SHALL
preserve the previous clamping behavior (realize as much of the surgery as the pool allows). The
generator MUST NOT silently clamp under the default policy.

#### Scenario: Default policy fails loudly on a censored surgery
- **WHEN** a pool-limited surgery's requested size exceeds its available pool and no censoring
  policy is explicitly configured
- **THEN** generation raises a clear validation error identifying the trajectory mode, the
  requested surgery size, and the available pool size, and no dataset is returned

#### Scenario: Opt-in clamping preserves the previous behavior
- **WHEN** the censoring policy is explicitly set to clamp and a pool-limited surgery's requested
  size exceeds its available pool
- **THEN** the generator realizes the largest surgery the pool allows and generation succeeds,
  matching the pre-policy behavior replicate for replicate at the same seed

#### Scenario: Uncensored surgeries are unaffected by the policy
- **WHEN** a pool-limited surgery's requested size does not exceed its available pool
- **THEN** the realized surgery equals the requested size under either policy value, and the
  sampled dataset at a given seed is identical to the pre-policy generator's output

### Requirement: Truth metadata makes requested-vs-realized surgery explicit
For every pool-limited surgery, truth metadata SHALL record the nominal (requested) surgery size
beside the realized surgery size, and a flag stating whether the surgery was censored (realized
size smaller than nominal). This makes the requested-effect → realized-surgery relationship
auditable per record, without regenerating data.

#### Scenario: Nominal and realized sizes are recorded together
- **WHEN** a dataset with a pool-limited surgery is generated successfully
- **THEN** truth metadata contains the nominal surgery size, the realized surgery size, and a
  censored flag that is true exactly when realized < nominal

#### Scenario: Clamped generation is identifiable downstream
- **WHEN** a dataset is generated under the clamping policy and the clamp binds
- **THEN** the truth metadata's censored flag is true, so downstream summaries can identify the
  record without access to the generator
