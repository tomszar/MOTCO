## ADDED Requirements

### Requirement: Baseline stage-program continuity is a generator axis
The generator SHALL expose a baseline continuity parameter ρ ∈ [0, 1) controlling how strongly
group A's per-stage methylation differential indicators persist across adjacent stages. Each CpG's
indicator sequence across stages SHALL follow a stationary first-order Markov chain whose per-stage
marginal is Bernoulli(`p_dmp`) at every ρ (first stage ~ Bernoulli(`p_dmp`); stay-differential
probability `p_dmp + ρ·(1 − p_dmp)`; become-differential probability `p_dmp·(1 − ρ)`), so that the
cross-stage indicator correlation is ρ^|t−s| while per-stage indicator counts, the per-omic effect
sizes, the cross-omic coupling derivation, and the meaning of `p_dmp` are unchanged along the axis.
At ρ = 0 the generator SHALL reproduce the independent per-stage draws of the pre-change generator
byte-identically at the same seed. Values outside [0, 1) SHALL be rejected with a clear validation
error, and truth metadata SHALL record the continuity value.

#### Scenario: Marginals are preserved along the axis
- **WHEN** datasets are generated at any ρ ∈ [0, 1)
- **THEN** each stage's methylation indicators are marginally Bernoulli(`p_dmp`), and the expected
  per-stage differential count does not depend on ρ

#### Scenario: Zero continuity reproduces the independent baseline exactly
- **WHEN** a dataset is generated with ρ = 0 and any seed
- **THEN** the baseline indicators, all derived indicators, and the sampled dataset are identical
  to the pre-change generator's output at that seed

#### Scenario: Continuity induces trending stage configurations
- **WHEN** ρ > 0
- **THEN** adjacent stages share differential programs with correlation ρ, so expected squared
  stage-mean distances grow with stage separation (proportional to 1 − ρ^|t−s|) instead of being
  equal for all pairs

#### Scenario: Continuity is recorded as truth
- **WHEN** generation succeeds
- **THEN** truth metadata records the baseline continuity value beside the other generator
  parameters

#### Scenario: Invalid continuity is rejected
- **WHEN** a caller passes a continuity value below 0 or at/above 1
- **THEN** generation raises a validation error naming the parameter and the allowed range

### Requirement: Expected surgery headroom accounts for baseline continuity
The generator's analytic expected-headroom computation for pool-limited surgeries SHALL use the
continuity-adjusted probability that a CpG is differential in at least one stage,
`1 − (1 − p_dmp)·(1 − p_dmp·(1 − ρ))^(n_stages − 1)`, which reduces to the independence union
probability at ρ = 0. Expected destination pools and saturating effects SHALL therefore reflect
that higher continuity shrinks the expected stage-active union (stage programs overlap more).

#### Scenario: Headroom at zero continuity is unchanged
- **WHEN** expected headroom is computed at ρ = 0
- **THEN** the expected pools and saturating effects equal the pre-change independence-based values

#### Scenario: Headroom tracks the continuity-adjusted union
- **WHEN** expected headroom is computed at ρ > 0 for an orientation or shape-relocate surgery
- **THEN** the expected stage-active fraction uses the continuity-adjusted union probability, and
  the reported saturating effect is correspondingly no smaller than at ρ = 0 for the same
  `p_dmp` and `n_stages`
