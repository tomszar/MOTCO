## MODIFIED Requirements

### Requirement: Phase 4 reporting relates operating characteristics to realized geometry

The study SHALL report per-statistic rejection rates and Monte Carlo uncertainty together with summaries of
every applicable realized-geometry checkpoint. It MUST interpret off-diagonal rejection in light of the
corresponding pre-integration geometry and MUST NOT label a response as estimator cross-talk solely because
the requested mode name differs from the responding statistic.

Materiality SHALL be determined in units of each checkpoint's own zero-effect null dispersion for the
statistic being judged, so that `delta`, `angle`, and `shape` are commensurable despite their
incommensurable native scales. A response counts as material at a checkpoint when its excess over the
zero-effect null at that same checkpoint is at least the configured number of null-dispersion units. A
single absolute threshold applied to all three statistics MUST NOT be used, because it renders a statistic
whose entire response range is smaller than that threshold structurally unclassifiable — reporting
`not_material` for a response the corresponding test rejects at high rate.

Where the zero-effect null has no usable dispersion — an analytically null construction leaves a population
checkpoint's dispersion at zero or at floating-point dust — materiality SHALL fall back to whether the
excess exceeds that dust tolerance, which is the limit of the dispersion rule as the null variance goes to
zero: a null with no variance is exceeded by any real difference. The system MUST NOT divide by a dust-valued
dispersion, and MUST NOT report such a checkpoint as unclassifiable merely because its null is degenerate.
Each emitted row MUST record whether the dispersion path or the fallback determined its verdict.

Changing the materiality rule MUST NOT silently change how an already-reported run's localization renders:
a run reported under an earlier rule MUST either continue to reproduce its recorded classification, or be
re-issued explicitly with the rule it was judged under recorded alongside it.

#### Scenario: Geometry-aware operating report is produced

- **WHEN** merged Phase 4 records are reported
- **THEN** structured tables summarize path lengths, `delta`, `angle`, and `shape` by mode, effect,
  checkpoint, and scope beside the rejection-rate tables
- **AND** the report identifies the first checkpoint where each material off-diagonal response appears

#### Scenario: Materiality is comparable across statistics

- **WHEN** an off-diagonal response is judged at a checkpoint whose zero-effect null has usable dispersion
- **THEN** its excess over the zero-effect null is expressed in that checkpoint's null-dispersion units for
  that statistic, and the reported row records both the excess and the dispersion it was scaled by

#### Scenario: A degenerate null does not silence a checkpoint

- **WHEN** the zero-effect null's dispersion at a checkpoint is zero or floating-point dust, as it is for an
  analytically null construction at a population checkpoint
- **THEN** the response is judged material when its excess exceeds the dust tolerance, no division by the
  degenerate dispersion occurs, and the row records that the fallback path determined the verdict

#### Scenario: A small-scale statistic remains classifiable

- **WHEN** a statistic's observed response and its null are both far smaller in native units than the
  responses of the other statistics, and the corresponding test rejects well above its nominal level
- **THEN** the localization output classifies the response at a checkpoint rather than reporting
  `not_material`

#### Scenario: A previously reported run does not change silently

- **WHEN** a run reported under an earlier materiality rule is regenerated from its merged records
- **THEN** its localization output either reproduces the recorded classification, or the regeneration is an
  explicit re-issue that records which rule each classification was produced under

#### Scenario: Measurement spaces are compared

- **WHEN** feature-space and PLS-latent checkpoints appear in one report
- **THEN** the report labels their measurement spaces and compares trends or retention without treating raw
  distances as scale-equivalent

#### Scenario: PLS and attribution stability are reported

- **WHEN** eligible PLS orientation replicates are available
- **THEN** the report summarizes selected component counts, attribution availability, observed-versus-captured
  retention, cross-replicate top-k selection and sign agreement, bootstrap stability, and generator-truth
  recovery by effect and transition
