# default-joint-magnitude-construction

## Why

The [Phase 5 magnitude re-measurement](../../../docs/reports/phase5-magnitude-remeasurement-2026-10-08.md)
lifted the withheld magnitude-specificity claim **for the `joint` construction only**. The magnitude
controls are met under `joint` (`angle` 0.024, `shape` 0.000 at e = 1.00), and they fail under the `all`
construction (0.276 / 0.742 in Phase 5). `adopt-joint-magnitude-construction` (design D2) deliberately left the
default at `'all'` until the re-measurement passed. It has now passed. Any caller that relies on the default
(the generator dataclass, `motco simulate`, the showcase, and any future study config that omits the key)
still gets the construction the paper must not describe as a pure size change. The roadmap lists this flip
as the first of the next three changes.

## What Changes

- **Flip the default.** `SemiSyntheticTrajectoryParams.magnitude_kind` defaults to `'joint'`, and so does
  `motco simulate --magnitude-kind`. `'all'` and `'extremes'` remain selectable and unchanged.
- **Pin `"all"` into every historical study config that omits the key.** That is the ten configs under
  `examples/trajectory_power_study/` other than the two Phase 5 magnitude profiles, which already name
  `"joint"`. `parameter_signature` hashes the resolved dataclass, so a pinned `"all"` keeps every enumerated
  cell's signature, cell id and matched seed byte-identical. Committed results stay resumable and
  reproducible. A regression test checks this against a fixture of pre-flip signatures.
- **Pin `magnitude_kind="all"` at the call site of the two historical diagnostics** that construct generator
  parameters directly, `realized_geometry_study.py` and `scripts/latent_rank_probe.py`. A re-run then
  regenerates what their committed results measured.
- **Let the showcase follow the default.** Its magnitude panel becomes the joint construction, the
  construction the paper describes. Its description is updated to match.
- **Document the scope boundary.** `magnitude_kind='extremes'` and `shape_kind='magnitude'` still scale
  methylation's δ only. The docs say so, so that "magnitude" is not read as joint-scaled everywhere.
- **Update the docs to the post-flip state.** This covers the generator module docstring,
  `docs/api/simulations.md`, `CLAUDE.md`, the examples README (why the pin exists, and that new configs must
  not copy it), and the roadmap ("Next three changes": item 1 retired; the one-sided acceptance specificity
  target recorded as its own follow-up).

Out of scope:
- the one-sided acceptance specificity target, which is a separate later change;
- making `extremes` or `shape_kind='magnitude'` joint-scaled;
- `specificity.py`'s comparison helpers, which keep their explicit `'all'` parameter defaults;
- any re-run, any statistic, RRPP, PLS, or gate change.

No compute is needed.

## Capabilities

### New Capabilities

(none)

### Modified Capabilities

- `semisynthetic-trajectory-generator`: the "extreme-stage variant" requirement, which made `all` the
  default, is replaced by "Magnitude kind selects the construction, joint by default", which also makes the
  `motco simulate` default follow the generator. A new requirement makes historical study configs and
  diagnostics pin `all` so their signatures and datasets are preserved.
- `magnitude-construction-diagnostic`: "A size-pure candidate construction is measured, not adopted" is
  replaced by "Diagnostic compares the joint and all constructions explicitly". A profile that names no kind
  now resolves to `joint`, and committed profiles name `all` explicitly.

## Impact

- `src/motco/simulations/semisynthetic.py`: dataclass default and module docstring (`MagnitudeKind` and
  `_MAGNITUDE_KINDS` are unchanged).
- `src/motco/cli.py`: `--magnitude-kind` default and help text.
- `src/motco/simulations/realized_geometry_study.py`, `scripts/latent_rank_probe.py`: explicit
  `magnitude_kind="all"`.
- `src/motco/simulations/showcase.py`: magnitude label and description only.
- `examples/trajectory_power_study/*.json` (10 files): `"magnitude_kind": "all"` added to `generator`.
  `examples/trajectory_power_study/README.md`.
- `tests/`: a new signature-preservation test with a committed pre-flip fixture; existing tests that rely on
  the `'all'` default are pinned explicitly or updated; a CLI default test.
- `docs/api/simulations.md`, `docs/roadmap.md`, `CLAUDE.md`.
- No effect on committed `results/`, reports, or the Phase 5 artefacts. No cluster time.
