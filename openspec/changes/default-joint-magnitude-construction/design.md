# Design

## Context

See proposal.md for motivation. These facts shape the approach:

- `parameter_signature` (`grid.py`) hashes `_to_jsonable(generator_params)`. That is every dataclass field
  with defaults already resolved. A config that names `"magnitude_kind": "all"` therefore serializes exactly
  as one that omitted the key under the old default. Cell ids and shard assignment
  (`partition_unit(cell_id, replicate)`) are built from the same resolved parameters, so they are preserved
  too.
- The censoring-policy precedent (`fix-effect-axis-censoring`, D4) added a *new* field, so every signature
  changed regardless and its pins only kept old configs loadable. This change is the first default flip on an
  *existing* field. Signature preservation is achievable here, but nothing in the codebase guarantees it yet.
  It needs its own test.
- `SemiSyntheticTrajectoryParams` is built directly, outside the study loader, in five places:
  - `cli.py`, which has its own argparse `default="all"`;
  - `showcase.py`;
  - `realized_geometry_study.py`, which includes a magnitude construction in `PHASE2_CONSTRUCTIONS`;
  - `specificity.py` (three sites), whose functions take an explicit `magnitude_kind: str = "all"`
    parameter;
  - `scripts/latent_rank_probe.py`, whose committed run (`results/latent-rank-probe-2026-09-03/`) used
    `--modes orientation none`, so no magnitude cell. `magnitude_kind` is inert outside magnitude mode.
- Committed `PROVENANCE.txt` files record each config's `config_sha256`. They are documentation, and no code
  verifies them. The study's own identity is the per-cell signature.

## Goals / Non-Goals

**Goals:**

- Every caller that relies on the default gets `joint`. Every committed config and historical diagnostic
  keeps its exact behavior through an explicit `all`.
- A test fails if any historical config's enumerated signatures, cell ids or cell set change.

**Non-Goals:**

- Removing the default, i.e. requiring `magnitude_kind` explicitly. This was rejected in exploration as too
  much friction for the CLI and quick experiments.
- Changing what `specificity.py`'s comparison helpers default to.

## Decisions

### D1. Pin in the config file, not in the loader

Each of the ten configs gains `"magnitude_kind": "all"` in its `generator` block. The two Phase 5 magnitude
profiles already name `"joint"` and are untouched. The pin goes next to `surgery_censoring` where a config
already names that, to follow the existing convention.

*Alternatives rejected:*
- A loader-side default keyed on a config version or date. It is hidden behavior, and a config would no
  longer say what it runs.
- Leaving the configs alone and accepting new signatures. That breaks resume and the "reproduce from the
  committed config" property of every committed result, for no benefit.

### D2. Signature preservation is checked against a fixture captured before the flip

The ordering is the point. **Before** any code edit, a one-off capture writes
`tests/data/historical_config_signatures.json`. For each of the ten configs it maps every enumerated cell's
`cell_id` to its `parameter_signature`. The capture runs on the pre-change code with the configs
unedited. After the pins and the flip, the test loads and enumerates each config and asserts:

- the same set of cell ids;
- the same signature per cell.

That covers signatures, cell ids, matched seeds (derived from the cell id and family) and shard assignment.
Enumeration is cheap and R-free, so the test runs in the fast suite. The fixture is committed with a note
naming the revision it was captured at.

*Alternative rejected:* computing "before" and "after" in the same test by monkeypatching the default. The
test would then check the pinning mechanism against itself, not against what was actually committed and run.

### D3. Diagnostics: explicit `all` where the committed run used magnitude, and also for the probe

- `realized_geometry_study.py` passes `magnitude_kind="all"`. Its magnitude construction's committed outputs
  (2026-09-01/02 runs) were produced under the old default.
- `scripts/latent_rank_probe.py` also passes `magnitude_kind="all"`. Its committed run did not include
  magnitude, so this pin is defensive: a re-run with `--modes magnitude` then means what the probe meant when
  it was written. It costs one keyword.
- `specificity.py` is unchanged. Its helpers already take `magnitude_kind` as an explicit parameter
  defaulting to `"all"`, and callers that compare constructions name both kinds.

### D4. CLI and showcase follow the generator

The argparse default becomes `"joint"`, so `motco simulate` agrees with the dataclass. The help text names
all three values and the default. The showcase passes no kind, so its magnitude panel becomes joint. Its
label and docstring say "size (delta), joint δ scaling". No showcase figure is committed, so nothing frozen
changes.

### D5. Config file hashes in committed provenance

Pinning changes the bytes, and so the sha256, of configs whose `PROVENANCE.txt` records a `config_sha256`.
Nothing verifies those hashes. Each `PROVENANCE.txt` also names the code revision at which its run
executed, and the config at that revision still matches the recorded hash. The examples README records that
the pin is the only edit and that D2's test proves the edit is behavior-neutral. Committed `PROVENANCE.txt`
files are frozen outputs and are not rewritten.

## Risks / Trade-offs

- [A config is missed] → D2's fixture covers exactly the ten configs. A second assertion requires every
  committed config under `examples/trajectory_power_study/` to name `generator.magnitude_kind`. After this
  change all twelve do, so a later config that omits the key fails the test and its author has to choose a
  construction deliberately.
- [Existing tests silently depend on the `'all'` default] → expected, notably the pinned-dataset tests in
  `tests/test_joint_magnitude.py` and magnitude-mode assertions in `test_semisynthetic.py`. Each is pinned to
  `"all"` explicitly where it tests the methylation-only construction, or updated to `joint` where it tests
  the default. No assertion is weakened.
- [Fixture captured on already-edited code] → D2's ordering is task 1.1, which runs before any source edit.
  The fixture note records the capture revision (`main` at `5d7359b` or the change branch's base).
- [Readers conflate `extremes` / `shape_kind='magnitude'` with joint scaling] → the docstring, API docs and
  the generator spec state that both remain methylation-only.

## Migration Plan

There are no user-facing breaks for committed artefacts. Code that relied on the default and wants the old
construction must pass `magnitude_kind="all"` (or `--magnitude-kind all`). This is noted in the API docs.
To roll back, revert the default and the CLI default. The pins are harmless under either default and can
stay.
