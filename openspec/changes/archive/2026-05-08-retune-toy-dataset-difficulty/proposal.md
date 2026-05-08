## Why

Decision D2 of the original toy-dataset proposal stated: *"Effect size 2.0 produces a clearly visible difference without making the data unrealistically clean."* In practice the dataset is *too* clean for the trajectory analysis: with `effect_size = 2.0`, the angle between groups A and B in the latent space is essentially pinned at ~93° with vanishing RRPP p-values. The pedagogical point of the tutorial — that real multi-omics group differences carry uncertainty that RRPP must adjudicate — is invisible because the injected signal saturates the test.

This change is sequenced after `fix-plsda-nested-cv-aggregation`. The fix-aggregation change made the CV table honest (mean-of-K AUROC); during implementation of the present change we discovered that the AUROC table value is *not* a useful difficulty target here. The notebook runs PLS-DA with `y = stage`, and stage classification on InterSIM cluster-driven data is intentionally easy regardless of the group-effect knobs. The variance the user actually cares about lives in the downstream trajectory analysis (`estimate_difference` + RRPP), not in the CV table. This change therefore retunes the **group-effect** parameters (`--effect-size`, new `--prop-affected-features` flag) to make the trajectory analysis itself more realistic — angle still detectable, p-values still informative, but no longer trivially pinned.

## What Changes

- **`src/motco/cli.py`** — adjust defaults of the `motco simulate` subcommand:
  - `--effect-size`: default lowered from `2.0` to `1.0`. Empirically (see design D5), this drops the latent-space angle from ~93° to ~72°, with RRPP p_angle moving from ≈0.005 to ≈0.035 — detectable but no longer trivially pinned.
  - `--prop-affected-features`: new flag exposing `SemiSyntheticTrajectoryParams.prop_affected_features`, default `0.1` (matches the existing `SemiSyntheticTrajectoryParams` default). Lowering this further (e.g. to 0.05) was shown empirically to *destroy* the orientation signal in the supervised PLS-DA latent space rather than make it more realistic, so the default is left at 0.1 and the flag is exposed for power-user experimentation only.
  - Early validation: `--prop-affected-features` outside `[0, 1]` exits before invoking R.
- **`examples/data/toy/`** — regenerate all nine output files with the new defaults (same `--seed 42`, `--n-samples 90`, `--trajectory-mode orientation`); commit the regenerated CSVs and `truth.json`.
- **`examples/motco_example.ipynb`** — narrative refresh in the markdown cells around the CV table and `motco de` outputs to (a) explain that `y = stage` AUROC remaining near 1.0 is expected (stage classification is the easy part; trajectory analysis is where uncertainty lives), and (b) match the now-realistic numerical outputs from RRPP. The summary cell's claim that "delta p-values are expected to be non-significant in orientation mode" is also corrected — this holds in the *original feature space*, not in the supervised PLS-DA latent space.
- **`README.md`** — update any quick-start AUROC numbers or screenshots that reference the old toy data.
- **Tests for `cmd_simulate`** — extend integration tests so the new `--prop-affected-features` flag is exercised, the early validator rejects out-of-range values, and `truth.json` records the value.

## Capabilities

### Modified Capabilities

- `simulate-command`: defaults of `motco simulate` change for `--effect-size` (2.0 → 1.0); new flag `--prop-affected-features` is added with default `0.1`; out-of-range values exit before invoking R.
- `toy-dataset`: the canonical regeneration command (and therefore the committed CSVs and `truth.json`) reflects the retuned defaults; the spec's "demonstrates orientation trajectory difference" scenario is preserved (a non-trivial angle is still detectable), and a new scenario asserts the realism target — RRPP p_angle for groups A vs B is below 0.1 but above 0 (signal detectable, not trivially so).

### New Capabilities

(none)

## Impact

- `src/motco/cli.py`: one subparser change (new flag + lowered default + early validator).
- `examples/data/toy/`: nine files regenerated; binary diff (each value shifts).
- `examples/motco_example.ipynb`: markdown narrative updated; outputs cleared.
- `README.md`: minor narrative fix if any specific AUROC value is quoted.
- `tests/`: new test asserting `--prop-affected-features` is wired through to `truth.json` and that out-of-range values exit cleanly; existing simulate integration tests keep passing.
- `openspec/specs/simulate-command/spec.md`, `openspec/specs/toy-dataset/spec.md`: updated requirements (covered by the spec-deltas in this change).
- Depends on `fix-plsda-nested-cv-aggregation` being merged first; the AUROC table from `motco plsr` is now mean-of-K (honest reporting). The present change does not move AUROC values; it tunes the trajectory-analysis signal.
- A separate follow-up proposal (`expose-intersim-cluster-mean-shift`) is suggested for exposing InterSIM's internal `cluster_mean_shift` parameter as a CLI knob — that knob would be the right one to control y=stage AUROC difficulty, but its implementation is out of scope here.
- Optional R + InterSIM runtime dependency for `motco simulate` is unchanged.
