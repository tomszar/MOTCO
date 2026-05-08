## Why

The bundled toy dataset still makes `motco plsr --label-col stage` report saturated stage classification (`AUROC = 1.0`, `AUROC_std = 0.0`) because InterSIM clusters are too cleanly separated. This is confusing for a usage example that should demonstrate realistic finite-sample uncertainty rather than a perfect classifier.

## What Changes

- Regenerate `examples/data/toy/` with the existing canonical toy parameters plus `--cluster-mean-shift 0.10`.
- Keep the same seed, sample count, trajectory mode, group effect size, and affected-feature proportion unless validation shows the downstream trajectory signal no longer lands in the intended range.
- Update README and notebook examples to show the new canonical regeneration command and, where relevant, set expectations that stage classification should be non-saturated.
- Verify the regenerated toy still supports the full tutorial path: PLS-DA latent-space generation, differential trajectory analysis, and repository-clone usability without R.

## Capabilities

### New Capabilities

- None.

### Modified Capabilities

- `toy-dataset`: Change the canonical pre-generated toy dataset regeneration command and acceptance expectations so the bundled toy has moderate, non-saturated stage-classification difficulty.

## Impact

- Affects `examples/data/toy/` generated CSV/JSON files.
- Affects README and example notebook text/commands.
- Affects tests that validate toy-data generation parameters or expected toy behavior.
- Requires R + InterSIM only during regeneration/verification; the committed toy dataset remains usable without R.
