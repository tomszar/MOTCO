## Why

The InterSIM bridge proves MOTCO can obtain realistic multi-omics matrices from R, but those outputs are not yet MOTCO trajectory simulation datasets. To evaluate power and Type I error, MOTCO needs a reproducible semi-synthetic generator that converts InterSIM clusters into ordered trajectory stages, assigns comparison groups, and records known trajectory truth.

## What Changes

- Add a semi-synthetic trajectory generator that consumes `InterSIMResult` or invokes `run_intersim()` through existing parameters
- Adopt an explicit clusters-as-stages assumption for the first generator: InterSIM cluster labels are mapped to ordered trajectory stages
- Assign samples to comparison groups with configurable group balance while preserving aligned methylation, expression, and proteomics matrices
- Produce a packaged dataset object containing omics matrices, sample metadata (`sample_id`, group, stage, cluster), and truth metadata
- Support initial trajectory truth modes: `none`, `translation`, `magnitude`, `orientation`, and `shape`
- Apply group-specific trajectory effects after InterSIM simulation, leaving InterSIM as the source of realistic within- and cross-omic structure
- Provide deterministic behavior through explicit seeds and record all generator parameters in metadata
- Keep this change limited to dataset generation and truth construction; MOTCO integration, RRPP execution, power estimation, Type I error summaries, and grid orchestration remain out of scope

## Capabilities

### New Capabilities

- `semisynthetic-trajectory-generator`: Generate MOTCO-ready semi-synthetic trajectory datasets from InterSIM outputs using the clusters-as-stages assumption, group assignment, trajectory effect injection, and truth metadata

### Modified Capabilities

*(none - the existing InterSIM bridge contract is consumed but not changed)*

## Impact

- **New code area**: generator module under `src/motco/simulations/`
- **Depends on**: `intersim-simulation-bridge`
- **API additions**: parameter and result models for semi-synthetic trajectory datasets, plus a generator function
- **Tests**: deterministic pure Python tests using small synthetic InterSIM-like fixtures, plus optional InterSIM-backed smoke coverage if useful
- **Documentation**: describe clusters-as-stages assumption, supported trajectory modes, and output dataset contract
- **No breaking changes** to existing MOTCO stats, InterSIM bridge, or CLI APIs
