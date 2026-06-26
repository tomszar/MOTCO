## 1. Add canonical logit to generator

- [x] 1.1 Add `logit(x, clip=1e-6)` function to `src/motco/simulations/generator.py` directly below `rev_logit`; clip x to `[clip, 1-clip]`, return `np.log(b / (1.0 - b))`
- [x] 1.2 Add `logit` to the module's `__all__` (or verify it is exported if not explicitly listed)

## 2. Update methylation_recovery to import from generator

- [x] 2.1 In `src/motco/simulations/methylation_recovery.py`, replace the local `beta_to_mvalue` definition with `from motco.simulations.generator import logit as beta_to_mvalue` (preserving the public alias `beta_to_mvalue` and the `_LOGIT_CLIP` constant reference in the docstring)

## 3. Apply M-value conversion in all integration helpers

- [x] 3.1 Import `logit` from `motco.simulations.generator` at the top of `src/motco/simulations/evaluation.py`
- [x] 3.2 In `_concat_integration`: after loading the methylation matrix (before standardisation), apply `logit` to its values — expression and proteomics are untouched
- [x] 3.3 In `_snf_integration`: after calling `getattr(dataset, layer)` for the methylation layer, apply `logit` — only the methylation layer
- [x] 3.4 In `_pls_integration`: in the loop over `_OMICS_ATTRS`, apply `logit` to the methylation values before standardisation

## 4. Add paper-grade study config

- [x] 4.1 Create `examples/trajectory_power_study/study.json` with `integration_method="pls"`, `n_replicates=500`, `permutations=999`, `n_samples=300`, same modes/effect-sizes/acceptance structure as smoke.json but paper-grade sizing

## 5. Verify and gate

- [x] 5.1 Run `uv run ruff check src/ tests/` — no new errors
- [x] 5.2 Run `uv run mypy src/motco/` — no new errors
- [x] 5.3 Run `MOTCO_TEST_PERMS=99 uv run pytest tests/ -m "not slow" --tb=short` — all tests pass (update any snapshots that change due to M-value input)
- [ ] 5.4 Run the smoke study end-to-end: 4 shards + merge + report; confirm specificity matrix is written without errors