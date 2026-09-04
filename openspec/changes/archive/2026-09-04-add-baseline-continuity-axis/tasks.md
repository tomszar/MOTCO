# Tasks: add-baseline-continuity-axis

## 1. Generator mechanism

- [x] 1.1 Add `markov_indicators(rng, n_feat, n_cell, p, rho)` to `generator.py` per design D2 (single uniform block, per-stage thresholds), with a docstring stating the stationary-marginal contract; verify with new unit tests: marginal frequency ≈ `p` at every stage for ρ ∈ {0, 0.5, 0.9}, adjacent-stage indicator correlation ≈ ρ, and `rho=0` output exactly equals `bernoulli_indicators` on the same fresh `rng`.
- [x] 1.2 Add `baseline_continuity: float = 0.0` to `SemiSyntheticTrajectoryParams`, validate `[0, 1)` in `_validate_params` (clear error naming parameter and range), and route `_baseline_methyl` through `markov_indicators`; verify with tests that out-of-range values raise and that generation succeeds at ρ = 0.9 for every trajectory mode.
- [x] 1.3 Byte-identity regression: verify `generate_semisynthetic_trajectory` at default `baseline_continuity=0.0` produces datasets identical (indicators, all three matrices, metadata, truth) to the pre-change generator at the same seed — pin via existing fixture-based tests passing unchanged plus an explicit equality test against values generated on `main` before the change.
- [x] 1.4 Record `baseline_continuity` in the truth `params` block and verify a test asserts its presence and value for ρ = 0 and ρ > 0.
- [x] 1.5 Geometry sanity test: at ρ = 0.9 vs ρ = 0 (same seed, `none` mode), verify the expected trending structure — mean squared population stage-mean distance grows with stage separation at high ρ and is flat at ρ = 0 (use `population_trajectories` means; tolerance-based, seeded).

## 2. Headroom generalization

- [x] 2.1 Generalize `expected_surgery_headroom`'s `active_fraction` to `1 − (1−p)·(1 − p(1−ρ))^(n−1)` per design D4; verify ρ = 0 reproduces the current values exactly (pin existing headroom tests unchanged) and ρ > 0 yields a larger saturating effect for orientation and shape-relocate at fixed `p_dmp`/`n_stages`.
- [x] 2.2 Monte Carlo adequacy test for the analytic pool at high ρ: generate indicator draws at ρ ∈ {0, 0.5, 0.9}, compare realized stage-active-union counts to the analytic expectation, and verify the 3σ guard band keeps the realized-censoring rate at the analytic saturating effect at zero across the sweep (widen the band per design risk note if not).
- [x] 2.3 Verify `study/enumerate.py` headroom rejection uses each cell's own continuity value with no further code change (it reads the cell's generator params): add an enumeration test where an orientation effect fails at ρ = 0 but enumerates at ρ = 0.9.

## 3. Study integration and reporting

- [x] 3.1 Add a config/enumeration test sweeping `generator.baseline_continuity` as an axis: deterministic cell ids, distinct parameter signatures per ρ, and signatures differing from pre-change ones (documenting the deliberate resume break).
- [x] 3.2 Implement the continuity-resolved report output per design D6 (emitted only when the merged set varies `baseline_continuity`): per continuity × mode × effect, per-statistic rejection rates, eigengap summaries from `config_spectrum`, and `null_summary["angle"]["q95"]` dispersion; verify with a synthetic-records test that the table is produced from records alone and is absent when continuity is constant.
- [x] 3.3 Verify existing study reports are byte-stable for continuity-free merged sets (run the report regression tests unchanged).

## 4. Docs and closure

- [x] 4.1 Update the `semisynthetic.py` module docstring (independent baseline becomes the ρ = 0 endpoint of a declared axis), `CLAUDE.md`'s semisynthetic bullet, and `docs/phase5-readiness.md` item 4 (the second lever now exists); verify by proofreading rendered diffs against the audit's P4 wording.
- [x] 4.2 Full pre-commit gate: `uv run ruff check src/ tests/ && uv run mypy src/motco/ && MOTCO_TEST_PERMS=99 uv run pytest tests/ -m "not slow" --tb=short` passes.
