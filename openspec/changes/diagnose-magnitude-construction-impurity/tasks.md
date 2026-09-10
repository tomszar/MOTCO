## 1. Block-decomposition reader

- [x] 1.1 Add a block-decomposition reader to `simulations/specificity.py` that takes a merged JSONL path and a trajectory mode and returns pooled-versus-per-block `delta`/`angle`/`shape` per checkpoint and scope beside the shared zero-effect anchor's values, built on `summarize_realized_geometry`; verify it reproduces the design's table from `results/phase5-2026-09-10/merged.jsonl` (magnitude joint `angle` 6.6978/12.1199/16.5890/20.3293, exact 0.0000 in every single-omic scope, `delta` nonzero only in methylation).
- [x] 1.2 Add tests for the reader on a small synthetic record set: per-block and joint scopes both surfaced, anchor identified and excluded from the mode rows, a checkpoint missing a scope reported as unavailable rather than dropped or zero-filled.
- [x] 1.3 Verify the reader is mode-agnostic by running it for `orientation` and `shape` as well, and record whether their off-target responses are also concentrated in the joint scope (this is free evidence for the exit review's cross-talk section, and the report should say what it found).

## 2. Materiality recalibration

- [x] 2.1 Add a rule selector to `localize_off_diagonal` with the legacy absolute-threshold rule and a new null-dispersion rule, and record the active rule and the dispersion used in every emitted row; verify the legacy rule is the default so far.
- [x] 2.2 Implement the null-dispersion rule (design D2a): excess over the anchor's normalized value at the same checkpoint divided by that anchor reference's dispersion across replicates, falling back — where the anchor's dispersion is zero or dust, as it is at every population checkpoint — to whether the excess exceeds the dust tolerance, with the deciding path recorded per row. Verify no division by a dust-valued dispersion occurs (measured anchor sd at `population_standardized`: `delta` exactly 0.0, `angle` 8.7e-07, `shape` 3.5e-16).
- [x] 2.3 Verify the recalibrated rule classifies magnitude/`shape` at a checkpoint instead of `not_material` on the Phase 5 records, and record the classification for both failing pairs (magnitude/`angle`, magnitude/`shape`) under both rules.
- [x] 2.4 Verify the committed Phase 5 `report/` still regenerates byte-identically across all 23 files under the default rule, exactly as the findings report asserts.
- [x] 2.5 Add tests for the new rule: commensurability (a small-scale statistic with a proportionally small null is classified), degenerate-dispersion handling, rule recorded in the output, and the legacy rule unchanged; update existing localization tests to the new columns without changing their legacy-rule expectations.

## 3. Uniform-δ candidate

- [x] 3.1 Add a probe-only uniform-δ candidate path in `simulations/semisynthetic.py` that scales every omic's δ by the same factor, reachable from the diagnostic entry point only; verify `_MAGNITUDE_KINDS` validation still rejects it as a `magnitude_kind` value.
- [x] 3.2 Add a test asserting a study configuration naming the candidate as a `magnitude_kind` is refused, so no committed profile can acquire it silently.
- [x] 3.3 Evaluate the candidate at the Phase 5 design point's generator parameters over the same effect grid at the standardized-geometry checkpoint only (no sampling, no RRPP, no PLS fit); verify whether joint `angle` and `shape` fall to the anchor's exact zero after per-block standardization, and record the per-block and joint values at every effect.
- [x] 3.4 Run `characterize_two_stage` for the magnitude mode at `n_stages=2` and record whether the joint orientation response persists where shape is undefinable; verify the two-stage and four-stage `angle` responses are reported side by side.

## 4. Driver and findings report

- [x] 4.1 Add a `scripts/` driver that runs the block decomposition, the recalibrated localization, the uniform-δ comparator and the two-stage isolation, writing its outputs under a dated results directory; verify it runs on the workstation with no cluster and no R runtime.
- [x] 4.2 Write `docs/reports/magnitude-construction-diagnostic-<date>.md` stating the mechanism verdict (with the block-decomposition table), the size-purity verdict for the uniform-δ candidate, the two-stage confirmation, and the recalibrated classification for both failing pairs under both rules; verify every number cites a committed output path and each verdict names the revision it implies (method or claim) without adopting one.
- [x] 4.3 Record in the report what the reader found for `orientation` and `shape` off-target responses (task 1.3), including whether the paper-grade run's predeclared "projection-associated" label for orientation/`shape` is supported by the block decomposition; verify the report does not restate the predeclared label as a finding.
- [ ] 4.4 Commit the diagnostic outputs and the report; verify no large regenerable artifact is staged and that `results/phase5-2026-09-10/` is untouched.

## 5. Docs and gate

- [x] 5.1 Update `docs/roadmap.md` ("Not yet established" — the magnitude-impurity line now has a measured mechanism; "Next three changes" — the exit review's inputs are in hand) and `docs/phase5-readiness.md` where it points at the exit review; verify no doc claims the mechanism is unverified.
- [x] 5.2 Run the pre-commit gate (`uv run ruff check src/ tests/ && uv run mypy src/motco/ && MOTCO_TEST_PERMS=99 uv run pytest tests/ -m "not slow" --tb=short`) and verify all three pass.
