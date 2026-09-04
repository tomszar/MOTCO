## Context

See proposal.md — Why. Eight documentation/spec defects (D1–D8) from the 2026-09-01 geometry
audit; seven are prose fixes, one (D8) is a small behavior change. Current state that shapes the
approach:

- `stats/design.py:get_model_matrix` documents and implements "category order deterministic:
  sorted by string representation" (`design.py:104-105`);
  `simulations/evaluation.py:build_simulation_trajectory_design` repeats the same `sorted()` on
  group and stage levels (`evaluation.py:385-386`).
- The semisynthetic generator labels stages `str(range(n_stages))` (`semisynthetic.py:680`), so
  every existing config uses single-digit labels — string sort and numeric sort agree today.
- `build_ls_means` takes level lists as arguments; it needs no change so long as the caller passes
  levels in the same order the model matrix used.
- Committed study records, resume signatures, and reference CSVs were all produced under
  single-digit stage labels and two string group labels; D8 must be byte-identical for those.

## Goals / Non-Goals

**Goals:**
- Every documented example runs; every doc statement matches the shipped estimator and generator.
- One authoritative SNF status message across CLAUDE.md, `evaluation.py`, and the roadmap.
- Stage ordering is correct for `n_stages ≥ 10` without perturbing any existing configuration.

**Non-Goals:**
- No change to any statistic, test decision, record schema, or resume signature.
- No re-run of any study; no new results.
- No general natural-sort of mixed alphanumeric labels (e.g. "stage10" still sorts
  lexicographically) — only fully integer-like label sets get numeric order.
- **Not regenerating the toy fixture.** D10 corrects what the `toy-dataset` spec *claims*; it does
  not resolve whether `examples/data/toy/` should be regenerated with the numpy generator. That
  touches committed example data, `examples/motco_example.ipynb`, the spec's scenarios, and
  `tests/test_toy_dataset.py`, so it is tracked as a cross-cutting item in `docs/roadmap.md` and
  needs its own change.

## Decisions

**D8 policy: numeric order for all-integer label sets, not an error guard.**
The audit allowed either a guard (raise at ambiguity) or a fix. Sorting numerically when *every*
label of a factor parses as an integer is strictly better: it makes the hazard impossible rather
than loud, and it is provably identical to the current order for every label set in use
(single digits; alphabetic group labels fall back to lexicographic). An error guard would leave
`n_stages ≥ 10` unusable rather than correct. Rejected alternative: full natural sort of mixed
labels ("stage2" < "stage10") — larger behavior surface, no current consumer, and it would change
order for label sets we cannot enumerate; the integer-only rule is exactly the set where
lexicographic order is *wrong on its own terms*.

Implementation shape: one private helper (`_sort_levels`) in `stats/design.py`, used by
`get_model_matrix`; `build_simulation_trajectory_design` imports it so harness and core can never
disagree. Leading zeros ("01" vs "1"): if two distinct labels collide numerically, fall back to
lexicographic for that factor — collision would otherwise make the order ill-defined.

**D7 policy: the roadmap's message wins.**
CLAUDE.md and the `evaluation.py` docstring call SNF a "production latent-space method"; the
roadmap says SNF's graph-spectral geometry is not aligned with the Euclidean trajectory statistics
and defers it pending graph-native statistics (Phase 7). The roadmap position is the
evidence-backed one (rung-2/rung-3 findings, metric-compatibility report), so CLAUDE.md and the
docstring adopt it: **PLS is the production measurement space**; SNF remains a supported
integration path whose use with the Euclidean statistics is deferred. `concat` stays
baseline/diagnostic. No code or CLI behavior changes — `snf` remains selectable.

**D5 mechanics: edit the main spec's purpose line directly.**
Delta specs carry requirement operations only; a purpose-line rewording is not a requirement
change. Per OpenSpec guidance, purpose edits go straight to
`openspec/specs/semisynthetic-trajectory-generator/spec.md` as part of implementation.

**D6 wording: state exactly what matched seeds share.**
The corrected claim, used verbatim in all three places (grid docstring, showcase docstring,
pivotality config note): matched seeds share the baseline indicator draw and the within-stage
group assignment; the mode transforms consume different RNG amounts, so sampled values differ
across modes, and the only common-random-numbers pair is `none`↔`magnitude` (whose transforms draw
nothing). Comparisons across modes are paired at the *generated reference*, not on identical
datasets.

**D2 example content:** rewrite with real flags only (`--seed`, `--n-samples`, `--n-stages`,
`--trajectory-mode`, `--effect-size`, `--p-dmp`, `--cluster-mean-shift`, `--out-dir`), comment
"numpy generator; no R at runtime". **D3 example content:** `SemiSyntheticTrajectoryParams` +
`generate_semisynthetic_trajectory` + `evaluate_semisynthetic_trajectory`, with parameters that
exist (`group_effect_size`, `p_dmp`); the snippet must execute as written in a fresh venv.

## Risks / Trade-offs

- [Numeric ordering changes design-matrix column order for hypothetical external users with
  integer-like labels ≥ 10] → That order was silently wrong for trajectories anyway (path length
  and shape depend on stage order); changelog-worthy note in the docstring. No repo-internal
  consumer is affected.
- [D3 example rot recurring] → Keep the snippet minimal (three imports, two calls); a follow-up
  doctest harness is out of scope here.
- [D7 wording drifting again] → Single source of truth: CLAUDE.md and the docstring now *point* at
  the roadmap's status rather than restating an independent claim.

## Migration Plan

Single PR; no data or schema migration. Rollback is a revert. Existing shards resume cleanly —
parameter signatures do not encode level ordering and no generator or evaluation parameter
changes.
