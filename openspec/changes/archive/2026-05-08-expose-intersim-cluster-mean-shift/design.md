## Context

InterSIM (the R package this project bridges to) generates clustered multi-omics data by drawing each omic's features from per-cluster Gaussians whose means are separated by per-omic delta parameters: `delta.methyl`, `delta.expr`, `delta.protein`. These deltas are the dominant knob for "how separable are the clusters" in the simulated data, and downstream MOTCO code interprets clusters as disease stages, so they are also the dominant knob for "how easy is `y = stage` PLS-DA classification."

The Python-side `InterSIMParams` dataclass (`src/motco/simulations/intersim.py:49`) already exposes these as optional fields with `None` meaning "use InterSIM's own default". The CLI wrapper (`src/motco/cli.py:cmd_simulate`) currently constructs `InterSIMParams` with only `seed` and `n_sample`, leaving all the deltas at InterSIM's defaults. There is no way for a CLI user to influence cluster separation without dropping into Python.

This change is motivated by the discovery during `retune-toy-dataset-difficulty` that `--effect-size` and `--prop-affected-features` do *not* control `y = stage` AUROC on InterSIM data — they only control the *group effect* used by trajectory analysis. Cluster-mean-shift is the right knob for AUROC difficulty.

## Goals / Non-Goals

**Goals:**
- Expose `delta.methyl`, `delta.expr`, `delta.protein` on the `motco simulate` CLI.
- Provide a convenience `--cluster-mean-shift` scalar that fans out to all three per-omic deltas when they are not individually set.
- Validate values are non-negative; exit before invoking R on out-of-range input.
- Preserve the current default behaviour: when none of the new flags are passed, `motco simulate` produces the same output as today.

**Non-Goals:**
- Regenerating `examples/data/toy/` with new defaults. (The current toy is fine; a future "harder tutorial" toy is a separate proposal if wanted.)
- Adding a tutorial section that sweeps cluster-mean-shift. (A future explore session may revisit this.)
- Exposing `cor_methyl_expr`, `cor_expr_protein`, `sigma_*`, or other InterSIM params. (Each could be its own follow-up; we don't bundle them here so this change has a clean acceptance criterion.)
- Introducing a generic "simulation difficulty preset" abstraction.

## Decisions

### D1 — Expose three per-omic flags AND a fanout convenience flag

The three per-omic flags (`--delta-methyl`, `--delta-expr`, `--delta-protein`) are the honest representation of how InterSIM parametrizes cluster separation. The fanout flag (`--cluster-mean-shift`) is a CLI ergonomics affordance: most tutorial users will want to sweep one knob, not three.

Resolution rule: each per-omic delta's effective value is `args.delta_<omic> if args.delta_<omic> is not None else args.cluster_mean_shift`. If both are `None`, the field stays `None` in `InterSIMParams` and InterSIM applies its own default.

Alternative considered: only the three per-omic flags. Rejected because it forces tutorial users to repeat the same value three times.

Alternative considered: only `--cluster-mean-shift` as a scalar that fans out. Rejected because power users may want to ablate per omic (e.g. set methylation high, expression low) and the fanout-only design forecloses that without dropping to the Python API.

### D2 — `--cluster-mean-shift` is a scalar, not `nargs=3`

Argparse supports `nargs=3` to accept three values: `--cluster-mean-shift 1.0 0.8 1.2`. Rejected because:
- Per-omic granularity is already covered by the three dedicated flags.
- A vector form would be redundant with the dedicated flags AND require ordering knowledge (which omic is first?).
- Scalar fanout matches the most common use case (sweep difficulty along one axis).

### D3 — Validation: non-negative; exit before invoking R on out-of-range

Pattern matches `--prop-affected-features` from the prior change: validate at the top of `cmd_simulate`, before `check_intersim_available` runs. Negative values are clearly invalid (deltas are magnitudes). Zero is allowed (collapses cluster separation entirely; a useful null case).

### D4 — Default values stay `None`

When neither the per-omic flag nor `--cluster-mean-shift` is set, the corresponding `InterSIMParams.delta_*` stays `None` and the Python-side bridge omits it from the R command-line. InterSIM's own defaults then apply. This guarantees that running `motco simulate` without any of the new flags produces byte-identical output to today.

### D5 — Acceptance verification

Before marking the change done, run a small empirical study on seed 42 to characterize the AUROC × cluster-mean-shift relationship:

```
motco simulate --seed 42 --cluster-mean-shift <X> --out-dir /tmp/sweep_<X>/
```

for `X ∈ {0.5×default, 0.75×default, default, 1.25×default, 1.5×default}`. For each, run the supervised PLS-DA path with `y = stage` (the notebook's PLS section) and record `AUROC_mean` and `AUROC_std`.

Acceptance criteria:
- The relationship is monotone (smaller delta → lower mean AUROC).
- At the smallest tested value, `AUROC_mean < 0.99` AND `AUROC_std > 0.01` — i.e. there exists a setting where stage classification is non-trivial. (This is what makes the new flag useful.)
- At the default and above, behaviour matches the current toy (AUROC saturates near 1.0).

If the small-delta point does not produce honest variance, either:
- Increase the swept range (try 0.25× default), or
- Document in the spec that cluster-mean-shift alone is insufficient and a noise term will be needed in a separate change.

## Risks / Trade-offs

- **[Risk] InterSIM's "default" delta values are not stable across versions.** If a future InterSIM release changes the default for `delta.methyl` etc., the meaning of `--cluster-mean-shift 1.0` shifts under us. *Mitigation:* none in scope here; the bridge accurately reports the resolved value via `truth.json` so users can audit.
- **[Risk] Cluster-mean-shift may interact non-linearly with `effect_size`.** Lowering cluster shift makes the latent space less stage-aligned, which can amplify or distort the apparent group effect. *Mitigation:* the acceptance verification in D5 measures AUROC only at default `effect_size`, leaving the interaction to a future grid study.
- **[Trade-off] Two flags for one conceptual knob is a bit confusing.** D1 spends one decision-doc paragraph addressing this; the help text in `--help` should be tight.
- **[Trade-off] We're not regenerating the bundled toy.** A future user may wonder "why doesn't the bundled toy show non-trivial AUROC?" — answer: by design, the toy demonstrates trajectory analysis, not classification difficulty. Sub-proposal `harden-toy-dataset-stage-classification` could revisit this if there's appetite.
