## 1. CLI flags

- [x] 1.1 Add four optional arguments to the `simulate` subparser in `src/motco/cli.py`:
  - `--delta-methyl` (float, default `None`)
  - `--delta-expr` (float, default `None`)
  - `--delta-protein` (float, default `None`)
  - `--cluster-mean-shift` (float, default `None`)
- [x] 1.2 At the top of `cmd_simulate` (alongside the existing `--prop-affected-features` validator), validate that each provided delta value is `>= 0`; exit with a non-zero status before `check_intersim_available` if any is negative
- [x] 1.3 Resolve effective per-omic deltas: `eff_<omic> = args.delta_<omic> if args.delta_<omic> is not None else args.cluster_mean_shift`
- [x] 1.4 Pass the resolved values into `InterSIMParams(delta_methyl=eff_methyl, delta_expr=eff_expr, delta_protein=eff_protein, ...)`
- [x] 1.5 Update the subparser help text to describe each flag and the fanout rule

## 2. Tests

- [x] 2.1 Add a test in `tests/test_cli.py`: `motco simulate --cluster-mean-shift 0.7 …` constructs `InterSIMParams` with all three deltas equal to `0.7` (mock the bridge; assert via `call_args`)
- [x] 2.2 Add a test: `motco simulate --cluster-mean-shift 0.7 --delta-expr 1.2 …` results in `delta_methyl=0.7, delta_expr=1.2, delta_protein=0.7` (per-omic wins on conflict)
- [x] 2.3 Add a test: `motco simulate --delta-methyl -0.1 …` exits non-zero before invoking R, with a message naming the value
- [x] 2.4 Add a test: `motco simulate --seed 0 …` (no new flags) constructs `InterSIMParams` with `delta_methyl=None, delta_expr=None, delta_protein=None` (default behaviour preserved)

## 3. Acceptance verification (manual; one-shot empirical sweep)

- [x] 3.1 With R + InterSIM available, run `motco simulate --seed 42 --cluster-mean-shift X --out-dir /tmp/sweep_X/` for `X ∈ {0.25, 0.5, 0.75, 1.0, 1.5, 2.0}` (or whatever covers below and above InterSIM's internal default)
- [x] 3.2 For each output dir, run the toy through the full PLS path with `y = stage` (concat + standardize + `plsda_doubleCV`); record `AUROC_mean` and `AUROC_std`
- [x] 3.3 Confirm monotonicity: smaller `cluster_mean_shift` → lower `AUROC_mean`; at the smallest tested value, `AUROC_mean < 0.99` and `AUROC_std > 0.01`
- [x] 3.4 Record the table in the change folder as `acceptance_sweep.md` (kept as evidence; not committed to specs)

## 4. Documentation

- [x] 4.1 Update `simulate-command` capability spec to list the new flags (covered by the spec deltas in this change)
- [x] 4.2 No README or notebook changes — the bundled toy is not regenerated

## 5. Pre-commit gate

- [x] 5.1 Run `uv run ruff check src/ tests/` — no errors
- [x] 5.2 Run `uv run mypy src/motco/` — no new type errors
- [x] 5.3 Run `MOTCO_TEST_PERMS=99 uv run pytest tests/ -m "not slow" --tb=short` — all pass
