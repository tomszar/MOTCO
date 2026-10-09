# Tasks

## 1. Capture the pre-flip baseline (before any source edit)

- [x] 1.1 On the unedited tree, load and enumerate each of the ten configs under `examples/trajectory_power_study/` that do not name `magnitude_kind`, and write `tests/data/historical_config_signatures.json`, a mapping of config file → {`cell_id` → `parameter_signature`}, with a top-level note naming the capture revision. Verify: the file lists ten configs, each with a non-empty cell map. `git status` shows only this new file.

## 2. Pin the historical configs and diagnostics

- [x] 2.1 Add `"magnitude_kind": "all"` to the `generator` block of the ten configs, next to `surgery_censoring` where present (design D1). Verify: `grep -L '"magnitude_kind"' examples/trajectory_power_study/*.json` prints nothing.
- [x] 2.2 Pass `magnitude_kind="all"` explicitly in `src/motco/simulations/realized_geometry_study.py` and `scripts/latent_rank_probe.py` (design D3). Verify: `uv run mypy src/motco/` is clean and `uv run python scripts/latent_rank_probe.py --help` runs.
- [x] 2.3 Add `tests/test_magnitude_default_pins.py`. It loads and enumerates each config in the fixture and asserts the same cell-id set and the same signature per cell. It also asserts that every committed config under `examples/trajectory_power_study/` names `generator.magnitude_kind`. Verify: the test passes **before** the default flip (task 3.1), confirming the pins alone are behavior-neutral.

## 3. Flip the default

- [x] 3.1 Change `SemiSyntheticTrajectoryParams.magnitude_kind` to default to `"joint"` and rewrite the module docstring's mode list. The docstring should name `joint` as the default and state that `extremes` and `shape_kind='magnitude'` remain methylation-only. Verify: `uv run pytest tests/test_magnitude_default_pins.py` still passes after the flip.
- [x] 3.2 Change `cli.py`'s `--magnitude-kind` default to `"joint"` and its help text to name the three values and the default. Add a CLI test: magnitude mode without the flag writes `truth.json` with `magnitude_kind: "joint"`, and `--magnitude-kind all` writes `"all"`. Verify: `uv run pytest tests/test_cli.py` passes.
- [x] 3.3 Update the showcase's magnitude label and docstring for joint δ scaling (design D4). Verify: `uv run pytest tests/test_showcase.py` passes.
- [x] 3.4 Fix tests that relied on the `'all'` default (expected in `tests/test_joint_magnitude.py`, `tests/test_semisynthetic.py`, and possibly others in magnitude mode). Pin `"all"` where a test checks the methylation-only construction, and switch to or add a `joint` assertion where a test checks the default. Add a test that an unset kind yields `joint` truth, and that explicit `"all"` reproduces the pinned pre-flip block statistics. Verify: `MOTCO_TEST_PERMS=99 uv run pytest tests/ -m "not slow" --tb=short` is green, and no assertion was deleted or loosened (review the diff of `tests/`).

## 4. Documentation

- [x] 4.1 Update `docs/api/simulations.md` (mode table and default, the migration note that the old construction needs explicit `"all"`, and the methylation-only status of `extremes` and `shape_kind='magnitude'`) and `CLAUDE.md`'s `magnitude_kind` sentence (default `joint`; `all` is the methylation-only historical construction pinned in committed configs). Verify: `grep -n "the default" CLAUDE.md docs/api/simulations.md` shows no line calling `'all'` the default.
- [x] 4.2 Add a section to `examples/trajectory_power_study/README.md` covering why the ten configs pin `"all"`, that the pin is behavior-neutral (test and fixture named), that committed `PROVENANCE.txt` `config_sha256` values refer to the file at the run revision (design D5), and that new configs must choose a kind deliberately rather than copy the pin. Verify: the README names the fixture and the test file.
- [x] 4.3 Update `docs/roadmap.md` "Next three changes": retire item 1 and record the one-sided acceptance specificity target as its own follow-up. That follow-up mirrors the gate's α-based SE, is an opt-in key defaulting to two-sided, and is omitted from `observations` when unset so committed reports regenerate byte-identically. Verify: no roadmap line still describes `'all'` as the default.

## 5. Gate

- [x] 5.1 Pre-commit gate: `uv run ruff check src/ tests/ && uv run mypy src/motco/ && MOTCO_TEST_PERMS=99 uv run pytest tests/ -m "not slow" --tb=short`. Verify: all three pass, and `git diff --stat -- results/ docs/reports/` is empty.
