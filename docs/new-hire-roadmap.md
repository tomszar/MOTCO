# MOTCO New-Hire Roadmap

## Goal

Bring MOTCO from "core methods implemented" to "scientifically trustworthy, faster, and usable by collaborating scientists" without changing the scientific intent of the package.

## Current State of the Repo

What appears to be in place:

- Core statistical modules exist in `src/motco/stats/`:
  - `pls.py`: PLS-DA with double cross-validation
  - `snf.py`: Similarity Network Fusion and spectral embedding
  - `sd.py`: trajectory-difference statistics, LS-means utilities, and RRPP
- There is a working CLI in `src/motco/cli.py`.
- There are smoke tests plus regression tests against bundled reference datasets in `tests/`.
- There are helper scripts for R/Python comparison and timing in `scripts/`.

What looks incomplete or fragile:

- The repo has a publish workflow but no normal CI workflow for tests/linting.
- Test execution depends on the environment being set up with `uv sync --extra test`.
- The expensive RRPP tests use `n_jobs=-1`; in restricted environments they fail because multiprocessing semaphores are blocked, even when the math is correct.
- The regression tests default to `MOTCO_TEST_PERMS=10000`, which is too heavy for normal developer feedback.
- Packaging/docs have drift:
  - `README.md` says Python `3.9+`
  - `pyproject.toml` requires Python `>=3.11`
  - `pyproject.toml` still has a placeholder homepage URL
- The CLI is functional but still low-level for end users:
  - limited input validation
  - raw CSV/JSON interfaces
  - no workflow-level command for a full analysis run
  - no model/export/reporting layer

Observed test status:

- `tests/test_sd_smoke.py`: passes
- `tests/test_sd_expected_example1.py`: passes when run outside the sandbox with reduced permutations
- `tests/test_sd_expected_example2.py`: passes when run outside the sandbox with reduced permutations
- The failures seen inside the sandbox were environment/process failures, not confirmed numerical mismatches

## Recommended Hire Profile

Hire a scientific Python engineer, not a generic full-stack developer.

Minimum fit:

- Strong Python, NumPy, pandas, scikit-learn
- Comfortable reading and validating statistical code
- Experience with reproducibility, benchmarking, and test design
- Able to work with scientists and convert methods into stable software

Strong bonus:

- Background in multivariate statistics, omics, or computational biology
- Experience comparing Python output against R reference implementations
- Experience optimizing numeric code and parallel workloads

## Top Priorities

The new hire should work in this order:

1. Stabilize the development and test workflow.
2. Confirm scientific correctness against the R/reference outputs.
3. Reduce runtime and memory cost in the expensive paths.
4. Improve the scientist-facing workflow and documentation.
5. Add release-quality engineering around CI, packaging, and examples.

## 30 / 60 / 90 Day Roadmap

### Days 1-30: Stabilization and Truth Baseline

Primary objective: make the project easy to run, test, and trust.

Deliverables:

- Reproducible local setup documented and validated with `uv`
- A CI workflow that runs a fast test suite on every push/PR
- A split test strategy:
  - fast smoke/unit tests for normal development
  - slower scientific regression tests for scheduled/manual runs
- A short validation note describing:
  - what is compared against reference data
  - what currently matches
  - what still needs deeper verification
- Cleanup of packaging/docs drift:
  - align Python version requirements
  - replace placeholder project metadata
  - make install/test commands consistent

Acceptance criteria:

- A new developer can clone the repo and run the fast tests without trial-and-error.
- CI catches breakage before release tags.
- Regression tests no longer fail just because multiprocessing is unavailable.

### Days 31-60: Scientific Validation and Performance

Primary objective: turn "it seems right" into "we know what is right and how fast it is."

Deliverables:

- A validation matrix covering:
  - PLS-DA outputs
  - SNF outputs
  - `sd.py` trajectory statistics
  - RRPP p-value behavior
- Benchmarks for the expensive functions, especially RRPP and trajectory comparisons
- Concrete performance improvements with before/after measurements
- A decision on parallel strategy:
  - robust serial fallback
  - optional multiprocessing
  - configuration exposed in CLI/API

Acceptance criteria:

- Scientific regression results are reproducible and documented.
- Runtime for typical datasets is measured and improved in the main bottlenecks.
- Parallel execution is optional and does not break local development or CI.

### Days 61-90: Usability and First Real Delivery

Primary objective: make MOTCO usable by other scientists without hand-holding.

Deliverables:

- A clearer CLI and end-to-end example workflow
- Better error messages and input validation
- A minimal "analysis recipe" for common tasks:
  - create latent space
  - run trajectory comparison
  - save interpretable outputs
- Example outputs and documentation for scientists
- A release checklist and versioning process

Acceptance criteria:

- A scientist can run a small example end-to-end from docs only.
- Outputs are understandable without opening source code.
- Release steps are routine rather than ad hoc.

## Concrete Workstreams

### 1. Environment and Reproducibility

Tasks:

- Standardize on one supported Python range and document it.
- Make `uv sync --extra test` the canonical setup path.
- Remove ambiguity around optional tools such as R.
- Add a developer setup section and a short troubleshooting section.

Likely files:

- `README.md`
- `pyproject.toml`
- `.github/workflows/`

### 2. Test Strategy

Tasks:

- Separate fast and slow tests with markers or dedicated commands.
- Make RRPP tests configurable by permutation count and safe in CI.
- Add coverage for CLI behavior and invalid inputs.
- Add direct tests for PLS and SNF, not only `sd.py`.

Likely files:

- `tests/`
- `pyproject.toml`
- `.github/workflows/`

### 3. Scientific Validation

Tasks:

- Write a short validation document describing what each reference dataset proves.
- Compare Python and R outputs where parity matters.
- Identify any tolerated differences versus true defects.
- Decide which outputs must match exactly and which only need statistical agreement.

Likely files:

- `tests/data/reference/`
- `scripts/compare_lsmeans.py`
- `scripts/lsmeans_obs.R`
- `docs/`

### 4. Performance

Tasks:

- Benchmark `RRPP`, `estimate_difference`, and any PLS/SNF bottlenecks.
- Profile realistic dataset sizes.
- Reduce unnecessary pandas-to-NumPy conversions and repeated work.
- Make expensive parallel paths optional and predictable.

Likely files:

- `src/motco/stats/sd.py`
- `src/motco/stats/pls.py`
- `src/motco/stats/snf.py`
- `scripts/time_estimate_difference.py`

### 5. Scientist-Facing Product Improvements

Tasks:

- Design one opinionated, documented workflow around the current CLI.
- Add stronger validation for matrix alignment, shapes, and required columns.
- Improve output naming and output schemas.
- Add one or two realistic examples that mirror actual usage.

Likely files:

- `src/motco/cli.py`
- `README.md`
- `docs/`

## First Two Weeks: Specific Task List

This is the handoff sequence I would give the new hire immediately.

Week 1:

1. Set up the repo from scratch and document every issue encountered.
2. Add a normal CI workflow for fast tests.
3. Reduce day-to-day regression test cost by introducing a lower default permutation count for routine runs.
4. Make RRPP tests robust when multiprocessing is unavailable.
5. Align `README.md` and `pyproject.toml` on supported Python versions and install/test instructions.

Week 2:

1. Audit the scientific surface area:
   - what is validated
   - what is assumed
   - what has no tests yet
2. Add missing tests for CLI, PLS, and SNF.
3. Produce a short benchmark report for the slowest paths.
4. Propose the first round of performance work with estimated impact and risk.

## Definition of Done for the First Milestone

The first milestone is complete when all of the following are true:

- Fast tests run reliably in CI and locally.
- Slow scientific tests are still available but no longer block normal iteration.
- The current correctness claims are documented.
- The repo setup is predictable for a new contributor.
- There is a prioritized optimization plan backed by measurements.

## What Not to Do First

The new hire should avoid these early traps:

- Do not redesign the algorithms before establishing a correctness baseline.
- Do not build a GUI or web app.
- Do not optimize blindly without benchmarks.
- Do not broaden scope to many new methods before the current methods are hardened.

## Suggested Weekly Reporting Format

Ask the hire to report weekly in this structure:

- What was validated this week
- What changed in tests/CI
- What changed in performance
- What user-facing friction was removed
- Risks, open questions, and decisions needed from you

## Short Management Summary

MOTCO already has a meaningful scientific core. The immediate need is not greenfield development; it is hardening, validating, and packaging what already exists. The best first hire is someone who can act as a scientific software engineer: rigorous on numerical correctness, practical on developer tooling, and capable of making the existing methods easier for other scientists to run.
