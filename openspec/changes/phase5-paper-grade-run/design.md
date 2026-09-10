## Context

See [proposal.md](proposal.md) — Why. Everything this change executes is already committed and frozen:
the config (`examples/trajectory_power_study/phase5_power_study.json`), the report contract it declares, the
Phase 4 gate roles, the acceptance targets, and the findings-report template. No source file changes here.

The operational facts that shape the approach:

- **Enumeration, verified locally 2026-09-10.** `enumerate_study` on the committed profile yields 19 cells
  (2 baseline Type I controls + 1 shared zero-effect anchor + 4 modes × 4 nonzero effects) × 500 replicates =
  **9,500 units**, with no cell rejected by `_require_surgery_headroom` — so under the default
  `surgery_censoring: "error"` no surgery is censored and no unit can fail for that reason.
- **`n_jobs` is part of the cell parameter signature.** RRPP seeds one RNG stream per worker, so the worker
  count changes the realized permutation draws. The contract sets `n_jobs_override: forbid`; the runner exits
  2 before enumerating if `--n-jobs` / `STUDY_N_JOBS` differs from `evaluation.n_jobs = 1`. All parallelism
  must therefore be across shards.
- **The cluster.** `ing` = `cluster.ing.uc.cl`, SLURM, partition `512x1024` (5 nodes, ≥64 cores, ≥256 GB),
  AMD EPYC 7662; `MaxArraySize = 6000`. `~/MOTCO` there is a working clone with `.venv` on Python 3.11.16 /
  motco 0.6.0, currently one commit behind `main` (missing the paper-grade profile). `/home1` is a shared
  filesystem at 100% use with ~115 GB free; the ladder run's 8,000 units cost 107 MB of JSONL, so this run's
  footprint (~150–250 MB) is not a concern but is not zero either.
- **Two directly comparable precedents.** The design-point pilot (6,500 units) and the latent-rank ladder
  (8,000 units) both ran as `--array=0-99 --cpus-per-task=1 --mem=2G --time=6:00:00` on `512x1024` with BLAS
  pinned, both finished in ~20 min wall with 0 failures, and their `PROVENANCE.txt` files are the template
  for this one.

## Goals / Non-Goals

**Goals:**

- Produce the 9,500-unit paper-grade record set exactly as the committed config specifies, reproducibly from
  that config plus a named code revision.
- Make the run's completeness auditable: every enumerated (cell, replicate) unit present once, matching
  parameter signature, zero failed records, zero censored surgeries — asserted from the merged JSONL, not
  assumed from SLURM exit codes.
- Report the predeclared gate verdict as it comes out, with orientation power read beside the recorded
  eigengap and `angle` null-width dispersion.

**Non-Goals:**

- Tuning anything that would change the measurement (config, rank rule, design point, permutation count,
  seeds, gate roles, acceptance targets).
- Implementing a remedy if the gate fails — this change records and explains; the revision is the Phase 5
  exit review's own change.
- Automating `PROVENANCE.txt` or the findings report.

## Decisions

**D1 — Run on `ing` as a 100-shard array with one CPU per task, matching both pilots.**
`sbatch -p 512x1024 --cpus-per-task=1 --mem=2G --time=6:00:00 --array=0-99 scripts/motco_study_array.sbatch`
with `--export=ALL,STUDY_CONFIG=…,STUDY_OUT=…,N_SHARDS=100` and BLAS pinned at submit. The sbatch template's
own `#SBATCH` defaults (8 CPUs, 16 GB, 24 h) are deliberately overridden on the command line: 95 units/shard
at ~57 s each is ~1.5 h, so 6 h is ample margin, and one CPU per task is the *only* setting consistent with
`n_jobs = 1`. `STUDY_N_JOBS` stays unset — with this contract, setting it fails all 100 tasks identically.
*Alternative considered:* fewer, fatter shards. Rejected — it buys nothing (units are serial either way) and
lengthens the tail. *Alternative considered:* running locally in ~10 shards. Rejected — ~12 h wall on this
16-core workstation against ~2 h on the cluster, and 2 GB × 16 exceeds free RAM here.

**D2 — Run from a clean tree at a named `main` revision, not a working-tree sync.**
The cluster clone is fast-forwarded to `main` (currently `b8f7daa`) and `uv sync` is run before submission;
`PROVENANCE.txt` records that revision and the fact that the tree is clean. This change contributes no code,
so unlike the ladder run (which rsynced an unmerged working tree and had to record a tracked-file diff hash)
there is nothing to sync — and a run pinned to a merged commit is reproducible by `git checkout` alone.
A pre-submit `git status --porcelain` check and a `config_sha256` comparison against the workstation copy
guard against silent drift.

**D3 — Run directory `results/phase5-<UTC launch date>/`, matching the established naming.**
The findings report and `PROVENANCE.txt` take the same date. If the run spans midnight UTC, the launch date
wins and `PROVENANCE.txt` records the wall interval.

**D4 — Completeness is verified from the merged records, and failures are re-driven by resubmitting only the
affected array ids.** `--error-policy record` keeps a failed unit from aborting its shard. After merge, the
check is: 9,500 units present exactly once, every parameter signature matching enumeration, `failed`
records = 0, and `censored` = 0 in the realized-surgery summary. Because shard JSONL is signature-guarded and
resumable, a failed or timed-out task is fixed by `sbatch --array=<ids>` with identical exports — completed
replicates are skipped. *Alternative considered:* `--error-policy raise`. Rejected — one bad unit would cost
a whole shard and the study is meant to record failures, not hide them.

**D5 — Merge on the cluster, report on the workstation, from the merged JSONL.**
Same split as the ladder run: `motco_study.py merge` runs where the shards are, the merged JSONL is rsynced
back (~150–250 MB), and `motco_study.py report` runs here against the committed config. Reporting is cheap
and single-threaded, and running it on the workstation keeps the figures and the committed `report/` produced
by the same tree that commits them. The merged and shard JSONL stay gitignored on both sides.

**D6 — Cross-check against the pilots before writing anything.** The paper-grade run's orientation `angle`
power at e = 1.00 must agree with the two independent matched-seed families that measured the same design
point (0.88 design-point pilot, 0.85 ladder CV column) within Monte Carlo error; magnitude `delta` ≈ 1.00 and
shape `shape` ≈ 0.95–1.00 likewise, and the anchor at nominal level on all three statistics. A material
discrepancy is a signal that something about the execution differs — it is investigated *before* the
findings report is written, not explained inside it.

**D7 — The gate verdict is reported, not negotiated.** `report/phase4_gate.json` (name kept: it names the
mechanism, not the phase) is the predeclared pass/fail. Whatever it returns goes into the findings report with
each deviation explained as a method or claim revision. The 0.80 orientation floor in particular is close to
the pilot estimate; if the paper-grade measurement lands under it, this change states that plainly, carries
the n-conditional framing and the eigengap/null-width covariates as the explanation, and hands the revision
decision to the exit review rather than resolving it here. Raising the replicate count in response is
explicitly forbidden by the Phase 4 exit gate.

**D8 — `PROVENANCE.txt` is hand-written to the template's field list**, following the two prior files
verbatim in structure: date, code revision and tree state, config path and sha256, host/partition/CPU model,
the full launch line, BLAS settings, unit and cell counts, package versions, job id, outcome (task exit
codes, unit/signature/failure/censoring counts), wall interval, per-unit timings, and the merge/report split.

## Risks / Trade-offs

- **The gate's orientation floor may not clear at paper-grade precision** (pilot point estimates 0.85/0.88
  against a 0.80 floor; pilot SE ≈ 0.032, paper-grade SE ≈ 0.014) → D7: the outcome is predeclared to be
  reportable either way, and the covariates that explain it are already recorded per replicate. No config
  edit, no replicate-count increase.
- **Cluster clone drift or a stale venv silently changes the measurement** → D2: fast-forward, `uv sync`,
  clean-tree check, and a `config_sha256` match against the workstation copy before submission; versions
  recorded in `PROVENANCE.txt`.
- **`/home1` is at 100% use with ~115 GB free** → the run needs ~250 MB; checked before submit, and the
  merged JSONL is removed from the cluster after it is rsynced back if space is tight.
- **A shard hits the wall clock or a node fails** → D4: resumable, signature-guarded shards; resubmit only
  the affected array ids with identical exports.
- **Accidentally passing `STUDY_N_JOBS`** (habit from other jobs) → all 100 tasks exit 2 immediately and
  visibly; the failure is loud by design, and the submit line in `PROVENANCE.txt` records that it was unset.
- **Attribution cost at n = 1200 was only rehearsed at one replicate** (median 42.9 s vs 43.0 s for
  non-attribution units) → if the array's orientation shards run materially longer than the others, the
  budget absorbs it (6 h limit against a ~1.5 h expectation); `bootstrap_replicates` must not be lowered
  after the run has started, per the config's own note.
- **Reporting on the workstation while the records were produced on EPYC nodes** → reporting is deterministic
  post-processing of recorded statistics; no re-fitting occurs. The version block records both environments.
