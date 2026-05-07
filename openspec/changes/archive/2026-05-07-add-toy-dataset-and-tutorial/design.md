## Context

The existing notebook (`examples/motco_example.ipynb`) uses a pre-processed 2D dataset (`evo_649_sm_example1.csv`) that skips omics integration entirely. There is no CLI entry point for generating example data. New users have no concrete path from "I have multi-omics data" to "I have trajectory analysis results."

The simulation infrastructure (`intersim.py`, `semisynthetic.py`, `evaluation.py`) is complete and tested, but only accessible via the Python API. `build_simulation_trajectory_design` in `evaluation.py` already produces all design objects needed for `motco de`.

## Goals / Non-Goals

**Goals:**
- A single `motco simulate` command produces all files needed for a complete pipeline run.
- Pre-generated toy data in the repo so users without R can run the tutorial.
- A rewritten notebook that demonstrates the full pipeline end-to-end.
- A README quick-start section.

**Non-Goals:**
- Supporting simulation backends other than InterSIM (pure-Python fallback is not implemented).
- Exposing all InterSIM parameters via CLI (seed, n-samples, trajectory-mode, effect-size are sufficient for a tutorial; researchers needing fine control use the Python API).
- Changing the simulation or evaluation internals.

## Decisions

### D1 — Pre-generate and commit data; don't download on demand

The toy dataset is committed directly to `examples/data/toy/` rather than fetched from a URL or generated at test time. Rationale: zero-friction tutorial (no network, no R required), deterministic content (no generation code path in CI), and the files are small enough for git (~10 CSVs, each a few hundred KB at most for 90 samples).

Alternative considered: a `make examples` or `download-data` script. Rejected because it adds a step and creates a network dependency.

### D2 — Orientation mode with effect size 2.0 for the toy dataset

`trajectory_mode="orientation"` injects a group effect where group B's trajectory points in a different direction from group A's, with equal magnitude. This is the most distinctive MOTCO use case — undetectable by simple group-mean or magnitude tests. Effect size 2.0 produces a clearly visible difference without making the data unrealistically clean.

Alternative: `magnitude` mode (easier to visualize intuitively). Rejected because it doesn't showcase what differentiates MOTCO from standard approaches.

### D3 — `motco simulate` outputs design files directly

The command writes `model_full.csv`, `model_reduced.csv`, `ls_means.csv`, and `contrast.json` alongside the omics matrices. This keeps the tutorial self-contained: `motco de` can be called immediately after `motco simulate` + one integration step, with no manual design construction. The alternative (separate `motco design` subcommand) adds a step and requires the user to understand the design matrix API before they've seen results.

### D4 — Notebook shows both PLS and SNF paths

The notebook demonstrates both integration paths (PLS-DA supervised by stage, SNF unsupervised) and uses the same design files for `motco de` on both. This reinforces that the two paths are complementary and converge at the same downstream analysis.

### D5 — InterSIM unavailability produces a clear SystemExit

`cmd_simulate` calls `check_intersim_available()` before any computation and raises `SystemExit` with a message that names the dependency and where to install it. This matches the error-handling pattern in the rest of `cli.py`.

## Risks / Trade-offs

- **[Risk] Pre-generated data goes stale if simulation code changes** — If `semisynthetic.py` or `intersim.py` behaviour changes, the committed data may no longer match what `motco simulate` would produce. Mitigation: `truth.json` records the exact params; a comment in the notebook shows the regeneration command.
- **[Risk] 90-sample toy dataset is small enough that RRPP p-values vary with seed** — Mitigation: use a fixed seed in the notebook; note that more permutations are needed for stable p-values in real analyses.
- **[Trade-off] Rewriting the notebook breaks any existing links to specific cells** — The existing notebook is already minimal and serves a different data format. A clean rewrite is better than a patch that leaves two conflicting examples.
- **[Risk] Large committed CSVs slow git operations** — Mitigation: 90 samples × ~500 features per layer ≈ 500 KB total; acceptable for a public tutorial asset.
