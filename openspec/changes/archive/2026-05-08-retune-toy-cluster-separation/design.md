## Context

The canonical toy dataset maps InterSIM clusters to ordered stages. This makes the README PLS-DA example easy to understand, but InterSIM's default cluster deltas currently make `stage` almost perfectly recoverable from the concatenated omics matrix. After the nested-CV aggregation fix, a printed `AUROC = 1.0` with `AUROC_std = 0.0` now means every outer fold is saturated, not that reporting is cherry-picking.

The existing `--cluster-mean-shift` CLI flag is the intended control for this problem. A prior acceptance sweep found that the InterSIM default and values at or above about `0.25` saturate the toy, while `--cluster-mean-shift 0.10` produces non-trivial stage classification (`AUROC` around `0.79`, non-zero fold dispersion) on the same seed/sample/features.

## Goals / Non-Goals

**Goals:**
- Regenerate the bundled toy data with moderate cluster separation using `--cluster-mean-shift 0.10`.
- Preserve the current tutorial shape: users still build a PLS-DA latent space from omics, then run `motco de` with the pre-generated design files.
- Make the README/example PLS-DA table non-saturated enough to demonstrate finite-sample uncertainty.
- Keep the downstream trajectory signal detectable and not pinned at a permutation floor.
- Record the new regeneration command and expectations in OpenSpec, README, and the notebook.

**Non-Goals:**
- Do not change the conceptual clusters-as-stages assumption in this change.
- Do not add a new simulation model, dependency, or CLI flag.
- Do not make the toy a benchmark dataset; it remains a compact usage example.
- Do not optimize `plsda_doubleCV` performance here, except by using example-friendly parameters in docs if needed.

## Decisions

### D1. Use `--cluster-mean-shift 0.10` as the canonical retune

Use the existing fanout flag rather than per-omic deltas so the toy's stage separability is controlled uniformly across methylation, expression, and proteomics.

Alternatives considered:
- `0.25`: nearly saturated in the acceptance sweep (`AUROC` about `0.999`), so it does not solve the README confusion.
- Values below `0.10`: may make stage classification too weak and could undermine downstream trajectory examples.
- Per-omic deltas: more flexible, but unnecessary for a usage example and harder to explain in README.

### D2. Keep `effect-size 1.0` and `prop-affected-features 0.1`

This change targets InterSIM cluster/stage separability, not the semisynthetic group trajectory effect. The existing group-effect settings should remain stable unless verification shows the downstream trajectory test no longer has a useful signal.

### D3. Validate both PLS-DA classification and downstream trajectory analysis

The regenerated toy must pass two checks:
- `motco plsr` on `y = stage` should no longer report all-fold saturation.
- `motco de` on the resulting PLS-DA latent space should still show a detectable orientation difference in the intended tutorial range.

### D4. Keep committed toy files sufficient for no-R usage

The repository should continue to include the generated omics files, metadata, design matrices, contrast, and truth JSON. Users should not need R or InterSIM to run the README pipeline after cloning.

## Risks / Trade-offs

- Lower cluster separation may make downstream trajectory statistics less stable -> verify `motco de` with at least 199 permutations before committing regenerated data.
- Reducing stage separability may vary across CV parameters -> specify acceptance bands using moderate CV settings rather than one fragile exact AUROC.
- The README can still be slow if it uses rigorous CV defaults -> keep example parameters modest or explicitly label larger settings as analysis-grade.
- Regenerated CSVs produce large diffs -> keep the change focused and avoid unrelated notebook output churn.
