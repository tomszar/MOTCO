## Context

The trajectory power study derives one deterministic seed per
`(cell, replicate)` unit and fans it out to InterSIM (R), the
semi-synthetic generator (Python / numpy), and RRPP (Python / numpy).
The R leg routes through a `Rscript` subprocess that calls
`as.integer(seed); set.seed(seed)`. R's `as.integer()` silently
yields `NA` for any value outside `[-2³¹, 2³¹ − 1]`, and
`set.seed(NA)` aborts the subprocess with
"supplied seed is not a valid integer".

`derive_replicate_seed` (grid.py:280-288) produces seeds via:

```python
return int(_stable_digest(payload, length=8), 16)
```

`_stable_digest(..., length=8)` returns the first 8 hex characters of
a SHA-256 digest. `int(..., 16)` parses them as **unsigned 32-bit**,
so the result spans `[0, 2³² − 1]`. The high half of that range
(≈ half of all draws) overflows R. The trajectory-power-study smoke
captured this exactly: 64 of 144 replicates failed, every failure was
the same R error, and every failed seed value (e.g. 2 797 983 684)
sits above `2³¹ − 1`.

`run_shard` in `study/sharding.py` resumes by `(cell_id,
replicate_index)` keys and trusts `parameter_signature` to detect
parameter changes. Today's signature payload (grid.py:294-302) does
not capture seed-derivation logic, so a fix to the derivation
function silently leaves stale shards' failed records in place: the
resume guard sees the unit as "already attempted" and skips
re-execution.

## Goals / Non-Goals

**Goals:**

- Eliminate the R-side seed overflow at the Python boundary so that
  ~half of all replicates stop failing for this reason.
- Preserve full determinism of the seed chain: a given
  `(base_seed, cell_id, replicate_index)` must map to a fixed,
  reproducible seed.
- Surface any future out-of-range seed as a clear Python error
  before it crosses into R.
- Invalidate stale shard JSONLs from the buggy derivation
  automatically, so resume does the right thing without operator
  intervention.

**Non-Goals:**

- Re-architecting seed derivation (no move to
  `numpy.random.SeedSequence`, no hierarchical RNG split). The
  current SHA-based scheme is fine; we only need to constrain its
  range.
- Fixing the unrelated `magnitude`-mode design issue (per-stage
  shifts use a fixed direction vector, so magnitude perturbations
  also rotate trajectory orientation). Separate proposal.
- Changing the InterSIM R helper. R-side validation is redundant
  once Python guarantees in-range values.

## Decisions

### D1. Mask the derived seed to 31 bits at the source.

In `derive_replicate_seed`, change

```python
return int(_stable_digest(payload, length=8), 16)
```

to

```python
return int(_stable_digest(payload, length=8), 16) & 0x7FFFFFFF
```

so the returned value is always in `[0, 2³¹ − 1]` — R's signed-32-bit
non-negative range. Update the docstring to "31-bit unsigned" and
note the R compatibility constraint.

**Alternatives considered.**

- *Slice 7 hex chars instead of 8.* Yields `[0, 2²⁸ − 1]`. Also safe,
  but leaves 3 bits of unused entropy and is less self-documenting.
- *Mask at the R boundary in `_build_rscript_command`.* Splits the
  invariant across two files; the seed stored in
  `SimulationReplicateResult` would differ from what R actually saw,
  complicating post-hoc analysis.
- *`numpy.random.SeedSequence`-based derivation.* Larger refactor, no
  benefit for the current use case, and would change every seed
  value in flight.

Masking at the source keeps `replicate_seed == intersim_seed ==
generator_seed == evaluation_seed` (the existing invariant in
`run_simulation_replicate`), preserves 31 bits of entropy
(≈ 2.15 × 10⁹ distinct seeds — comfortably more than any plausible
`n_replicates × n_cells`), and writes the same value into the result
record that R observed.

### D2. Add a defensive bounds check at the InterSIM boundary.

In `_build_rscript_command`, validate `params.seed` ∈
`[-2³¹, 2³¹ − 1]` and raise `InterSIMError` with a clear message
otherwise. This is belt-and-suspenders: D1 guarantees the value is
in range when it comes from `derive_replicate_seed`, but any external
caller (e.g. a notebook driving `run_intersim` with an
ad-hoc seed) gets a Python exception rather than a silent R coercion
to `NA`.

**Alternatives considered.**

- *Validate inside the R helper script.* Symmetric defense, but
  redundant given Python is already strict and harder to test
  cleanly.
- *Clamp silently.* Hides the caller's mistake; degrades
  reproducibility because the recorded seed no longer matches what R
  used.

### D3. Bump `parameter_signature` with a derivation-version tag.

Add `"seed_derivation_version": 2` to the payload of
`parameter_signature`. The signature changes for every cell, so
`run_shard`'s existing "different signature, refuse to resume" guard
fires automatically on stale shards from the buggy derivation. Fresh
runs are unaffected by the constant.

**Alternatives considered.**

- *Encode the derivation version into `replicate_seed` itself.*
  Couples version metadata to the seed value, making future
  derivation changes harder.
- *Document "delete shards before re-running".* Footgun: easy to
  forget; a paper-grade run that silently skips failed replicates
  would publish biased numbers.
- *Validate stored seed against re-derived seed in
  `_completed_index`.* More code, same effect, slower resume.

The signature constant carries no semantic load; it's a salt that
forces invalidation. The label `2` reflects the second logical
version of the derivation function — the original was the buggy 32-
bit unsigned mapping.

## Risks / Trade-offs

- **[Risk] Existing in-flight studies will be re-executed in full
  after this lands** → Acceptable. Without the signature bump the
  only safe alternative is operator-led shard deletion, which is
  error-prone. Communicate via the trajectory-power-study README
  changelog.

- **[Risk] 31 bits of entropy might feel "small" compared to 32**
  → Negligible. Birthday-collision probability for `n_seeds ≪ 2¹⁵·⁵`
  (≈ 46 k) is well under 10⁻⁹; we expect at most a few thousand
  replicates per cell, with millions in total.

- **[Trade-off] Boundary check duplicates information already
  guaranteed at the source** → Worth it: the bridge is a public
  Python entry point usable outside the study harness, and the
  failure mode without the check (silent `NA` in R) is exceptionally
  hard to diagnose.

- **[Risk] A future contributor changes `_stable_digest`'s length
  parameter and forgets the mask** → Mitigated by unit tests that
  exercise the masked range directly, plus the boundary check in D2
  catches anything that slips through to R.

## Migration Plan

1. Land the source change (D1) and boundary check (D2) together with
   their tests.
2. Land the signature bump (D3) in the same change so resume
   automatically invalidates stale shards.
3. Delete or re-run the trajectory-power-study smoke
   (`/tmp/motco-smoke`) — the bumped signature will trigger
   full re-execution.
4. Update the trajectory-power-study README's troubleshooting
   section to mention the signature bump as a one-time
   invalidation event (optional; not required for correctness).

No rollback complexity: reverting the change reverts the signature
constant, and any shards produced under the fixed derivation become
stale under the old code — exactly the symmetric situation we're
solving for.

## Open Questions

- None. The fix surface is fully scoped at the Python layer; the R
  helper is untouched.
