## Context

See `proposal.md` for motivation. The current trajectory shape implementation is a close Python translation of the legacy R `pgpa`/`pPsup` procedure: it centers and scales each group trajectory, iteratively aligns each group to the mean of the other groups, then reports Euclidean distances between flattened aligned configurations.

That leave-one-out GPA procedure is not yet proven to satisfy the production contract for strict morphometric shape. The inverse-PLS diagnostic exposed the key failure candidate: a proper rigid rotation in the latent score plane preserved all pairwise stage distances but produced a substantial nonzero `shape` value before feature reconstruction.

## Goals / Non-Goals

**Goals:**

- Establish deterministic evidence for how current shape behaves under translation, uniform scale, proper rotation, reflection, and true configuration changes.
- Define a production implementation path where `shape` is a strict pairwise morphometric residual after translation, proper rotation, and uniform scale are removed.
- Preserve legacy GPA behavior only as a diagnostic/reference comparator unless it satisfies the strict contract.
- Add permanent tests for the invariance contract and for expected positive shape under genuine bends.

**Non-Goals:**

- Redesign `delta` or `angle`.
- Change trajectory simulation mode semantics beyond whatever is necessary to measure shape invariance.
- Add graph-native SNF shape statistics.
- Claim biological causality or unique feature-space inverses from shape results.

## Decisions

### 1. Treat strict pairwise Procrustes distance as the production target

The production `shape(A, B)` target is the residual norm after centering both trajectories, scaling them to unit centroid size, and optimally aligning one to the other by a proper orthogonal rotation. If two trajectories differ only by translation, positive uniform scaling, and proper rotation, the distance must be numerical zero.

Alternative considered: keep the legacy leave-one-out GPA distance because it matches the original R procedure. Rejected for production semantics if it fails rigid-rotation invariance; matching a legacy implementation is weaker than matching the scientific definition of shape.

### 2. Keep reflections distinct by default

The default alignment should use proper rotations only. A mirror reflection is therefore a different configuration unless a future explicit option allows reflection removal.

Rationale: the existing implementation already contains determinant-sign handling that attempts to control reflection, and preserving handedness is the conservative default for trajectory configurations in a shared feature/latent coordinate system. The audit should still include reflected cases so this policy is observable rather than implicit.

Alternative considered: remove reflections by default. This is defensible in some morphometric workflows, but it would silently collapse mirror-image trajectories and should be a deliberate future option if needed.

### 3. Decompose the audit before fixing behavior

The audit should report the output of:

- centered-only configurations;
- centered and centroid-size-scaled configurations;
- direct pairwise Procrustes distance;
- current leave-one-out GPA distance;
- legacy R `pgpa`/`pPsup` output when the R environment is available.

This separates implementation defects from estimator semantics. The likely diagnostic signature is:

```text
direct pairwise Procrustes:  shape(A, rotate(A)) ~= 0
current leave-one-out GPA:   shape(A, rotate(A)) > 0
```

If that signature appears, the conceptual mismatch is the GPA comparison scheme, not the centering/scaling definition.

### 4. Use minimal deterministic trajectories as the first gate

The first tests should use small two-dimensional trajectories with at least three stages. Good fixtures include a non-collinear triangle and a four-stage bent path so rotation/reflection degeneracy is avoided.

Suggested transformations:

```text
A                       identical
A + t                   translation only
c * A                   uniform scale only, c > 0
R * A                   proper rotation only
reflect(A)              mirror reflection
A with middle point moved  genuine bend
```

Collinear three-stage paths are useful as secondary degeneracy checks, but not as the main invariant fixture because rotations around a line and rank-deficient Procrustes fits can mask errors.

### 5. Keep the public output shape unchanged

`estimate_difference` should continue returning symmetric `deltas`, `angles`, and `shapes` matrices. This change should alter the numerical semantics of `shapes`, not the public result structure.

## Risks / Trade-offs

- [Historical result drift] Existing shape values and p-values may change after enforcing strict morphometric semantics. → Document this as an intentional correction and keep the audit report with before/after examples.
- [R reference mismatch] The legacy R procedure may reproduce the current non-invariant behavior. → Treat R as a legacy comparator, not as the production oracle, when it conflicts with the strict morphometric contract.
- [Reflection ambiguity] Users may expect reflected shapes to align away. → Document the default proper-rotation policy and leave an explicit future option for reflection-permissive shape if needed.
- [Permutation null changes] Corrected shape distances may change RRPP rejection behavior. → Rerun focused fast tests first, then repeat the medium pilot before any paper-grade grid.
- [Degenerate trajectories] Near-zero centroid size or collinear configurations can make rotation unstable. → Add explicit handling and tests for degenerate inputs, with clear tolerances or unavailable shape where necessary.

## Migration Plan

1. Add audit tests and diagnostics that expose current behavior without changing production semantics.
2. Update the shape estimator to satisfy the strict invariance contract if the audit confirms the current behavior fails it.
3. Update documentation and findings to mark older shape operating-characteristic results as superseded where they depended on the legacy behavior.
4. Run focused trajectory tests, permutation tests that cover shape, and the fast non-slow test suite before moving to the medium pilot.
