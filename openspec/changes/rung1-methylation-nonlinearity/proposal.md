## Why

Rung 0 (`archive`/`rung0-gaussian-existence-proof/findings.md`) established a **clean linear floor**: under a pure-linear PCA projector, the production estimators isolate magnitude (`delta`) and orientation (`angle`) with no cross-talk. The cross-talk the earlier specificity study found (magnitude leaking into orientation/shape) therefore enters at a *higher* rung. The Rung-0 findings name the prime suspect explicitly: the methylation `rev.logit` nonlinearity, which makes a similarity transform clean in the *generative* frame (M-value space) but not in the *measurement* frame (β space). Rung 1 adds **exactly that one factor** and nothing else, to test whether `rev.logit` alone produces cross-talk.

## What Changes

- Add a Rung-1 **methylation test bed** that reuses the Rung-0 generation machinery unchanged but reinterprets the feature matrix as CpG **M-values**: inject the same known 2-stage geometry (`none`/`magnitude`/`orientation`) in M-value space, then pass the drawn samples through InterSIM's `rev_logit` (reused from `generator.py`, not reimplemented) to obtain β values, project with inline PCA, and measure `delta`/`angle` via the existing `stats/trajectory.py` estimators. **Methylation-only:** no expression/protein layer is generated or measured — InterSIM's downstream coupling transmits the differential *support* but not the *magnitude* (gene shift size is set by an independent `delta_expr`), so a continuous cascade would conflate factors; Rung 1 keeps the nonlinearity as the single new variable.
- **Operating point is the independent variable.** `rev.logit` is locally linear at M ≈ 0 (slope ≈ 0.25) and saturates on the tails. The test bed adds a baseline M-value offset `m_baseline` and **sweeps it from the sigmoid center out to the tails**, reporting `delta`/`angle` recovery and cross-talk as a function of operating point. A secondary sweep over step scale (how much of the sigmoid a single step spans) is included.
- **Stage = 2 only**, so Procrustes `shape` is degenerate and excluded — the scope stays magnitude and orientation, identical to Rung 0.
- **Distortion characterization:** quantify (a) absolute distortion of measured β-space `delta`/`angle` against the intended M-space geometry, and (b) cross-talk (magnitude→`angle`, orientation→`delta`) against the Rung-0 null floor, establishing whether `rev.logit` alone reproduces the specificity-study cross-talk.
- **Inverse design (secondary, linearized):** because the nonlinearity breaks the exact PCA round-trip, extend the Rung-0 inverse design to a first-order (Jacobian `diag(β(1−β))`) preimage at the operating point, to keep the feature-intuition thread continuous; mark it explicitly first-order.
- Add a committed **findings writeup** with the operating-point distortion curves and the gate decision for Rung 2.
- **Out of scope (deferred to later proposals):** the PCA → PLS projector swap, the cross-omic cascade and full InterSIM generator path, the `evaluation.py` integration method, per-feature standardization, real per-CpG `mean_M` baselines (the sweep uses synthetic operating points), and any purity metric or power study.

## Capabilities

### New Capabilities
- `methylation-geometry-recovery`: A Rung-1 test bed that layers the methylation `rev.logit` nonlinearity onto the Rung-0 geometry injection and characterizes how trajectory `delta`/`angle` recovery and cross-talk depend on the sigmoid operating point under a linear (PCA) projector at two stages.

### Modified Capabilities
<!-- None: this is a self-contained new test bed; it does not change requirements of the evaluation harness, generator, or trajectory estimators. -->

## Impact

- **New code:** a Rung-1 test-bed module under `src/motco/simulations/` (e.g. `methylation_recovery.py`) plus its unit tests under `tests/`, and a driver script under `scripts/`.
- **New docs:** a findings writeup in the change folder (matching the Rung-0 pattern).
- **Reused, unchanged:** `stats/trajectory.py` estimators, `generator.rev_logit`, and the Rung-0 geometry-injection / inverse-design helpers in `simulations/linear_recovery.py`.
- **Dependencies:** `numpy`, `scikit-learn` (PCA) — both already present.
- **No changes** to `evaluation.py`, `semisynthetic.py`, `generator.py`, `reference.py`, or any existing spec.
