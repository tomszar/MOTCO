## Rung 1 — Methylation rev.logit Nonlinearity: Findings

### What this experiment does

Rung 0 proved the linear floor is clean: under a pure-linear PCA projector the production estimators isolate magnitude (`delta`) and orientation (`angle`) with no cross-talk. Rung 1 adds **exactly one** new factor — InterSIM's methylation nonlinearity — and asks whether *it* is the source of the cross-talk the specificity study found.

We reuse the Rung-0 generation **unchanged**, but reinterpret the feature matrix as CpG **M-values**. The known 2-stage geometry (`none`/`magnitude`/`orientation`) is injected in M-value space exactly as in Rung 0; the drawn samples are then passed through InterSIM's `rev_logit` (`β = 1/(1+e^{−M})`, reused from `generator.py`) to obtain β values, which are projected with inline PCA and measured with `estimate_difference`. **Methylation-only:** no expression/protein layer is generated — InterSIM's downstream coupling transmits the differential *support* but not the *magnitude*, so a cascade would conflate factors. `rev_logit` is the single new variable versus Rung 0.

`rev_logit` is locally linear at M ≈ 0 (slope `β(1−β) = 0.25`) and saturates on the tails (slope → 0). Distortion therefore depends on **where on the sigmoid the trajectory sits** and **how much of the sigmoid a single step spans**. These are the two independent variables, swept separately.

**Two integration representations.** The data carried through the pipeline is always β (what InterSIM passes to gene expression), but the standard analysis practice is to transform methylation to **M-values** before integration (homoscedastic, ~Gaussian). Axes 1–2 below integrate in **β-space** to expose the failure mode; the **Resolution** section then contrasts that against **M-value integration** — the `integration_space` knob, defaulting to `"mvalue"` as the recommended pipeline choice. Because `logit` is the exact inverse of the generative `rev_logit`, M-integration recovers the clean linear geometry (and is why the field already prefers M-values).

---

### Reproduction parameters

| Parameter              | Value  |
|------------------------|--------|
| `n_features` (CpGs)    | 50     |
| `n_samples_per_cell`   | 40     |
| `signal_scale` (‖a‖, M)| 2.0    |
| `noise_scale` (σ, M)   | 0.3    |
| `scale_c` (magnitude)  | 2.0    |
| `angle_theta` (orient.)| 45°    |
| `n_components` (PCA)   | 2      |
| Seeds averaged over    | 0–9    |

```bash
.venv/bin/python scripts/methylation_recovery_probe.py
```

Operating-point figure: `build/rung1_operating_point.png`. **Axes 1–2 below integrate in β-space** (`integration_space="beta"`) to expose the failure mode; the Resolution section contrasts both arms.

**Why `k = 2`.** The 2-group × 2-stage signal lives in ≤ 2 dimensions, and Rung 0 showed the angle null floor *grows with each retained noise PC*. Rung 1 probes a small (≈ 5–10°) cross-talk signal, so it retains only the signal subspace — keeping `rev_logit`, not the k-noise floor already characterized in Rung 0, as the variable under test.

---

### Axis 1 — Operating point (baseline M-value), step scale fixed

Mean ± SD across 10 seeds. The baseline M is applied uniformly to every CpG; `slope = β(1−β)` is the local sigmoid gain.

| m_baseline | β_base | slope | manip | δ mean ± SD | θ mean ± SD |
|-----------:|-------:|------:|-------|-------------|-------------|
| 0.0 | 0.500 | 0.250 | none        | 0.012 ± 0.009 | 5.3° ± 3.9° |
| 0.0 | 0.500 | 0.250 | magnitude   | **0.447 ± 0.016** | 4.4° ± 3.5° |
| 0.0 | 0.500 | 0.250 | orientation | 0.019 ± 0.015 | **47.3° ± 2.8°** |
| 2.0 | 0.881 | 0.105 | none        | 0.006 ± 0.004 | 4.7° ± 3.8° |
| 2.0 | 0.881 | 0.105 | magnitude   | **0.268 ± 0.026** | 3.5° ± 2.6° |
| 2.0 | 0.881 | 0.105 | orientation | 0.009 ± 0.006 | **47.3° ± 3.8°** |
| 4.0 | 0.982 | 0.018 | none        | 0.001 ± 0.001 | 4.2° ± 4.0° |
| 4.0 | 0.982 | 0.018 | magnitude   | **0.058 ± 0.008** | 3.8° ± 3.8° |
| 4.0 | 0.982 | 0.018 | orientation | 0.002 ± 0.001 | **47.2° ± 4.3°** |

(Full grid `m_baseline ∈ {0,1,2,3,4}` in the driver output.)

**What the operating-point sweep shows:**

- **Magnitude `delta` compresses monotonically** as the baseline saturates: 0.447 → 0.401 → 0.268 → 0.136 → 0.058. The compression tracks the sigmoid slope `β(1−β)` (0.25 → 0.018): deep in the tail, a magnitude difference of the same M-space size is squeezed into a vanishing β-space difference. This is a **loss of magnitude sensitivity**, not a bias toward another statistic.
- **Orientation `angle` is essentially invariant** — 47.3° → 47.3° → 47.3° → 47.2° → 47.2° across the entire sweep. A uniform operating point gives every CpG the *same* Jacobian, so the β-space step is (to first order) a *scalar* multiple of the M-space step: length is rescaled, **direction is preserved**.
- **No cross-talk.** Magnitude's angle stays at the `none` floor (3–4°) and orientation's delta stays at its floor at every operating point.

**Verdict for Axis 1: a uniform `rev_logit` operating point does NOT reproduce the cross-talk. It only compresses magnitude.**

**Population confirmation — `delta ≈ β(1−β) · delta_M`.** The measured δ is not a noise or estimator artifact. Computing the geometry on the *noiseless population* (apply `rev_logit` straight to the cell means — no sampling, no PCA, no estimator) gives δ = 0.446 at center versus the measured 0.447: the pipeline faithfully reports the generative→β-frame geometry. And the compression follows the first-order prediction that `rev_logit` scales an M-space step by the local slope `β(1−β)`, so β-space δ ≈ `β(1−β)` × (M-space δ), where the M-space δ here is `signal_scale·(c−1) = 2.0`:

| m_baseline | slope β(1−β) | population δ | δ / slope |
|-----------:|-------------:|-------------:|----------:|
| 0 | 0.250 | 0.446 | 1.78 |
| 1 | 0.197 | 0.359 | 1.83 |
| 2 | 0.105 | 0.223 | 2.12 |
| 3 | 0.045 | 0.109 | 2.41 |
| 4 | 0.018 | 0.045 | 2.58 |

δ / slope stays near the M-space δ of 2.0, drifting up only as second-order curvature accumulates across the step's extent deep in the tail. **Compression is governed by the local sigmoid gain at the operating point.**

---

### Axis 2 — Step scale (effect size), operating point fixed at center (M = 0)

The `none` floor falls as `signal_scale` rises (higher SNR → less angle estimation noise), so cross-talk is read against a *shrinking* floor.

| signal_scale | manip | δ mean ± SD | θ mean ± SD | θ vs none-floor |
|-------------:|-------|-------------|-------------|----------------|
| 1.0 | none        | 0.012 ± 0.010 | 9.9° ± 7.4° | — |
| 1.0 | magnitude   | 0.250 ± 0.012 | 8.3° ± 6.4° | at floor |
| 1.0 | orientation | 0.022 ± 0.017 | 44.7° ± 8.4° | — |
| 4.0 | none        | 0.012 ± 0.008 | 2.8° ± 2.1° | — |
| 4.0 | magnitude   | 0.679 ± 0.036 | 2.9° ± 2.3° | at floor |
| 4.0 | orientation | 0.024 ± 0.018 | 46.1° ± 1.5° | — |
| 6.0 | none        | 0.012 ± 0.007 | 2.0° ± 1.5° | — |
| 6.0 | magnitude   | 0.748 ± 0.044 | **6.9° ± 2.7°** | **3.5× floor** |
| 6.0 | orientation | 0.035 ± 0.027 | 45.9° ± 1.6° | — |
| 8.0 | none        | 0.013 ± 0.007 | 1.5° ± 1.1° | — |
| 8.0 | magnitude   | 0.739 ± 0.036 | **9.4° ± 1.5°** | **6.3× floor** |
| 8.0 | orientation | 0.047 ± 0.036 | 46.1° ± 2.2° | — |

**What the step-scale sweep shows:**

- **Magnitude `delta` saturates and plateaus:** 0.250 → 0.447 → 0.679 → 0.748 → 0.739. Once both groups' steps reach the upper tail, the β-space gap between a `1·a` step and a `2·a` step stops growing — the β cap (≈ 0.74 here) bounds the measurable magnitude difference.
- **Magnitude→angle cross-talk *emerges* at large effect size.** Below `signal_scale ≈ 4` the magnitude angle sits at the `none` floor (no leakage). At `signal_scale = 6` it is 6.9° against a 2.0° floor (3.5×), and at `8.0` it is 9.4° against a 1.5° floor (6.3×). **This is the specificity-study leak, reproduced by `rev_logit` alone.**
- **Orientation remains robust** — angle ≈ 45–47°, delta at floor — across all step scales.

**Direct confirmation — a pure scaling acquires a real rotation.** The cleanest way to see the cross-talk is to measure it with *no PCA, no estimator, and no noise* — just the angle between group A's β-space step and group B's β-space step, computed straight from the cell means at center (M = 0). For the magnitude manipulation the two M-space steps are **exactly collinear by construction** (`b = 2a`, angle 0°), so *any* β-space angle is purely `rev_logit` bending:

| signal_scale (step span) | angle(βA, βB), population | ‖βA‖ | ‖βB‖ |
|-------------------------:|--------------------------:|-----:|-----:|
| 1  | 0.5° | 0.249 | 0.491 |
| 2  | 1.8° | 0.491 | 0.937 |
| 4  | 4.8° | 0.937 | 1.620 |
| 6  | 7.1° | 1.314 | 2.064 |
| 8  | 8.5° | 1.620 | 2.355 |
| 10 | 9.3° | 1.866 | 2.554 |

A pure 2× scaling in M-space emerges as a several-degree rotation in β-space, growing monotonically with step span. The full pipeline tracks this exactly: population 8.5° at `signal_scale = 8` versus the measured 9.4° (the extra is estimation noise layered on the real bending).

**Mechanism.** `rev_logit` acts coordinate-wise, so the per-coordinate ratio `βB_i / βA_i` is **not constant across coordinates**: a coordinate with a large `|a_i|` has its 2× step pushed further into saturation and grows by *less than* 2×, while a small-`|a_i|` coordinate is still linear and grows by nearly 2×. Hence `βB` is not a scalar multiple of `βA` — the two vectors point in different directions, and that misalignment *is* the magnitude→orientation cross-talk. When the step is small (or a saturated baseline compresses everything by ~the same factor) the ratio is near-constant → direction preserved → no cross-talk. When the step is **large enough to span the sigmoid's curved region**, the spread in coordinate magnitudes maps to a spread in growth ratios → the β-direction **bends**, and the `2×` step bends more than the `1×` step → spurious `angle`. Cross-talk is driven by **effect size (step span relative to sigmoid curvature)**; the operating point drives **compression**. Two distinct, separable effects.

---

### Why the two axes differ — "where" vs "how far"

Both axes push the step into nonlinear territory, but they engage different parts of the sigmoid:

- **Operating point** moves a *short* step (fixed span) deeper into saturation. Over that short span the slope `β(1−β)` is nearly constant, so every coordinate is scaled by ~the same factor → the step stays collinear → **compress, don't rotate.**
- **Effect size** holds the baseline at center but makes the step *long enough to reach from the steep middle into the flat tail*. Now coordinates experience very different local slopes → non-uniform scaling → **rotate.**

Compression needs the step to *sit* somewhere flat; rotation needs the step to *cross* between steep and flat. They are orthogonal knobs — which is why the operating-point sweep shows pure compression with a frozen ~47° orientation angle, while the step-scale sweep shows angle cross-talk with a *saturating* δ. This separation is the central result of Rung 1.

#### Reproducing the noiseless population tables

```python
# Population geometry behind the δ/slope and angle(βA,βB) tables:
# no sampling, no PCA, no estimator — straight rev_logit of the cell means.
import numpy as np
from motco.simulations.generator import rev_logit
from motco.simulations.linear_recovery import generate_dataset, LinearRecoveryParams

def step_vectors(manip, signal):
    d = generate_dataset(LinearRecoveryParams(
        seed=0, n_features=50, signal_scale=signal, manipulation=manip, scale_c=2.0))
    return d.step_A, d.step_B  # M-space steps (group A baseline, group B transform)

# Axis 1 — magnitude δ vs operating point (M-space steps collinear, signal=2)
aA, aB = step_vectors("magnitude", 2.0)
for m in (0, 1, 2, 3, 4):
    base = np.full(50, float(m))
    dA = rev_logit(base + aA) - rev_logit(base)
    dB = rev_logit(base + aB) - rev_logit(base)
    print(m, np.linalg.norm(dB) - np.linalg.norm(dA))

# Axis 2 — angle(βA, βB) at center vs step span (any angle is pure rev_logit bending)
for s in (1, 2, 4, 6, 8, 10):
    aA, aB = step_vectors("magnitude", float(s))
    dA, dB = rev_logit(aA) - 0.5, rev_logit(aB) - 0.5  # rev_logit(0) = 0.5
    cos = dA @ dB / (np.linalg.norm(dA) * np.linalg.norm(dB))
    print(s, np.degrees(np.arccos(np.clip(cos, -1, 1))))
```

---

### Resolution — M-value integration removes the distortion

Everything above integrates in **β-space**. But β is only the *carried* representation; the standard analysis practice is to transform methylation to **M-values** before integration. In our generator the signal is injected in M-space and mapped to β by the *exact* logistic link `rev_logit`, so `logit(β)` is its **exact inverse** — M-integration undoes the nonlinearity entirely.

The step-scale sweep, run under both representations (mean ± SD over seeds 0–9):

| signal_scale | manip | β-int: δ | β-int: θ | **M-int: δ** | **M-int: θ** |
|-------------:|-------|---------:|---------:|-------------:|-------------:|
| 2.0 | none        | 0.012 | 5.3° | 0.051 | 5.0° |
| 2.0 | magnitude   | 0.447 | 4.4° | **2.049** | 4.0° |
| 2.0 | orientation | 0.019 | 47.3° | 0.077 | 47.3° |
| 4.0 | magnitude   | 0.679 | 2.9° | **4.045** | 2.0° |
| 6.0 | magnitude   | 0.748 | **6.9°** | **6.044** | 1.3° |
| 8.0 | none        | 0.013 | 1.5° | 0.052 | 1.3° |
| 8.0 | magnitude   | 0.739 | **9.4°** | **8.044** | **1.0°** |
| 8.0 | orientation | 0.047 | 46.1° | 0.075 | 45.4° |

Under M-integration:

- **Magnitude `delta` recovers the exact M-space truth, with no saturation:** δ = 2.05, 4.05, 6.04, 8.04 at `signal_scale` = 2, 4, 6, 8 — i.e. δ ≈ `signal_scale·(c−1)` linearly at every effect size. The β-arm's plateau (≈ 0.74) is gone; magnitude sensitivity is fully restored.
- **The cross-talk vanishes:** the magnitude angle that reached 9.4° in β at `signal_scale = 8` drops to **1.0°**, at the `none` floor. `logit` inverts `rev_logit` coordinate-wise, so the differential-saturation bending is undone before the projector sees it.
- **Orientation is preserved** (≈ 45°), as in β.

It is literally Rung 0 again: a clean linear problem. (M is up to a global scale — natural-log `logit` vs the log2-based M-value convention — which leaves angles invariant and rescales magnitudes uniformly.)

**Caveats.**

1. *Clean by construction.* The inversion is exact because the generative link and the integration transform are exact inverses *and* the noise is additive in M-space. Real methylation deviates (non-exact link, probe effects, noise not strictly M-additive); within the InterSIM-faithful model the result is exact, and as a real-world claim it restates why the field already prefers M-values for analysis.
2. *Numerical edge at saturation.* `logit(β) → ±∞` as β → 0/1, so β is clipped to `[1e−6, 1−1e−6]` before transform (`beta_to_mvalue`). The edge check at `m_baseline = 4` (β up to 0.997) stayed well inside the clip; deeper baselines lean on it, and the clipping itself is a real, characterizable M-integration failure mode.

---

### Gate decision

The Rung-1 result has two layers:

1. **In β-space, `rev_logit` is a real distortion** — and it is *two* distinct distortions, not one. The **operating point** (baseline saturation) compresses magnitude `delta` while preserving direction (no cross-talk: a uniform baseline rescales but cannot rotate). The **effect size** (step span across the sigmoid) produces genuine magnitude→`angle` cross-talk via differential per-coordinate saturation, onsetting around `signal_scale ≳ 6`. So β-integration both *loses magnitude sensitivity* and *manufactures spurious orientation* — the specificity-study leak, reproduced.

2. **M-value integration removes both** — exactly, at every operating point and effect size — because `logit` is the inverse of the generative link. Since `logit` acts coordinate-wise, this holds regardless of per-CpG baseline *heterogeneity* too: heterogeneous operating points would also be inverted.

**Decision: M-value integration is the correct pipeline representation for methylation**, and the methylation `rev_logit` nonlinearity is therefore **not** a standing cross-talk source for a correctly-built pipeline — it is a cautionary failure mode of integrating in β-space.

**Consequence for the ladder.** Because M-integration inverts the link coordinate-wise, the planned Rung-2 (per-CpG baseline *heterogeneity*) is **mooted under M-integration** — heterogeneous baselines are inverted just as cleanly as a uniform one. The real cross-talk must live downstream of the methylation representation: the **projector** (PCA → PLS, SNF), **per-feature standardization**, **concatenation** of heterogeneous omics, or the **cross-omic coupling**. The ladder should advance to those, with methylation fixed in M-space. The next rung is therefore re-scoped from "heterogeneous methylation baselines" to the **projector / integration step** as the next single factor.
