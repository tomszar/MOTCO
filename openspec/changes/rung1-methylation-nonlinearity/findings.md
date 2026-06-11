## Rung 1 — Methylation rev.logit Nonlinearity: Findings

### What this experiment does

Rung 0 proved the linear floor is clean: under a pure-linear PCA projector the production estimators isolate magnitude (`delta`) and orientation (`angle`) with no cross-talk. Rung 1 adds **exactly one** new factor — InterSIM's methylation nonlinearity — and asks whether *it* is the source of the cross-talk the specificity study found.

We reuse the Rung-0 generation **unchanged**, but reinterpret the feature matrix as CpG **M-values**. The known 2-stage geometry (`none`/`magnitude`/`orientation`) is injected in M-value space exactly as in Rung 0; the drawn samples are then passed through InterSIM's `rev_logit` (`β = 1/(1+e^{−M})`, reused from `generator.py`) to obtain β values, which are projected with inline PCA and measured with `estimate_difference`. **Methylation-only:** no expression/protein layer is generated — InterSIM's downstream coupling transmits the differential *support* but not the *magnitude*, so a cascade would conflate factors. `rev_logit` is the single new variable versus Rung 0.

`rev_logit` is locally linear at M ≈ 0 (slope `β(1−β) = 0.25`) and saturates on the tails (slope → 0). Distortion therefore depends on **where on the sigmoid the trajectory sits** and **how much of the sigmoid a single step spans**. These are the two independent variables, swept separately.

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

Operating-point figure: `build/rung1_operating_point.png`

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

**Mechanism.** `a_feat` is a random vector, so its coordinates have *different magnitudes*. When the step is small (or the baseline tail compresses everything uniformly), every coordinate experiences nearly the same local slope → direction preserved → no cross-talk. When the step is **large enough to span the sigmoid's curved region**, large-magnitude coordinates saturate while small ones stay linear → the β-space direction **bends** away from the M-space direction; the `2×`-scaled magnitude step bends *more*, so groups A and B end up pointing in different directions → a spurious `angle`. Cross-talk is therefore driven by **effect size (step span relative to sigmoid curvature)**, while the operating point drives **compression**. Two distinct, separable effects.

---

### Gate decision

The Rung-1 result is two-sided and sharper than "does `rev_logit` cause cross-talk?":

- **Operating point (baseline saturation) → magnitude compression, direction preserved, NO cross-talk.** A uniform `rev_logit` baseline rescales the step but cannot rotate it.
- **Effect size (step span) → magnitude→angle cross-talk.** Once a step traverses enough sigmoid curvature, differential per-coordinate saturation bends the direction and the magnitude manipulation leaks into `angle`. In this regime the onset is around `signal_scale ≳ 6` (β endpoints ≈ 0.997).

So `rev_logit` **is** a sufficient mechanism for the magnitude→orientation cross-talk — but only at large effect sizes; for modest effects at a uniform baseline it costs only magnitude *sensitivity*, not specificity. This implicates **effect size and per-CpG baseline heterogeneity** (real CpGs sit at different operating points, so even a modest uniform M-step lands on different local slopes) as the live suspects above this rung.

**Next step: Rung 2** — introduce per-CpG baseline heterogeneity (the real `mean_M` operating points), holding the projector linear, to test whether realistic operating-point spread converts a modest-effect magnitude manipulation into angle cross-talk without requiring a large step. The PCA→PLS projector swap remains a separate, later rung.
