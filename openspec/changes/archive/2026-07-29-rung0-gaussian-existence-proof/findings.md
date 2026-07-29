## Rung 0 — Gaussian Existence Proof: Findings

### What this experiment does

The goal here is to establish a **clean baseline** for the MOTCO trajectory pipeline before adding any biological complexity. The question is: given perfectly controlled, mathematically simple data, do the production estimators (`estimate_difference`) correctly recover the intended geometric differences between two groups?

We inject two groups (A and B), each measured at two stages (0 and 1). Every sample is drawn as a Gaussian: `x = μ_{group, stage} + noise`, where the noise is isotropic (`Σ = σ²I`) and the means are set by hand. Both groups start at the same point (the origin) at stage 0. Group A moves to a fixed position at stage 1, defining its "step vector" `a`. Group B's step is a controlled transform of `a`:

- **none** — Group B takes the exact same step as A (null control: should produce zero delta, zero angle).
- **magnitude** — Group B's step is scaled by `c = 2`: twice as long, same direction. This should appear as a large delta (size difference) with zero angle (no direction difference).
- **orientation** — Group B's step is rotated by `θ = 45°` relative to A: same length, different direction. This should appear as a ~45° angle with zero delta.

After generating the data we fit a PCA (no standardization), project into the latent space, and run `estimate_difference` — exactly the same code path the real pipeline uses. We then read off the measured `delta` (magnitude difference) and `angle` (direction difference in degrees).

Because we know the ground truth exactly, this is an **existence proof**: if the estimators fail here, the problem is in the math, not in biology.

---

### Reproduction parameters

| Parameter              | Value  |
|------------------------|--------|
| `n_features`           | 50     |
| `n_samples_per_cell`   | 40     |
| `signal_scale` (‖a‖)   | 5.0    |
| `noise_scale` (σ)      | 1.0    |
| `scale_c` (magnitude)  | 2.0    |
| `angle_theta` (orient.)| 45°    |
| Seeds averaged over    | 0–9    |

```bash
uv run python scripts/linear_recovery_probe.py
```

Trajectory figure: `build/rung0_latent_trajectories.png`

---

### Existence-proof table (k = 10 PCs)

Mean ± SD across 10 seeds, inline PCA retaining `k = 10` components out of 50 features:

| manipulation  | δ mean ± SD      | θ mean ± SD        |
|---------------|------------------|--------------------|
| none (null)   | 0.16 ± 0.08      | 15.6° ± 4.4°       |
| magnitude     | **5.00 ± 0.17**  | 12.0° ± 3.8°       |
| orientation   | 0.23 ± 0.15      | **47.0° ± 3.8°**   |

**Verdict: the floor is clean.**

- `magnitude` drives delta to ≈ 5.0 (which equals `signal_scale × (c − 1) = 5 × 1`, exactly as expected), and the angle stays within the null baseline.
- `orientation` drives angle to ≈ 47° (target: 45°; < 5% error), and delta stays at the null floor.
- Neither manipulation crosses over into the other's statistic.

#### Why is the "none" angle 15.6° and not 0°?

With a null manipulation both groups have the same ground-truth step direction, so ideally the measured angle should be 0°. The 15.6° is **finite-sample estimation noise**, not a failure. Here is what happens: the LS-mean estimator uses 40 samples per cell to estimate the true cell mean. Each estimate carries noise on the order of `σ / √n ≈ 0.16` per feature. In a `k`-dimensional latent space, that per-feature noise accumulates: the estimated step vectors for A and B drift slightly away from each other, and the angle between two slightly-drifted estimates of the same direction can be substantial in higher dimensions (more on this in the sweep below). This floor shrinks with more samples or higher signal-to-noise ratio. The key point is that both `magnitude` (12°) and `orientation` (47°) are evaluated against this same floor — magnitude's angle is *at* the floor (no cross-talk), and orientation's angle is *well above* it (real signal).

---

### Does the number of PCs matter? (k = 3 … 10 sweep)

To answer your question directly: yes, we ran this with `k = 10`, but the result holds across the full range `k = 3..10`. Here is the full sweep with the same 10 seeds and same base parameters — only `k` changes.

| k  | none δ      | none θ      | magnitude δ  | magnitude θ  | orientation δ | orientation θ |
|----|-------------|-------------|--------------|--------------|---------------|---------------|
|  3 | 0.15 ± 0.09 |  9.3° ± 4.9° | 5.04 ± 0.17 |  7.8° ± 3.8° | 0.24 ± 0.17  | 45.6° ± 4.0°  |
|  4 | 0.15 ± 0.08 | 10.8° ± 4.1° | 5.03 ± 0.17 |  8.5° ± 3.8° | 0.24 ± 0.17  | 45.7° ± 3.9°  |
|  5 | 0.15 ± 0.08 | 11.4° ± 4.3° | 5.03 ± 0.17 |  8.9° ± 3.8° | 0.24 ± 0.17  | 45.8° ± 4.0°  |
|  6 | 0.15 ± 0.08 | 12.6° ± 4.8° | 5.02 ± 0.17 |  9.8° ± 3.9° | 0.24 ± 0.16  | 46.3° ± 3.9°  |
|  7 | 0.16 ± 0.08 | 13.1° ± 4.4° | 5.02 ± 0.17 | 10.4° ± 3.4° | 0.24 ± 0.15  | 46.5° ± 3.8°  |
|  8 | 0.16 ± 0.08 | 13.7° ± 4.2° | 5.01 ± 0.17 | 10.9° ± 3.2° | 0.24 ± 0.15  | 46.6° ± 3.8°  |
|  9 | 0.16 ± 0.08 | 15.0° ± 4.3° | 5.01 ± 0.17 | 11.4° ± 3.7° | 0.23 ± 0.15  | 46.8° ± 3.7°  |
| 10 | 0.16 ± 0.08 | 15.6° ± 4.4° | 5.00 ± 0.17 | 12.0° ± 3.8° | 0.23 ± 0.15  | 47.0° ± 3.8°  |

#### What the sweep shows

**Delta is completely unaffected by k.** Across every value of `k`, the magnitude manipulation recovers δ ≈ 5.0 and the null floor stays at δ ≈ 0.15. The delta statistic does not care how many PCs you retain, at least for this SNR.

**The angle null floor grows with k.** The "none" baseline angle rises from 9.3° at `k = 3` to 15.6° at `k = 10`. This is a pure noise effect: each additional PC contributes an independent noise dimension to the latent-space step estimate. Retaining more PCs does not add signal (the signal lives in the top 2 PCs; see below), but it does add more directions for the estimated vectors to drift apart. Importantly, the cross-talk from `magnitude` onto angle (the "magnitude θ" column) tracks the null floor closely at all `k`, confirming it is noise — not leakage from the magnitude manipulation.

**The orientation angle is stable and accurate across all k.** The angle for `orientation` stays near 45–47° regardless of `k`. This might seem surprising — shouldn't retaining more PCs that capture `û` (the orthogonal component of group B's step) improve accuracy? The reason it doesn't change is that with two strong signal directions present in the data — both `a` (group A's step, scale 5) and `step_B` (group B's rotated step, also scale 5) — PCA always places both signal directions in PC1 and PC2, regardless of how many additional noise PCs are retained. With `k ≥ 2`, the signal subspace is already fully captured; adding more PCs adds only noise. The cumulative explained variance rises with `k` (from ≈ 15% at k=2 to 43% at k=10), but the trajectory signal is accounted for in the first two components.

#### Practical implication

For this Gaussian setting, **k has no meaningful effect on the primary signals.** The choice of k only affects the noise level in the angle statistic (higher k → noisier angle floor). For Rung 1+, where biology and nonlinearity may spread the signal across more PCs, this tradeoff becomes more relevant and will be re-examined.

---

### Inverse-design recipes (seed 0, k = 10)

One of the goals of Rung 0 is to answer: "if we want to create a specific latent-space manipulation, what is the minimal feature-space change that achieves it?" These are the **minimum-norm preimages**:

| Recipe               | Latent target Δy              | Feature recipe Δx       |
|----------------------|-------------------------------|-------------------------|
| Magnitude (c = 2)    | `(c − 1) · a` (scale a by 1) | `Vk @ (c−1) · a`        |
| Orientation (θ = 45°)| `(R − I) · a` (rotate a 45°) | `Vk @ (R−I) · a`        |

where `Vk` (shape 50 × 10) is the PCA loading matrix and `R` is a 10×10 Givens rotation. The round-trip `Vk.T @ Δx = Δy` holds to machine precision in both cases.

**What do the minimum-norm Δx vectors look like?**

| Recipe       | Features with signal (of 50) | Participation ratio | Top-3 features carry |
|--------------|------------------------------|---------------------|----------------------|
| Magnitude    | 44                           | 31.5                | 16.5% of ‖Δx‖₁      |
| Orientation  | 46                           | 31.7                | 17.5% of ‖Δx‖₁      |
| Support overlap | 40 out of 50 shared       | —                   | —                    |

Both Δx vectors are **dense**: the minimum-norm change involves 44–46 of the 50 features with no single dominant feature (the top 3 carry only ~17% of the signal). This is expected because PCA loadings mix all features; the minimum-norm preimage of any latent-space change inherits that mixing. The 80% support overlap means both manipulations touch the same features. The implication is that the minimum-norm answer to "what feature change creates a 45° rotation?" involves mostly the same features as "what feature change creates a 2× scaling?" — just combined differently. Magnitude and orientation are not separable at the individual-feature level; they are separable only in latent space.

---

### Gate decision

The Rung-0 floor is **clean**:

- The production estimators correctly isolate magnitude and orientation with no cross-talk, across all values of k tested.
- Delta recovery is exact and k-invariant.
- Orientation angle recovery is accurate (< 5% error) and k-invariant for k ≥ 3.
- The growing angle noise floor with k is a finite-sample effect, not a signal-leakage effect.

Because this is a purely linear Gaussian system, any failure here would implicate the estimator or projector itself. No such failure was found. The cross-talk observed in the earlier specificity study therefore arises from something introduced at higher rungs — most likely the methylation `rev.logit` nonlinearity or the data-dependent projector.

**Next step: Rung 1** — repeat with the `rev.logit` methylation transform layered on top to isolate its contribution to cross-talk.
