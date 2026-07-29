## Rung 2 — Projector Cross-talk: Findings

### What this experiment does

Rung 0 proved the linear floor is clean under one deliberately benign projector (mean-centered PCA). Rung 1 showed the methylation `rev.logit` is inverted coordinate-wise by M-value integration, so it is not a standing cross-talk source — the leak must enter **downstream of the methylation representation**, in the integration/projection step. Rung 2 adds **exactly one** new factor: the **projector**.

We reuse the Rung-0 geometry injection **unchanged** — the known 2-stage geometry (`none`/`magnitude`/`orientation`) in a clean linear feature space (the methylation **M-value frame**, so there is no `rev.logit` distortion) — and measure `delta`/`angle` through four projectors, holding the estimator path (`get_model_matrix`/`build_ls_means`/`estimate_difference`, two-group contrast) identical to Rung 0/1:

- **`pca`** — mean-centered PCA (the Rung-0/1 reference floor).
- **`standardize`** — per-feature z-score (the transform inside the production `concat` integration, `evaluation.py:_concat_integration`), then PCA.
- **`plsda`** — supervised PLS-DA latent space, conditioned by default on the **stage** label (`stats/pls.fit_plsda_transform`), matching the production trajectory pipeline (`viz.plot_trajectory_from_plsr`), which orients the latent axes to separate disease stages — the trajectory itself — before group A-vs-B differences are measured within that space. (The adversarial *group*-conditioned variant is exercised separately by the supervised-leakage probe, Axis 4.)
- **`snf`** — graph-spectral embedding (`stats/snf`). SNF *fusion* requires ≥ 2 networks, so on this single-block test bed the arm reduces to `get_affinity_matrix → get_spectral` — the per-block core of the production `snf` path.

Cross-talk is read against each projector's own `none` null and against the PCA floor. A second pass under **heteroscedastic (anisotropic) per-feature noise** stresses the projectors where they differ most.

---

### Reproduction parameters

| Parameter               | Value  |
|-------------------------|--------|
| `n_features`            | 50     |
| `n_samples_per_cell`    | 40 (→ 160 samples) |
| `signal_scale` (‖a‖)    | 5.0    |
| `noise_scale` (σ)       | 1.0    |
| `scale_c` (magnitude)   | 2.0    |
| `angle_theta` (orient.) | 45°    |
| `n_components`          | 2 (headline); sweep 2/3/5/10 |
| `anisotropy`            | 0 (headline); 1.5 (stress) |
| `snf_K` / `snf_eps`     | 20 / 0.5 |
| Seeds averaged over     | 0–9    |

```bash
.venv/bin/python scripts/projector_recovery_probe.py
```

Intended geometry: magnitude target `delta = signal_scale·(c−1) = 5.0`; orientation target `angle = 45°`.

---

### Axis 1 — Per-projector recovery on the clean isotropic floor

mean ± SD over seeds 0–9, `n_components = 2`:

| projector   | manip       | δ mean | δ SD | θ mean | θ SD |
|-------------|-------------|-------:|-----:|-------:|-----:|
| **pca**     | none        | 0.17 | 0.13 | 6.6° | 5.2 |
| **pca**     | magnitude   | **5.17** | 0.16 | 5.4° | 4.2 |
| **pca**     | orientation | 0.26 | 0.21 | **47.8°** | 3.9 |
| standardize | none        | 0.15 | 0.12 | 6.4° | 4.8 |
| standardize | magnitude   | 3.91 | 0.17 | 5.1° | 3.5 |
| standardize | orientation | 0.25 | 0.18 | **49.6°** | 4.4 |
| plsda       | none        | 0.15 | 0.11 | 3.2° | 2.1 |
| plsda       | magnitude   | 3.69 | 0.17 | 15.1° | 2.6 |
| plsda       | orientation | 0.25 | 0.14 | **10.9°** | 5.7 |
| snf         | none        | 0.02 | 0.01 | 14.1° | 8.2 |
| snf         | magnitude   | 0.17 | 0.02 | **70.1°** | 1.5 |
| snf         | orientation | 0.04 | 0.02 | 66.0° | 2.6 |

Reading it:

- **PCA is clean** — it reproduces the Rung-0 floor exactly: magnitude δ = 5.17 ≈ the target 5.0 with θ at the null floor, orientation θ = 47.8° ≈ 45° with δ at the null floor. **No cross-talk.**
- **`standardize` preserves direction but rescales magnitude.** Orientation recovers cleanly (49.6°) and angles stay at the null floor for every manipulation, so **standardization introduces no magnitude→orientation cross-talk**. Its only effect is a *uniform* magnitude rescale (δ 5.17 → 3.91): z-scoring divides each feature by its total (signal+noise) std, shrinking the signal-carrying features. This is a units change, not a rotation.
- **Stage-conditioned PLS-DA does not leak, but it *compresses* the geometry.** The `none` null is clean and stable (θ = 3.2° ± 2.1, δ = 0.15) — no spurious separation. But the recovered geometry is attenuated: magnitude δ = 3.69 (shrunk from PCA's 5.17, because PLS-DA scales features internally, like `standardize`), and — the key effect — **orientation comes out only 10.9° versus the 45° truth**, a severe *under*-recovery, with a mild magnitude→angle bleed (15.1°). Mechanism: PLS-DA on a 2-stage label defines one dominant discriminant axis (the shared stage-separating direction), and at the production-typical `k = 2` the projection collapses both groups' trajectories toward that common axis, squashing the group-specific orientation difference. So the failure mode is the **opposite** of leakage — not invented direction, but *lost* direction. (The gross 110° rotation and unstable 19.9° ± 40.6° null reported in earlier drafts were artifacts of conditioning PLS-DA on the **group** label; see Axis 4.)
- **SNF produces magnitude→angle cross-talk.** A pure `magnitude` manipulation registers **70°** of angle. SNF magnitudes are tiny and embedding-relative (δ ≈ 0.02–0.17, not comparable in units to the feature-space target), so we report SNF as recovery-vs-null: it *does* separate signal from `none`, but the spectral embedding's metric is not commensurable with the injected geometry, and it rotates magnitude into angle.

### Axis 2 — Heteroscedastic noise: where the projectors diverge

Same run with `anisotropy = 1.5` (per-feature noise std `× exp(1.5·z)`):

| projector   | manip       | δ mean | θ mean | θ SD |
|-------------|-------------|-------:|-------:|-----:|
| **pca**     | none        | 4.92 | **139.6°** | 32.8 |
| **pca**     | magnitude   | 5.90 | **129.5°** | 44.2 |
| **pca**     | orientation | 4.50 | **144.7°** | 24.2 |
| **standardize** | none    | 0.12 | **4.1°** | 2.3 |
| **standardize** | magnitude | 4.55 | **4.5°** | 2.3 |
| **standardize** | orientation | 0.59 | **45.8°** | 5.2 |
| plsda       | none        | 0.11 | 2.1° | 1.5 |
| plsda       | magnitude   | 4.36 | 13.2° | 3.3 |
| plsda       | orientation | 0.42 | 29.5° | 10.6 |
| snf         | none        | 0.39 | 93.1° | 64.5 |
| snf         | magnitude   | 0.43 | 88.4° | 66.1 |
| snf         | orientation | 0.39 | 102.2° | 59.1 |

This is the decisive contrast:

- **Raw PCA collapses.** Under heteroscedastic noise the high-variance noise features dominate the top components, so the signal falls out of the retained subspace and *everything* — including `none` — registers ~130–145° of spurious angle and large spurious δ. Mean-centered PCA has no defense against features on different scales.
- **`standardize` rescues it completely.** Per-feature z-scoring equalizes the noise scales, putting the signal back in the top components: orientation recovers (45.8°), magnitude stays direction-clean (4.5°), and the `none` floor returns to ~4°. **This is exactly why the production `concat` path standardizes** — and it is clean.
- **Stage-conditioned PLS-DA** is also robust to anisotropy (it scales features internally, so it ignores feature-scale heterogeneity): the `none` floor stays at ~2° and the signal arms are unchanged. Its orientation recovery is actually *better* here (29.5°) than on the isotropic floor (10.9°) — the per-feature scaling spreads the discriminant axes — but it is still well short of the 45° truth, i.e. the geometry stays compressed.
- **SNF** stays scrambled (~90–100°, huge variance): its Euclidean affinity is dominated by the high-variance features just as raw PCA is.

### Axis 3 — Dimensionality and effect-size sweeps

- **Dimensionality (k = 2/3/5/10):** the `none` angle floor rises with retained components for *every* projector (the Rung-0 k-noise effect: each extra noise PC adds angular spread) — pca `none` θ: 6.6 → 9.4 → 11.3 → 15.9; stage-conditioned plsda `none` θ tracks it: 3.2 → 6.2 → 9.0 → 12.2 (no leakage inflation). SNF magnitude→angle stays ~70° at every k (intrinsic). Stage-conditioned PLS-DA's orientation *recovers* as k grows — 10.9° → 21.7° → 30.9° → 35.7°, climbing toward the 45° truth — because retaining more axes gives the group-specific direction room beyond the single dominant stage discriminant; its magnitude stays attenuated (~3.66 at every k). So for PLS-DA the orientation compression is a low-dimensionality effect that eases with more components, but at the cost of the rising k-noise null — there is no k that is both faithful and tight.
- **Effect size (`signal_scale` = 2/5/8/12):** PCA magnitude δ tracks the target linearly with no onset (2.17 / 5.17 / 8.16 / 12.15 ≈ scale·(c−1)) and angles stay at the floor. `standardize` magnitude δ **compresses progressively** (2.02 / 3.91 / 4.99 / 5.83) — as the signal grows it contributes more to per-feature variance, so z-scoring divides out more — but angles remain clean throughout (no cross-talk onset). Stage-conditioned PLS-DA magnitude δ tracks but stays attenuated (1.72 / 3.69 / 4.82 / 5.71), and its orientation *improves* with effect size (9.2° → 10.9° → 14.4° → 20.7° toward 45°) while its magnitude→angle bleed *shrinks* (18.7° → 15.1° → 12.0° → 9.6°) — a stronger trajectory is easier for the discriminant to lock onto. SNF magnitude→angle persists (~70–76°) while its δ collapses toward 0 at large signal (graph saturates).

### Axis 4 — Supervised-leakage probe (**group**-conditioned PLS-DA vs PCA on the `none` null)

This probe forces the adversarial worst case: PLS-DA conditioned on the **group** (A/B) label — *not* the stage-conditioned headline arm — so it tests whether a group-supervised projection can invent direction on a null trajectory. (`run_leakage_probe` sets `plsda_label="group"` regardless of the base params.)

| k  | pca θ (null) | plsda θ (null, group-conditioned) |
|---:|-------------:|----------------------------------:|
| 2  | 6.6° ± 5.2   | **19.9° ± 40.6** |
| 3  | 9.4° ± 4.8   | 18.6° ± 29.7 |
| 5  | 11.3° ± 3.9  | 12.2° ± 4.3 |
| 10 | 15.9° ± 4.4  | 16.9° ± 3.6 |

**Group**-conditioned PLS-DA inflates and destabilizes the **null angle** at low component counts — a group-supervised projection inventing direction where there is none — and converges to the PCA floor only once enough components are retained. The null **δ** is never inflated. This confirms *why* the production pipeline conditions PLS-DA on **stage**, not group: the stage-conditioned headline arm (Axis 1) has a clean, stable null (3.2° ± 2.1), whereas this group-conditioned variant leaks specifically into *orientation*, worst exactly where a practitioner would use it (few components, two classes).

---

### Gate decision

**Which projectors are clean for trajectory geometry?**

| projector   | orientation faithful? | magnitude→angle cross-talk? | robust to feature-scale heterogeneity? | verdict |
|-------------|:---------------------:|:---------------------------:|:--------------------------------------:|---------|
| **pca** (raw) | yes (isotropic only) | no (isotropic only) | **no** — collapses under anisotropy | conditional |
| **standardize** (`concat`) | **yes** | **no** | **yes** | **clean** |
| **plsda** (stage) | **no** — compresses (45°→11° at k=2) | mild (15°), eases with effect size | yes | **lossy, but no leakage** |
| **plsda** (group) | n/a | leaks into null at low k | yes | **leaks** (adversarial, Axis 4) |
| **snf** | no | **yes (~70°)** | no | **leaks** |

1. **The production `concat` projector (per-feature standardize) is clean.** It preserves orientation, introduces no magnitude→orientation cross-talk, and — crucially — is *robust to feature-scale heterogeneity* where raw PCA fails. Its only distortion is a uniform magnitude rescale/compression (δ no longer in source units, compressing with effect size), which does not rotate the geometry and so does not threaten the angle-specificity the trajectory test depends on. The specificity-study cross-talk does **not** originate in single-block standardization.

2. **The production `snf` projector leaks.** Single-block spectral embedding turns a pure magnitude change into ~70° of spurious angle and yields embedding-relative magnitudes that collapse under large effects or heteroscedastic noise. Its latent metric is not commensurable with the injected geometry. **Recommendation:** prefer `concat` over `snf` for trajectory-geometry analysis, or treat `snf` geometry as embedding-relative (recovery-vs-null only), never as a unit-faithful magnitude/angle.

3. **Supervised PLS-DA (stage-conditioned, as production uses it) does not leak, but it is not geometry-faithful either.** Conditioned on the **stage** label — the production configuration — its null is clean and stable (no invented separation), but a 2-stage label yields one dominant discriminant axis onto which the projection collapses, so at the production-typical `k = 2` it *compresses* the trajectory: magnitude attenuated (5.17 → 3.69, like `standardize`'s internal scaling) and orientation severely under-recovered (45° → 11°). Recovery improves with more components and larger effects, but only by trading the orientation compression for a rising k-noise null. The earlier "manufactures 110° / unstable null / unsafe" verdict was an artifact of conditioning PLS-DA on the **group** label; that adversarial configuration *does* leak (Axis 4) and is the reason production conditions on stage. **Net:** stage-conditioned PLS-DA is safe against false positives (clean null) but lossy for magnitude/orientation at low k — use it for visualization of the stage trajectory, but `standardize`/`concat` remains the geometry-faithful projector for the `delta`/`angle` tests.

**Consequence for the ladder.** On a **single** clean block, the production `concat` path is faithful — so the magnitude→orientation cross-talk the specificity study saw is **not** explained by single-block standardization. The remaining untested factors on the clean path are (a) **heterogeneous multi-omic concatenation** — the real pipeline z-scores and concatenates three omic blocks of *different dimensionality and correlation structure*, and unequal block sizes can tilt the pooled PCA toward the larger/most-correlated block, rotating the combined geometry; and (b) the **cross-omic coupling** in the generator, which correlates the blocks. Single-block standardization being clean makes **heterogeneous multi-omic concatenation the next single factor**.

**Decision: Rung 3 = heterogeneous multi-omic concatenation** (methylation in M-space + a second/third block of different dimension and scale, standardized and concatenated as in `concat`), measuring whether unequal block geometry rotates magnitude into orientation. SNF is recorded here as a known-leaky projector; stage-conditioned PLS-DA is recorded as non-leaking but lossy (a visualization projector, not the measurement projector), and group-conditioned PLS-DA as a known leakage hazard — all settled at this rung.

**Exact reproduction:** seeds 0–9; `n_features = 50`, `n_samples_per_cell = 40`, `signal_scale = 5.0`, `noise_scale = 1.0`, `scale_c = 2.0`, `angle_theta = 45°`, `n_components = 2` (sweep 2/3/5/10), `anisotropy ∈ {0, 1.5}`, `snf_K = 20`, `snf_eps = 0.5`; driver `scripts/projector_recovery_probe.py`.
