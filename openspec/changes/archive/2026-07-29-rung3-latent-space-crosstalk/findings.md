## Rung 3 — Cross-talk through the production latent spaces: Findings

### What this experiment does

Rungs 0–2 isolated the cross-talk factors on a generator-free test bed. Rung 3 switches to the **full production path** — semi-synthetic generator → integration (latent space) → `estimate_difference` → RRPP rejection rates, via `specificity.py` — and runs the dominant-specificity study through three latent spaces on identical seeds, effect size, stages, and permutation count. The only swept factor is the latent space:

- **`concat`** — the standardized-feature-concatenation **baseline** (what the specificity study has always used).
- **`snf`** — the graph-spectral production latent space.
- **`pls`** — the stage-conditioned, double-CV-sized production latent space (added in PR #23).

A *specific* construction rejects predominantly on its target statistic (`magnitude`→`delta`, `orientation`→`angle`, `shape`→`shape`); off-target rejections are cross-talk. The group-in-stage fraction (GIS) reports how much of the injected group signal lies in the stage subspace of each latent space.

---

### Reproduction parameters

| Parameter | Value |
|-----------|-------|
| `n_replicates` | 6 |
| `permutations` | 49 |
| `n_samples` | 160 |
| `n_stages` | 4 |
| `effect_size` | 1.0 |
| `p_dmp` | 0.2 |
| PLS CV | `n_repeats=2, cv2=4, cv1=3, max_components=8` |
| Integration | serial (`n_jobs=1`) |

```bash
.venv/bin/python scripts/latent_space_crosstalk_probe.py \
    --reps 6 --perms 49 --n-samples 160 --n-stages 4 --n-jobs 1 \
    --pls-repeats 2 --pls-cv2 4 --pls-cv1 3 --pls-max-components 8
```

**Resolution caveat.** 6 replicates → rejection-rate granularity of 1/6 ≈ 0.17; values of 0.17/0.33/0.50 are 1/6, 2/6, 3/6. These are *qualitative direction* reads, not power estimates. Read `pls`/`snf` relative to the `concat` baseline *on the same generator* — not against the Rung-2 test-bed numbers (the production path also carries generator coupling + `rev.logit`).

---

### Per-statistic rejection rates by latent space

| latent space | mode | delta | angle | shape | group-in-stage |
|--------------|------|-------|-------|-------|----------------|
| concat (baseline) | `none` | 0.17 | 0.00 | 0.17 | 0.14 |
| concat (baseline) | `magnitude` | **1.00** | 0.00 | 1.00 | 0.09 |
| concat (baseline) | `orientation` | 0.17 | **0.33** | 1.00 | 0.07 |
| concat (baseline) | `shape` | 1.00 | 0.67 | **1.00** | 0.18 |
| snf | `none` | 0.00 | 0.00 | 0.33 | 0.43 |
| snf | `magnitude` | **0.00** | 0.00 | 1.00 | 0.15 |
| snf | `orientation` | 0.00 | **0.00** | 1.00 | 0.53 |
| snf | `shape` | 0.00 | 0.00 | **1.00** | 0.53 |
| pls | `none` | 0.17 | 0.00 | 0.00 | 1.00 |
| pls | `magnitude` | **1.00** | 0.17 | 0.00 | 1.00 |
| pls | `orientation` | 0.33 | **0.33** | 1.00 | 1.00 |
| pls | `shape` | 1.00 | 0.50 | **1.00** | 1.00 |

---

### Finding 1 — The feared "magnitude → orientation" cross-talk is essentially absent; the real cross-talk is the **shape statistic**

Across *all three* latent spaces the magnitude→`angle` bleed is small (concat 0.00, snf 0.00, pls 0.17). The dominant off-target signal is **`shape`**:

- `magnitude` triggers `shape` (concat 1.00, snf 1.00) — a pure size change reads as a shape change.
- `orientation` triggers `shape` (1.00 in *every* latent space).
- `shape` triggers `delta` (concat 1.00, pls 1.00) and `angle` (concat 0.67, pls 0.50) — the shape mode is the least specific construction.

So the cross-talk the ladder has been chasing is predominantly **Procrustes-`shape` promiscuity**, and it is *largely invariant to the latent space* (it appears under concat, snf, and pls alike). That points the next rung at the `shape` estimator/test, not the projector.

### Finding 2 — PLS **removes** the magnitude→shape contamination (the part that *was* a baseline artifact)

The one cross-talk that the latent space *does* fix: under `concat`, a pure `magnitude` change rejects `shape` at 1.00; under **`pls` it drops to 0.00** (and `none`→`shape` also goes 0.17 → 0.00). The stage-conditioned, low-dimensional PLS space strips the size-driven shape artifact, giving a clean `magnitude`→`delta`=1.00 with `shape`=0.00. This is the concrete payoff of measuring in a real latent space rather than the concat baseline — but it only cures the *magnitude*-side shape contamination; `orientation`→`shape` and `shape`→`delta`/`angle` persist under PLS.

PLS otherwise matches the Rung-2 prediction: clean `none` null (`angle`=0.00, `shape`=0.00), strong `magnitude`→`delta` (1.00), only a mild `magnitude`→`angle` bleed (0.17). Orientation power is modest (`angle`=0.33, same as concat) — the predicted compression shows as weak-but-not-absent orientation sensitivity.

### Finding 3 — SNF has **no `delta`/`angle` power** and a saturated/anti-conservative `shape`

Under `snf`, `delta` and `angle` reject at 0.00 for *every* mode (including the true `magnitude`/`orientation` movers), while `shape` rejects at 1.00 everywhere — and even `none`→`shape`=0.33 (anti-conservative). The spectral embedding's metric is not commensurable with the injected size/orientation geometry (Rung 2's leak, end-to-end): it cannot detect magnitude or orientation at all, and its shape test is saturated. **SNF is unusable as the measurement latent space for `delta`/`angle`.**

### Finding 4 — Structural: PLS confines the group signal to the stage subspace (GIS = 1.00)

The group-in-stage fraction is ~0.1 under `concat` (the injected group difference is mostly *orthogonal* to the stage subspace in full feature space) but **1.00 under `pls`** for every mode. This is by construction: PLS builds the space from the stage label, so any measured group difference necessarily lies in the stage-relevant subspace. This is a double-edged property — it focuses the test on disease-relevant variation, but it also means group variation *orthogonal* to the stage axis is discarded before measurement. Worth keeping in view when interpreting PLS-space power.

---

### Gate decision

1. **The projector is not the dominant cross-talk source.** The magnitude→orientation leak that motivated the ladder is essentially absent on the real generator (≤0.17 everywhere). The dominant cross-talk is the **`shape` (Procrustes) statistic**, which is promiscuous across *all* latent spaces — `orientation`→`shape`=1.00 and `shape`→`delta`/`angle` are large under both `concat` and `pls`.

2. **PLS is the best production measurement space for size/orientation**, and it specifically removes the `magnitude`→`shape` artifact that the `concat` baseline manufactured. SNF is recorded as unusable for `delta`/`angle` (no power, saturated shape).

3. **Decision: Rung 4 = specificity of the Procrustes `shape` statistic.** Characterize why `orientation` rejects `shape` (1.00 everywhere) and why `shape` rejects `delta`/`angle`, isolating whether it is the GPA alignment, the shape-distance null, or the `n_stages≥3` trajectory construction. This is the largest residual cross-talk and — unlike the projector — it is *latent-space-invariant*, so it lives in the estimator/test, not the integration step. (The `concat`→`pls` shape fix for `magnitude` suggests dimensionality matters; Rung 4 should sweep the latent dimension as a secondary axis.)

**Exact reproduction:** seeds 0–5; `n_samples=160`, `n_stages=4`, `effect_size=1.0`, `p_dmp=0.2`, `permutations=49`; PLS CV `n_repeats=2, cv2=4, cv1=3, max_components=8`; serial; driver `scripts/latent_space_crosstalk_probe.py`. Coarse resolution (6 reps) — qualitative reads; a higher-replicate confirmation is advisable before the Rung-4 writeup leans on the exact rates.
