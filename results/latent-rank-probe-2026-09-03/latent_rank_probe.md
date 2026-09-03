# Latent-rank scaling probe

Run: 100 reps/cell, 199 perms, `n_samples=300`, `n_stages=4`, `effect_size=1.0`, `p_dmp=0.2`, `alpha=0.05`, `base_seed=400`, `surgery_censoring=clamp`, serial (`n_jobs=1`).

Matched seeds: every rank in a row measures the same generated datasets; only the retained latent rank varies (`integration_params["forced_components"]`).

## Per-rank latent shape response

| mode | rank | reps | observed shape (sd) | reject | pooled eigengap | population shape | excess |
|------|------|------|---------------------|--------|-----------------|------------------|--------|
| `none` | 3 | 100 | 0.0108 (0.0053) | 0.10 | 0.044 | 0.0000 | +0.0108 |
| `none` | 4 | 100 | 0.0068 (0.0024) | 0.00 | 0.044 | 0.0000 | +0.0068 |
| `none` | 6 | 100 | 0.0066 (0.0024) | 0.00 | 0.044 | 0.0000 | +0.0066 |
| `none` | 9 | 100 | 0.0065 (0.0024) | 0.00 | 0.044 | 0.0000 | +0.0065 |
| `none` | 12 | 100 | 0.0065 (0.0023) | 0.00 | 0.044 | 0.0000 | +0.0065 |
| `orientation` | 3 | 100 | 0.0694 (0.0245) | 0.97 | 0.044 | 0.0424 | +0.0270 |
| `orientation` | 4 | 100 | 0.0541 (0.0186) | 1.00 | 0.043 | 0.0424 | +0.0117 |
| `orientation` | 6 | 100 | 0.0195 (0.0070) | 0.36 | 0.043 | 0.0424 | -0.0229 |
| `orientation` | 9 | 100 | 0.0135 (0.0051) | 0.08 | 0.043 | 0.0424 | -0.0289 |
| `orientation` | 12 | 100 | 0.0136 (0.0051) | 0.09 | 0.043 | 0.0424 | -0.0288 |

`excess` is the latent response minus the joint standardized population value — the quantity that must shrink with rank if the response is a projection artifact.

## Decay measurement

- `orientation`: observed latent shape decays non-monotonically with rank — 0.0694 at rank 3 → 0.0136 at rank 12 (-80.4%). Rejection rate 0.97 → 0.09. It crosses the population value 0.0424 between rank 4 and rank 6.
- `none`: observed latent shape decays monotonically with rank — 0.0108 at rank 3 → 0.0065 at rank 12 (-39.5%). Rejection rate 0.10 → 0.00.

Reading: a **decaying** `orientation` response — falling toward (and possibly through) the population value as rank grows, with the rejection rate collapsing alongside it — supports the rank-limited-projection account: the orientation→shape response is an artifact of measuring a rotated trajectory in a rank-limited latent space, not a shape difference in the population. A response flat in rank does not support it. The `none` control shows how much of any decay is generic to rank rather than specific to the orientation surgery.

Caveat: the population column is measured in the standardized observed space, not in the latent space, so the two are not on a common scale — the crossing rank is indicative, and the load-bearing evidence is the decay and the rejection-rate collapse, not exact agreement with the population value.
