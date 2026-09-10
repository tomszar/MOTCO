# Magnitude-construction diagnostic — 2026-09-10

**Run:** `results/magnitude-construction-2026-09-10/` · **Driver:**
`scripts/magnitude_construction_diagnostic.py` · **Records read:**
`results/phase5-2026-09-10/merged.jsonl` (the paper-grade run, 9,500 units)

The [paper-grade Phase 5 run](phase5-paper-grade-2026-09-10.md) returned gate decision **HOLD** on the two
magnitude mandatory controls (magnitude/`angle` 0.276, magnitude/`shape` 0.742 against a 0.0695 bound). This
diagnostic establishes what causes those responses and whether they are fixable, so the Phase 5 exit review
can choose between a method revision and a claim revision on evidence.

**Both verdicts point the same way: method revision.** The magnitude surgery's off-target response is
entirely an artefact of scaling one omic block's δ and not the others, and a construction that scales all
three together is exactly size-pure in the measurement space. Nothing about the `angle` or `shape`
estimators is implicated.

## 1. Mechanism: the response exists only in the concatenation

`magnitude_kind='all'` (`semisynthetic.py`) scales `delta_methyl` alone and leaves `delta_expr` and
`delta_protein` at baseline, while the measurement space standardizes and concatenates all three omic blocks
(`evaluation.py`). The per-omic scopes recorded in the paper-grade run show the consequence directly
(`block_decomposition.csv`, checkpoint `population_standardized`, median over 500 replicates):

| statistic | scope | e = 0.25 | 0.50 | 0.75 | 1.00 | anchor |
|---|---|---|---|---|---|---|
| `angle` | methylation / expression / proteomics | **0.0000** | **0.0000** | **0.0000** | **0.0000** | 0.0 |
| `angle` | **joint** | 6.6978 | 12.1199 | 16.5890 | **20.3293** | 0.0 |
| `shape` | methylation / expression / proteomics | **0.0000** | **0.0000** | **0.0000** | **0.0000** | 0.0 |
| `shape` | **joint** | 0.0076 | 0.0138 | 0.0187 | **0.0226** | 0.0 |
| `delta` | methylation | 9.9148 | 18.1365 | 24.8473 | 30.3300 | 0.0 |
| `delta` | expression / proteomics | **0.0000** | **0.0000** | **0.0000** | **0.0000** | 0.0 |

`block_localization_summary.csv` records the verdict as `joint_only = True` for magnitude/`angle` and
magnitude/`shape` at `population_standardized`: every individual block is flat while the joint scope moves.

**Within each omic block the surgery is exactly size-pure.** The zeros are analytic, not small — `angle` is
0.0000 and `shape` lands at 8.4e-18, floating-point dust from the Procrustes routine. Only `delta` moves, and
only in the block whose δ was scaled. The off-target response is created by concatenation: lengthen one
component of a vector and leave the others, and the vector rotates. That is ordinary geometry, not a defect
in any estimator.

Across all four checkpoints the causal chain is fully attributable:

| checkpoint | per-block `angle` | joint `angle` |
|---|---|---|
| `population_native` | 0.0000 at every effect | *(no joint scope recorded)* |
| `population_standardized` | 0.0000 at every effect | 6.70 → 20.33 |
| `observed_standardized` | constant in effect (7.91 expression, 6.24 proteomics — a sampling floor) | 10.70 → 21.96 |
| `pls_latent` | *(joint only)* | 3.30 → 7.59 |

Two things follow that correct beliefs recorded in the repository:

- **It is not the methylation `rev.logit` nonlinearity.** `specificity.py` previously stated that magnitude
  bends shape through that nonlinearity. At `population_native` — before any standardization and before any
  sampling — the per-omic `angle` and `shape` are 0.0000 at every effect, so the nonlinearity does not bend
  the methylation trajectory at all. The module docstring is corrected in this change.
- **The PLS projection *attenuates* the rotation rather than creating it.** Joint `angle` is 20.33° at
  `population_standardized` and 7.59° at `pls_latent`. Whatever the latent space does here, it reduces the
  off-target geometry it is handed.

## 2. Constructibility: a size-pure magnitude change is realizable

The open question was whether scaling every omic's δ together survives per-block standardization, which
z-scores each block on pooled data and so divides back out part of what the scaling adds. It does
(`uniform_delta_comparison.csv`, population-standardized geometry, `n_stages = 4`):

| construction | e | joint `delta` | joint `angle` | joint `shape` | max block `angle` |
|---|---|---|---|---|---|
| production (`delta_methyl` only) | 0.25 | 5.9456 | 6.2898 | 1.095e-02 | 0.000e+00 |
| production | 1.00 | 18.1360 | **18.9097** | **3.288e-02** | 0.000e+00 |
| uniform probe (all three δ) | 0.25 | 14.9623 | **8.538e-07** | **9.558e-16** | 8.538e-07 |
| uniform probe | 1.00 | 44.7980 | **2.561e-06** | **9.995e-16** | 1.479e-06 |
| *anchor (e = 0, either)* | 0.00 | 0.0000 | 0.000e+00 | 1.008e-15 | 0.000e+00 |

The uniform-δ construction grows the trajectory strongly (joint `delta` 14.96 → 44.80) while holding joint
`angle` at 8.5e-07 → 2.6e-06 and joint `shape` at ~1e-15 — the anchor's own floating-point floor — at every
effect. Per-block standardization does not defeat it. **A genuinely size-only magnitude change is
constructible in this measurement space.**

One calibration caveat for whoever adopts it: the effect axis is not comparable between constructions. At
e = 1.00 the uniform probe reaches joint `delta` 44.80 against production's 18.14, because three blocks grow
instead of one. A corrected mode needs its own effect-size calibration before its power curve can be read
beside the existing one.

## 3. Shape-free confirmation

At two stages the trajectory is a single step and Procrustes `shape` is undefinable, so the orientation
response can be observed with no possibility of shape contaminating it
(`uniform_delta_comparison.csv`, `n_stages = 2`):

| n_stages | construction | e | joint `delta` | joint `angle` | joint `shape` |
|---|---|---|---|---|---|
| 2 | production | 1.00 | 5.4250 | **19.0470** | *(undefinable)* |
| 2 | uniform probe | 1.00 | 12.9072 | 1.000e-06 | *(undefinable)* |
| 4 | production | 1.00 | 18.1360 | **18.9097** | 3.288e-02 |
| 4 | uniform probe | 1.00 | 44.7980 | 2.561e-06 | 9.995e-16 |

The production construction's joint `angle` is 19.05° at two stages against 18.91° at four — indistinguishable.
The rotation is real and independent of the shape-removal step; it is not an artefact of Procrustes alignment
in the four-stage design. This was a confirmation of an already-analytic finding and it confirmed it.

## 4. Localization: the instrument, recalibrated

`localize_off_diagonal` normalizes `delta` by mean path length and `angle` by 180°, but passes `shape`
through raw. At e = 1.00 the joint values normalize to `angle` 20.3293/180 = 0.1129 (≥ the 0.05 cut → material)
and `shape` 0.0226 (< 0.05 → `not_material`) — while the RRPP `shape` test rejects at 0.742. A single absolute
cut cannot serve a statistic whose entire response range lies below it.

The recalibrated `null_dispersion` rule judges each excess in units of the zero-effect null's dispersion at
the same checkpoint. Where that dispersion is degenerate — zero or dust, as it is at every population
checkpoint for an analytically null construction (measured: `delta` sd exactly 0.0, `angle` sd 8.7e-07,
`shape` sd 3.5e-16) — it falls back to "excess exceeds the dust tolerance", the limit of the rule as the null
variance goes to zero. Each row records which path decided it.

Classifications under both rules (`localization_by_rule.csv`); 12 of 32 pairs change:

| mode / statistic | e | absolute rule | null-dispersion rule |
|---|---|---|---|
| magnitude / `angle` | 0.25–1.00 | `construction_present` | `construction_present` |
| **magnitude / `shape`** | 0.25–1.00 | **`not_material`** | **`construction_present`** |
| orientation / `delta` | 0.25–0.75 | `not_material` | `construction_present` |
| **orientation / `delta`** | 1.00 | **`projection_associated`** | **`construction_present`** |
| orientation / `shape` | 0.25 | `not_material` | `construction_present` |
| orientation / `shape` | 0.50–1.00 | `construction_present` | `construction_present` |
| shape / `angle` | 0.25–1.00 | `construction_present` | `construction_present` |
| shape / `delta` | 0.25–0.75 | `not_material` | `construction_present` |
| shape / `delta` | 1.00 | `construction_present` | `construction_present` |
| translation / `angle` | 0.25–1.00 | `not_material` | `not_material` |
| translation / `shape` | 0.25–1.00 | `not_material` | `not_material` |

Two properties of that table matter more than the individual rows:

- **The negative control is preserved.** Translation stays `not_material` on all eight rows under the
  recalibrated rule. The rule is calibrated, not merely permissive — its population off-target geometry is
  exactly zero, so nothing promotes it.
- **No response anywhere remains `projection_associated`.** The paper-grade run's single such
  classification, orientation/`delta` at e = 1.00, becomes `construction_present`. Under the absolute rule
  its population-standardized excess fell below the 0.05 cut, so localization skipped forward to
  `pls_latent`; once the population checkpoint is judged against its own (degenerate) null, the response is
  seen where it actually first appears.

**Interpretive shift this implies, stated plainly.** Because population checkpoints carry analytically zero
nulls, the recalibrated rule makes them very sensitive: any non-dust population-level off-target geometry now
localizes as `construction_present`. That is the right standard for a *localization* instrument — a
construction with any real off-target population geometry is impure, by definition, and the old cut was
hiding real impurity behind an arbitrary threshold. But it answers "is this construction exactly pure?", not
"is the impurity large enough to matter". The latter is read from the response magnitude, which the same
tables carry. The exit review should not read `construction_present` as a severity claim.

## 5. What this hands the exit review

| deviation from the paper-grade run | what this diagnostic establishes | revision implied |
|---|---|---|
| `mandatory_control[magnitude,angle]` 0.276 | Block-asymmetric δ scaling; exactly size-pure within every block; rotation confirmed shape-free at two stages; a uniform-δ construction is exactly pure in the joint space | **Method revision.** Correct the magnitude construction to scale all omics' δ together, recalibrate its effect axis, and re-measure. The `angle` estimator is not implicated. |
| `mandatory_control[magnitude,shape]` 0.742 | Same mechanism; and the response was real all along — the 0.05 absolute materiality cut mis-reported it as `not_material` | **Method revision**, same fix. The diagnostic prerequisite the paper-grade report named is discharged: the threshold is recalibrated and the pair now classifies. |
| orientation/`shape` predeclared "projection-associated" | Not supported. Orientation's response is present *within* each omic block (max block `angle` 90.08° against joint 89.93°; max block `shape` 0.108 against joint 0.059), and under the recalibrated rule no pair in the study is `projection_associated` | **Claim revision.** Drop the projection-associated framing for orientation; it is construction-present cross-talk of a per-omic feature permutation. No gate impact — both orientation off-diagonals are descriptive roles. |
| `power[orientation,angle]` monotonicity | Untouched by this diagnostic — it is a property of the `angle` null's pivotality, not of a construction | **Claim revision**, as the paper-grade report stated. Out of scope here. |

Nothing here adopts a revision. The uniform-δ construction is reachable only from the diagnostic entry point
(`generate_semisynthetic_trajectory(..., _probe_uniform_delta=True)`); it is not a `magnitude_kind` value, and
a study configuration naming it is refused at load time. Adopting it as the production magnitude mode is the
exit review's decision.

## 6. Reproduction

```bash
export LD_LIBRARY_PATH=/nix/store/61a1nwx3w6rqyaisj5rn1sal1981apm7-zlib-1.3.2/lib:$LD_LIBRARY_PATH
python scripts/magnitude_construction_diagnostic.py \
    --merged results/phase5-2026-09-10/merged.jsonl \
    --out-dir results/magnitude-construction-2026-09-10
```

Steps 1 and 2 read committed records only. Step 3 generates data but computes analytic population geometry —
no sampling, no RRPP, no PLS fit — so the whole diagnostic runs on a workstation in minutes with no cluster
and no R runtime. The `LD_LIBRARY_PATH` export is the local numpy/zlib workaround, not a project requirement.
