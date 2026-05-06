## Context

The current simulation mockup lives in `src/simulations`, outside the packaged `src/motco` namespace. It sketches a future grid runner and trajectory injection strategy, but its first unresolved dependency is basic InterSIM invocation. InterSIM is an R package that returns three aligned omics matrices (`dat.methyl`, `dat.expr`, `dat.protein`) and a clustering assignment table. MOTCO needs those outputs in Python before downstream trajectory-specific simulation work can be designed safely.

This change creates a narrow bridge: Python calls R InterSIM, normalizes the result into MOTCO-owned Python structures, and verifies that sample rows and cluster metadata remain aligned.

## Goals / Non-Goals

**Goals:**

- Provide a Python API for invoking InterSIM with explicit simulation parameters and seed
- Return methylation, gene expression, protein expression, sample IDs, cluster assignments, and run metadata
- Keep InterSIM as an optional external dependency with clear availability checks and skip-friendly tests
- Place new code under the packaged `motco` namespace so it can be imported and tested consistently
- Establish the data contract future trajectory simulation code will consume

**Non-Goals:**

- Inject MOTCO trajectory effects into the simulated matrices
- Build the full simulation grid runner or batch orchestration layer
- Estimate power, Type I error, or summary reports
- Add a public CLI command for simulations
- Support arbitrary feature counts beyond what InterSIM returns natively
- Reimplement InterSIM in Python

## Decisions

### Use an `Rscript` subprocess bridge for the first implementation

Python will invoke a small package-owned R helper script with `subprocess.run`. The helper will load `InterSIM`, call `set.seed(seed)`, run `InterSIM::InterSIM(...)`, and write output files into a temporary directory. Python will read those files into pandas DataFrames and return a typed result object.

Rationale: this avoids adding `rpy2` as a Python runtime dependency and keeps R failures isolated in a child process. It also works in environments where R is available but embedding R into Python is not configured.

Alternative considered: `rpy2` in-process calls. This gives direct object conversion, but it adds a compiled Python dependency and tighter coupling to the local R installation. That is too much surface area for the first proof.

### Use file-based exchange, not stdout JSON for matrices

The R helper will write separate CSV files for methylation, expression, protein, and cluster assignment. Stdout will be reserved for status messages or remain quiet. Python will parse the CSV files and construct the result.

Rationale: matrices can be large enough that stdout JSON becomes brittle and hard to debug. Separate files make partial failures inspectable and avoid requiring an extra R JSON package.

Alternative considered: serialize the full result as JSON on stdout. Rejected because it requires additional R-side JSON handling and is less robust for numeric matrices.

### Introduce a small Python result model

The Python API should expose a result object equivalent to:

```python
InterSIMResult(
    methylation=pd.DataFrame,
    expression=pd.DataFrame,
    proteomics=pd.DataFrame,
    sample_ids=pd.Index,
    clusters=pd.Series,
    metadata=dict,
)
```

Rows MUST be aligned across all matrices and metadata. The object can be a dataclass to keep the API simple and inspectable.

### Support InterSIM's native parameter surface first

The initial parameter model will map directly to InterSIM arguments:

- `n_sample`
- `cluster_sample_prop`
- `delta_methyl`
- `delta_expr`
- `delta_protein`
- `p_dmp`
- `p_deg`
- `p_dep`
- `sigma_methyl`
- `sigma_expr`
- `sigma_protein`
- `cor_methyl_expr`
- `cor_expr_protein`
- `seed`

The wrapper can expose Pythonic names while translating to R names internally. Parameters not supplied by Python should use InterSIM defaults.

### Make dependency checks explicit

The bridge will include a lightweight availability check that verifies `Rscript` is on `PATH` and that `requireNamespace("InterSIM", quietly = TRUE)` succeeds. Missing dependencies should raise a MOTCO-owned exception with installation guidance.

Tests that require the real R package should use this availability check and skip when unavailable. Pure Python normalization tests should not require R.

## Risks / Trade-offs

- **R package missing in developer or CI environments** -> Provide clear availability checks and mark integration smoke tests as skippable when InterSIM is absent.
- **CSV exchange is slower than direct memory conversion** -> Acceptable for the proof stage; InterSIM generation and downstream RRPP dominate expected runtime.
- **InterSIM output feature counts are fixed by package data** -> Do not expose arbitrary `features_per_omic` yet; future generators can subset/resample after the bridge is stable.
- **R helper script path may be fragile after packaging** -> Store the helper under the packaged `motco` namespace and resolve it with `importlib.resources`.
- **Randomness spans Python and R** -> Treat R's `set.seed(seed)` as authoritative for InterSIM generation; record the seed in metadata.

## Migration Plan

This is additive. No existing MOTCO API changes are required. The current `src/simulations` mockup can remain untouched during implementation or be moved later once the bridge is proven.
