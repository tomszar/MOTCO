### Requirement: design module exposes public symbols
`motco.stats.design` SHALL export `center_matrix`, `get_model_matrix`, and `build_ls_means`.

#### Scenario: Direct import from design submodule
- **WHEN** a caller does `from motco.stats.design import get_model_matrix`
- **THEN** the import succeeds and the function is callable

#### Scenario: Flat namespace unchanged
- **WHEN** a caller does `from motco.stats import get_model_matrix`
- **THEN** the import succeeds and returns the same object as the direct submodule import

### Requirement: trajectory module exposes public symbols
`motco.stats.trajectory` SHALL export `pair_difference`, `estimate_difference`, `estimate_betas`, and `get_observed_vectors`.

#### Scenario: Direct import from trajectory submodule
- **WHEN** a caller does `from motco.stats.trajectory import estimate_difference`
- **THEN** the import succeeds and the function is callable

### Requirement: permutation module exposes public symbols
`motco.stats.permutation` SHALL export `RRPP`.

#### Scenario: Direct import from permutation submodule
- **WHEN** a caller does `from motco.stats.permutation import RRPP`
- **THEN** the import succeeds and the function is callable

### Requirement: sd module no longer exists
`motco.stats.sd` SHALL NOT be importable after this change.

#### Scenario: Direct import from sd fails
- **WHEN** a caller does `import motco.stats.sd`
- **THEN** an `ImportError` or `ModuleNotFoundError` is raised
