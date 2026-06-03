## ADDED Requirements

### Requirement: fit_plsda_model function
`stats/pls.py` SHALL expose a public function `fit_plsda_model(X, y, n_components) -> PLSRegression` that fits a `PLSRegression(n_components=n_components, scale=True)` on the full arrays `X` and one-hot-encoded `y` (matching `fit_plsda_transform`/`plsda_doubleCV` encoding) and returns the fitted estimator. No cross-validation is performed.

#### Scenario: Returns a fitted projector
- **WHEN** called with valid `X`, `y`, and `n_components=k`
- **THEN** the return value is a fitted `PLSRegression` whose `.transform()` maps an `(m, p)` feature matrix to `(m, k)` scores

#### Scenario: fit_plsda_transform is consistent with fit_plsda_model
- **WHEN** `fit_plsda_transform(X, y, k)` and `fit_plsda_model(X, y, k).transform(X)` are computed
- **THEN** the two score matrices are equal

#### Scenario: Exported from the stats package
- **WHEN** a caller writes `from motco.stats import fit_plsda_model`
- **THEN** the import succeeds
