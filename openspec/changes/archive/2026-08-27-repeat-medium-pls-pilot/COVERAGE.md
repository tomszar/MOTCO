# Requirement scenario coverage

Every scenario in this change's delta specs, mapped to the task, test, structured output, or findings
section that covers it. Verified 2026-08-27 against a completed 1,900-unit Phase 4 pilot.

## `simulation-evaluation-harness`

| Scenario | Covered by |
|---|---|
| Attribution is enabled for PLS | Tasks 2.1, 3.1. `evaluation._attribution_diagnostics`; `tests/test_evaluation_attribution.py::test_attribution_receives_the_estimator_used_for_measurement` (asserts `model.x_scores_` *is* the latent matrix, so no second fit exists) |
| Attribution is disabled | Task 1.1. `tests/test_evaluation_attribution.py::test_disabled_and_enabled_evaluations_agree`, `::test_p_values_match_with_attribution_enabled`, `::test_disabled_evaluation_records_not_requested` |
| Attribution is requested for a non-PLS method | Task 1.1. `evaluation._validate_attribution_settings`; `tests/test_evaluation_attribution.py::test_attribution_rejected_for_non_pls_methods` (concat and snf), `tests/test_study_attribution_config.py::test_study_config_rejects_attribution_without_pls` |
| Compact diagnostic is produced | Tasks 3.1, 3.4. `attribution_diagnostics.compute_attribution_diagnostics`; `tests/test_evaluation_attribution.py::test_attribution_record_is_bounded_and_versioned`, `::test_top_feature_records_carry_both_unit_bases`; `tests/test_attribution_diagnostics.py::test_top_k_truncates_and_never_exceeds_the_feature_count`. Output: `report/phase4_attribution.csv`; unit basis recorded as `units.methylation_units = "mvalue"` |
| Generator truth is available | Task 3.2. `attribution_diagnostics.derive_truth_driver_features`; `tests/test_attribution_diagnostics.py::test_truth_includes_propagated_expression_and_protein_drivers`, `::test_truth_covers_effect_size_only_constructions`. Findings: "Attribution diagnostics" (423 truth drivers/replicate, top-20 precision 1.00) |
| A transition is unavailable | Task 3.1. `_retention` returns `available: False` with `None` metrics; `tests/test_attribution_diagnostics.py::test_unavailable_transition_is_marked_not_zero` (path length preserved, direction unavailable) |
| Scores and attribution share one final fit | Tasks 2.1, 2.3. `_pls_integration` fits once via `fit_plsda_model`; `tests/test_evaluation_attribution.py::test_disabled_and_enabled_evaluations_agree` (identical selected LV, statistics, p-values) |

## `trajectory-power-study`

| Scenario | Covered by |
|---|---|
| Phase 4 config is loaded | Task 6.3. `examples/trajectory_power_study/phase4_pilot_100x199.json`; `tests/test_phase4_end_to_end.py::test_shards_cover_every_expected_work_unit`. Run: 19 cells, 1,900 units, no R |
| Primary cells use matched seeds | Tasks 4.1, 4.2. `grid.derive_replicate_seed` seed-family branch; `tests/test_matched_seeds_persistence.py::test_matched_seeds_are_shared_across_mode_and_effect_cells`, `::test_seed_family_change_changes_the_signature`, `tests/test_phase4_end_to_end.py::test_matched_seeds_pair_primary_cells_in_the_run` |
| Zero-effect cells are not duplicated across modes | Task 4.1a, 4.6. `enumerate._zero_effect_anchor_cell`, `_require_distinct_primary_datasets`; `tests/test_matched_seeds_persistence.py::test_single_zero_effect_anchor_is_emitted`, `::test_every_mode_resolves_its_zero_point_to_the_anchor`, `::test_no_two_primary_cells_generate_identical_datasets` |
| Gate parameters come from the configuration | Task 1.2, 5.5. `Phase4GateConfig` supplies every threshold; `phase4.py` hard-codes none. `tests/test_study_attribution_config.py::test_gate_requires_complete_parameters`; `tests/test_study_phase4.py` drives every outcome by varying config alone |
| Orientation diagnostics are selected without significance conditioning | Task 1.3. `AttributionSelector.selects` runs at enumeration; `tests/test_study_attribution_config.py::test_selector_enables_only_nonzero_primary_orientation_cells`, `::test_selection_does_not_depend_on_observed_p_values`. Run: 400 eligible replicates, all computed |
| Phase 4 replicate round-trips | Tasks 4.3, 4.4. `tests/test_matched_seeds_persistence.py::test_completed_record_carries_integration_and_attribution`, `tests/test_phase4_end_to_end.py::test_merged_records_round_trip` |
| Ineligible cell is persisted | Task 4.3. `attribution_status = "not_requested"`; `tests/test_matched_seeds_persistence.py::test_ineligible_cell_is_marked_not_requested`, `tests/test_phase4_end_to_end.py::test_only_selected_orientation_cells_carry_attribution` |
| Legacy record is loaded | Task 4.4. `tests/test_matched_seeds_persistence.py::test_legacy_records_load_with_empty_additive_fields`, `::test_legacy_signature_cannot_satisfy_a_phase_4_resume` |
| Geometry-aware operating report is produced | Tasks 5.1, 5.4, 6.1. `report/phase4_geometry.csv`, `report/phase4_operating.csv`, `report/phase4_localization.csv`. Findings: "Why orientation is flat", "Off-diagonal responses, localized" |
| Measurement spaces are compared | Task 5.4. `MEASUREMENT_SPACES` labels every geometry row; `localize_off_diagonal` normalizes per checkpoint (delta by path length, angle by 180°) and compares only against that checkpoint's own null. `tests/test_study_phase4.py::test_localization_never_compares_raw_distances_across_spaces` |
| PLS and attribution stability are reported | Tasks 5.2, 5.3. `report/phase4_pls_selection.csv`, `report/phase4_attribution.csv`; `tests/test_study_phase4.py::test_pls_selection_summary_reports_components_and_missing`, `::test_attribution_summary_reports_agreement_and_availability`. Findings: "PLS representation", "Attribution diagnostics" |
| Type I inflation gate is evaluated | Task 5.5. `evaluate_type_i_inflation` over `none` + every translation level. Run: 18 tests, all passed. Findings: "Type I control" |
| A single control statistic is marginally above its bound | Task 5.6. `_is_marginal` + `max_marginal_exceedances`; `tests/test_study_phase4.py::test_single_marginal_control_exceedance_is_indeterminate`, `::test_two_marginal_exceedances_hold`, `::test_material_control_exceedance_holds`. Not engaged in this run (no exceedance occurred) |
| Magnitude gate is evaluated | Task 5.5. `evaluate_power_rule` + `evaluate_control_rule`; `tests/test_study_phase4.py::test_magnitude_off_diagonal_control_failure_holds`, `::test_small_power_reversal_is_tolerated`, `::test_material_power_reversal_holds`. Findings: "Magnitude" |
| Orientation and shape gates are evaluated | Task 5.5. Findings: "Shape", "Orientation — the failing gate", "Off-diagonal responses, localized". `tests/test_study_phase4.py::test_mixed_construction_off_diagonals_are_descriptive_only` proves descriptive pairs never gate |
| Diagnostic completeness gate is evaluated | Task 5.6. `evaluate_completeness`; `tests/test_study_phase4.py::test_incomplete_work_units_block_proceed`, `::test_missing_integration_metadata_blocks_proceed`, `::test_eligible_attribution_without_result_blocks_proceed`, `::test_recorded_attribution_failure_is_acceptable` |
| A mandatory gate fails or lacks evidence | Task 5.6. `tests/test_study_phase4.py::test_power_floor_failure_holds` (hold), `::test_unavailable_statistic_yields_indeterminate_not_proceed` (indeterminate). Run: HOLD on `mandatory_power[orientation,angle]` |
| Findings report is committed | Tasks 8.4, 8.5. `docs/reports/phase4-medium-pls-pilot-2026-08-27.md` with decision, interpretation, limitations, and exact shard/merge/report commands; every claim cites a CSV/JSON under `results/phase4-2026-08-27/report/`. July report unchanged and labeled superseded in `mkdocs.yml` and `docs/roadmap.md` |

## Deviations from the design worth recording

- **Design decision 5** states parallel worker counts "do not alter statistical output". RRPP seeds one RNG
  stream per worker, so `n_jobs` changes the realized permutation draws; it is part of `evaluation_params`
  and therefore of the cell signature. The sbatch script now forwards `--n-jobs` only when `STUDY_N_JOBS` is
  set, the runner warns on override, and the pilot ran at the config's `n_jobs=1`.
- **Design decision 6** localization was specified without a materiality rule. A ratio against the
  zero-effect null is unusable — an exactly-null construction leaves the anchor angle near `1e-8`. It is
  implemented as a scale-free *excess* instead (delta normalized by path length, angle by its 180° maximum),
  still compared only within a checkpoint.
