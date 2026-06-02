from __future__ import annotations

import numpy as np
import pytest

from motco.simulations import (
    InterSIMParams,
    SemiSyntheticTrajectoryParams,
    SimulationEvaluationParams,
    SimulationEvaluationResult,
    SimulationGrid,
    SimulationGridError,
    SimulationReplicateResult,
    SimulationRunConfig,
    append_replicate_results,
    derive_replicate_seed,
    enumerate_power_grid,
    enumerate_type_i_grid,
    make_simulation_cell,
    parameter_signature,
    read_replicate_results,
    rejection_indicator,
    run_simulation_grid,
    run_simulation_replicate,
    summarize_rejection_rates,
)


def baseline_intersim() -> InterSIMParams:
    return InterSIMParams(seed=1, n_sample=20, cluster_sample_prop=(0.5, 0.5))


def baseline_generator() -> SemiSyntheticTrajectoryParams:
    return SemiSyntheticTrajectoryParams(
        seed=2,
        trajectory_mode="magnitude",
        group_effect_size=0.2,
        group_ratio=0.5,
    )


def baseline_evaluation() -> SimulationEvaluationParams:
    return SimulationEvaluationParams(integration_method="concat", permutations=0, seed=3)


def fake_result(
    p_values: dict[str, float] | None = None,
    pair_statistics: dict[str, float] | None = None,
    truth_seed: int = 0,
) -> SimulationEvaluationResult:
    matrix = np.zeros((2, 2), dtype=float)
    return SimulationEvaluationResult(
        observed_deltas=matrix,
        observed_angles=matrix,
        observed_shapes=matrix,
        pair_statistics=pair_statistics or {"delta": 1.0, "angle": 2.0, "shape": float("nan")},
        p_values=p_values or {"delta": 0.01, "angle": 0.20},
        latent_matrix_metadata={"integration_method": "concat"},
        truth_metadata={"seed": truth_seed},
        runtime_metadata={"runtime_seconds": 0.1},
        evaluation_params=baseline_evaluation(),
        group_levels=["A", "B"],
        stage_levels=["0", "1"],
        contrast=[[0, 1], [2, 3]],
    )


def test_type_i_grid_enumeration_is_stable_and_null() -> None:
    axes = {
        "intersim.n_sample": [20, 40],
        "generator.group_ratio": [0.5, 0.7],
        "evaluation.permutations": [0, 2],
    }
    first = enumerate_type_i_grid(
        baseline_intersim_params=baseline_intersim(),
        baseline_generator_params=baseline_generator(),
        evaluation_params=baseline_evaluation(),
        axes=axes,
        n_replicates=2,
        base_seed=101,
    )
    second = enumerate_type_i_grid(
        baseline_intersim_params=baseline_intersim(),
        baseline_generator_params=baseline_generator(),
        evaluation_params=baseline_evaluation(),
        axes=axes,
        n_replicates=2,
        base_seed=101,
    )

    assert [cell.cell_id for cell in first.cells] == [cell.cell_id for cell in second.cells]
    assert len(first.cells) == 4
    assert first.cells[0].phase == "type_i_baseline"
    assert all(cell.generator_params.trajectory_mode == "none" for cell in first.cells)
    assert all(cell.generator_params.group_effect_size == 0.0 for cell in first.cells)
    assert {cell.metadata["varied_axis"] for cell in first.cells[1:]} == {
        "intersim.n_sample",
        "generator.group_ratio",
        "evaluation.permutations",
    }


def test_power_grid_enumeration_includes_modes_effects_and_axes() -> None:
    grid = enumerate_power_grid(
        baseline_intersim_params=baseline_intersim(),
        baseline_generator_params=baseline_generator(),
        evaluation_params=baseline_evaluation(),
        trajectory_modes=["magnitude", "orientation"],
        effect_sizes=[0.1, 0.2],
        axes={"intersim.n_sample": [20, 40]},
    )

    assert len(grid.cells) == 8
    primary = [cell for cell in grid.cells if cell.phase == "power_primary"]
    ofat = [cell for cell in grid.cells if cell.phase == "power_ofat"]
    assert len(primary) == 4
    assert len(ofat) == 4
    assert {(cell.generator_params.trajectory_mode, cell.generator_params.group_effect_size) for cell in primary} == {
        ("magnitude", 0.1),
        ("magnitude", 0.2),
        ("orientation", 0.1),
        ("orientation", 0.2),
    }


def test_invalid_grid_inputs_are_rejected() -> None:
    with pytest.raises(SimulationGridError, match="n_replicates"):
        make_simulation_cell(
            phase="type_i_baseline",
            intersim_params=baseline_intersim(),
            generator_params=baseline_generator(),
            n_replicates=0,
        )

    with pytest.raises(SimulationGridError, match="namespace prefix"):
        enumerate_type_i_grid(
            baseline_intersim_params=baseline_intersim(),
            baseline_generator_params=baseline_generator(),
            axes={"n_sample": [20, 40]},
        )


def test_replicate_seed_derivation_is_deterministic() -> None:
    cell = make_simulation_cell(
        phase="type_i_baseline",
        intersim_params=baseline_intersim(),
        generator_params=baseline_generator(),
        n_replicates=2,
        base_seed=99,
    )

    assert derive_replicate_seed(cell, 0) == derive_replicate_seed(cell, 0)
    assert derive_replicate_seed(cell, 0) != derive_replicate_seed(cell, 1)
    with pytest.raises(SimulationGridError, match="replicate_index"):
        derive_replicate_seed(cell, 2)


def test_replicate_seed_always_fits_r_signed_int32() -> None:
    R_MAX = 2**31 - 1

    seeds: list[int] = []
    for base_seed in (0, 1, 42, 100, 2_797_983_684, -1):
        for cell_index in range(8):
            cell = make_simulation_cell(
                cell_id=f"probe-{cell_index}",
                phase="power_primary",
                intersim_params=baseline_intersim(),
                generator_params=baseline_generator(),
                n_replicates=16,
                base_seed=base_seed,
            )
            for replicate_index in range(cell.n_replicates):
                seeds.append(derive_replicate_seed(cell, replicate_index))

    assert all(0 <= s <= R_MAX for s in seeds)
    # Sanity: at least some draws land in the previously-overflowing high half
    # under the *unmasked* 32-bit derivation, so the test actually exercises
    # the mask rather than coincidentally staying small.
    assert max(seeds) > 2**30


def test_replicate_seed_masks_known_pre_fix_value() -> None:
    # Smoke-run failure surfaced an unmasked seed of 2_797_983_684 (high bit set,
    # = 0xa6b8f084). Clearing the high bit gives 0x26b8f084 = 650_500_036,
    # which fits R's signed-32-bit range.
    unmasked = 2_797_983_684
    assert unmasked & 0x7FFFFFFF == 650_500_036
    assert 650_500_036 <= 2**31 - 1


def test_parameter_signature_includes_seed_derivation_version(monkeypatch: pytest.MonkeyPatch) -> None:
    import motco.simulations.grid as grid_module

    cell = make_simulation_cell(
        phase="type_i_baseline",
        intersim_params=baseline_intersim(),
        generator_params=baseline_generator(),
        n_replicates=2,
        base_seed=77,
    )
    current_signature = parameter_signature(cell)

    # Reproduce the signature is stable across calls at the current version.
    assert parameter_signature(cell) == current_signature

    # Strip the derivation-version field from the payload before digesting;
    # the resulting "legacy" signature must differ from the current one. This
    # is the property the resume guard relies on to invalidate pre-fix shards.
    original_stable_digest = grid_module._stable_digest

    def stable_digest_without_version(payload: object, *, length: int | None = None) -> str:
        if isinstance(payload, dict) and "seed_derivation_version" in payload:
            payload = {k: v for k, v in payload.items() if k != "seed_derivation_version"}
        return original_stable_digest(payload, length=length)

    monkeypatch.setattr(grid_module, "_stable_digest", stable_digest_without_version)
    assert parameter_signature(cell) != current_signature


def test_run_replicate_uses_injectable_evaluator_and_records_seeds() -> None:
    cell = make_simulation_cell(
        phase="type_i_baseline",
        intersim_params=baseline_intersim(),
        generator_params=baseline_generator(),
        evaluation_params=baseline_evaluation(),
    )

    def evaluator(
        intersim_params: InterSIMParams,
        generator_params: SemiSyntheticTrajectoryParams,
        evaluation_params: SimulationEvaluationParams,
    ) -> SimulationEvaluationResult:
        assert intersim_params.seed == generator_params.seed
        assert evaluation_params.seed == 3
        return fake_result(truth_seed=generator_params.seed)

    record = run_simulation_replicate(cell, 0, evaluator=evaluator)

    assert record.status == "completed"
    assert record.intersim_seed == record.generator_seed
    assert record.evaluation_seed == 3
    assert record.truth_metadata["seed"] == record.replicate_seed
    assert record.parameter_signature == parameter_signature(cell)


def test_run_replicate_error_policy_records_failures() -> None:
    cell = make_simulation_cell(
        phase="type_i_baseline",
        intersim_params=baseline_intersim(),
        generator_params=baseline_generator(),
    )

    def evaluator(
        intersim_params: InterSIMParams,
        generator_params: SemiSyntheticTrajectoryParams,
        evaluation_params: SimulationEvaluationParams,
    ) -> SimulationEvaluationResult:
        raise RuntimeError("boom")

    record = run_simulation_replicate(cell, 0, evaluator=evaluator, error_policy="record")

    assert record.status == "failed"
    assert record.error_type == "RuntimeError"
    assert record.error_message == "boom"


def test_jsonl_persistence_read_and_resume(tmp_path) -> None:
    path = tmp_path / "results.jsonl"
    cell = make_simulation_cell(
        phase="type_i_baseline",
        intersim_params=baseline_intersim(),
        generator_params=baseline_generator(),
        n_replicates=2,
    )
    grid = SimulationGrid(cells=(cell,))
    calls = 0

    def evaluator(
        intersim_params: InterSIMParams,
        generator_params: SemiSyntheticTrajectoryParams,
        evaluation_params: SimulationEvaluationParams,
    ) -> SimulationEvaluationResult:
        nonlocal calls
        calls += 1
        return fake_result(truth_seed=generator_params.seed)

    first = run_simulation_grid(grid, config=SimulationRunConfig(output_path=path), evaluator=evaluator)
    second = run_simulation_grid(grid, config=SimulationRunConfig(output_path=path), evaluator=evaluator)
    loaded = read_replicate_results(path)

    assert len(first) == 2
    assert second == []
    assert calls == 2
    assert len(loaded) == 2
    assert {record.replicate_index for record in loaded} == {0, 1}


def test_resume_detects_parameter_mismatch(tmp_path) -> None:
    path = tmp_path / "results.jsonl"
    cell = make_simulation_cell(
        phase="type_i_baseline",
        intersim_params=baseline_intersim(),
        generator_params=baseline_generator(),
        cell_id="shared",
    )
    append_replicate_results(
        path,
        [
            run_simulation_replicate(
                cell,
                0,
                evaluator=lambda intersim, generator, evaluation: fake_result(),
            )
        ],
    )
    changed = make_simulation_cell(
        phase="type_i_baseline",
        intersim_params=baseline_intersim(),
        generator_params=SemiSyntheticTrajectoryParams(seed=2, group_ratio=0.7),
        cell_id="shared",
    )

    with pytest.raises(SimulationGridError, match="different parameter signature"):
        run_simulation_grid(SimulationGrid(cells=(changed,)), config=SimulationRunConfig(output_path=path))


def test_rejection_summaries_handle_available_and_missing_statistics() -> None:
    records = [
        SimulationReplicateResult(
            cell_id="cell-a",
            phase="type_i_baseline",
            replicate_index=0,
            replicate_seed=1,
            intersim_seed=1,
            generator_seed=1,
            evaluation_seed=1,
            parameter_signature="sig",
            status="completed",
            p_values={"delta": 0.01, "angle": 0.20},
        ),
        SimulationReplicateResult(
            cell_id="cell-a",
            phase="type_i_baseline",
            replicate_index=1,
            replicate_seed=2,
            intersim_seed=2,
            generator_seed=2,
            evaluation_seed=2,
            parameter_signature="sig",
            status="completed",
            p_values={"delta": 0.10, "angle": 0.03},
        ),
    ]

    summaries = summarize_rejection_rates(records, alpha=0.05, statistics=("delta", "angle", "shape"))
    by_stat = {summary.statistic: summary for summary in summaries}

    assert rejection_indicator(0.01, alpha=0.05)
    assert rejection_indicator(None, alpha=0.05) is None
    assert by_stat["delta"].rejection_rate == 0.5
    assert by_stat["delta"].monte_carlo_se == pytest.approx((0.5 * 0.5 / 2) ** 0.5)
    assert by_stat["angle"].rejected_replicates == 1
    assert by_stat["shape"].available_replicates == 0
    assert by_stat["shape"].rejection_rate is None
    assert by_stat["shape"].unavailable_replicates == 2
