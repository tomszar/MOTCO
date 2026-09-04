"""The baseline stage-program continuity axis (``baseline_continuity``, ρ).

Group A's per-stage methylation indicators are drawn along a stationary
first-order Markov chain rather than independently per stage. The axis exists to
give the baseline trajectory *a direction to differ in*: at ρ = 0 the stage
means form a near-regular simplex whose PC1 is ill-determined, and the geometry
audit (``docs/reports/geometry-audit-2026-09-01.md``, F1/F5) traced the
orientation power shortfall to that anisotropy. The contracts pinned here are:

- the per-stage Bernoulli(``p_dmp``) marginal is preserved at *every* ρ, so
  comparisons across the axis isolate geometry rather than abundance;
- ρ = 0 reproduces the pre-axis generator byte-identically;
- the analytic surgery headroom follows the continuity-adjusted union
  probability, so enumeration's fail-loud guarantee carries along the axis.

``PINNED_PRECHANGE`` was recorded against the generator as it stood before the
axis existed (commit ``e46a512``). Like ``test_effect_axis_censoring``'s pins it
hashes the differential **indicators** — produced by RNG draws alone, so exact
on every machine — rather than the sampled matrices, whose last bits depend on
the BLAS behind the MVN factorization. The indicators pin the RNG stream through
the surgery; everything drawn afterwards consumes an unchanged stream in an
unchanged order.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from motco.simulations import (
    SemiSyntheticTrajectoryError,
    SemiSyntheticTrajectoryParams,
    SimulationEvaluationParams,
    expected_stage_active_fraction,
    expected_surgery_headroom,
    generate_semisynthetic_trajectory,
    load_reference,
    make_simulation_cell,
    parameter_signature,
)
from motco.simulations.generator import (
    GeneratorError,
    bernoulli_indicators,
    markov_indicators,
)
from motco.simulations.grid import SimulationReplicateResult
from motco.simulations.semisynthetic import list_trajectory_modes
from motco.simulations.study import (
    AcceptanceTargets,
    PowerMonotonicityTarget,
    SpecificityTarget,
    StudyConfig,
    StudyConfigError,
    TypeIControlTarget,
    enumerate_study,
)
from motco.simulations.study.report import build_report_frames, write_report_csvs
from motco.simulations.study.spectrum import (
    record_continuity,
    resolve_orientation_by_continuity,
)


@pytest.fixture(scope="module")
def reference():
    return load_reference()


def params(**overrides) -> SemiSyntheticTrajectoryParams:
    base: dict = dict(seed=11, n_samples=240, n_stages=4, p_dmp=0.2)
    base.update(overrides)
    return SemiSyntheticTrajectoryParams(**base)  # type: ignore[arg-type]


def indicator_digest(dataset) -> str:
    """Machine-stable digest of both groups' differential indicators."""

    hasher = hashlib.sha256()
    for group in dataset.truth["group_labels"]:
        for layer in ("methylation", "expression", "proteomics"):
            values = np.ascontiguousarray(
                dataset.truth["indicators"][group][layer], dtype=float
            )
            hasher.update(str(values.shape).encode())
            hasher.update(values.tobytes())
    return hasher.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# 1. The Markov mechanism
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("rho", [0.0, 0.5, 0.9])
def test_markov_indicators_preserve_the_per_stage_marginal(rho: float) -> None:
    """Stationarity is the contract that makes ρ a pure-geometry knob."""

    indicators = markov_indicators(np.random.default_rng(3), 200_000, 5, 0.2, rho)

    assert indicators.shape == (200_000, 5)
    assert set(np.unique(indicators)) <= {0.0, 1.0}
    # 3 SE of a Bernoulli(0.2) mean over 200k draws is ~0.0027; the slack below
    # covers the extra dependence-driven spread at high rho.
    assert np.allclose(indicators.mean(0), 0.2, atol=0.006)


@pytest.mark.parametrize("rho", [0.0, 0.5, 0.9])
def test_markov_indicators_have_the_declared_cross_stage_correlation(rho: float) -> None:
    """``corr(x_t, x_s) = rho ** |t - s|`` is what orders the stage means."""

    indicators = markov_indicators(np.random.default_rng(4), 200_000, 4, 0.2, rho)

    for lag in (1, 2, 3):
        observed = [
            float(np.corrcoef(indicators[:, t], indicators[:, t + lag])[0, 1])
            for t in range(4 - lag)
        ]
        assert np.allclose(observed, rho**lag, atol=0.01), f"lag {lag}: {observed}"


def test_zero_continuity_is_the_independent_draw_bit_for_bit() -> None:
    """Same block, same shape, same stream position — so nothing downstream shifts."""

    independent = bernoulli_indicators(np.random.default_rng(7), 5_000, 4, 0.2)
    markov = markov_indicators(np.random.default_rng(7), 5_000, 4, 0.2, 0.0)

    assert np.array_equal(independent, markov)

    # The stream is also left in the same place, which is what carries the
    # byte-identity past the indicators into the sampled matrices.
    left, right = np.random.default_rng(7), np.random.default_rng(7)
    bernoulli_indicators(left, 5_000, 4, 0.2)
    markov_indicators(right, 5_000, 4, 0.2, 0.0)
    assert np.array_equal(left.random(16), right.random(16))


@pytest.mark.parametrize("rho", [-0.01, 1.0, 1.5])
def test_markov_indicators_reject_out_of_range_continuity(rho: float) -> None:
    with pytest.raises(GeneratorError, match=r"rho must be in \[0, 1\)"):
        markov_indicators(np.random.default_rng(0), 10, 3, 0.2, rho)


# --------------------------------------------------------------------------- #
# 2. The generator parameter
# --------------------------------------------------------------------------- #


def test_default_continuity_is_the_isotropic_endpoint() -> None:
    assert SemiSyntheticTrajectoryParams(seed=0).baseline_continuity == 0.0


@pytest.mark.parametrize("value", [-0.1, 1.0, 2.0])
def test_out_of_range_continuity_is_rejected(value: float, reference) -> None:
    with pytest.raises(SemiSyntheticTrajectoryError, match=r"baseline_continuity"):
        generate_semisynthetic_trajectory(
            params(baseline_continuity=value), reference=reference
        )


@pytest.mark.parametrize("mode", list_trajectory_modes())
def test_every_trajectory_mode_generates_at_high_continuity(mode: str, reference) -> None:
    """The axis lives in the baseline, so it composes with every surgery."""

    dataset = generate_semisynthetic_trajectory(
        params(trajectory_mode=mode, group_effect_size=0.25, baseline_continuity=0.9),
        reference=reference,
    )

    assert dataset.methylation.shape[0] == 240
    assert dataset.truth["baseline_continuity"] == 0.9
    assert not dataset.truth["transform"].get("censored", False)


@pytest.mark.parametrize("rho", [0.0, 0.6])
def test_continuity_is_recorded_as_truth(rho: float, reference) -> None:
    dataset = generate_semisynthetic_trajectory(
        params(baseline_continuity=rho), reference=reference
    )

    assert dataset.truth["baseline_continuity"] == rho


def test_per_stage_indicator_counts_do_not_track_continuity(reference) -> None:
    """Preserved marginals mean the axis moves geometry, not abundance."""

    counts = {}
    for rho in (0.0, 0.9):
        dataset = generate_semisynthetic_trajectory(
            params(baseline_continuity=rho, seed=23), reference=reference
        )
        counts[rho] = np.array(dataset.truth["indicator_counts"]["A"]["methylation"])

    expected = 0.2 * reference.n_cpg
    for rho, observed in counts.items():
        # ~4 SD of Binomial(n_cpg, 0.2) at rho=0; the high-rho draws share the
        # same marginal but are correlated across stages, not within one.
        assert np.all(np.abs(observed - expected) < 0.25 * expected), f"rho={rho}: {observed}"


# --------------------------------------------------------------------------- #
# 3. Byte-identity at rho = 0 against the pre-axis generator
# --------------------------------------------------------------------------- #

#: (params overrides, indicator digest, group-A per-stage methylation counts,
#: transform truth) recorded on ``e46a512``, before ``baseline_continuity``
#: existed.
PINNED_PRECHANGE: tuple[tuple[dict, str, list[int], dict], ...] = (
    ({"trajectory_mode": "none"}, "77859bd89c6c3974", [82, 75, 63, 80], {}),
    (
        {"trajectory_mode": "orientation", "group_effect_size": 0.3},
        "1cc7c03fa01902ce",
        [82, 75, 63, 80],
        {"censored": False, "orientation_nominal": 65, "orientation_relocated": 65},
    ),
    (
        {"trajectory_mode": "shape", "group_effect_size": 0.3},
        "e29fac3172da4267",
        [82, 75, 63, 80],
        {
            "censored": False,
            "shape_kind": "relocate",
            "shape_nominal": 19,
            "shape_relocated": 19,
            "shape_stage": 2,
        },
    ),
    (
        {"trajectory_mode": "magnitude", "group_effect_size": 0.5},
        "77859bd89c6c3974",
        [82, 75, 63, 80],
        {"delta_methyl_scale": 1.5, "magnitude_kind": "all"},
    ),
)


@pytest.mark.parametrize(
    "overrides,digest,counts,transform",
    PINNED_PRECHANGE,
    ids=[case[0]["trajectory_mode"] for case in PINNED_PRECHANGE],
)
def test_default_continuity_reproduces_the_prechange_generator(
    overrides: dict, digest: str, counts: list[int], transform: dict, reference
) -> None:
    dataset = generate_semisynthetic_trajectory(params(**overrides), reference=reference)

    assert indicator_digest(dataset) == digest
    assert dataset.truth["indicator_counts"]["A"]["methylation"] == counts
    assert dataset.truth["transform"] == transform
    assert dataset.truth["baseline_continuity"] == 0.0


def test_explicit_zero_continuity_equals_the_default(reference) -> None:
    default = generate_semisynthetic_trajectory(
        params(trajectory_mode="orientation", group_effect_size=0.3), reference=reference
    )
    explicit = generate_semisynthetic_trajectory(
        params(trajectory_mode="orientation", group_effect_size=0.3, baseline_continuity=0.0),
        reference=reference,
    )

    for layer in ("methylation", "expression", "proteomics"):
        np.testing.assert_array_equal(
            getattr(default, layer).to_numpy(), getattr(explicit, layer).to_numpy()
        )
    assert default.metadata.equals(explicit.metadata)
    assert indicator_digest(default) == indicator_digest(explicit)


# --------------------------------------------------------------------------- #
# 4. The geometry the axis exists to produce
# --------------------------------------------------------------------------- #


def stage_distance_by_separation(dataset) -> dict[int, float]:
    """Mean squared population stage-mean distance, keyed by stage separation."""

    means = dataset.population_trajectories.layers["methylation"].xs("A", level="group")
    values = means.to_numpy()
    by_separation: dict[int, list[float]] = {}
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            by_separation.setdefault(j - i, []).append(
                float(np.sum((values[i] - values[j]) ** 2))
            )
    return {sep: float(np.mean(d)) for sep, d in by_separation.items()}


def test_continuity_makes_stage_configurations_trend(reference) -> None:
    """The point of the axis: distances grow with separation instead of being flat.

    At ρ = 0 the expected squared distance is the same for every pair (a regular
    simplex, no dominant PC1); at ρ > 0 it scales with ``1 - rho ** |t - s|``.
    """

    flat = stage_distance_by_separation(
        generate_semisynthetic_trajectory(params(baseline_continuity=0.0), reference=reference)
    )
    trending = stage_distance_by_separation(
        generate_semisynthetic_trajectory(params(baseline_continuity=0.9), reference=reference)
    )

    separations = sorted(flat)
    assert separations == [1, 2, 3]

    # Isotropic: no pair separation is systematically farther than another.
    flat_values = [flat[sep] for sep in separations]
    assert max(flat_values) / min(flat_values) < 1.2, flat

    # Trending: strictly monotone in separation, and the near neighbours have
    # collapsed relative to the isotropic baseline.
    trending_values = [trending[sep] for sep in separations]
    assert trending_values == sorted(trending_values), trending
    assert trending_values[-1] / trending_values[0] > 1.5, trending
    assert trending[1] < flat[1]


# --------------------------------------------------------------------------- #
# 5. Headroom under the continuity-adjusted union
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_stages", [2, 3, 4, 6])
@pytest.mark.parametrize("p_dmp", [0.05, 0.2, 0.5])
def test_zero_continuity_active_fraction_is_the_independence_union(
    p_dmp: float, n_stages: int
) -> None:
    assert expected_stage_active_fraction(p_dmp, n_stages, 0.0) == pytest.approx(
        1.0 - (1.0 - p_dmp) ** n_stages
    )


@pytest.mark.parametrize("rho", [0.0, 0.5, 0.9])
def test_analytic_active_fraction_matches_monte_carlo(rho: float) -> None:
    """The formula is what enumeration trusts; pin it against real draws."""

    draws = markov_indicators(np.random.default_rng(19), 200_000, 4, 0.2, rho)
    realized = float(draws.max(1).mean())

    assert realized == pytest.approx(
        expected_stage_active_fraction(0.2, 4, rho), abs=0.005
    )


@pytest.mark.parametrize("mode,kwargs", [("orientation", {}), ("shape", {"shape_kind": "relocate"})])
def test_continuity_enlarges_the_saturating_effect(mode: str, kwargs: dict, reference) -> None:
    """Overlapping stage programs shrink the active union, so the pool grows."""

    saturating = [
        expected_surgery_headroom(
            params(trajectory_mode=mode, group_effect_size=0.5, baseline_continuity=rho, **kwargs),
            reference=reference,
        ).saturating_effect
        for rho in (0.0, 0.5, 0.9)
    ]

    assert saturating == sorted(saturating), saturating
    assert saturating[-1] > saturating[0] * 1.5


def test_zero_continuity_headroom_is_unchanged(reference) -> None:
    """The headroom generalization must be a no-op at the isotropic endpoint."""

    headroom = expected_surgery_headroom(
        params(trajectory_mode="orientation", group_effect_size=0.5), reference=reference
    )
    n = float(reference.n_cpg)
    active = 1.0 - (1.0 - 0.2) ** 4

    assert headroom.pool == pytest.approx((1.0 - active) * n)
    assert headroom.nominal == pytest.approx(0.5 * active * n)


@pytest.mark.parametrize("rho", [0.0, 0.5, 0.9])
def test_guard_band_keeps_the_saturating_effect_uncensored(rho: float, reference) -> None:
    """The 3σ band must hold across the axis, not only at ρ = 0.

    Generating *at* the analytic saturating effect under the fail-loud default
    is the sharpest test of the band: if it were mis-sized for the Markov
    baseline's union variance, an unlucky draw would raise here.
    """

    saturating = expected_surgery_headroom(
        params(trajectory_mode="orientation", group_effect_size=0.5, baseline_continuity=rho),
        reference=reference,
    ).saturating_effect

    for seed in range(5):
        dataset = generate_semisynthetic_trajectory(
            params(
                seed=100 + seed,
                trajectory_mode="orientation",
                group_effect_size=saturating,
                baseline_continuity=rho,
            ),
            reference=reference,
        )
        assert dataset.truth["transform"]["censored"] is False


# --------------------------------------------------------------------------- #
# 6. Study enumeration along the axis
# --------------------------------------------------------------------------- #


def study_config(**overrides) -> StudyConfig:
    defaults: dict = {
        "generator": SemiSyntheticTrajectoryParams(
            seed=2, trajectory_mode="orientation", n_samples=60, n_stages=4, p_dmp=0.2
        ),
        "evaluation": SimulationEvaluationParams(
            integration_method="concat", permutations=0, seed=3
        ),
        "trajectory_modes": ("orientation",),
        "effect_sizes": (1.0,),
        "axes": {},
        "n_replicates": 2,
        "base_seed": 100,
        "alpha": 0.05,
        "acceptance": AcceptanceTargets(
            type_i=(TypeIControlTarget(alpha=0.05),),
            power=(
                PowerMonotonicityTarget(
                    trajectory_mode="orientation", statistic="angle", min_power_at_top=0.8
                ),
            ),
            specificity=(
                SpecificityTarget(trajectory_mode="translation", statistic="angle", alpha=0.05),
            ),
        ),
    }
    defaults.update(overrides)
    return StudyConfig(**defaults)


def test_effect_rejected_at_zero_continuity_enumerates_at_high_continuity() -> None:
    """Headroom is checked per cell, so the axis moves the fail-loud boundary."""

    with pytest.raises(StudyConfigError, match="surgery than the expected pool headroom"):
        enumerate_study(study_config())

    grid = enumerate_study(
        study_config(
            generator=SemiSyntheticTrajectoryParams(
                seed=2,
                trajectory_mode="orientation",
                n_samples=60,
                n_stages=4,
                p_dmp=0.2,
                baseline_continuity=0.9,
            )
        )
    )

    assert any(cell.phase == "power_primary" for cell in grid.cells)


def test_continuity_axis_yields_deterministic_distinct_cells() -> None:
    """A swept continuity axis behaves like any other generic generator axis."""

    config = study_config(
        generator=SemiSyntheticTrajectoryParams(
            seed=2,
            trajectory_mode="orientation",
            n_samples=60,
            n_stages=4,
            p_dmp=0.2,
            baseline_continuity=0.9,
        ),
        axes={"generator.baseline_continuity": (0.9, 0.5)},
    )

    first, second = enumerate_study(config), enumerate_study(config)
    assert [cell.cell_id for cell in first.cells] == [cell.cell_id for cell in second.cells]

    by_continuity: dict[float, set[str]] = {}
    for cell in first.cells:
        by_continuity.setdefault(
            cell.generator_params.baseline_continuity, set()
        ).add(parameter_signature(cell))
    assert set(by_continuity) == {0.5, 0.9}
    assert not (by_continuity[0.5] & by_continuity[0.9])


#: Parameter signature of ``signature_probe_cell`` on ``e46a512``, before the
#: generator carried ``baseline_continuity``.
PRECHANGE_SIGNATURE = "0f087ed9be3b11a3e240e7063e25125aee30cc526065a7183c815a52111bb093"


def test_the_new_field_breaks_resume_by_design() -> None:
    """The params hash *is* the version; pre-change shards must refuse to resume.

    Established policy for generation-affecting changes (the
    ``SHAPE_STATISTIC_VERSION`` precedent), recorded here so the break is a
    documented consequence rather than a surprise.
    """

    cell = make_simulation_cell(
        phase="power_primary",
        generator_params=SemiSyntheticTrajectoryParams(
            seed=5, trajectory_mode="orientation", group_effect_size=0.25, n_samples=60, n_stages=3
        ),
        evaluation_params=SimulationEvaluationParams(
            integration_method="concat", permutations=0, seed=3
        ),
        n_replicates=1,
        base_seed=7,
        cell_id="signature-probe",
    )

    assert parameter_signature(cell) != PRECHANGE_SIGNATURE


# --------------------------------------------------------------------------- #
# 7. Continuity-resolved reporting
# --------------------------------------------------------------------------- #


def orientation_record(
    *,
    cell_id: str,
    replicate_index: int,
    continuity: float | None,
    eigengap: float | None = 0.05,
    angle_null_q95: float | None = 1.2,
    p_value: float = 0.01,
) -> SimulationReplicateResult:
    truth: dict = {"trajectory_mode": "orientation"}
    if continuity is not None:
        truth["baseline_continuity"] = continuity
    return SimulationReplicateResult(
        cell_id=cell_id,
        phase="power_primary",
        replicate_index=replicate_index,
        replicate_seed=replicate_index,
        generator_seed=replicate_index,
        evaluation_seed=replicate_index,
        parameter_signature="sig",
        status="completed",
        p_values={"angle": p_value, "delta": 0.5, "shape": 0.5},
        truth_metadata=truth,
        cell_metadata={"trajectory_mode": "orientation", "effect_size": 1.0},
        integration_metadata={"component_selection": "cv"},
        null_summary=({} if angle_null_q95 is None else {"angle": {"q95": angle_null_q95}}),
        config_spectrum=(
            {}
            if eigengap is None
            else {"version": 1, "pooled": {"relative_eigengap": eigengap}, "groups": {}}
        ),
    )


def continuity_records() -> list[SimulationReplicateResult]:
    """Twenty replicates per continuity value; the high-ρ cell rejects, the low one does not."""

    records: list[SimulationReplicateResult] = []
    for index in range(20):
        records.append(
            orientation_record(
                cell_id="orient-rho00",
                replicate_index=index,
                continuity=0.0,
                eigengap=0.02,
                angle_null_q95=1.4,
                p_value=0.5,
            )
        )
        records.append(
            orientation_record(
                cell_id="orient-rho09",
                replicate_index=index,
                continuity=0.9,
                eigengap=0.30,
                angle_null_q95=0.4,
                p_value=0.001,
            )
        )
    return records


def test_record_continuity_separates_absent_from_zero() -> None:
    assert record_continuity(
        orientation_record(cell_id="c", replicate_index=0, continuity=0.0)
    ) == 0.0
    assert (
        record_continuity(orientation_record(cell_id="c", replicate_index=0, continuity=None))
        is None
    )


def test_continuity_resolved_table_is_built_from_records_alone() -> None:
    frame = resolve_orientation_by_continuity(continuity_records(), alpha=0.05)

    assert not frame.empty
    assert sorted(frame["baseline_continuity"].unique()) == [0.0, 0.9]
    assert set(frame["statistic"]) == {"delta", "angle", "shape"}

    angle = frame[frame["statistic"] == "angle"].set_index("baseline_continuity")
    assert angle.loc[0.0, "rejection_rate"] == 0.0
    assert angle.loc[0.9, "rejection_rate"] == 1.0
    # The linking observable: power rises with the recorded eigengap, and the
    # angle null narrows — not merely with the knob.
    assert angle.loc[0.9, "median_eigengap"] > angle.loc[0.0, "median_eigengap"]
    assert angle.loc[0.9, "median_angle_null_q95"] < angle.loc[0.0, "median_angle_null_q95"]
    assert angle.loc[0.0, "n_replicates"] == 20
    assert angle.loc[0.0, "n_cells"] == 1
    for column in ("q33_eigengap", "median_eigengap", "q67_eigengap"):
        assert angle[column].notna().all()


def test_constant_continuity_produces_no_resolved_view() -> None:
    single = [record for record in continuity_records() if record.cell_id == "orient-rho09"]

    assert resolve_orientation_by_continuity(single).empty


def test_records_without_the_field_are_not_folded_into_the_zero_bin() -> None:
    legacy = [
        orientation_record(cell_id="legacy", replicate_index=index, continuity=None)
        for index in range(10)
    ]

    assert resolve_orientation_by_continuity(legacy).empty


def test_report_writes_the_table_only_when_the_axis_varies(tmp_path) -> None:
    records = continuity_records()

    swept = build_report_frames([], [], records)
    assert not swept.continuity_resolved_orientation.empty
    paths = write_report_csvs(swept, tmp_path / "swept")
    assert paths["continuity_resolved_orientation"].exists()

    fixed = build_report_frames(
        [], [], [record for record in records if record.cell_id == "orient-rho09"]
    )
    assert fixed.continuity_resolved_orientation.empty
    fixed_paths = write_report_csvs(fixed, tmp_path / "fixed")
    assert "continuity_resolved_orientation" not in fixed_paths
    assert not (tmp_path / "fixed" / "continuity_resolved_orientation.csv").exists()
    # Every pre-existing output is still written unchanged.
    assert {
        "specificity_matrix",
        "power_curves",
        "type_i_table",
        "config_spectrum",
        "eigengap_stratified_power",
        "realized_surgery",
    } <= set(fixed_paths)


def test_continuity_resolved_table_round_trips_through_csv(tmp_path) -> None:
    frames = build_report_frames([], [], continuity_records())
    paths = write_report_csvs(frames, tmp_path)

    text = paths["continuity_resolved_orientation"].read_text()
    assert "baseline_continuity" in text.splitlines()[0]
    assert json.dumps(list(frames.continuity_resolved_orientation.columns))
