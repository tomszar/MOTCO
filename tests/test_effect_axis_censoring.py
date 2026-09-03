"""Censoring policy for pool-limited surgeries (orientation, translation, shape).

The pinned digests in ``PINNED_CLAMPED`` were recorded against the generator as
it stood *before* the policy parameter existed (silent clamping). They are the
contract that ``surgery_censoring="clamp"`` reproduces the old generator
replicate-for-replicate at the same seed.

They hash the differential **indicators** — what the surgery actually produces —
rather than the sampled omic matrices. The indicators come from ``rng.choice``
draws alone, so they are exact and identical on every machine; the sampled
matrices go through the MVN covariance factorization, whose last bits depend on
the BLAS in use and therefore differ between a workstation and CI. That the RNG
stream itself is unchanged is pinned by the realized surgery sizes beside each
digest, and that the *sampled* data is policy-independent is checked in-process
by :func:`test_uncensored_generation_is_unchanged_by_the_policy`.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from motco.simulations import (  # noqa: E402
    SemiSyntheticTrajectoryError,
    SemiSyntheticTrajectoryParams,
    SimulationGrid,
    SimulationGridError,
    SimulationReplicateResult,
    SimulationRunConfig,
    append_replicate_results,
    expected_surgery_headroom,
    generate_semisynthetic_trajectory,
    load_reference,
    make_simulation_cell,
    parameter_signature,
    run_simulation_grid,
    summarize_realized_surgery,
    summarize_rejection_rates,
)
from motco.simulations.generator import bernoulli_indicators  # noqa: E402
from motco.simulations.study.config import StudyConfigError, load_study_config  # noqa: E402
from motco.simulations.study.enumerate import enumerate_study  # noqa: E402
from motco.simulations.study.report import (  # noqa: E402
    build_realized_surgery,
    build_report_frames,
    find_duplicate_constructions,
    write_report_csvs,
)

PHASE4_PILOT = Path("examples/trajectory_power_study/phase4_pilot_100x199.json")


@pytest.fixture(scope="module")
def reference():
    return load_reference()


def indicator_digest(dataset) -> str:
    """Machine-stable digest of both groups' differential indicators.

    These are integer-valued and produced by RNG draws alone, so the digest is
    reproducible across platforms — unlike the sampled omic matrices, whose
    floating-point bits depend on the BLAS behind the MVN sampling.
    """

    hasher = hashlib.sha256()
    for group in dataset.truth["group_labels"]:
        for layer in ("methylation", "expression", "proteomics"):
            values = np.ascontiguousarray(
                dataset.truth["indicators"][group][layer], dtype=float
            )
            hasher.update(str(values.shape).encode())
            hasher.update(values.tobytes())
    return hasher.hexdigest()[:16]


def clamped_params(**overrides) -> SemiSyntheticTrajectoryParams:
    base = dict(seed=11, n_samples=240, n_stages=4, p_dmp=0.2, surgery_censoring="clamp")
    base.update(overrides)
    return SemiSyntheticTrajectoryParams(**base)  # type: ignore[arg-type]


#: (params overrides, realized-size truth key, realized size, indicator digest)
#: recorded against the pre-policy generator.
PINNED_CLAMPED: tuple[tuple[dict, str, int, str], ...] = (
    (
        {"trajectory_mode": "orientation", "group_effect_size": 1.0},
        "orientation_relocated",
        151,
        "505192fed8ce4d7b",
    ),
    (
        {"trajectory_mode": "translation", "group_effect_size": 1.0},
        "translation_set_size",
        41,
        "c8a1ed21d8059afe",
    ),
    (
        {
            "trajectory_mode": "shape",
            "shape_kind": "relocate",
            "group_effect_size": 1.0,
            "p_dmp": 0.5,
        },
        "shape_relocated",
        25,
        "9f4cd2aa13bd3e6d",
    ),
)

#: Same shape, for surgeries whose requested size fits inside the pool.
PINNED_UNCENSORED: tuple[tuple[dict, str, int, str], ...] = (
    (
        {"trajectory_mode": "orientation", "group_effect_size": 0.3},
        "orientation_relocated",
        65,
        "1cc7c03fa01902ce",
    ),
    (
        {"trajectory_mode": "translation", "group_effect_size": 0.3},
        "translation_set_size",
        22,
        "0e3d37ffc640547b",
    ),
)


@pytest.mark.parametrize(("overrides", "key", "realized", "digest"), PINNED_CLAMPED)
def test_clamp_policy_reproduces_the_pre_policy_generator(
    reference, overrides, key, realized, digest
) -> None:
    dataset = generate_semisynthetic_trajectory(clamped_params(**overrides), reference=reference)

    assert dataset.truth["transform"][key] == realized
    assert indicator_digest(dataset) == digest


@pytest.mark.parametrize(("overrides", "key", "realized", "digest"), PINNED_UNCENSORED)
def test_uncensored_generation_is_unchanged_by_the_policy(
    reference, overrides, key, realized, digest
) -> None:
    """An uncensored surgery samples the same dataset under either policy value."""

    datasets = {}
    for policy in ("error", "clamp"):
        dataset = generate_semisynthetic_trajectory(
            clamped_params(surgery_censoring=policy, **overrides), reference=reference
        )
        assert dataset.truth["transform"][key] == realized
        assert dataset.truth["transform"]["censored"] is False
        assert indicator_digest(dataset) == digest
        datasets[policy] = dataset

    # The sampled matrices are compared in-process rather than against a pinned
    # constant: they are BLAS-dependent in their last bits, but two runs on one
    # machine must agree exactly, which is what "the policy changes nothing when
    # the clamp does not bind" actually asserts.
    for layer in ("methylation", "expression", "proteomics"):
        np.testing.assert_array_equal(
            getattr(datasets["error"], layer).to_numpy(),
            getattr(datasets["clamp"], layer).to_numpy(),
        )


def test_default_policy_is_error() -> None:
    assert SemiSyntheticTrajectoryParams(seed=0).surgery_censoring == "error"


def test_invalid_policy_is_rejected(reference) -> None:
    with pytest.raises(SemiSyntheticTrajectoryError, match="Unknown surgery_censoring"):
        generate_semisynthetic_trajectory(
            clamped_params(trajectory_mode="none", surgery_censoring="bogus"),
            reference=reference,
        )


@pytest.mark.parametrize(
    ("overrides", "mode"),
    [
        ({"trajectory_mode": "orientation", "group_effect_size": 1.0}, "orientation"),
        ({"trajectory_mode": "translation", "group_effect_size": 1.0}, "translation"),
        (
            {
                "trajectory_mode": "shape",
                "shape_kind": "relocate",
                "group_effect_size": 1.0,
                "p_dmp": 0.5,
            },
            "shape",
        ),
    ],
)
def test_error_policy_fails_loudly_on_a_censored_surgery(reference, overrides, mode) -> None:
    params = clamped_params(surgery_censoring="error", **overrides)

    with pytest.raises(SemiSyntheticTrajectoryError) as excinfo:
        generate_semisynthetic_trajectory(params, reference=reference)

    message = str(excinfo.value)
    assert mode in message
    assert "requested" in message
    assert "pool" in message
    assert "saturat" in message


def test_error_policy_consumes_no_rng_draws_before_failing(reference) -> None:
    """The policy check must not perturb the RNG sequence the clamp path uses.

    A censored orientation surgery under ``"error"`` fails at the same point the
    ``"clamp"`` path would have drawn, so a lower effect that fits the pool
    samples identically under either policy — verified by the shared digests in
    ``test_uncensored_generation_is_unchanged_by_the_policy``. Here we check the
    complementary direction: raising is decided from pool sizes alone.
    """

    params = clamped_params(trajectory_mode="orientation", group_effect_size=1.0)
    baseline = generate_semisynthetic_trajectory(params, reference=reference)

    # Same seed, same pools: the error path sees the pool the clamp path realized.
    with pytest.raises(SemiSyntheticTrajectoryError) as excinfo:
        generate_semisynthetic_trajectory(
            clamped_params(
                trajectory_mode="orientation", group_effect_size=1.0, surgery_censoring="error"
            ),
            reference=reference,
        )
    assert str(baseline.truth["transform"]["orientation_relocated"]) in str(excinfo.value)


def test_truth_records_nominal_realized_and_censored_flag(reference) -> None:
    dataset = generate_semisynthetic_trajectory(
        clamped_params(trajectory_mode="orientation", group_effect_size=1.0), reference=reference
    )
    transform = dataset.truth["transform"]

    assert transform["orientation_nominal"] == 216  # all stage-active CpGs requested
    assert transform["orientation_relocated"] == 151  # only the inactive pool available
    assert transform["censored"] is True


def test_shape_magnitude_has_no_pool_limited_surgery(reference) -> None:
    """``shape_kind='magnitude'`` scales a stage; it relocates nothing."""

    dataset = generate_semisynthetic_trajectory(
        clamped_params(
            trajectory_mode="shape",
            shape_kind="magnitude",
            group_effect_size=1.0,
            surgery_censoring="error",
        ),
        reference=reference,
    )
    transform = dataset.truth["transform"]

    assert "shape_relocated" not in transform
    assert "censored" not in transform


# --------------------------------------------------------------------------- #
# Expected headroom
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "overrides",
    [
        {"trajectory_mode": "none"},
        {"trajectory_mode": "magnitude", "group_effect_size": 1.0},
        {"trajectory_mode": "shape", "shape_kind": "magnitude", "group_effect_size": 1.0},
        {"trajectory_mode": "orientation", "group_effect_size": 0.0},
    ],
)
def test_headroom_is_absent_without_a_pool_limited_surgery(reference, overrides) -> None:
    params = clamped_params(**overrides)

    assert expected_surgery_headroom(params, reference=reference) is None


@pytest.mark.parametrize(("p_dmp", "n_stages"), [(0.2, 4), (0.2, 3), (0.5, 4), (0.1, 3)])
def test_expected_pools_match_empirical_pools_across_seeds(reference, p_dmp, n_stages) -> None:
    """The analytic pool expectations track the pools the generator actually draws."""

    inactive, candidates = [], []
    for seed in range(120):
        methyl = bernoulli_indicators(
            np.random.default_rng(seed), reference.n_cpg, n_stages, p_dmp
        )
        stage_active = methyl.sum(1) > 0
        inactive.append(int((~stage_active).sum()))
        used_genes = reference.incidence_cpg_gene[stage_active].sum(0) > 0
        cpg_gene = reference.incidence_cpg_gene.argmax(1)
        candidates.append(int(((~stage_active) & (~used_genes[cpg_gene])).sum()))

    def headroom(mode: str):
        return expected_surgery_headroom(
            clamped_params(
                trajectory_mode=mode, group_effect_size=1.0, p_dmp=p_dmp, n_stages=n_stages
            ),
            reference=reference,
        )

    # orientation and shape/relocate both target the globally stage-inactive pool
    assert headroom("orientation").pool == pytest.approx(np.mean(inactive), abs=2.0)
    assert headroom("shape").pool == pytest.approx(np.mean(inactive), abs=2.0)
    # translation's fresh pool is computed from the CpG→gene incidence
    assert headroom("translation").pool == pytest.approx(np.mean(candidates), abs=2.0)


def test_guard_band_keeps_the_verdict_below_the_expected_pool(reference) -> None:
    """The verdict uses the pool *less* a guard band, since pools are random."""

    headroom = expected_surgery_headroom(
        clamped_params(trajectory_mode="orientation", group_effect_size=1.0), reference=reference
    )

    assert headroom.guard_band > 0
    assert headroom.available == pytest.approx(headroom.pool - headroom.guard_band)
    assert 0 < headroom.available < headroom.pool
    # e = 1.00 requests every stage-active CpG, far beyond the inactive pool
    assert headroom.nominal > headroom.pool
    assert headroom.fits is False
    # the reported saturating effect is exactly where the request meets available
    assert headroom.saturating_effect == pytest.approx(
        headroom.available / (headroom.nominal / headroom.group_effect_size)
    )


def test_headroom_verdict_brackets_the_generator(reference) -> None:
    """An effect the headroom accepts generates; one it rejects raises."""

    def fits(effect: float) -> bool:
        return expected_surgery_headroom(
            clamped_params(trajectory_mode="orientation", group_effect_size=effect),
            reference=reference,
        ).fits

    accepted = 0.5
    rejected = 0.75
    assert fits(accepted) and not fits(rejected)

    dataset = generate_semisynthetic_trajectory(
        clamped_params(
            trajectory_mode="orientation", group_effect_size=accepted, surgery_censoring="error"
        ),
        reference=reference,
    )
    assert dataset.truth["transform"]["censored"] is False

    with pytest.raises(SemiSyntheticTrajectoryError):
        generate_semisynthetic_trajectory(
            clamped_params(
                trajectory_mode="orientation",
                group_effect_size=rejected,
                surgery_censoring="error",
                seed=3,
            ),
            reference=reference,
        )


# --------------------------------------------------------------------------- #
# Enumeration-time headroom validation
# --------------------------------------------------------------------------- #


def write_config(directory: Path, data: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def phase4_config_data() -> dict:
    return json.loads(PHASE4_PILOT.read_text(encoding="utf-8"))


def test_phase4_shaped_config_fails_enumeration_under_the_default_policy(tmp_path) -> None:
    """The audit's acceptance criterion: the censored Phase-4 axis is caught.

    The committed pilot carries ``surgery_censoring="clamp"`` as a historical
    record; drop the flag and it is exactly the grid the audit found — four
    stages, ``p_dmp = 0.2``, orientation and translation up to ``e = 1.00``.
    """

    data = phase4_config_data()
    assert data["generator"].pop("surgery_censoring") == "clamp"

    with pytest.raises(StudyConfigError) as excinfo:
        enumerate_study(load_study_config(write_config(tmp_path, data)))

    message = str(excinfo.value)
    assert "saturates at group_effect_size" in message
    assert "surgery_censoring='clamp'" in message
    # exactly the censored cells: orientation above e ~= 0.56 and translation
    # above e ~= 0.29, and nothing from magnitude (no pool-limited surgery).
    assert "trajectory_mode='magnitude'" not in message
    for mode, censored_effects in (
        ("orientation", ("0.75", "1")),
        # 0.5/0.75/1.00 power cells plus the negative control, also at 1.00
        ("translation", ("0.5", "0.75", "1", "1")),
    ):
        offenders = [line for line in message.splitlines() if f"trajectory_mode={mode!r}" in line]
        effects = sorted(
            line.split("group_effect_size=")[1].split(" ")[0] for line in offenders
        )
        assert effects == sorted(censored_effects), (mode, effects)


def test_clamping_configs_are_exempt_from_the_headroom_check() -> None:
    """The committed historical configs stay enumerable with the flag set."""

    grid = enumerate_study(load_study_config(PHASE4_PILOT))

    assert grid.cells
    assert all(cell.generator_params.surgery_censoring == "clamp" for cell in grid.cells)


def test_every_censored_cell_is_named_not_just_the_first(tmp_path) -> None:
    """One enumeration reports every offender, so a config is fixed in one pass."""

    data = phase4_config_data()
    data["generator"].pop("surgery_censoring")
    data["trajectory_modes"] = ["orientation"]
    data["effect_sizes"] = [0.0, 0.75, 1.0]

    with pytest.raises(StudyConfigError) as excinfo:
        enumerate_study(load_study_config(write_config(tmp_path, data)))

    message = str(excinfo.value)
    offenders = [line for line in message.splitlines() if line.startswith("  - cell ")]
    # both orientation power cells plus the translation negative control at 1.00
    assert len(offenders) == 3
    assert sum("trajectory_mode='orientation'" in line for line in offenders) == 2
    assert all("saturates at group_effect_size" in line for line in offenders)


def test_headroom_respecting_config_enumerates_with_unchanged_cells(tmp_path) -> None:
    """Within the headroom, enumeration is unaffected by the check."""

    data = phase4_config_data()
    data["generator"].pop("surgery_censoring")
    # 0.25 is inside the headroom for every mode here, including the
    # `translation` negative control that enumeration adds at the top effect.
    data["trajectory_modes"] = ["magnitude", "orientation"]
    data["effect_sizes"] = [0.0, 0.25]
    config_path = write_config(tmp_path, data)

    grid = enumerate_study(load_study_config(config_path))

    # The same config with clamping opted in enumerates the same cells apart
    # from the policy field itself, so the check adds no cells and drops none.
    clamped = copy.deepcopy(data)
    clamped["generator"]["surgery_censoring"] = "clamp"
    clamped_grid = enumerate_study(
        load_study_config(write_config(tmp_path / "clamped", clamped))
    )

    def shape_of(cells):
        # cell ids (and the control seed families keyed on them) hash the
        # generator params, so the policy field itself moves them; everything
        # that describes *what* each cell measures must be identical.
        return [
            (cell.phase, {k: v for k, v in cell.metadata.items() if k != "seed_family"})
            for cell in cells
        ]

    assert shape_of(grid.cells) == shape_of(clamped_grid.cells)
    assert [cell.base_seed for cell in grid.cells] == [
        cell.base_seed for cell in clamped_grid.cells
    ]


# --------------------------------------------------------------------------- #
# Summaries and study report
# --------------------------------------------------------------------------- #


def surgery_record(
    cell_id: str,
    replicate_index: int,
    *,
    mode: str,
    effect_size: float,
    transform: dict | None = None,
    seed_family: str | None = "primary",
) -> SimulationReplicateResult:
    metadata: dict = {"trajectory_mode": mode, "effect_size": effect_size, "varied_axis": None}
    if seed_family is not None:
        metadata["seed_family"] = seed_family
    return SimulationReplicateResult(
        cell_id=cell_id,
        phase="power_primary",
        replicate_index=replicate_index,
        replicate_seed=100 + replicate_index,
        generator_seed=100 + replicate_index,
        evaluation_seed=None,
        parameter_signature="sig",
        status="completed",
        p_values={"delta": 0.01, "angle": 0.2, "shape": 0.3},
        truth_metadata={"trajectory_mode": mode, "transform": transform or {}},
        cell_metadata=metadata,
    )


def orientation_records(
    cell_id: str, *, effect_size: float, nominal: int, realized: int, n: int = 10
) -> list[SimulationReplicateResult]:
    return [
        surgery_record(
            cell_id,
            index,
            mode="orientation",
            effect_size=effect_size,
            transform={
                "orientation_relocated": realized,
                "orientation_nominal": nominal,
                "censored": realized < nominal,
            },
        )
        for index in range(n)
    ]


def test_summarize_realized_surgery_reports_nominal_realized_and_censoring() -> None:
    records = [
        *orientation_records("orient-075", effect_size=0.75, nominal=160, realized=151, n=8),
        *orientation_records("orient-025", effect_size=0.25, nominal=54, realized=54, n=8),
    ]

    summaries = {s.cell_id: s for s in summarize_realized_surgery(records)}

    censored = summaries["orient-075"]
    assert censored.nominal_size == 160
    assert censored.realized_mean == 151
    assert (censored.realized_min, censored.realized_max) == (151, 151)
    assert censored.censored_fraction == 1.0

    clean = summaries["orient-025"]
    assert clean.nominal_size == clean.realized_mean == 54
    assert clean.censored_fraction == 0.0


def test_summarize_realized_surgery_reports_absence_not_zero() -> None:
    """A mode with no pool-limited surgery must not read as a zero-size surgery."""

    records = [
        surgery_record("mag-100", index, mode="magnitude", effect_size=1.0,
                       transform={"magnitude_kind": "all", "delta_methyl_scale": 2.0})
        for index in range(5)
    ]

    (summary,) = summarize_realized_surgery(records)

    assert summary.trajectory_mode == "magnitude"
    assert summary.completed_replicates == 5
    assert summary.surgery_replicates == 0
    assert summary.nominal_size is None
    assert summary.realized_mean is None
    assert summary.censored_fraction is None


def test_realized_surgery_table_is_written_by_the_study_report(tmp_path) -> None:
    records = [
        *orientation_records("orient-075", effect_size=0.75, nominal=160, realized=151, n=6),
        *orientation_records("orient-025", effect_size=0.25, nominal=54, realized=54, n=6),
    ]
    summaries = summarize_rejection_rates(records)

    frames = build_report_frames(summaries, [], records)
    paths = write_report_csvs(frames, tmp_path)

    assert paths["realized_surgery"].exists()
    written = pd.read_csv(paths["realized_surgery"])
    assert set(written["cell_id"]) == {"orient-075", "orient-025"}
    row = written.set_index("cell_id").loc["orient-075"]
    assert row["nominal_size"] == 160
    assert row["realized_mean"] == 151
    assert row["censored_fraction"] == 1.0

    # the power-curve frame carries the annotation so curves can be marked
    curves = frames.power_curves.set_index(["cell_id", "statistic"])
    assert curves.loc[("orient-075", "delta"), "censored_fraction"] == 1.0
    assert curves.loc[("orient-025", "delta"), "censored_fraction"] == 0.0


def test_duplicate_constructions_are_flagged_reproducing_the_audit_pattern() -> None:
    """The audit's 80/100 pattern: two matched-seed cells censored identically."""

    duplicated = [
        # e = 0.75 and e = 1.00 both clamp to the same 151-site relocation, so
        # at each replicate index the two cells generate the same dataset.
        *orientation_records("orient-075", effect_size=0.75, nominal=160, realized=151, n=100),
        *orientation_records("orient-100", effect_size=1.0, nominal=216, realized=151, n=100),
    ]

    duplicates = find_duplicate_constructions(duplicated)

    assert duplicates == {"orient-075": {"orient-100"}, "orient-100": {"orient-075"}}
    frame = build_realized_surgery(duplicated).set_index("cell_id")
    assert bool(frame.loc["orient-075", "duplicate_construction"]) is True
    assert frame.loc["orient-100", "duplicate_of"] == "orient-075"


def test_uncensored_cells_are_not_flagged_as_duplicates() -> None:
    """Distinct realized sizes are independent measurements, not duplicates."""

    records = [
        *orientation_records("orient-025", effect_size=0.25, nominal=54, realized=54, n=100),
        *orientation_records("orient-050", effect_size=0.5, nominal=108, realized=108, n=100),
    ]

    assert find_duplicate_constructions(records) == {}
    frame = build_realized_surgery(records)
    assert not frame["duplicate_construction"].any()
    assert frame["duplicate_of"].isna().all()


def test_cells_in_different_seed_families_are_never_duplicates() -> None:
    """Without a shared seed family the datasets differ even at equal sizes."""

    records = [
        *[
            surgery_record(
                "orient-a", index, mode="orientation", effect_size=0.75,
                transform={"orientation_relocated": 151, "orientation_nominal": 160,
                           "censored": True},
                seed_family="control:orient-a",
            )
            for index in range(50)
        ],
        *[
            surgery_record(
                "orient-b", index, mode="orientation", effect_size=1.0,
                transform={"orientation_relocated": 151, "orientation_nominal": 216,
                           "censored": True},
                seed_family="control:orient-b",
            )
            for index in range(50)
        ],
    ]

    assert find_duplicate_constructions(records) == {}


# --------------------------------------------------------------------------- #
# The parameter-signature break
# --------------------------------------------------------------------------- #


def test_old_signature_records_refuse_to_resume_into_a_new_signature_cell(tmp_path) -> None:
    """Adding the policy field changes every cell's signature, by design.

    A shard written before the policy existed carries a stale signature. Resume
    must refuse it rather than mix records generated under silent clamping with
    records generated under the fail-loud default.
    """

    generator = SemiSyntheticTrajectoryParams(
        seed=5, trajectory_mode="magnitude", n_samples=60, n_stages=3
    )
    cell = make_simulation_cell(
        phase="power_primary",
        generator_params=generator,
        n_replicates=1,
        cell_id="resume-me",
    )
    # A record written under the pre-policy signature; what matters is only that
    # it does not match the signature the cell hashes to now.
    stale = SimulationReplicateResult(
        cell_id="resume-me",
        phase="power_primary",
        replicate_index=0,
        replicate_seed=1,
        generator_seed=1,
        evaluation_seed=None,
        parameter_signature="pre-policy-signature",
        status="completed",
    )
    assert parameter_signature(cell) != stale.parameter_signature

    path = tmp_path / "results.jsonl"
    append_replicate_results(path, [stale])

    with pytest.raises(SimulationGridError, match="different parameter signature"):
        run_simulation_grid(
            SimulationGrid(cells=(cell,)),
            config=SimulationRunConfig(output_path=path, resume=True),
            evaluator=lambda generator_params, evaluation_params: pytest.fail(
                "resume must refuse the stale record before running anything"
            ),
        )
