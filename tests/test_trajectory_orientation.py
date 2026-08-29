"""Invariance contract for the trajectory orientation estimator.

The committed R fixtures (`results_example1.csv`, `results_example2.csv`) have
benign geometry: every candidate sign convention reproduces them exactly, so
they cannot detect an orientation sign flip. These tests construct the failure
geometry on purpose and assert the invariances the `angle` statistic must have
to be a difference in *direction of progression*.

See `openspec/changes/fix-orientation-sign-anchor/`.
"""

from __future__ import annotations

import numpy as np
import pytest

from motco.stats.trajectory import _estimate_orientation, estimate_difference

# Tolerance for angles that must be exactly zero up to floating point.
ANGLE_TOL_DEG = 1e-6
# Angles between trajectories that differ only by noise of `NOISE_SD` must stay
# well inside this band; the geometry's extent is ~5 units, so the true angular
# perturbation is a fraction of a degree.
NOISE_ANGLE_TOL_DEG = 5.0
NOISE_SD = 0.01

LEVELS = [0, 1, 2, 3]

# A bent four-stage trajectory whose first stage projects *exactly* zero onto
# its own centered principal axis. Constructed so the centered configuration
# has zero x/y cross-covariance (making PC1 the x axis) with the first stage
# sitting at the centroid's x coordinate: the trajectory departs along +x,
# returns past its start to -x, and ends part-way back. The centroid is offset
# from the coordinate origin so translation effects stay distinguishable.
BENT_TRAJECTORY = np.array(
    [
        [0.0, -2.5],
        [4.0, 0.5],
        [-3.0, 0.0],
        [-1.0, 2.0],
    ],
    dtype=float,
) + np.array([10.0, 5.0])


def _angle_between(left: np.ndarray, right: np.ndarray) -> float:
    """Pairwise trajectory `angle` (degrees) via the production estimator."""
    vectors = np.vstack([left, right])
    n_levels = left.shape[0]
    contrast = [list(range(n_levels)), list(range(n_levels, 2 * n_levels))]
    identity = np.eye(vectors.shape[0])
    _, angles, _ = estimate_difference(vectors, identity, identity, contrast)
    return float(angles[0, 1])


def _jitter(trajectory: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Perturb every stage by noise small relative to the trajectory's extent."""
    return trajectory + rng.normal(scale=NOISE_SD, size=trajectory.shape)


def test_bent_geometry_is_the_degenerate_anchor_case():
    """Guard the fixture: the old anchor really is at zero for this geometry."""
    centered = BENT_TRAJECTORY - BENT_TRAJECTORY.mean(axis=0, keepdims=True)
    orientation = _estimate_orientation(BENT_TRAJECTORY, LEVELS)

    assert float(orientation @ centered[0, :]) == pytest.approx(0.0, abs=1e-12)
    net_displacement = BENT_TRAJECTORY[-1, :] - BENT_TRAJECTORY[0, :]
    assert abs(float(orientation @ net_displacement)) > 0.5


# ── 1.1 / 1.2 / 1.3 — the defect ─────────────────────────────────────────────


def test_bent_identical_trajectories_report_zero_angle():
    """Two copies of a bent trajectory carry no orientation difference.

    They are separated only by noise two orders of magnitude below the
    trajectory's extent — exactly identical inputs would share the SVD's
    arbitrary sign and hide the defect, so the copies are perturbed instead.
    """
    rng = np.random.default_rng(0)

    for _ in range(50):
        left = _jitter(BENT_TRAJECTORY, rng)
        right = _jitter(BENT_TRAJECTORY, rng)

        assert _angle_between(left, right) < NOISE_ANGLE_TOL_DEG


def test_orientation_sign_is_stable_under_small_perturbation():
    """The returned direction must not change sign across repeated draws."""
    rng = np.random.default_rng(1)
    unperturbed = _estimate_orientation(BENT_TRAJECTORY, LEVELS)

    for _ in range(200):
        orientation = _estimate_orientation(_jitter(BENT_TRAJECTORY, rng), LEVELS)

        assert float(orientation @ unperturbed) > 0.0


@pytest.mark.parametrize(
    "transform",
    [
        pytest.param(lambda t: t + np.array([7.5, -13.0]), id="translation"),
        pytest.param(lambda t: 3.25 * t, id="uniform-scale"),
        pytest.param(lambda t: t, id="identity"),
    ],
)
def test_null_configurations_never_approach_antiparallel(transform):
    """No configuration carrying zero orientation difference may report ~180°."""
    rng = np.random.default_rng(2)

    for _ in range(100):
        left = _jitter(BENT_TRAJECTORY, rng)
        right = transform(_jitter(BENT_TRAJECTORY, rng))

        assert _angle_between(left, right) < NOISE_ANGLE_TOL_DEG


# ── 3.1 — direction semantics ────────────────────────────────────────────────


def test_orientation_points_along_net_displacement():
    """The returned vector is a direction, aligned with progression."""
    rng = np.random.default_rng(3)

    for _ in range(50):
        trajectory = rng.standard_normal((5, 4)) + np.array([3.0, -1.0, 0.5, 2.0])
        orientation = _estimate_orientation(trajectory, list(range(5)))
        net_displacement = trajectory[-1, :] - trajectory[0, :]

        assert float(orientation @ net_displacement) >= 0.0


def test_reversing_stage_order_negates_orientation():
    forward = _estimate_orientation(BENT_TRAJECTORY, LEVELS)
    reversed_ = _estimate_orientation(BENT_TRAJECTORY[::-1, :], LEVELS)

    np.testing.assert_allclose(reversed_, -forward, atol=1e-12)


def test_two_stage_orientation_is_the_unit_transition_vector():
    transition = np.array([3.0, -4.0, 12.0])
    trajectory = np.vstack([np.array([1.5, 2.5, -0.5]), np.array([1.5, 2.5, -0.5]) + transition])

    orientation = _estimate_orientation(trajectory, [0, 1])

    np.testing.assert_allclose(orientation, transition / np.linalg.norm(transition), atol=1e-12)


# ── 3.2 — translation and uniform scale ──────────────────────────────────────


def test_translated_trajectory_has_zero_angle():
    translated = BENT_TRAJECTORY + np.array([-31.0, 17.5])

    assert _angle_between(BENT_TRAJECTORY, translated) == pytest.approx(0.0, abs=ANGLE_TOL_DEG)


def test_trajectories_on_opposite_sides_of_the_origin_have_zero_angle():
    """The regime the reference's raw-row anchor gets wrong.

    A straight two-stage trajectory and its copy reflected through the origin
    have the same direction of progression, so the angle must be zero — the
    raw anchor reports 180 degrees here.
    """
    straight = np.array([[1.0, 0.5], [4.0, 2.5]])
    other_side = straight - np.array([6.0, 4.0])
    assert np.all(straight[:, 0] > 0.0) and np.all(other_side[:, 0] < 0.0)

    assert _angle_between(straight, other_side) == pytest.approx(0.0, abs=ANGLE_TOL_DEG)


@pytest.mark.parametrize("factor", [0.125, 3.0, 250.0])
def test_uniformly_scaled_trajectory_has_zero_angle(factor: float):
    assert _angle_between(BENT_TRAJECTORY, factor * BENT_TRAJECTORY) == pytest.approx(
        0.0, abs=ANGLE_TOL_DEG
    )


# ── 3.3 — genuine differences are preserved ──────────────────────────────────


@pytest.mark.parametrize("expected_deg", [17.0, 45.0, 90.0, 118.0, 165.0])
def test_known_angle_is_recovered(expected_deg: float):
    """Rotating a trajectory by a known angle must report exactly that angle.

    Cases above 90 degrees are included so the fix cannot be satisfied by
    collapsing every angle toward zero.
    """
    theta = np.deg2rad(expected_deg)
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=float
    )
    centroid = BENT_TRAJECTORY.mean(axis=0, keepdims=True)
    rotated = (BENT_TRAJECTORY - centroid) @ rotation.T + centroid

    assert _angle_between(BENT_TRAJECTORY, rotated) == pytest.approx(expected_deg, abs=1e-8)


# ── The anchor decision, pinned ──────────────────────────────────────────────
#
# The four regimes that decided the sign convention. Reported in
# docs/reports/orientation-sign-anchor-2026-08-28.md; pinned here so the
# decision table traces to a committed test rather than to a scratch script.


def _orientation_with_anchor(trajectory: np.ndarray, anchor: str) -> np.ndarray:
    """PC1 of the centered configuration, signed by one of three anchors."""
    centered = trajectory - trajectory.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    orientation = Vt[0, :]
    projection = {
        # the shipped convention before this change
        "centered": orientation @ centered[0, :],
        # a faithful port of evo_649_sm_suppmat.r:64
        "raw": orientation @ trajectory[0, :],
        # the convention adopted here
        "net": orientation @ (trajectory[-1, :] - trajectory[0, :]),
    }[anchor]
    return -orientation if float(projection) < 0.0 else orientation


def _angle_with_anchor(left: np.ndarray, right: np.ndarray, anchor: str) -> float:
    dot = float(
        _orientation_with_anchor(left, anchor) @ _orientation_with_anchor(right, anchor)
    )
    return float(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))))


def _rotate(trajectory: np.ndarray, degrees: float) -> np.ndarray:
    theta = np.deg2rad(degrees)
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=float
    )
    centroid = trajectory.mean(axis=0, keepdims=True)
    return (trajectory - centroid) @ rotation.T + centroid


def _four_regimes() -> dict[str, tuple[np.ndarray, np.ndarray, float]]:
    """Each regime: (left, right, true angle in degrees)."""
    rng = np.random.default_rng(11)
    straight = np.array([[1.0, 0.5], [4.0, 2.5]])
    return {
        "bent-4-stage-identical-groups": (
            _jitter(BENT_TRAJECTORY, rng),
            _jitter(BENT_TRAJECTORY, rng),
            0.0,
        ),
        "straight-2-stage-either-side-of-origin": (
            straight,
            straight - np.array([6.0, 4.0]),
            0.0,
        ),
        # Shifted far enough that the first stage crosses the plane through the
        # origin orthogonal to PC1 — which is all it takes to flip the raw anchor.
        "translation-control": (
            BENT_TRAJECTORY,
            BENT_TRAJECTORY + np.array([-30.0, 8.0]),
            0.0,
        ),
        "genuine-90-degree-difference": (
            BENT_TRAJECTORY,
            _rotate(BENT_TRAJECTORY, 90.0),
            90.0,
        ),
    }


@pytest.mark.parametrize(
    "regime,expected",
    [
        # regime -> (centered, raw, net); the failures are the large entries.
        ("bent-4-stage-identical-groups", (180.0, 0.0, 0.0)),
        ("straight-2-stage-either-side-of-origin", (0.0, 180.0, 0.0)),
        ("translation-control", (0.0, 180.0, 0.0)),
        ("genuine-90-degree-difference", (90.0, 90.0, 90.0)),
    ],
)
def test_four_regime_anchor_comparison(regime: str, expected: tuple[float, float, float]):
    """Net displacement is the only anchor correct in all four regimes.

    The faithful port trades one failure for two, and one of the two is MOTCO's
    own translation null control.
    """
    left, right, truth = _four_regimes()[regime]

    observed = tuple(
        _angle_with_anchor(left, right, anchor) for anchor in ("centered", "raw", "net")
    )

    for got, want in zip(observed, expected, strict=True):
        assert got == pytest.approx(want, abs=1.0)
    # Whatever the other anchors do, net displacement always recovers the truth.
    assert observed[2] == pytest.approx(truth, abs=1.0)
