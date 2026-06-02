from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import pytest

from motco.simulations import (
    InterSIMDependencyError,
    InterSIMError,
    InterSIMMalformedOutputError,
    InterSIMParams,
    check_intersim_available,
    intersim,
    run_intersim,
)


def _write_intersim_outputs(output_dir: Path, *, aligned: bool = True) -> None:
    sample_ids = ["subject1", "subject2", "subject3"]
    pd.DataFrame(
        {"m1": [0.1, 0.2, 0.3], "m2": [0.4, 0.5, 0.6]},
        index=sample_ids,
    ).to_csv(output_dir / "methylation.csv")
    pd.DataFrame(
        {"g1": [1.0, 2.0, 3.0]},
        index=sample_ids if aligned else ["subject1", "subject3", "subject2"],
    ).to_csv(output_dir / "expression.csv")
    pd.DataFrame(
        {"p1": [4.0, 5.0, 6.0]},
        index=sample_ids,
    ).to_csv(output_dir / "proteomics.csv")
    pd.DataFrame(
        {"sample_id": sample_ids, "cluster": [1, 2, 1]},
    ).to_csv(output_dir / "clusters.csv", index=False)


def test_build_rscript_command_translates_python_params(tmp_path: Path) -> None:
    params = InterSIMParams(
        seed=123,
        n_sample=20,
        cluster_sample_prop=(0.2, 0.3, 0.5),
        delta_methyl=1.0,
        p_dmp=0.1,
        sigma_methyl="indep",
        cor_methyl_expr=0.25,
    )

    cmd = intersim._build_rscript_command(params, output_dir=tmp_path)

    assert cmd[0] == "Rscript"
    assert "--output-dir" in cmd
    assert "--seed" in cmd
    assert "123" in cmd
    assert "--n-sample" in cmd
    assert "20" in cmd
    assert "--cluster-sample-prop" in cmd
    assert "0.2,0.3,0.5" in cmd
    assert "--delta-methyl" in cmd
    assert "--p-dmp" in cmd
    assert "--sigma-methyl" in cmd
    assert "--cor-methyl-expr" in cmd
    assert "--p-deg" not in cmd


@pytest.mark.parametrize("seed", [2**31 - 1, 0, -(2**31)])
def test_build_rscript_command_accepts_in_range_seeds(seed: int, tmp_path: Path) -> None:
    cmd = intersim._build_rscript_command(InterSIMParams(seed=seed), output_dir=tmp_path)
    assert str(seed) in cmd


@pytest.mark.parametrize("seed", [2**31, -(2**31) - 1, 2_797_983_684])
def test_build_rscript_command_rejects_out_of_range_seeds(seed: int, tmp_path: Path) -> None:
    with pytest.raises(InterSIMError, match=r"signed-32-bit range"):
        intersim._build_rscript_command(InterSIMParams(seed=seed), output_dir=tmp_path)


def test_load_result_normalizes_outputs(tmp_path: Path) -> None:
    _write_intersim_outputs(tmp_path)

    result = intersim._load_result(tmp_path, params=InterSIMParams(seed=99, n_sample=3))

    assert result.methylation.shape == (3, 2)
    assert result.expression.shape == (3, 1)
    assert result.proteomics.shape == (3, 1)
    assert result.sample_ids.tolist() == ["subject1", "subject2", "subject3"]
    assert result.clusters.tolist() == [1, 2, 1]
    assert result.clusters.index.tolist() == result.sample_ids.tolist()
    assert result.metadata["seed"] == 99
    assert result.metadata["n_sample"] == 3
    assert result.metadata["n_features"] == {
        "methylation": 2,
        "expression": 1,
        "proteomics": 1,
    }


def test_load_result_rejects_misaligned_sample_ids(tmp_path: Path) -> None:
    _write_intersim_outputs(tmp_path, aligned=False)

    with pytest.raises(InterSIMMalformedOutputError, match="sample ID alignment"):
        intersim._load_result(tmp_path, params=InterSIMParams(seed=99))


def test_availability_reports_missing_rscript(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(intersim, "which", lambda _: None)

    availability = check_intersim_available()

    assert not availability.available
    assert "Rscript" in availability.message


def test_availability_reports_missing_intersim_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(intersim, "which", lambda _: "/usr/bin/Rscript")

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=args, returncode=1, stdout="", stderr="no package")

    monkeypatch.setattr(intersim.subprocess, "run", fake_run)

    availability = check_intersim_available()

    assert not availability.available
    assert "InterSIM" in availability.message
    assert "no package" in availability.message


def test_run_intersim_raises_dependency_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        intersim,
        "check_intersim_available",
        lambda _: intersim.InterSIMAvailability(False, "Rscript dependency is missing"),
    )

    with pytest.raises(InterSIMDependencyError, match="Rscript"):
        run_intersim(InterSIMParams(seed=1))


def test_run_intersim_invokes_rscript_and_loads_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        intersim,
        "check_intersim_available",
        lambda _: intersim.InterSIMAvailability(True, "ok", "/usr/bin/Rscript"),
    )

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        output_dir = Path(cmd[cmd.index("--output-dir") + 1])
        _write_intersim_outputs(output_dir)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(intersim.subprocess, "run", fake_run)

    result = run_intersim(InterSIMParams(seed=7, n_sample=3))

    assert result.sample_ids.tolist() == ["subject1", "subject2", "subject3"]
    assert result.metadata["seed"] == 7


def test_real_intersim_smoke_invocation_when_available() -> None:
    availability = check_intersim_available()
    if not availability.available:
        pytest.skip(availability.message)

    result = run_intersim(
        InterSIMParams(
            seed=42,
            n_sample=12,
            cluster_sample_prop=(0.25, 0.25, 0.5),
            delta_methyl=0.5,
            delta_expr=0.5,
            delta_protein=0.5,
            p_dmp=0.1,
        )
    )

    assert result.methylation.shape[0] == 12
    assert result.expression.shape[0] == 12
    assert result.proteomics.shape[0] == 12
    assert result.clusters.shape[0] == 12


def test_real_intersim_seed_reproducibility_when_available() -> None:
    availability = check_intersim_available()
    if not availability.available:
        pytest.skip(availability.message)

    params = InterSIMParams(
        seed=1203,
        n_sample=12,
        cluster_sample_prop=(0.25, 0.25, 0.5),
        delta_methyl=0.5,
        delta_expr=0.5,
        delta_protein=0.5,
        p_dmp=0.1,
    )
    first = run_intersim(params)
    second = run_intersim(params)

    pd.testing.assert_frame_equal(first.methylation, second.methylation)
    pd.testing.assert_frame_equal(first.expression, second.expression)
    pd.testing.assert_frame_equal(first.proteomics, second.proteomics)
    pd.testing.assert_series_equal(first.clusters, second.clusters)
