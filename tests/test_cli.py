# tests/test_cli.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from motco.cli import main
from motco.stats.sd import build_ls_means, get_model_matrix


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def plsr_csv(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    n, p = 30, 5
    df = pd.DataFrame(rng.standard_normal((n, p)), columns=[f"f{i}" for i in range(p)])
    df["label"] = ["A"] * 15 + ["B"] * 15
    path = tmp_path / "plsr_data.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def snf_csvs(tmp_path: Path) -> list[Path]:
    rng = np.random.default_rng(0)
    paths = []
    for i in range(2):
        df = pd.DataFrame(rng.standard_normal((15, 4)))
        p = tmp_path / f"omics{i}.csv"
        df.to_csv(p, index=False)
        paths.append(p)
    return paths


@pytest.fixture()
def de_files(tmp_path: Path) -> dict[str, Path]:
    rng = np.random.default_rng(0)
    n = 20
    Y = pd.DataFrame(rng.standard_normal((n, 3)))
    factors = pd.DataFrame({
        "group": ["A"] * 10 + ["B"] * 10,
        "level": ["t0", "t1"] * 10,
    })
    M_full = get_model_matrix(factors, group_col="group", level_col="level", full=True)
    M_red = get_model_matrix(factors, group_col="group", level_col="level", full=False)
    LS = build_ls_means(["A", "B"], ["t0", "t1"], full=True)
    contrast = [[0, 1], [2, 3]]

    paths = {}
    paths["Y"] = tmp_path / "Y.csv"
    paths["model"] = tmp_path / "model.csv"
    paths["model_full"] = tmp_path / "model_full.csv"
    paths["model_red"] = tmp_path / "model_red.csv"
    paths["ls"] = tmp_path / "ls.csv"
    paths["contrast"] = tmp_path / "contrast.json"

    Y.to_csv(paths["Y"], index=False)
    pd.DataFrame(M_full).to_csv(paths["model"], index=False)
    pd.DataFrame(M_full).to_csv(paths["model_full"], index=False)
    pd.DataFrame(M_red).to_csv(paths["model_red"], index=False)
    pd.DataFrame(LS).to_csv(paths["ls"], index=False)
    paths["contrast"].write_text(json.dumps(contrast))
    return paths


# ── plsr subcommand ───────────────────────────────────────────────────────────

def test_plsr_saves_csv(tmp_path: Path, plsr_csv: Path) -> None:
    out = tmp_path / "table.csv"
    main([
        "plsr", "--data", str(plsr_csv), "--label-col", "label",
        "--cv1-splits", "3", "--cv2-splits", "3", "--n-repeats", "2",
        "--max-components", "3", "--out-table", str(out),
    ])
    assert out.exists()
    df = pd.read_csv(out)
    assert df.shape[0] == 2  # n_repeats rows


def test_plsr_prints_to_stdout(capsys: pytest.CaptureFixture, plsr_csv: Path) -> None:
    main([
        "plsr", "--data", str(plsr_csv), "--label-col", "label",
        "--cv1-splits", "3", "--cv2-splits", "3", "--n-repeats", "2",
        "--max-components", "3",
    ])
    out = capsys.readouterr().out
    assert "AUROC" in out


def test_plsr_bad_label_col_exits(plsr_csv: Path) -> None:
    with pytest.raises(SystemExit):
        main([
            "plsr", "--data", str(plsr_csv), "--label-col", "nonexistent",
            "--cv1-splits", "3", "--cv2-splits", "3", "--n-repeats", "2",
            "--max-components", "3",
        ])


# ── snf subcommand ────────────────────────────────────────────────────────────

def test_snf_saves_fused_matrix(tmp_path: Path, snf_csvs: list[Path]) -> None:
    out = tmp_path / "fused.csv"
    main([
        "snf",
        "--input", str(snf_csvs[0]), "--input", str(snf_csvs[1]),
        "--K", "5", "--k", "5", "--t", "3", "--out-fused", str(out),
    ])
    assert out.exists()
    df = pd.read_csv(out)
    assert df.shape == (15, 15)


def test_snf_saves_embedding(tmp_path: Path, snf_csvs: list[Path]) -> None:
    out_fused = tmp_path / "fused.csv"
    out_emb = tmp_path / "emb.csv"
    main([
        "snf",
        "--input", str(snf_csvs[0]), "--input", str(snf_csvs[1]),
        "--K", "5", "--k", "5", "--t", "3",
        "--out-fused", str(out_fused), "--out-embedding", str(out_emb),
    ])
    assert out_emb.exists()
    emb = pd.read_csv(out_emb)
    assert emb.shape == (15, 10)


def test_snf_requires_two_inputs() -> None:
    with pytest.raises(SystemExit):
        main(["snf", "--input", "only_one.csv"])


# ── de subcommand — estimate path ─────────────────────────────────────────────

def test_de_estimate_saves_json(tmp_path: Path, de_files: dict[str, Path]) -> None:
    out = tmp_path / "result.json"
    main([
        "de",
        "--Y", str(de_files["Y"]),
        "--model-matrix", str(de_files["model"]),
        "--ls-means", str(de_files["ls"]),
        "--contrast", str(de_files["contrast"]),
        "--out-json", str(out),
    ])
    assert out.exists()
    result = json.loads(out.read_text())
    assert {"deltas", "angles", "shapes"} <= result.keys()
    # 2 groups → 2×2 matrices
    assert len(result["deltas"]) == 2
    assert len(result["deltas"][0]) == 2


# ── de subcommand — RRPP path ─────────────────────────────────────────────────

def test_de_rrpp_saves_json(tmp_path: Path, de_files: dict[str, Path]) -> None:
    out = tmp_path / "rrpp.json"
    main([
        "de",
        "--Y", str(de_files["Y"]),
        "--model-full", str(de_files["model_full"]),
        "--model-reduced", str(de_files["model_red"]),
        "--ls-means", str(de_files["ls"]),
        "--contrast", str(de_files["contrast"]),
        "--rrpp-permutations", "2",
        "--out-json", str(out),
    ])
    assert out.exists()
    result = json.loads(out.read_text())
    assert {"deltas", "angles", "shapes"} <= result.keys()
    assert len(result["deltas"]) == 2  # 2 permutations
