from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _summary_values(out: dict[str, Any], name: str) -> np.ndarray:
    if name in out["features"]:
        return np.asarray(out["features"][name], dtype=float)
    return np.zeros((0,), dtype=float)


def _empty_column(name: str, length: int) -> np.ndarray:
    if name == "valid_control_mask":
        return np.zeros(length, dtype=bool)
    return np.zeros(length, dtype=np.float32)


def _load_index_with_source_index(index_path: Path, source_index_path: Path) -> pd.DataFrame:
    index_df = pd.read_parquet(index_path).reset_index(names="filtered_index")
    if "source_index" in index_df.columns:
        return index_df
    if source_index_path.exists():
        source_df = pd.read_parquet(source_index_path).reset_index(names="source_index")
        source_df = source_df[(source_df["difficulty"] >= 2.0) & (source_df["difficulty"] <= 6.0)].reset_index(
            drop=True
        )
        same_length = len(source_df) == len(index_df)
        same_maps = (
            same_length
            and source_df["beatmap_id"].astype(str).tolist() == index_df["beatmap_id"].astype(str).tolist()
        )
        if same_maps:
            index_df.insert(1, "source_index", source_df["source_index"].astype(int).to_numpy())
            return index_df
    index_df.insert(1, "source_index", index_df["filtered_index"].astype(int).to_numpy())
    return index_df


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_rev_parse(revision: str) -> str:
    try:
        result = subprocess.run(["git", "rev-parse", revision], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def _git_dirty() -> bool:
    try:
        result = subprocess.run(["git", "status", "--porcelain"], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return True
    return bool(result.stdout.strip())
