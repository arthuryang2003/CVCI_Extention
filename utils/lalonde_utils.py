"""Shared LaLonde dataset helpers used by both OBS and RCT modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import pandas as pd

LALONDE_RAW_COLUMNS: List[str] = [
    "treatment",
    "age",
    "education",
    "black",
    "hispanic",
    "married",
    "nodegree",
    "re74",
    "re75",
    "re78",
]

LALONDE_CATEGORY_COLUMNS: List[str] = ["treatment", "black", "hispanic", "married", "nodegree", "u74", "u75"]
LALONDE_COLUMNS: List[str] = LALONDE_RAW_COLUMNS
CATEGORY_COLUMNS: List[str] = LALONDE_CATEGORY_COLUMNS

OBS_DEFAULT_X_COLS: List[str] = [
    "age",
    "education",
    "black",
    "hispanic",
    "married",
    "nodegree",
    "re74",
    "re75",
]


@dataclass
class LalondeBuildSummary:
    output_path: str
    n_rows: int
    n_files: int
    groups: List[str]


def infer_group_from_filename(path: Path) -> str:
    """Infer group label from source TXT filename."""
    tokens = path.stem.split("_")
    if len(tokens) < 2:
        raise ValueError(f"Unexpected LaLonde filename format: {path.name}")
    return tokens[0] if tokens[1] == "controls" else tokens[1]


def read_lalonde_txt_file(path: Path) -> pd.DataFrame:
    """Read one raw LaLonde TXT file into canonical tabular schema."""
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] != len(LALONDE_RAW_COLUMNS):
        raise ValueError(
            f"Invalid column count for {path.name}: expected {len(LALONDE_RAW_COLUMNS)}, got {df.shape[1]}"
        )
    df.columns = LALONDE_RAW_COLUMNS
    df["group"] = infer_group_from_filename(path)
    return df


def collect_lalonde_txt_files(data_dir: Path, pattern: str = "*.txt") -> List[Path]:
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No txt files found in {data_dir} with pattern '{pattern}'.")
    return files


def add_lalonde_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add common engineered columns used across OBS/RCT experiments."""
    out = df.copy()
    if "re74" in out.columns and "u74" not in out.columns:
        out["u74"] = (out["re74"] == 0).astype(int)
    if "re75" in out.columns and "u75" not in out.columns:
        out["u75"] = (out["re75"] == 0).astype(int)
    if "age" in out.columns and "age2" not in out.columns:
        out["age2"] = out["age"] ** 2
    return out


def build_lalonde_dataframe(txt_files: Iterable[Path]) -> pd.DataFrame:
    """Build full LaLonde dataframe from TXT files with shared engineered features."""
    frames = [read_lalonde_txt_file(path) for path in txt_files]
    lalonde = pd.concat(frames, ignore_index=True)
    lalonde = add_lalonde_engineered_features(lalonde)

    for col in LALONDE_CATEGORY_COLUMNS:
        if col in lalonde.columns:
            lalonde[col] = lalonde[col].astype("category")

    return lalonde


def get_repo_root_from_file(anchor_file: str, levels_up: int) -> Path:
    """Resolve repository root path from an anchor file path and parent depth."""
    return Path(anchor_file).resolve().parents[levels_up]


def load_lalonde_csv(lalonde_path: str) -> pd.DataFrame:
    """Load ``lalonde.csv`` and ensure common engineered features exist."""
    raw_df = pd.read_csv(lalonde_path)
    raw_df.columns = [str(col) for col in raw_df.columns]
    return add_lalonde_engineered_features(raw_df)


def split_obs_target_groups(df: pd.DataFrame, obs_source: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split LaLonde dataframe into RCT and OBS-target raw subsets.

    RCT: treated + control
    OBS-target: treated + obs_source
    """
    if "group" not in df.columns:
        raise ValueError("Expected a `group` column in Lalonde data.")

    obs_source = str(obs_source).lower()
    group_lower = df["group"].astype(str).str.lower()
    available_groups = set(group_lower.unique().tolist())
    if obs_source not in available_groups:
        raise ValueError(f"obs_source={obs_source} not found in data groups: {sorted(available_groups)}")

    df_rct_raw = df[group_lower.isin(["treated", "control"])].copy()
    df_obs_raw = df[group_lower.isin(["treated", obs_source])].copy()
    return df_rct_raw, df_obs_raw


def ensure_required_columns(df: pd.DataFrame, required_cols: Sequence[str], context: str) -> None:
    """Raise a clear error when required columns are missing."""
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for {context}: {missing_cols}")
