"""Data entrypoints for RCT-target workflows.

Canonical LaLonde raw parsing remains in ``utils.lalonde_utils``.
This module provides RCT-facing adapters and split helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from utils.lalonde_utils import (
    CATEGORY_COLUMNS,
    LALONDE_COLUMNS,
    LalondeBuildSummary,
    add_lalonde_engineered_features,
    build_lalonde_dataframe,
    collect_lalonde_txt_files,
    ensure_required_columns,
    infer_group_from_filename,
    load_lalonde_csv,
    read_lalonde_txt_file,
    split_obs_target_groups,
)


def get_repo_root() -> Path:
    """Return repository root path, inferred from this module location."""
    return Path(__file__).resolve().parents[1]


def generate_lalonde_csv(
    data_dir: Optional[str] = None,
    output_path: Optional[str] = None,
    pattern: str = "*.txt",
) -> LalondeBuildSummary:
    """Generate ``lalonde.csv`` from raw TXT files."""
    repo_root = get_repo_root()
    data_dir_path = Path(data_dir) if data_dir is not None else repo_root / "data"
    txt_files = collect_lalonde_txt_files(data_dir_path, pattern=pattern)
    lalonde = build_lalonde_dataframe(txt_files)

    output = Path(output_path) if output_path is not None else repo_root / "lalonde.csv"
    lalonde.to_csv(output, index=False)

    return LalondeBuildSummary(
        output_path=str(output),
        n_rows=int(lalonde.shape[0]),
        n_files=len(txt_files),
        groups=sorted(lalonde["group"].astype(str).unique().tolist()),
    )


def lalonde_get_data(df: pd.DataFrame, group: str, variables: Sequence[str], subsample_idx=None):
    """Select structured RCT and OBS arrays from a LaLonde dataframe.

    Returns arrays shaped as [X..., T, Y].
    """
    variables = list(variables)
    ensure_required_columns(df, ["group", "treatment", "re78", *variables], context="lalonde_get_data")

    x_exp_df = df[df["group"].isin(["control", "treated"])].copy()
    z_exp = x_exp_df[variables].to_numpy() if variables else np.empty((x_exp_df.shape[0], 0))
    a_exp = x_exp_df["treatment"].to_numpy()
    y_exp = x_exp_df["re78"].to_numpy()
    x_exp = np.concatenate((z_exp, a_exp.reshape(-1, 1), y_exp.reshape(-1, 1)), axis=1)

    if subsample_idx is not None:
        x_exp = x_exp[subsample_idx]
        x_obs_df = df[df["group"] == group].copy()
    else:
        x_obs_df = df[df["group"].isin(["treated", group])].copy()

    z_obs = x_obs_df[variables].to_numpy() if variables else np.empty((x_obs_df.shape[0], 0))
    a_obs = x_obs_df["treatment"].to_numpy()
    y_obs = x_obs_df["re78"].to_numpy()
    x_obs = np.concatenate((z_obs, a_obs.reshape(-1, 1), y_obs.reshape(-1, 1)), axis=1)

    if subsample_idx is not None:
        x_treated_df = df[df["group"].isin(["control", "treated"])].iloc[subsample_idx]
        x_treated_df = x_treated_df[x_treated_df["group"] == "treated"]
        z_t = x_treated_df[variables].to_numpy() if variables else np.empty((x_treated_df.shape[0], 0))
        a_t = x_treated_df["treatment"].to_numpy()
        y_t = x_treated_df["re78"].to_numpy()
        x_exp_treated = np.concatenate((z_t, a_t.reshape(-1, 1), y_t.reshape(-1, 1)), axis=1)
        x_obs = np.concatenate((x_obs, x_exp_treated), axis=0)

    return x_exp, x_obs
