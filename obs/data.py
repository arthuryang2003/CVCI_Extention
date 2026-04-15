"""
Data utilities for OBS-target causal inference on LaLonde-style data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from utils.lalonde_utils import OBS_DEFAULT_X_COLS, get_lalonde_default_covariates, load_lalonde_csv, load_lalonde_split


DEFAULT_X_COLS = list(OBS_DEFAULT_X_COLS)


@dataclass
class ObsTargetDataBundle:
    df_rct: pd.DataFrame
    df_obs: pd.DataFrame
    x_cols: List[str]
    metadata: Dict[str, object]


def _pick_first_existing_column(df: pd.DataFrame, candidates: Sequence[str], field_name: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Cannot find column for {field_name}. Tried candidates: {list(candidates)}")


def prepare_obs_target_dataframe(
    df: pd.DataFrame,
    x_cols: Sequence[str],
    a_col: Optional[str] = None,
    y_col: Optional[str] = None,
    g_col: Optional[str] = None,
    group_col: Optional[str] = None,
    rct_label: int = 1,
    obs_label: int = 0,
) -> pd.DataFrame:
    """
    Build a standardized dataframe with fields:
    - X columns from `x_cols`
    - A: treatment indicator
    - Y: outcome
    - G: data source indicator (RCT=1, OBS=0)

    Args:
        df: Raw input dataframe.
        x_cols: Covariate columns to keep as X.
        a_col: Optional treatment column name in input data.
        y_col: Optional outcome column name in input data.
        g_col: Optional source indicator column name in input data.
        group_col: Optional source text label column name if `g_col` not provided.
        rct_label: Value in `g_col` that means RCT.
        obs_label: Value in `g_col` that means OBS.

    Returns:
        Dataframe with selected X columns and standardized columns `A`, `Y`, `G`.
    """
    local_df = df.copy()

    if a_col is None:
        a_col = _pick_first_existing_column(local_df, ["A", "a", "treatment", "treat"], "treatment A")
    if y_col is None:
        y_col = _pick_first_existing_column(local_df, ["Y", "y", "outcome", "re78"], "outcome Y")

    missing_x = [col for col in x_cols if col not in local_df.columns]
    if missing_x:
        raise ValueError(f"Missing requested x_cols in dataframe: {missing_x}")

    result_df = local_df[list(x_cols)].copy()
    result_df["A"] = local_df[a_col].astype(float)
    result_df["Y"] = local_df[y_col].astype(float)

    if g_col is not None:
        result_df["G"] = local_df[g_col].astype(float)
    elif group_col is not None:
        group_values = local_df[group_col].astype(str).str.lower()
        result_df["G"] = 0.0
        result_df.loc[group_values.eq("rct"), "G"] = float(rct_label)
        result_df.loc[group_values.eq("obs"), "G"] = float(obs_label)
    elif "G" in local_df.columns:
        result_df["G"] = local_df["G"].astype(float)
    else:
        raise ValueError("Cannot construct `G`. Provide `g_col` or `group_col`, or include `G` in input df.")

    return result_df


def load_lalonde_obs_target_data(
    lalonde_path: str = "lalonde.csv",
    obs_source: str = "cps",
    x_cols: Optional[Sequence[str]] = None,
    extra_feature_cols: Optional[Sequence[str]] = None,
) -> ObsTargetDataBundle:
    """
    Load LaLonde data and construct OBS-target split.

    OBS-target setting used here:
    - RCT data: NSW treated + NSW control (groups: treated + control), with G=1
    - OBS data: NSW treated + selected observational controls (e.g., cps / psid), with G=0

    Args:
        lalonde_path: Path to `lalonde.csv`.
        obs_source: Observational controls source, e.g. `cps` or `psid`.
        x_cols: Covariates for X; default uses common LaLonde baseline covariates.

    Returns:
        ObsTargetDataBundle containing standardized `df_rct`, `df_obs`, `x_cols`, and metadata.
    """
    if x_cols is None:
        raw_df = load_lalonde_csv(lalonde_path)
        x_cols = get_lalonde_default_covariates(raw_df)
    x_cols = list(x_cols)
    extra_feature_cols = [] if extra_feature_cols is None else list(extra_feature_cols)
    all_feature_cols = list(dict.fromkeys(x_cols + extra_feature_cols))
    df_rct_split, df_obs_split, split_summary = load_lalonde_split(
        target_mode="obs",
        obs_source=obs_source,
        x_cols=all_feature_cols,
        lalonde_path=lalonde_path,
    )

    df_rct = prepare_obs_target_dataframe(df_rct_split, x_cols=all_feature_cols, a_col="T", y_col="Y", g_col="G")
    df_obs = prepare_obs_target_dataframe(df_obs_split, x_cols=all_feature_cols, a_col="T", y_col="Y", g_col="G")

    metadata = dict(split_summary)
    metadata.update(
        {
            "n_rct_treated": int(df_rct["A"].sum()),
            "n_rct_control": int((1 - df_rct["A"]).sum()),
            "n_obs_treated": int(df_obs["A"].sum()),
            "n_obs_control": int((1 - df_obs["A"]).sum()),
        }
    )

    return ObsTargetDataBundle(df_rct=df_rct, df_obs=df_obs, x_cols=x_cols, metadata=metadata)
