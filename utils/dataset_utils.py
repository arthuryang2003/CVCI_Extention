from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

from utils.lalonde_utils import load_lalonde_split


def _find_first_existing_col(df: pd.DataFrame, candidates: Sequence[str], field_name: str) -> str:
    cols = {str(c).lower(): str(c) for c in df.columns}
    for cand in candidates:
        key = str(cand).lower()
        if key in cols:
            return cols[key]
    raise ValueError(
        f"Failed to infer {field_name}. Please pass --{field_name.replace('_', '-')} explicitly. "
        f"Candidates={list(candidates)}, available_columns={list(df.columns)}"
    )


def _coerce_treatment_binary(series: pd.Series, col_name: str) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    uniq = sorted(pd.Series(s.dropna().unique()).tolist())
    if len(uniq) < 2:
        raise ValueError(f"Treatment column '{col_name}' must contain at least two values.")
    if set(uniq).issubset({0, 1}):
        return s.astype(float)
    if set(uniq).issubset({1, 2}):
        return (s == 2).astype(float)
    if len(uniq) == 2:
        return (s == max(uniq)).astype(float)
    raise ValueError(
        f"Treatment column '{col_name}' is not binary-like. unique_values={uniq}. "
        "Please provide a binary treatment indicator column via --treatment-col."
    )


def _is_excluded_covariate(col: str, treatment_col: str, outcome_col: str, site_col: Optional[str]) -> bool:
    c = str(col)
    c_low = c.lower()
    explicit = {
        str(treatment_col).lower(),
        str(outcome_col).lower(),
        "t",
        "y",
        "g",
        "source",
        "group",
        "site",
        "site_name",
        "center",
        "location",
    }
    if site_col is not None:
        explicit.add(str(site_col).lower())
    if c_low in explicit:
        return True
    if c_low == "id" or c_low.endswith("_id") or c_low.startswith("id_"):
        return True
    return False


def _validate_treatment_arms(df: pd.DataFrame, frame_name: str) -> None:
    t = df["T"].to_numpy(dtype=float)
    n_t = int(np.sum(t == 1))
    n_c = int(np.sum(t == 0))
    if n_t == 0 or n_c == 0:
        raise ValueError(
            f"{frame_name} must contain both treated and control groups after preprocessing. "
            f"treated={n_t}, control={n_c}."
        )


def _load_metadata_for_sim_dataset(data_path: str) -> Dict[str, object]:
    base_path = Path(data_path)
    metadata_candidates = []
    stem = base_path.stem
    if stem.endswith("_combined"):
        metadata_candidates.append(base_path.with_name(f"{stem[:-9]}_metadata.json"))
    metadata_candidates.append(base_path.with_name(f"{stem}_metadata.json"))
    metadata_candidates.append(base_path.with_suffix(".json"))

    for candidate in metadata_candidates:
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as fh:
                return dict(json.load(fh))
    raise ValueError(
        f"Failed to locate simulation metadata json for data_path={data_path}. "
        f"Tried candidates={[str(p) for p in metadata_candidates]}"
    )


def _load_obs_target_sim_split(
    dataset: str,
    data_path: str,
    target_mode: str,
    x_cols: Optional[Sequence[str]],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    raw_df = pd.read_csv(data_path)
    raw_df.columns = [str(c) for c in raw_df.columns]
    if "G" not in raw_df.columns:
        raise ValueError(f"Simulation dataset '{dataset}' must contain column 'G'.")
    if "T" not in raw_df.columns or "Y" not in raw_df.columns:
        raise ValueError(f"Simulation dataset '{dataset}' must contain columns 'T' and 'Y'.")

    metadata = _load_metadata_for_sim_dataset(data_path)
    default_feature_cols = metadata.get("candidate_feature_cols") or metadata.get("x_cols")
    if default_feature_cols is None:
        raise ValueError(f"Simulation metadata for dataset '{dataset}' must include candidate_feature_cols or x_cols.")

    if x_cols is None:
        x_cols_working = [str(c) for c in default_feature_cols]
    else:
        x_cols_working = [str(c) for c in x_cols]
    missing = [c for c in [*x_cols_working, "T", "Y", "G"] if c not in raw_df.columns]
    if missing:
        raise ValueError(f"Simulation dataset '{dataset}' is missing required columns: {missing}")

    df_rct = raw_df[raw_df["G"] == 1].copy()
    df_obs = raw_df[raw_df["G"] == 0].copy()
    if df_rct.empty or df_obs.empty:
        raise ValueError(
            f"Simulation dataset '{dataset}' must contain both G=1 and G=0 samples. "
            f"n_rct={df_rct.shape[0]}, n_obs={df_obs.shape[0]}"
        )

    keep_cols = list(dict.fromkeys([*x_cols_working, "T", "Y", "G", *[str(c) for c in raw_df.columns if c not in x_cols_working]]))
    df_rct = df_rct[keep_cols].copy()
    df_obs = df_obs[keep_cols].copy()

    for frame in (df_rct, df_obs):
        frame["T"] = pd.to_numeric(frame["T"], errors="coerce").astype(float)
        frame["Y"] = pd.to_numeric(frame["Y"], errors="coerce").astype(float)
        frame["G"] = pd.to_numeric(frame["G"], errors="coerce").astype(float)
    _validate_treatment_arms(df_rct, "df_rct")
    _validate_treatment_arms(df_obs, "df_obs")

    summary: Dict[str, object] = {
        "dataset": dataset,
        "target_mode": target_mode,
        "data_path": data_path,
        "n_rct": int(df_rct.shape[0]),
        "n_obs": int(df_obs.shape[0]),
        "n_rct_treated": int((df_rct["T"] == 1).sum()),
        "n_rct_control": int((df_rct["T"] == 0).sum()),
        "n_obs_treated": int((df_obs["T"] == 1).sum()),
        "n_obs_control": int((df_obs["T"] == 0).sum()),
        "x_cols": x_cols_working,
        "base_x_cols": metadata.get("x_cols", ["X1", "X2", "X3"]),
        "iv_cols": metadata.get("iv_cols", []),
        "shadow_cols": metadata.get("shadow_cols", []),
        "treatment_col": "T",
        "outcome_col": "Y",
        "site_col": None,
        "g_semantics": metadata.get("g_semantics", {"1": "RCT", "0": "OBS"}),
        "simulation_metadata": metadata,
    }
    return df_rct, df_obs, summary


def _postprocess_tabular_dataset(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    site_col: Optional[str],
    x_cols: Optional[Sequence[str]],
    warnings: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()

    n_before = out.shape[0]
    out[treatment_col] = pd.to_numeric(out[treatment_col], errors="coerce")
    out[outcome_col] = pd.to_numeric(out[outcome_col], errors="coerce")
    out = out.dropna(subset=[treatment_col, outcome_col]).copy()
    dropped = n_before - out.shape[0]
    if dropped > 0:
        warnings.append(f"Dropped {dropped} rows due to missing treatment/outcome.")

    out[treatment_col] = _coerce_treatment_binary(out[treatment_col], treatment_col)
    out[outcome_col] = pd.to_numeric(out[outcome_col], errors="coerce").astype(float)

    if x_cols is None:
        candidate_cols = []
        for c in out.columns:
            if not _is_excluded_covariate(c, treatment_col=treatment_col, outcome_col=outcome_col, site_col=site_col):
                candidate_cols.append(str(c))
        x_cols_working = candidate_cols
    else:
        x_cols_working = [str(c) for c in x_cols]
        missing = [c for c in x_cols_working if c not in out.columns]
        if missing:
            raise ValueError(f"User-provided --x-cols contains missing columns: {missing}")

    num_cols: List[str] = []
    cat_cols: List[str] = []
    for c in x_cols_working:
        if pd.api.types.is_numeric_dtype(out[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)

    x_num = out[num_cols].copy() if num_cols else pd.DataFrame(index=out.index)
    for c in num_cols:
        x_num[c] = pd.to_numeric(x_num[c], errors="coerce")
        x_num[c] = x_num[c].fillna(x_num[c].median())

    x_cat = out[cat_cols].copy() if cat_cols else pd.DataFrame(index=out.index)
    for c in cat_cols:
        x_cat[c] = x_cat[c].astype(str).fillna("missing")
    x_cat = pd.get_dummies(x_cat, drop_first=True) if cat_cols else x_cat

    x_df = pd.concat([x_num, x_cat], axis=1)
    x_df.columns = [str(c) for c in x_df.columns]

    result = pd.concat(
        [
            x_df,
            out[[treatment_col, outcome_col]].rename(columns={treatment_col: "T", outcome_col: "Y"}),
        ],
        axis=1,
    )
    result["T"] = pd.to_numeric(result["T"], errors="coerce").astype(float)
    result["Y"] = pd.to_numeric(result["Y"], errors="coerce").astype(float)

    bad_x = [c for c in x_df.columns if c.lower() in {"t", "y", "g", "site", "source", "group", "id"}]
    if bad_x:
        raise ValueError(f"x_cols contains forbidden columns: {bad_x}")

    return result, list(x_df.columns)


def load_dataset_split(
    dataset: str,
    data_path: Optional[str] = None,
    target_mode: str = "rct",
    x_cols: Optional[Sequence[str]] = None,
    seed: int = 2024,
    obs_source: str = "cps",
    lalonde_path: str = "lalonde.csv",
    data_mode: str = "real",
    semisynth_config: Optional[dict] = None,
    treatment_col: Optional[str] = None,
    outcome_col: Optional[str] = None,
    site_col: Optional[str] = None,
    jtpa_rct_site: str = "Coosa Valley",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    dataset = str(dataset).lower()
    target_mode = str(target_mode).lower()
    warnings: List[str] = []

    if target_mode not in {"rct", "obs"}:
        raise ValueError(f"target_mode must be one of ['rct', 'obs'], got {target_mode}")

    if dataset == "lalonde":
        df_rct, df_obs, summary = load_lalonde_split(
            target_mode=target_mode,
            obs_source=obs_source,
            x_cols=x_cols,
            lalonde_path=lalonde_path,
            data_mode=data_mode,
            semisynth_config=semisynth_config,
        )
        summary = dict(summary)
        summary.update(
            {
                "dataset": "lalonde",
                "data_path": lalonde_path,
                "treatment_col": "T",
                "outcome_col": "Y",
                "site_col": None,
            }
        )
        if warnings:
            summary["warnings"] = warnings
        return df_rct, df_obs, summary

    if dataset in {"sim_obs_iv", "sim_obs_shadow"}:
        if not data_path:
            raise ValueError(f"dataset='{dataset}' requires --data-path to the generated combined csv.")
        return _load_obs_target_sim_split(dataset=dataset, data_path=data_path, target_mode=target_mode, x_cols=x_cols)

    if dataset not in {"actg", "jtpa"}:
        raise ValueError(f"Unsupported dataset '{dataset}'. choices=['lalonde','actg','jtpa','sim_obs_iv','sim_obs_shadow']")

    if not data_path:
        raise ValueError(f"dataset='{dataset}' requires --data-path")

    rng = np.random.default_rng(int(seed))
    raw_df = pd.read_csv(data_path)
    raw_df.columns = [str(c) for c in raw_df.columns]

    if dataset == "actg":
        t_col = treatment_col or _find_first_existing_col(
            raw_df, ["T", "A", "treatment", "treat", "treated", "trt"], "treatment_col"
        )
        y_col = outcome_col or _find_first_existing_col(
            raw_df,
            ["Y", "outcome", "cd4", "cd40", "cd4_post", "post_cd4", "posttreatment_cd4", "cd4_posttreatment"],
            "outcome_col",
        )
        age_col = _find_first_existing_col(raw_df, ["age", "Age"], "age_col")

        age_num = pd.to_numeric(raw_df[age_col], errors="coerce")
        obs_pool = raw_df[(age_num < 30) | (age_num > 50)].copy()
        if obs_pool.shape[0] < 558:
            warnings.append(
                f"ACTG OBS pool size {obs_pool.shape[0]} < 558; using all eligible samples as OBS."
            )
            obs_idx = obs_pool.index
        else:
            obs_idx = rng.choice(obs_pool.index.to_numpy(), size=558, replace=False)

        df_obs_raw = raw_df.loc[obs_idx].copy()
        df_rct_raw = raw_df.drop(index=obs_idx).copy()
        s_col = None

    else:  # jtpa
        t_col = treatment_col or _find_first_existing_col(
            raw_df, ["T", "A", "treatment", "treat", "assigned", "program"], "treatment_col"
        )
        y_col = outcome_col or _find_first_existing_col(
            raw_df, ["Y", "outcome", "earnings", "income", "employment", "employed"], "outcome_col"
        )
        s_col = site_col
        if s_col is None:
            s_col = _find_first_existing_col(raw_df, ["site", "site_name", "center", "location"], "site_col")

        site_text = raw_df[s_col].astype(str)
        is_rct = site_text.str.contains(str(jtpa_rct_site), case=False, regex=False, na=False)
        if int(is_rct.sum()) == 0:
            raise ValueError(
                f"No rows matched jtpa_rct_site='{jtpa_rct_site}' in site_col='{s_col}'. "
                "Please check --jtpa-rct-site or --site-col."
            )
        df_rct_raw = raw_df[is_rct].copy()
        df_obs_raw = raw_df[~is_rct].copy()

    df_rct_std, resolved_x_cols = _postprocess_tabular_dataset(
        df=df_rct_raw,
        treatment_col=t_col,
        outcome_col=y_col,
        site_col=s_col,
        x_cols=x_cols,
        warnings=warnings,
    )
    df_obs_std, obs_x_cols = _postprocess_tabular_dataset(
        df=df_obs_raw,
        treatment_col=t_col,
        outcome_col=y_col,
        site_col=s_col,
        x_cols=x_cols,
        warnings=warnings,
    )

    x_union = list(dict.fromkeys([*resolved_x_cols, *obs_x_cols]))
    for c in x_union:
        if c not in df_rct_std.columns:
            df_rct_std[c] = 0.0
        if c not in df_obs_std.columns:
            df_obs_std[c] = 0.0
    ordered_cols = x_union + ["T", "Y"]
    df_rct = df_rct_std[ordered_cols].copy()
    df_obs = df_obs_std[ordered_cols].copy()

    df_rct["G"] = 1.0
    df_obs["G"] = 0.0

    for frame in (df_rct, df_obs):
        frame["T"] = pd.to_numeric(frame["T"], errors="coerce").astype(float)
        frame["Y"] = pd.to_numeric(frame["Y"], errors="coerce").astype(float)
        frame["G"] = pd.to_numeric(frame["G"], errors="coerce").astype(float)

    _validate_treatment_arms(df_rct, "df_rct")
    _validate_treatment_arms(df_obs, "df_obs")

    summary: Dict[str, object] = {
        "dataset": dataset,
        "target_mode": target_mode,
        "data_path": data_path,
        "n_rct": int(df_rct.shape[0]),
        "n_obs": int(df_obs.shape[0]),
        "n_rct_treated": int((df_rct["T"] == 1).sum()),
        "n_rct_control": int((df_rct["T"] == 0).sum()),
        "n_obs_treated": int((df_obs["T"] == 1).sum()),
        "n_obs_control": int((df_obs["T"] == 0).sum()),
        "x_cols": x_union,
        "treatment_col": t_col,
        "outcome_col": y_col,
        "site_col": s_col,
        "g_semantics": {"1": "RCT", "0": "OBS"},
    }
    if warnings:
        summary["warnings"] = warnings

    return df_rct, df_obs, summary
