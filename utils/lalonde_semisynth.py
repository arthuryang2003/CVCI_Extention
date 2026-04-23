"""Semi-synthetic LaLonde data generation with sample-level ground truth effects."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def _safe_sigma(residual: np.ndarray) -> float:
    residual = np.asarray(residual, dtype=float).reshape(-1)
    if residual.shape[0] <= 1:
        return 1e-6
    return float(max(np.std(residual, ddof=1), 1e-6))


def _split_lalonde_raw(df: pd.DataFrame, obs_source: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "group" not in df.columns:
        raise ValueError("Expected `group` column in raw LaLonde dataframe.")
    obs_source = str(obs_source).lower()
    group_lower = df["group"].astype(str).str.lower()
    available = set(group_lower.unique().tolist())
    if obs_source not in available:
        raise ValueError(f"obs_source={obs_source} not found in raw group labels: {sorted(available)}")
    df_rct_raw = df[group_lower.isin(["treated", "control"])].copy()
    df_obs_raw = df[group_lower.isin(["treated", obs_source])].copy()
    return df_rct_raw, df_obs_raw


def _fit_truth_model(
    df_truth: pd.DataFrame,
    x_cols: List[str],
    effect_mode: str,
) -> Tuple[LinearRegression, np.ndarray, np.ndarray, np.ndarray]:
    x_mat = df_truth[x_cols].to_numpy(dtype=float) if x_cols else np.zeros((df_truth.shape[0], 0), dtype=float)
    t_vec = df_truth["treatment"].to_numpy(dtype=float).reshape(-1, 1)
    y_vec = df_truth["re78"].to_numpy(dtype=float)

    if effect_mode == "constant":
        design = np.concatenate((t_vec, x_mat), axis=1)
    elif effect_mode == "linear":
        tx_mat = x_mat * t_vec if x_mat.shape[1] > 0 else np.zeros((df_truth.shape[0], 0), dtype=float)
        design = np.concatenate((t_vec, x_mat, tx_mat), axis=1)
    else:
        raise ValueError(f"Unsupported effect_mode={effect_mode}.")

    model = LinearRegression()
    model.fit(design, y_vec)
    return model, design, x_mat, y_vec


def _compute_mu0_tau(
    x_mat: np.ndarray,
    model: LinearRegression,
    effect_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    d = x_mat.shape[1]
    coef = np.asarray(model.coef_, dtype=float).reshape(-1)
    intercept = float(model.intercept_)
    beta_t = float(coef[0]) if coef.size > 0 else 0.0
    beta_x = coef[1 : 1 + d] if d > 0 else np.zeros(0, dtype=float)

    mu0 = intercept + (x_mat @ beta_x if d > 0 else 0.0)
    if effect_mode == "constant":
        tau = np.full(x_mat.shape[0], beta_t, dtype=float)
        return np.asarray(mu0, dtype=float).reshape(-1), tau

    beta_tx = coef[1 + d : 1 + 2 * d] if d > 0 else np.zeros(0, dtype=float)
    tau = beta_t + (x_mat @ beta_tx if d > 0 else 0.0)
    return np.asarray(mu0, dtype=float).reshape(-1), np.asarray(tau, dtype=float).reshape(-1)


def _build_semisynth_split(
    df_base: pd.DataFrame,
    x_cols: List[str],
    sigma: float,
    model: LinearRegression,
    effect_mode: str,
    g_value: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df_base.index)
    for col in x_cols:
        out[col] = df_base[col].to_numpy(dtype=float)

    x_mat = df_base[x_cols].to_numpy(dtype=float) if x_cols else np.zeros((df_base.shape[0], 0), dtype=float)
    t_vec = df_base["treatment"].to_numpy(dtype=float).reshape(-1)
    mu0, tau = _compute_mu0_tau(x_mat=x_mat, model=model, effect_mode=effect_mode)
    eps = rng.normal(loc=0.0, scale=float(max(sigma, 1e-6)), size=df_base.shape[0]).reshape(-1)
    y0_true = mu0 + eps
    y1_true = mu0 + tau + eps
    y_obs = t_vec * y1_true + (1.0 - t_vec) * y0_true

    out["T"] = t_vec.astype(float)
    out["Y"] = y_obs.astype(float)
    out["G"] = float(g_value)
    out["tau_true"] = tau.astype(float)
    out["y0_true"] = y0_true.astype(float)
    out["y1_true"] = y1_true.astype(float)
    return out


def build_lalonde_semisynth_data(
    raw_df: pd.DataFrame,
    obs_source: str,
    x_cols: List[str],
    effect_mode: str = "constant",
    truth_source: str = "rct",
    noise_mode: str = "groupwise",
    seed: int = 2024,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Generate semi-synthetic LaLonde splits with sample-level potential outcomes.

    The real sample structure/covariates/treatment/source are preserved while observed
    outcome Y is regenerated from a parametric truth model and additive noise.
    """
    effect_mode = str(effect_mode).lower()
    truth_source = str(truth_source).lower()
    noise_mode = str(noise_mode).lower()
    if effect_mode not in {"constant", "linear"}:
        raise ValueError(f"effect_mode must be one of ['constant', 'linear'], got {effect_mode}.")
    if truth_source not in {"rct", "pooled"}:
        raise ValueError(f"truth_source must be one of ['rct', 'pooled'], got {truth_source}.")
    if noise_mode not in {"shared", "groupwise"}:
        raise ValueError(f"noise_mode must be one of ['shared', 'groupwise'], got {noise_mode}.")

    x_cols = [str(c) for c in x_cols]
    required_cols = x_cols + ["group", "treatment", "re78"]
    missing = [c for c in required_cols if c not in raw_df.columns]
    if missing:
        raise ValueError(f"Missing required columns for semi-synthetic Lalonde generation: {missing}")

    df_rct_raw, df_obs_raw = _split_lalonde_raw(raw_df, obs_source=obs_source)
    if truth_source == "rct":
        df_truth = df_rct_raw.copy()
    else:
        group_lower = raw_df["group"].astype(str).str.lower()
        df_truth = raw_df[group_lower.isin(["treated", "control", str(obs_source).lower()])].copy()

    truth_model, truth_design, _, truth_y = _fit_truth_model(df_truth=df_truth, x_cols=x_cols, effect_mode=effect_mode)
    truth_resid = truth_y - truth_model.predict(truth_design)

    x_rct = df_rct_raw[x_cols].to_numpy(dtype=float) if x_cols else np.zeros((df_rct_raw.shape[0], 0), dtype=float)
    x_obs = df_obs_raw[x_cols].to_numpy(dtype=float) if x_cols else np.zeros((df_obs_raw.shape[0], 0), dtype=float)
    t_rct = df_rct_raw["treatment"].to_numpy(dtype=float).reshape(-1, 1)
    t_obs = df_obs_raw["treatment"].to_numpy(dtype=float).reshape(-1, 1)
    if effect_mode == "constant":
        pred_rct = truth_model.predict(np.concatenate((t_rct, x_rct), axis=1))
        pred_obs = truth_model.predict(np.concatenate((t_obs, x_obs), axis=1))
    else:
        tx_rct = x_rct * t_rct if x_rct.shape[1] > 0 else np.zeros((df_rct_raw.shape[0], 0), dtype=float)
        tx_obs = x_obs * t_obs if x_obs.shape[1] > 0 else np.zeros((df_obs_raw.shape[0], 0), dtype=float)
        pred_rct = truth_model.predict(np.concatenate((t_rct, x_rct, tx_rct), axis=1))
        pred_obs = truth_model.predict(np.concatenate((t_obs, x_obs, tx_obs), axis=1))
    resid_rct = df_rct_raw["re78"].to_numpy(dtype=float) - pred_rct
    resid_obs = df_obs_raw["re78"].to_numpy(dtype=float) - pred_obs

    if noise_mode == "shared":
        sigma_shared = _safe_sigma(truth_resid)
        sigma_rct = sigma_shared
        sigma_obs = sigma_shared
    else:
        sigma_rct = _safe_sigma(resid_rct)
        sigma_obs = _safe_sigma(resid_obs)

    rng = np.random.default_rng(int(seed))
    df_rct_syn = _build_semisynth_split(
        df_base=df_rct_raw,
        x_cols=x_cols,
        sigma=sigma_rct,
        model=truth_model,
        effect_mode=effect_mode,
        g_value=1.0,
        rng=rng,
    )
    df_obs_syn = _build_semisynth_split(
        df_base=df_obs_raw,
        x_cols=x_cols,
        sigma=sigma_obs,
        model=truth_model,
        effect_mode=effect_mode,
        g_value=0.0,
        rng=rng,
    )

    truth_summary: Dict[str, object] = {
        "effect_mode": effect_mode,
        "truth_source": truth_source,
        "noise_mode": noise_mode,
        "true_ate_rct": float(df_rct_syn["tau_true"].mean()),
        "true_ate_obs": float(df_obs_syn["tau_true"].mean()),
        "true_ate_all": float(
            pd.concat([df_rct_syn["tau_true"], df_obs_syn["tau_true"]], axis=0, ignore_index=True).mean()
        ),
        "sigma_rct": float(sigma_rct),
        "sigma_obs": float(sigma_obs),
        "x_cols": list(x_cols),
        "seed": int(seed),
        "n_rct": int(df_rct_syn.shape[0]),
        "n_obs": int(df_obs_syn.shape[0]),
    }
    return df_rct_syn, df_obs_syn, truth_summary

