"""Core selection-IV implementation shared by CVCI and RHC hosts."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from utils.screening_utils import partial_abs_corr


def _decision_logit(model: LogisticRegression, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(x), dtype=float).reshape(-1)
    p = np.clip(model.predict_proba(x)[:, 1], 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def select_iv_candidates(
    df: pd.DataFrame,
    candidate_cols: Sequence[str],
    t_col: str = "T",
    y_col: str = "Y",
    g_col: str = "G",
    relevance_threshold: float = 0.02,
    exclusion_threshold: float = 0.02,
    allow_empty_fallback: bool = True,
) -> Dict[str, object]:
    """
    Empirical selection-IV screening over all candidate covariates.

    - relevance: association with source indicator G conditional on T and remaining X
    - exclusion proxy: low association with Y conditional on T and remaining X
    """
    x_cols = [str(c) for c in candidate_cols]
    required = [*x_cols, t_col, y_col, g_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for IV screening: {missing}")
    if not x_cols:
        raise ValueError("candidate_cols must be non-empty for IV screening.")

    g_vec = df[g_col].to_numpy(dtype=float)
    y_vec = df[y_col].to_numpy(dtype=float)
    t_vec = df[t_col].to_numpy(dtype=float)

    logs: List[Dict[str, object]] = []
    selected_iv_cols: List[str] = []

    for col in x_cols:
        remaining = [x for x in x_cols if x != col]
        z = df[col].to_numpy(dtype=float)

        nuisance_parts = [t_vec.reshape(-1, 1)]
        if remaining:
            nuisance_parts.append(df[remaining].to_numpy(dtype=float))
        nuisance = np.hstack(nuisance_parts)

        relevance_score = float(partial_abs_corr(z, g_vec, nuisance))
        exclusion_score = float(partial_abs_corr(z, y_vec, nuisance))
        passed = bool(relevance_score >= relevance_threshold and exclusion_score <= exclusion_threshold)
        if passed:
            selected_iv_cols.append(col)

        logs.append(
            {
                "column": col,
                "relevance_score": relevance_score,
                "exclusion_score": exclusion_score,
                "selected": passed,
                "relevance_threshold": float(relevance_threshold),
                "exclusion_threshold": float(exclusion_threshold),
            }
        )

    if not selected_iv_cols:
        if not allow_empty_fallback:
            raise ValueError(
                "No IV candidates passed screening. "
                "Set allow_empty_fallback=True to select a best empirical fallback."
            )
        best_idx = int(np.argmax([entry["relevance_score"] - entry["exclusion_score"] for entry in logs]))
        selected_iv_cols = [logs[best_idx]["column"]]
        logs[best_idx]["selected"] = True
        logs[best_idx]["fallback_selected"] = True

    xz_set = set(selected_iv_cols)
    xc_cols = [x for x in x_cols if x not in xz_set]
    return {
        "selected_iv_cols": selected_iv_cols,
        "Xc_cols": xc_cols,
        "screening_logs": logs,
        "relevance_threshold": float(relevance_threshold),
        "exclusion_threshold": float(exclusion_threshold),
        "allow_empty_fallback": bool(allow_empty_fallback),
    }


def fit_selection_bias_model(
    df: pd.DataFrame,
    xczyt_cols: Sequence[str],
    g_col: str = "G",
    max_iter: int = 2000,
) -> LogisticRegression:
    """Fit selection-bias model eta(x, y, z, t) via logistic regression."""
    model = LogisticRegression(max_iter=max_iter)
    model.fit(df[list(xczyt_cols)].to_numpy(dtype=float), df[g_col].to_numpy(dtype=float))
    return model


def predict_eta(
    eta_model: LogisticRegression,
    baseline_logit: np.ndarray,
    xczyt: np.ndarray,
) -> np.ndarray:
    """Predict eta as the log-odds residual over baseline participation log-odds."""
    full_logit = _decision_logit(eta_model, xczyt)
    baseline_logit = np.asarray(baseline_logit, dtype=float).reshape(-1)
    return full_logit - baseline_logit


def fit_iv_pipeline(
    df: pd.DataFrame,
    Xc_cols: Sequence[str],
    Xz_cols: Sequence[str],
    t_col: str = "T",
    y_col: str = "Y",
    g_col: str = "G",
    y_ref: float = 0.0,
    prob_clip_eps: float = 1e-6,
    weight_clip_min: float = 0.05,
    weight_clip_max: float = 20.0,
    max_iter: int = 2000,
) -> np.ndarray:
    """
    Fit selection-IV participation models and return final correction weights only.

    Returns:
        weights: numpy array with one weight per row in df
    """
    xc_cols = [str(c) for c in Xc_cols]
    xz_cols = [str(c) for c in Xz_cols]
    xczt_cols = list(dict.fromkeys(xc_cols + xz_cols + [t_col]))
    xczyt_cols = list(dict.fromkeys(xczt_cols + [y_col]))

    for col in [*xczt_cols, *xczyt_cols, g_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required column for IV pipeline: {col}")

    baseline_model = LogisticRegression(max_iter=max_iter)
    baseline_model.fit(df[xczt_cols].to_numpy(dtype=float), df[g_col].to_numpy(dtype=float))
    baseline_logit = _decision_logit(baseline_model, df[xczt_cols].to_numpy(dtype=float))

    eta_model = fit_selection_bias_model(df, xczyt_cols=xczyt_cols, g_col=g_col, max_iter=max_iter)
    xczyt = df[xczyt_cols].to_numpy(dtype=float)
    eta = predict_eta(eta_model, baseline_logit=baseline_logit, xczyt=xczyt)

    _ = y_ref  # reserved parameter for compatibility with baseline-reference extensions
    pi = expit(eta + baseline_logit)
    pi = np.clip(pi, prob_clip_eps, 1.0 - prob_clip_eps)

    weights = (1.0 - pi) / pi
    weights = np.asarray(weights, dtype=float)
    weights = np.clip(weights, weight_clip_min, weight_clip_max)
    weights = weights / np.mean(weights)
    return weights
