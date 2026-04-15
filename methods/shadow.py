"""Core shadow-variable implementation shared by CVCI and RHC hosts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from utils.screening_utils import partial_abs_corr


@dataclass
class ConditionalGaussianModel:
    """Gaussian conditional model: linear mean + homoskedastic residual std."""

    feature_cols: List[str]
    y_col: str
    mean_model: LinearRegression
    residual_sigma: float

    def predict_mean(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return self.mean_model.predict(x)

    def sample(self, x: np.ndarray, n_samples: int = 2000, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[0] != 1:
            raise ValueError("ConditionalGaussianModel.sample currently expects exactly one row.")
        means = np.repeat(self.predict_mean(x), int(n_samples))
        if rng is None:
            rng = np.random.default_rng()
        return means + self.residual_sigma * rng.standard_normal(int(n_samples))


def clip_prob(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Clip probabilities into [eps, 1-eps]."""
    return np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)


def fit_classifier(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    max_iter: int = 2000,
) -> LogisticRegression:
    """Fit logistic regression classifier with probabilistic output."""
    x = df[list(feature_cols)].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)
    if np.unique(y).size < 2:
        raise ValueError(f"Cannot fit classifier for {target_col}: only one class present.")
    model = LogisticRegression(max_iter=max_iter)
    model.fit(x, y)
    return model


def predict_prob(model: LogisticRegression, x: np.ndarray) -> np.ndarray:
    """Predict class-1 probability from a fitted classifier."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return model.predict_proba(x)[:, 1]


def fit_conditional_distribution_model(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    y_col: str,
) -> ConditionalGaussianModel:
    """Fit linear-mean Gaussian conditional distribution model for continuous outcomes."""
    x = df[list(feature_cols)].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    if y.size == 0:
        raise ValueError("Cannot fit conditional distribution model: empty training set.")

    mean_model = LinearRegression()
    mean_model.fit(x, y)
    residuals = y - mean_model.predict(x)
    sigma = float(np.std(residuals, ddof=1)) if y.size > 1 else 0.0
    sigma = max(sigma, 1e-6)

    return ConditionalGaussianModel(
        feature_cols=list(feature_cols),
        y_col=y_col,
        mean_model=mean_model,
        residual_sigma=sigma,
    )


def screen_shadow_candidates(
    df: pd.DataFrame,
    X_cols: Sequence[str],
    t_col: str = "T",
    y_col: str = "Y",
    g_col: str = "G",
    relevance_threshold: float = 0.02,
    independence_threshold: float = 0.02,
    allow_empty_fallback: bool = False,
) -> Dict[str, object]:
    """
    Screen shadow variables from X_cols.

    Conditions:
    - Relevance: association with Y conditional on T and remaining X.
    - Shadow-independence: near-independence with G conditional on Y, T, and remaining X.
    """
    x_cols = list(X_cols)
    missing = [c for c in [*x_cols, t_col, y_col, g_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for shadow screening: {missing}")
    if not x_cols:
        raise ValueError("X_cols must be non-empty for shadow screening.")

    logs: List[Dict[str, object]] = []
    selected_shadow_cols: List[str] = []

    t_vec = df[t_col].to_numpy(dtype=float)
    y_vec = df[y_col].to_numpy(dtype=float)
    g_vec = df[g_col].to_numpy(dtype=float)

    for col in x_cols:
        remaining = [x for x in x_cols if x != col]
        z_vec = df[col].to_numpy(dtype=float)

        nuisance_rel_parts = [t_vec.reshape(-1, 1)]
        if remaining:
            nuisance_rel_parts.append(df[remaining].to_numpy(dtype=float))
        nuisance_rel = np.hstack(nuisance_rel_parts)

        nuisance_ind_parts = [y_vec.reshape(-1, 1), t_vec.reshape(-1, 1)]
        if remaining:
            nuisance_ind_parts.append(df[remaining].to_numpy(dtype=float))
        nuisance_ind = np.hstack(nuisance_ind_parts)

        relevance_score = float(partial_abs_corr(z_vec, y_vec, nuisance_rel))
        independence_score = float(partial_abs_corr(z_vec, g_vec, nuisance_ind))
        passed = bool(relevance_score >= relevance_threshold and independence_score <= independence_threshold)
        if passed:
            selected_shadow_cols.append(col)

        logs.append(
            {
                "column": col,
                "relevance_score": relevance_score,
                "independence_score": independence_score,
                "selected": passed,
                "relevance_threshold": float(relevance_threshold),
                "independence_threshold": float(independence_threshold),
            }
        )

    if not selected_shadow_cols:
        if allow_empty_fallback:
            best_idx = int(np.argmax([entry["relevance_score"] - entry["independence_score"] for entry in logs]))
            selected_shadow_cols = [logs[best_idx]["column"]]
            logs[best_idx]["selected"] = True
            logs[best_idx]["fallback_selected"] = True
        else:
            raise ValueError(
                "No feature in X_cols passed shadow screening. "
                "Set allow_empty_fallback=True if you explicitly want fallback behavior."
            )

    selected_set = set(selected_shadow_cols)
    xc_cols = [x for x in x_cols if x not in selected_set]

    return {
        "selected_shadow_cols": selected_shadow_cols,
        "Xc_cols": xc_cols,
        "screening_logs": logs,
        "relevance_threshold": float(relevance_threshold),
        "independence_threshold": float(independence_threshold),
        "allow_empty_fallback": bool(allow_empty_fallback),
    }


def fit_shadow_pipeline(
    df: pd.DataFrame,
    X_cols: Sequence[str],
    selected_shadow_cols: Sequence[str],
    t_col: str = "T",
    y_col: str = "Y",
    g_col: str = "G",
) -> Dict[str, object]:
    """Fit all models needed for shadow-based recovery."""
    x_cols = list(X_cols)
    xs_cols = list(selected_shadow_cols)
    if not xs_cols:
        raise ValueError("selected_shadow_cols must be non-empty for fit_shadow_pipeline.")
    invalid = [c for c in xs_cols if c not in x_cols]
    if invalid:
        raise ValueError(f"selected_shadow_cols must be subset of X_cols, invalid: {invalid}")

    xc_cols = [x for x in x_cols if x not in set(xs_cols)]
    dist_feature_cols = xc_cols + xs_cols

    df_rct = df[df[g_col] == 1].copy()
    if df_rct.empty:
        raise ValueError("No RCT rows (G==1) available for shadow pipeline.")

    df_rct_t1 = df_rct[df_rct[t_col] == 1].copy()
    df_rct_t0 = df_rct[df_rct[t_col] == 0].copy()
    if df_rct_t1.empty or df_rct_t0.empty:
        raise ValueError("RCT rows must include both treatment arms for shadow pipeline.")

    model_f_t1 = fit_conditional_distribution_model(df_rct_t1, dist_feature_cols, y_col=y_col)
    model_f_t0 = fit_conditional_distribution_model(df_rct_t0, dist_feature_cols, y_col=y_col)

    gy_feature_cols = xc_cols + [t_col, y_col]
    model_g_y = fit_classifier(df, gy_feature_cols, g_col)

    gxs_feature_cols = xc_cols + [t_col] + xs_cols
    model_g_xs = fit_classifier(df, gxs_feature_cols, g_col)

    return {
        "model_f_t1": model_f_t1,
        "model_f_t0": model_f_t0,
        "model_g_y": model_g_y,
        "model_g_xs": model_g_xs,
        "X_cols": x_cols,
        "Xc_cols": xc_cols,
        "Xs_cols": xs_cols,
        "dist_feature_cols": dist_feature_cols,
        "g_y_feature_cols": gy_feature_cols,
        "g_xs_feature_cols": gxs_feature_cols,
        "t_col": t_col,
        "y_col": y_col,
        "g_col": g_col,
    }


def _split_x(
    x_all: np.ndarray,
    x_cols: Sequence[str],
    selected_shadow_cols: Sequence[str],
) -> Dict[str, np.ndarray]:
    x_cols = list(x_cols)
    xs_cols = list(selected_shadow_cols)
    xc_cols = [x for x in x_cols if x not in set(xs_cols)]

    x_all = np.asarray(x_all, dtype=float).reshape(-1)
    if x_all.shape[0] != len(x_cols):
        raise ValueError("x_all length must equal len(X_cols).")

    lookup = {name: x_all[idx] for idx, name in enumerate(x_cols)}
    xc_vec = np.asarray([lookup[col] for col in xc_cols], dtype=float)
    xs_vec = np.asarray([lookup[col] for col in xs_cols], dtype=float)
    return {"xc_vec": xc_vec, "xs_vec": xs_vec, "Xc_cols": xc_cols, "Xs_cols": xs_cols}


def predict_mu_t_shadow(
    models: Dict[str, object],
    xc_vec: np.ndarray,
    xs_vec: np.ndarray,
    t: int,
    M: int = 2000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Predict mu_t^shadow(xc, xs) for continuous Y using Monte Carlo."""
    if int(t) == 1:
        model_f = models["model_f_t1"]
    else:
        model_f = models["model_f_t0"]

    xc_vec = np.asarray(xc_vec, dtype=float).reshape(-1)
    xs_vec = np.asarray(xs_vec, dtype=float).reshape(-1)
    x_dist = np.concatenate([xc_vec, xs_vec], axis=0).reshape(1, -1)

    y_samples = model_f.sample(x_dist, n_samples=M, rng=rng)

    g_y_features = np.column_stack(
        [
            np.repeat(xc_vec.reshape(1, -1), int(M), axis=0),
            np.full((int(M), 1), float(t)),
            y_samples.reshape(-1, 1),
        ]
    )
    s = predict_prob(models["model_g_y"], g_y_features)
    s = clip_prob(s)
    r_y = (1.0 - s) / s

    denom = float(np.sum(r_y))
    if denom <= 0:
        raise ValueError("Invalid Monte Carlo denominator while computing mu_t_shadow.")
    return float(np.sum(y_samples * r_y) / denom)


def predict_tau_shadow(
    models: Dict[str, object],
    xc_vec: np.ndarray,
    xs_vec: np.ndarray,
    M: int = 2000,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """Predict shadow-corrected treatment effect for one sample."""
    mu1_shadow = predict_mu_t_shadow(models, xc_vec=xc_vec, xs_vec=xs_vec, t=1, M=M, rng=rng)
    mu0_shadow = predict_mu_t_shadow(models, xc_vec=xc_vec, xs_vec=xs_vec, t=0, M=M, rng=rng)
    tau_shadow = float(mu1_shadow - mu0_shadow)
    return {
        "mu1_shadow": float(mu1_shadow),
        "mu0_shadow": float(mu0_shadow),
        "tau_shadow": tau_shadow,
    }


def build_shadow_obs_outcomes_for_cvci(
    df_obs: pd.DataFrame,
    shadow_models: Dict[str, object],
    X_cols: Sequence[str],
    selected_shadow_cols: Sequence[str],
    t_col: str = "T",
    M: int = 2000,
    random_state: int = 2024,
) -> np.ndarray:
    """Build shadow-corrected observational outcomes for CVCI L_obs."""
    x_cols = list(X_cols)
    xs_cols = list(selected_shadow_cols)

    obs_outcomes = np.zeros(df_obs.shape[0], dtype=float)
    rng = np.random.default_rng(random_state)

    for idx, (_, row) in enumerate(df_obs.iterrows()):
        x_all = row[x_cols].to_numpy(dtype=float)
        split = _split_x(x_all, x_cols=x_cols, selected_shadow_cols=xs_cols)
        mu_t = predict_mu_t_shadow(
            shadow_models,
            xc_vec=split["xc_vec"],
            xs_vec=split["xs_vec"],
            t=int(row[t_col]),
            M=M,
            rng=rng,
        )
        obs_outcomes[idx] = mu_t

    return obs_outcomes


def build_shadow_corrected_targets_for_rhc(
    df_rct: pd.DataFrame,
    shadow_models: Dict[str, object],
    w_hat_rct: np.ndarray,
    X_cols: Sequence[str],
    selected_shadow_cols: Sequence[str],
    t_col: str = "T",
    M: int = 2000,
    random_state: int = 2024,
) -> Dict[str, object]:
    """Build corrected second-stage targets for RHC: tau_shadow - w_hat."""
    x_cols = list(X_cols)
    xs_cols = list(selected_shadow_cols)

    w_hat_rct = np.asarray(w_hat_rct, dtype=float).reshape(-1)
    if w_hat_rct.shape[0] != df_rct.shape[0]:
        raise ValueError("w_hat_rct length must match df_rct rows.")

    mu1_list: List[float] = []
    mu0_list: List[float] = []
    tau_list: List[float] = []

    rng = np.random.default_rng(random_state)

    for _, row in df_rct.iterrows():
        x_all = row[x_cols].to_numpy(dtype=float)
        split = _split_x(x_all, x_cols=x_cols, selected_shadow_cols=xs_cols)
        tau_obj = predict_tau_shadow(
            shadow_models,
            xc_vec=split["xc_vec"],
            xs_vec=split["xs_vec"],
            M=M,
            rng=rng,
        )
        mu1_list.append(tau_obj["mu1_shadow"])
        mu0_list.append(tau_obj["mu0_shadow"])
        tau_list.append(tau_obj["tau_shadow"])

    tau_shadow = np.asarray(tau_list, dtype=float)
    corrected_targets = tau_shadow - w_hat_rct

    return {
        "corrected_targets": corrected_targets,
        "diagnostics": {
            "mu1_shadow": np.asarray(mu1_list, dtype=float),
            "mu0_shadow": np.asarray(mu0_list, dtype=float),
            "tau_shadow": tau_shadow,
            "mean_tau_shadow": float(np.mean(tau_shadow)),
            "std_tau_shadow": float(np.std(tau_shadow)),
            "mean_corrected_target": float(np.mean(corrected_targets)),
            "std_corrected_target": float(np.std(corrected_targets)),
        },
    }
