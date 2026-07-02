"""Formal selection-IV core for source-selection correction.

Mathematical mapping:
- ``lambda(x, z, t)``: baseline participation log-odds at ``Y=0``.
- ``eta(x, y, z, t)``: selection-bias term on top of baseline log-odds.
- ``pi(x, y, z, t) = expit(eta + lambda)``: extended participation probability.
- ``w_iv = (1-pi)/pi``: density-ratio style correction weight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils.screening_utils import partial_abs_corr

ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series, Sequence[float]]


def _to_1d_float(values: ArrayLike) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def _to_2d_float(values: ArrayLike) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D input, got shape={arr.shape}.")
    return arr


def clip_prob(p: Union[float, np.ndarray], eps: float = 1e-6) -> Union[float, np.ndarray]:
    """Clip probability/probabilities into ``[eps, 1-eps]`` for numerical stability."""
    clipped = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    if np.asarray(p).ndim == 0:
        return float(clipped.reshape(-1)[0])
    return clipped


def fit_classifier(X: ArrayLike, y: ArrayLike, max_iter: int = 10000):
    """Fit logistic classifier with strict class-degeneracy checks."""
    x_mat = _to_2d_float(X)
    y_vec = _to_1d_float(y)
    if x_mat.shape[0] != y_vec.shape[0]:
        raise ValueError(f"X/y row mismatch: {x_mat.shape[0]} vs {y_vec.shape[0]}.")
    if x_mat.shape[0] == 0:
        raise ValueError("Cannot fit classifier on empty data.")
    if np.unique(y_vec).size < 2:
        raise ValueError("Cannot fit classifier: y has fewer than 2 classes.")

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=max_iter))
    model.fit(x_mat, y_vec)
    return model


def predict_prob(model: LogisticRegression, x: ArrayLike) -> Union[float, np.ndarray]:
    """Predict ``P(class=1)`` for one sample or a batch."""
    x_mat = _to_2d_float(x)
    proba = model.predict_proba(x_mat)[:, 1]
    return float(proba[0]) if proba.shape[0] == 1 else proba


def _decision_logit(model: LogisticRegression, x: ArrayLike) -> np.ndarray:
    x_mat = _to_2d_float(x)
    if hasattr(model, "decision_function"):
        return _to_1d_float(model.decision_function(x_mat))
    p = clip_prob(model.predict_proba(x_mat)[:, 1])
    return np.log(_to_1d_float(p) / (1.0 - _to_1d_float(p)))


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
    Heuristic IV screening from full covariate candidates.

    Rule of thumb:
    1. relevance: candidate is associated with source indicator ``G`` conditional on ``T`` and remaining covariates.
    2. exclusion proxy: candidate is not too directly associated with outcome ``Y`` conditional on ``T`` and remaining covariates.
    """
    cols = [str(c) for c in candidate_cols]
    required = [*cols, t_col, y_col, g_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for IV screening: {missing}")
    if not cols:
        raise ValueError("candidate_cols must be non-empty.")

    t_vec = _to_1d_float(df[t_col])
    y_vec = _to_1d_float(df[y_col])
    g_vec = _to_1d_float(df[g_col])

    logs: List[Dict[str, object]] = []
    selected: List[str] = []

    for col in cols:
        remaining = [x for x in cols if x != col]
        z_vec = _to_1d_float(df[col])

        nuisance_parts = [t_vec.reshape(-1, 1)]
        if remaining:
            nuisance_parts.append(df[remaining].to_numpy(dtype=float))
        nuisance = np.hstack(nuisance_parts)

        rel = float(partial_abs_corr(z_vec, g_vec, nuisance))
        exc = float(partial_abs_corr(z_vec, y_vec, nuisance))
        passed = bool(rel >= relevance_threshold and exc <= exclusion_threshold)
        if passed:
            selected.append(col)

        logs.append(
            {
                "column": col,
                "relevance_score": rel,
                "exclusion_score": exc,
                "selected": passed,
                "relevance_threshold": float(relevance_threshold),
                "exclusion_threshold": float(exclusion_threshold),
            }
        )

    if not selected:
        if not allow_empty_fallback:
            raise ValueError("No IV candidate passed screening and fallback is disabled.")
        best_idx = int(np.argmax([entry["relevance_score"] - entry["exclusion_score"] for entry in logs]))
        selected = [logs[best_idx]["column"]]
        logs[best_idx]["selected"] = True
        logs[best_idx]["fallback_selected"] = True

    xz_set = set(selected)
    xc_cols = [c for c in cols if c not in xz_set]
    return {
        "selected_iv_cols": selected,
        "Xc_cols": xc_cols,
        "screening_logs": logs,
        "relevance_threshold": float(relevance_threshold),
        "exclusion_threshold": float(exclusion_threshold),
        "allow_empty_fallback": bool(allow_empty_fallback),
    }


def select_iv_candidates_with_mode(
    df: pd.DataFrame,
    candidate_cols: Sequence[str],
    t_col: str = "T",
    y_col: str = "Y",
    g_col: str = "G",
    relevance_threshold: float = 0.02,
    exclusion_threshold: float = 0.02,
    allow_empty_fallback: bool = True,
    screening_mode: str = "screened",
    top_k: Optional[int] = None,
    force_candidate_cols: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Screen IV candidates under ablation modes while preserving return compatibility."""
    mode = str(screening_mode).lower()
    if mode not in {"screened", "all", "topk"}:
        raise ValueError(f"Unsupported screening_mode={screening_mode}. Expected one of screened/all/topk.")

    cols = [str(c) for c in (force_candidate_cols if force_candidate_cols is not None else candidate_cols)]
    if not cols:
        raise ValueError("candidate_cols must be non-empty after applying force_candidate_cols.")

    if mode == "screened":
        result = select_iv_candidates(
            df=df,
            candidate_cols=cols,
            t_col=t_col,
            y_col=y_col,
            g_col=g_col,
            relevance_threshold=relevance_threshold,
            exclusion_threshold=exclusion_threshold,
            allow_empty_fallback=allow_empty_fallback,
        )
        result["screening_mode"] = mode
        result["top_k"] = None
        result["candidate_cols"] = cols
        return result

    required = [*cols, t_col, y_col, g_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for IV screening: {missing}")

    t_vec = _to_1d_float(df[t_col])
    y_vec = _to_1d_float(df[y_col])
    g_vec = _to_1d_float(df[g_col])

    logs: List[Dict[str, object]] = []
    for col in cols:
        remaining = [x for x in cols if x != col]
        z_vec = _to_1d_float(df[col])
        nuisance_parts = [t_vec.reshape(-1, 1)]
        if remaining:
            nuisance_parts.append(df[remaining].to_numpy(dtype=float))
        nuisance = np.hstack(nuisance_parts)

        rel = float(partial_abs_corr(z_vec, g_vec, nuisance))
        exc = float(partial_abs_corr(z_vec, y_vec, nuisance))
        logs.append(
            {
                "column": col,
                "relevance_score": rel,
                "exclusion_score": exc,
                "score": float(rel - exc),
                "selected": False,
                "relevance_threshold": float(relevance_threshold),
                "exclusion_threshold": float(exclusion_threshold),
            }
        )

    if mode == "all":
        selected = list(cols)
    else:
        n_candidates = len(cols)
        if top_k is None:
            raise ValueError("top_k must be provided when screening_mode='topk'.")
        k = int(top_k)
        if k <= 0 or k > n_candidates:
            raise ValueError(f"Invalid top_k={top_k}. Expected integer in [1, {n_candidates}] for topk mode.")
        ranked = sorted(logs, key=lambda entry: float(entry["score"]), reverse=True)
        selected = [str(entry["column"]) for entry in ranked[:k]]

    selected_set = set(selected)
    for entry in logs:
        entry["selected"] = bool(entry["column"] in selected_set)
    xc_cols = [c for c in cols if c not in selected_set]
    resolved_top_k = None if mode != "topk" else int(top_k)
    return {
        "selected_iv_cols": selected,
        "Xc_cols": xc_cols,
        "screening_logs": logs,
        "relevance_threshold": float(relevance_threshold),
        "exclusion_threshold": float(exclusion_threshold),
        "allow_empty_fallback": bool(allow_empty_fallback),
        "screening_mode": mode,
        "top_k": resolved_top_k,
        "candidate_cols": cols,
    }


@dataclass
class SelectionBiasModel:
    """Parametric model for ``eta(x, y, z, t)`` used in extended participation log-odds."""

    feature_cols: List[str]
    coef_: np.ndarray
    intercept_: float

    def predict(self, x: ArrayLike) -> np.ndarray:
        x_mat = _to_2d_float(x)
        if x_mat.shape[1] != len(self.feature_cols):
            raise ValueError(
                f"SelectionBiasModel feature mismatch: expected {len(self.feature_cols)}, got {x_mat.shape[1]}."
            )
        return _to_1d_float(self.intercept_ + x_mat @ self.coef_)


@dataclass
class ConditionalGaussianModel:
    """Continuous-Y conditional model: linear mean plus homoskedastic Gaussian residual."""

    feature_cols: List[str]
    mean_model: LinearRegression
    residual_sigma: float

    def predict_mean(self, x: ArrayLike) -> np.ndarray:
        return self.mean_model.predict(_to_2d_float(x))

    def sample(
        self,
        x: ArrayLike,
        n_samples: int = 2000,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        x_mat = _to_2d_float(x)
        if x_mat.shape[0] != 1:
            raise ValueError("ConditionalGaussianModel.sample expects exactly one conditioning row.")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        if rng is None:
            rng = np.random.default_rng()
        mean_val = float(self.predict_mean(x_mat)[0])
        return mean_val + float(self.residual_sigma) * rng.standard_normal(int(n_samples))


def fit_conditional_distribution_model(
    X: Union[np.ndarray, pd.DataFrame],
    y: ArrayLike,
    feature_cols: Optional[Sequence[str]] = None,
) -> ConditionalGaussianModel:
    """Fit a lightweight continuous outcome distribution model for IV outcome recovery."""
    x_mat = _to_2d_float(X)
    y_vec = _to_1d_float(y)
    if x_mat.shape[0] == 0:
        raise ValueError("Cannot fit conditional distribution model on empty data.")
    if x_mat.shape[0] != y_vec.shape[0]:
        raise ValueError(f"X/y row mismatch: {x_mat.shape[0]} vs {y_vec.shape[0]}.")

    model = LinearRegression()
    model.fit(x_mat, y_vec)
    residual = y_vec - model.predict(x_mat)
    sigma = float(np.std(residual, ddof=1)) if y_vec.shape[0] > 1 else 0.0
    sigma = max(sigma, 1e-6)

    if feature_cols is None:
        feature_cols = [f"x{i}" for i in range(x_mat.shape[1])]
    return ConditionalGaussianModel(feature_cols=list(feature_cols), mean_model=model, residual_sigma=sigma)



def fit_selection_bias_model(
    df: pd.DataFrame,
    Xc_cols: Sequence[str],
    Xz_cols: Sequence[str],
    t_col: str,
    y_col: str,
    g_col: str,
    baseline_model: LogisticRegression,
    max_iter: int = 2000,
    l2_reg: float = 1e-6,
) -> SelectionBiasModel:
    """
    Fit ``eta(x,y,z,t)`` by logistic likelihood with fixed baseline offset ``lambda(x,z,t)``.

    We optimize:
    ``P(G=1|x,y,z,t) = expit( lambda(x,z,t) + eta_theta(x,y,z,t) )``
    where ``lambda`` comes from baseline model fitted at ``Y=0``.
    """
    xc_cols = [str(c) for c in Xc_cols]
    xz_cols = [str(c) for c in Xz_cols]
    eta_feature_cols = list(dict.fromkeys(xc_cols + xz_cols + [t_col, y_col]))

    for col in [*eta_feature_cols, g_col]:
        if col not in df.columns:
            raise ValueError(f"Missing column for fit_selection_bias_model: {col}")

    y_zero = np.zeros(df.shape[0], dtype=float)
    base_input_cols = eta_feature_cols[:-1] + [y_col]
    x_base = df[base_input_cols].copy()
    x_base[y_col] = y_zero
    p_y0 = _to_1d_float(predict_prob(baseline_model, x_base.to_numpy(dtype=float)))
    p_y0 = _to_1d_float(clip_prob(p_y0))
    lambda_offset = np.log(p_y0 / (1.0 - p_y0))

    x_eta = df[eta_feature_cols].to_numpy(dtype=float)
    g = _to_1d_float(df[g_col])

    if np.unique(g).size < 2:
        raise ValueError("Cannot fit selection-bias model: g has fewer than 2 classes.")

    n_features = x_eta.shape[1]

    def objective(params: np.ndarray) -> float:
        intercept = float(params[0])
        coef = params[1:]
        eta_val = intercept + x_eta @ coef
        p = _to_1d_float(clip_prob(expit(lambda_offset + eta_val)))
        nll = -np.mean(g * np.log(p) + (1.0 - g) * np.log(1.0 - p))
        reg = 0.5 * l2_reg * float(np.sum(coef**2))
        return nll + reg

    def gradient(params: np.ndarray) -> np.ndarray:
        intercept = float(params[0])
        coef = params[1:]
        eta_val = intercept + x_eta @ coef
        p = _to_1d_float(expit(lambda_offset + eta_val))
        residual = p - g
        grad_intercept = float(np.mean(residual))
        grad_coef = np.mean(x_eta * residual.reshape(-1, 1), axis=0) + l2_reg * coef
        return np.concatenate(([grad_intercept], grad_coef), axis=0)

    init = np.zeros(n_features + 1, dtype=float)
    opt = minimize(
        objective,
        x0=init,
        jac=gradient,
        method="L-BFGS-B",
        options={"maxiter": max_iter},
    )
    if not opt.success:
        raise RuntimeError(f"Selection-bias optimization failed: {opt.message}")

    return SelectionBiasModel(
        feature_cols=eta_feature_cols,
        coef_=_to_1d_float(opt.x[1:]),
        intercept_=float(opt.x[0]),
    )


def predict_eta(
    model_eta: SelectionBiasModel,
    xc_vec: ArrayLike,
    xz_vec: ArrayLike,
    t: float,
    y: float,
) -> float:
    """Predict scalar ``eta(x, y, z, t)`` for one sample."""
    xc = _to_1d_float(xc_vec)
    xz = _to_1d_float(xz_vec)
    x_row = np.concatenate([xc, xz, np.asarray([float(t), float(y)], dtype=float)])
    eta = model_eta.predict(x_row.reshape(1, -1))
    return float(_to_1d_float(eta)[0])


def fit_iv_pipeline(
    df: pd.DataFrame,
    Xc_cols: Sequence[str],
    Xz_cols: Sequence[str],
    t_col: str = "T",
    y_col: str = "Y",
    g_col: str = "G",
    prob_clip_eps: float = 1e-6,
    weight_clip_min: float = 0.05,
    weight_clip_max: float = 20.0,
    max_iter: int = 2000,
) -> np.ndarray:
    """
    Fit IV-based source-selection correction and return only per-sample weights.

    The implementation follows:
    1. Fit baseline participation model at ``Y=0``.
    2. Fit separate selection-bias model ``eta`` with baseline offset.
    3. Compute ``pi = expit(eta + lambda)``.
    4. Compute ``w = (1-pi)/pi``.
    """
    xc_cols = [str(c) for c in Xc_cols]
    xz_cols = [str(c) for c in Xz_cols]
    xczt_cols = list(dict.fromkeys(xc_cols + xz_cols + [t_col]))
    xczyt_cols = list(dict.fromkeys(xczt_cols + [y_col]))

    required = [*xczyt_cols, g_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for fit_iv_pipeline: {missing}")
    if df.shape[0] == 0:
        raise ValueError("Cannot run fit_iv_pipeline on empty dataframe.")

    df_y0 = df.copy()
    df_y0[y_col] = 0.0
    model_lambda = fit_classifier(df_y0[xczyt_cols], df_y0[g_col], max_iter=max_iter)

    model_eta = fit_selection_bias_model(
        df=df,
        Xc_cols=xc_cols,
        Xz_cols=xz_cols,
        t_col=t_col,
        y_col=y_col,
        g_col=g_col,
        baseline_model=model_lambda,
        max_iter=max_iter,
    )

    weights: List[float] = []
    for _, row in df.iterrows():
        xc_vec = _to_1d_float(row[xc_cols]) if xc_cols else np.asarray([], dtype=float)
        xz_vec = _to_1d_float(row[xz_cols]) if xz_cols else np.asarray([], dtype=float)

        base_input = np.concatenate([xc_vec, xz_vec, np.asarray([float(row[t_col]), 0.0])])
        p_y0 = predict_prob(model_lambda, base_input)
        p_y0 = float(clip_prob(float(p_y0), eps=prob_clip_eps))
        lambda_val = float(np.log(p_y0 / (1.0 - p_y0)))

        eta_val = predict_eta(
            model_eta,
            xc_vec=xc_vec,
            xz_vec=xz_vec,
            t=float(row[t_col]),
            y=float(row[y_col]),
        )

        pi_val = float(clip_prob(float(expit(eta_val + lambda_val)), eps=prob_clip_eps))
        weight_val = (1.0 - pi_val) / pi_val
        weights.append(float(weight_val))

    w = _to_1d_float(weights)
    if np.any(~np.isfinite(w)):
        raise ValueError("Non-finite IV weights detected before clipping.")
    w = np.clip(w, weight_clip_min, weight_clip_max)
    if np.any(~np.isfinite(w)):
        raise ValueError("Non-finite IV weights detected after clipping.")
    mean_w = float(np.mean(w))
    if mean_w <= 0 or not np.isfinite(mean_w):
        raise ValueError(f"Invalid IV weight mean: {mean_w}")
    return w / mean_w


def fit_iv_or_pipeline(
    df: pd.DataFrame,
    Xc_cols: Sequence[str],
    Xz_cols: Sequence[str],
    t_col: str = "T",
    y_col: str = "Y",
    g_col: str = "G",
    source_g: Union[int, float] = 1,
    target_g: Union[int, float] = 0,
    max_iter: int = 2000,
) -> Dict[str, object]:
    """
    Fit IV outcome-recovery models for an RCT-to-OBS CATE signal.

    This is a parallel regression-recovery path. It reuses the same IV
    source-selection components as ``fit_iv_pipeline`` but returns fitted models
    instead of sample weights.
    """
    xc_cols = [str(c) for c in Xc_cols]
    xz_cols = [str(c) for c in Xz_cols]
    xczt_cols = list(dict.fromkeys(xc_cols + xz_cols + [t_col]))
    xczyt_cols = list(dict.fromkeys(xczt_cols + [y_col]))
    dist_cols = list(dict.fromkeys(xc_cols + xz_cols))

    required = [*xczyt_cols, g_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for fit_iv_or_pipeline: {missing}")
    if df.shape[0] == 0:
        raise ValueError("Cannot run fit_iv_or_pipeline on empty dataframe.")

    df_y0 = df.copy()
    df_y0[y_col] = 0.0
    model_lambda = fit_classifier(df_y0[xczyt_cols], df_y0[g_col], max_iter=max_iter)
    model_eta = fit_selection_bias_model(
        df=df,
        Xc_cols=xc_cols,
        Xz_cols=xz_cols,
        t_col=t_col,
        y_col=y_col,
        g_col=g_col,
        baseline_model=model_lambda,
        max_iter=max_iter,
    )

    df_source = df[df[g_col] == source_g].copy()
    if df_source.empty:
        raise ValueError(f"No source rows (G=={source_g}) available for IV outcome recovery.")
    df_source_t1 = df_source[df_source[t_col] == 1].copy()
    df_source_t0 = df_source[df_source[t_col] == 0].copy()
    if df_source_t1.empty or df_source_t0.empty:
        raise ValueError(f"Source rows (G=={source_g}) must contain both treatment arms for IV outcome recovery.")

    model_f_t1 = fit_conditional_distribution_model(
        X=df_source_t1[dist_cols],
        y=df_source_t1[y_col],
        feature_cols=dist_cols,
    )
    model_f_t0 = fit_conditional_distribution_model(
        X=df_source_t0[dist_cols],
        y=df_source_t0[y_col],
        feature_cols=dist_cols,
    )

    return {
        "model_lambda": model_lambda,
        "model_eta": model_eta,
        "model_f_t1": model_f_t1,
        "model_f_t0": model_f_t0,
        "Xc_cols": xc_cols,
        "Xz_cols": xz_cols,
        "dist_feature_cols": dist_cols,
        "lambda_feature_cols": xczyt_cols,
        "eta_feature_cols": model_eta.feature_cols,
        "t_col": t_col,
        "y_col": y_col,
        "g_col": g_col,
        "source_g": source_g,
        "target_g": target_g,
    }


def predict_mu_t_iv_or(
    models: Dict[str, object],
    xc_vec: ArrayLike,
    xz_vec: ArrayLike,
    t: int,
    M: int = 2000,
    random_state: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    prob_clip_eps: float = 1e-6,
) -> float:
    """Recover ``E[Y(t)|target, X]`` using IV selection-bias exponential tilting."""
    if int(t) not in (0, 1):
        raise ValueError(f"t must be 0 or 1, got {t}")
    if M <= 0:
        raise ValueError("M must be positive.")
    if rng is None:
        rng = np.random.default_rng(random_state)

    xc = _to_1d_float(xc_vec)
    xz = _to_1d_float(xz_vec)
    model_f = models["model_f_t1"] if int(t) == 1 else models["model_f_t0"]
    y_samples = _to_1d_float(model_f.sample(np.concatenate([xc, xz]), n_samples=M, rng=rng))

    base_input = np.concatenate([xc, xz, np.asarray([float(t), 0.0], dtype=float)])
    p_y0 = float(clip_prob(float(predict_prob(models["model_lambda"], base_input)), eps=prob_clip_eps))
    lambda_val = float(np.log(p_y0 / (1.0 - p_y0)))

    eta_rows = np.column_stack(
        [
            np.repeat(xc.reshape(1, -1), int(M), axis=0),
            np.repeat(xz.reshape(1, -1), int(M), axis=0),
            np.full((int(M), 1), float(t)),
            y_samples.reshape(-1, 1),
        ]
    )
    eta_vals = _to_1d_float(models["model_eta"].predict(eta_rows))
    pi_vals = _to_1d_float(clip_prob(expit(lambda_val + eta_vals), eps=prob_clip_eps))
    recovery_weights = (1.0 - pi_vals) / pi_vals

    denom = float(np.sum(recovery_weights))
    if not np.isfinite(denom) or denom <= 0:
        raise ValueError(f"Invalid Monte Carlo denominator in predict_mu_t_iv_or: {denom}")
    numerator = float(np.sum(y_samples * recovery_weights))
    if not np.isfinite(numerator):
        raise ValueError("Invalid Monte Carlo numerator in predict_mu_t_iv_or.")
    return float(numerator / denom)


def predict_tau_iv_or(
    models: Dict[str, object],
    xc_vec: ArrayLike,
    xz_vec: ArrayLike,
    M: int = 2000,
    random_state: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """Compute ``tau_iv_or = mu1_iv_or - mu0_iv_or`` for one sample."""
    if rng is None:
        rng = np.random.default_rng(random_state)
    mu1 = predict_mu_t_iv_or(models, xc_vec=xc_vec, xz_vec=xz_vec, t=1, M=M, rng=rng)
    mu0 = predict_mu_t_iv_or(models, xc_vec=xc_vec, xz_vec=xz_vec, t=0, M=M, rng=rng)
    return {"mu1_iv_or": float(mu1), "mu0_iv_or": float(mu0), "tau_iv_or": float(mu1 - mu0)}
