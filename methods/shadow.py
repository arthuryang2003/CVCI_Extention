"""Formal shadow-recovery core for source-selection correction.

Mathematical mapping:
- ``model_f_t1 / model_f_t0``: baseline outcome distributions
  ``p(y|G=source_g, Xc, Xz, T=t)``.
- ``model_g_y``: source-membership model ``P(G=source_g|Xc, T, Y)``.
- ``mu_t^shadow``: Monte Carlo normalized weighted expectation under shadow recovery.
- ``tau_shadow = mu1_shadow - mu0_shadow``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from utils.screening_utils import partial_abs_corr

ArrayLike = Union[np.ndarray, pd.Series, Sequence[float]]
GroupLike = Optional[Union[float, int]]


def _canonical_group_value(value: Union[float, int]) -> Union[int, float]:
    val = float(value)
    if np.isclose(val, round(val)):
        return int(round(val))
    return val


def _infer_shadow_direction_label(source_g: Union[float, int], target_g: Union[float, int]) -> str:
    src = _canonical_group_value(source_g)
    tgt = _canonical_group_value(target_g)
    if src == 0 and tgt == 1:
        return "obs_to_rct"
    if src == 1 and tgt == 0:
        return "rct_to_obs"
    return f"custom_{src}_to_{tgt}"


def _resolve_shadow_direction(
    shadow_direction: Optional[str] = None,
    source_g: GroupLike = None,
    target_g: GroupLike = None,
    *,
    default_source_g: Union[float, int] = 1,
    default_target_g: Union[float, int] = 0,
) -> Tuple[Union[float, int], Union[float, int], str]:
    """Resolve source/target groups and canonical direction label.

    source_g: group where baseline density p(y|G=source_g, Xc, Xz, T=t) is fitted.
    target_g: group recovered by ratio weighting r(y)=(1-s)/s with s=P(G=source_g|Xc,T,Y).
    """
    direction_map = {
        "obs_to_rct": (0, 1),
        "rct_to_obs": (1, 0),
    }
    if shadow_direction is not None:
        shadow_direction = str(shadow_direction).lower()
        if shadow_direction not in direction_map:
            raise ValueError(f"Unsupported shadow_direction={shadow_direction}.")
        mapped_source, mapped_target = direction_map[shadow_direction]
    else:
        mapped_source, mapped_target = None, None

    resolved_source = mapped_source if source_g is None else source_g
    resolved_target = mapped_target if target_g is None else target_g
    if resolved_source is None and resolved_target is None:
        resolved_source, resolved_target = default_source_g, default_target_g
    elif resolved_source is None or resolved_target is None:
        raise ValueError("source_g and target_g must be provided together when shadow_direction is not given.")

    resolved_source = _canonical_group_value(resolved_source)
    resolved_target = _canonical_group_value(resolved_target)
    if resolved_source == resolved_target:
        raise ValueError(f"source_g and target_g must be different, got {resolved_source}.")
    if mapped_source is not None and (
        _canonical_group_value(mapped_source) != resolved_source
        or _canonical_group_value(mapped_target) != resolved_target
    ):
        raise ValueError(
            "shadow_direction conflicts with explicit source_g/target_g: "
            f"direction={shadow_direction}, source_g={resolved_source}, target_g={resolved_target}."
        )

    resolved_direction = (
        shadow_direction if shadow_direction is not None else _infer_shadow_direction_label(resolved_source, resolved_target)
    )
    return resolved_source, resolved_target, resolved_direction


def _to_1d_float(values: ArrayLike) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def _to_2d_float(values: Union[np.ndarray, pd.DataFrame, Sequence[float]]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D input, got shape={arr.shape}.")
    return arr


def clip_prob(p: Union[float, np.ndarray], eps: float = 1e-6) -> Union[float, np.ndarray]:
    """Clip probability/probabilities into ``[eps, 1-eps]``."""
    clipped = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    if np.asarray(p).ndim == 0:
        return float(clipped.reshape(-1)[0])
    return clipped


def fit_classifier(
    X: Union[np.ndarray, pd.DataFrame],
    y: ArrayLike,
    max_iter: int = 2000,
) -> LogisticRegression:
    """Fit source classifier with explicit empty/degenerate-class checks."""
    x_mat = _to_2d_float(X)
    y_vec = _to_1d_float(y)
    if x_mat.shape[0] == 0:
        raise ValueError("Cannot fit classifier on empty data.")
    if x_mat.shape[0] != y_vec.shape[0]:
        raise ValueError(f"X/y row mismatch: {x_mat.shape[0]} vs {y_vec.shape[0]}.")
    if np.unique(y_vec).size < 2:
        raise ValueError("Cannot fit classifier: y has fewer than 2 classes.")

    model = LogisticRegression(max_iter=max_iter)
    model.fit(x_mat, y_vec)
    return model


def predict_prob(model: LogisticRegression, x: Union[np.ndarray, Sequence[float]]) -> Union[float, np.ndarray]:
    """Predict class-1 probability for one sample or a batch."""
    x_mat = _to_2d_float(x)
    p = model.predict_proba(x_mat)[:, 1]
    return float(p[0]) if p.shape[0] == 1 else p


@dataclass
class ConditionalGaussianModel:
    """Continuous-Y conditional model: linear mean + homoskedastic Gaussian residual."""

    feature_cols: List[str]
    mean_model: LinearRegression
    residual_sigma: float

    def predict_mean(self, x: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
        return self.mean_model.predict(_to_2d_float(x))

    def sample(
        self,
        x: Union[np.ndarray, Sequence[float]],
        n_samples: int = 2000,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Sample ``Y`` from Normal(mean(x), sigma^2) for one conditioning row."""
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
    """Fit continuous-Y baseline conditional distribution model."""
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


def screen_shadow_candidates(
    df: pd.DataFrame,
    X_cols: Sequence[str],
    t_col: str = "T",
    y_col: str = "Y",
    g_col: str = "G",
    relevance_threshold: float = 0.02,
    independence_threshold: float = 0.02,
    allow_empty_fallback: bool = False,
    relevance_group: Optional[str] = None,
    shadow_direction: Optional[str] = None,
    source_g: GroupLike = None,
    target_g: GroupLike = None,
) -> Dict[str, object]:
    """
    Heuristic shadow-variable screening over all candidate covariates.

    - relevance proxy: association with outcome ``Y`` given ``T`` and remaining covariates
    - independence proxy: weak association with source ``G`` given ``Y, T`` and remaining covariates
    """
    x_cols = [str(c) for c in X_cols]
    required = [*x_cols, t_col, y_col, g_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for shadow screening: {missing}")
    if not x_cols:
        raise ValueError("X_cols must be non-empty.")

    resolved_source_g, resolved_target_g, resolved_direction = _resolve_shadow_direction(
        shadow_direction=shadow_direction,
        source_g=source_g,
        target_g=target_g,
    )

    relevance_group_normalized = None if relevance_group is None else str(relevance_group).lower()
    if relevance_group_normalized not in {None, "target", "source"}:
        raise ValueError("relevance_group must be one of {None, 'target', 'source'}.")

    if relevance_group_normalized is None:
        df_rel = df
    elif relevance_group_normalized == "target":
        df_rel = df[df[g_col] == resolved_target_g]
    else:
        df_rel = df[df[g_col] == resolved_source_g]
    if df_rel.empty:
        raise ValueError(
            f"Shadow relevance subset is empty for relevance_group={relevance_group_normalized}, "
            f"source_g={resolved_source_g}, target_g={resolved_target_g}."
        )

    t_vec_rel = _to_1d_float(df_rel[t_col])
    y_vec_rel = _to_1d_float(df_rel[y_col])
    t_vec = _to_1d_float(df[t_col])
    y_vec = _to_1d_float(df[y_col])
    g_vec = _to_1d_float(df[g_col])

    selected: List[str] = []
    logs: List[Dict[str, object]] = []

    for col in x_cols:
        remaining = [x for x in x_cols if x != col]
        z_vec_rel = _to_1d_float(df_rel[col])
        z_vec = _to_1d_float(df[col])

        nuisance_rel_parts = [t_vec_rel.reshape(-1, 1)]
        nuisance_ind_parts = [y_vec.reshape(-1, 1), t_vec.reshape(-1, 1)]
        if remaining:
            remain_mat_rel = df_rel[remaining].to_numpy(dtype=float)
            remain_mat = df[remaining].to_numpy(dtype=float)
            nuisance_rel_parts.append(remain_mat_rel)
            nuisance_ind_parts.append(remain_mat)

        rel = float(partial_abs_corr(z_vec_rel, y_vec_rel, np.hstack(nuisance_rel_parts)))
        ind = float(partial_abs_corr(z_vec, g_vec, np.hstack(nuisance_ind_parts)))
        passed = bool(rel >= relevance_threshold and ind <= independence_threshold)
        if passed:
            selected.append(col)

        logs.append(
            {
                "column": col,
                "relevance_score": rel,
                "independence_score": ind,
                "selected": passed,
                "relevance_threshold": float(relevance_threshold),
                "independence_threshold": float(independence_threshold),
            }
        )

    if not selected:
        if not allow_empty_fallback:
            raise ValueError("No shadow candidate passed screening and fallback is disabled.")
        best_idx = int(np.argmax([entry["relevance_score"] - entry["independence_score"] for entry in logs]))
        selected = [logs[best_idx]["column"]]
        logs[best_idx]["selected"] = True
        logs[best_idx]["fallback_selected"] = True

    xz_set = set(selected)
    xc_cols = [x for x in x_cols if x not in xz_set]
    return {
        "selected_shadow_cols": selected,
        "Xc_cols": xc_cols,
        "Xz_cols": selected,
        "screening_logs": logs,
        "relevance_threshold": float(relevance_threshold),
        "independence_threshold": float(independence_threshold),
        "allow_empty_fallback": bool(allow_empty_fallback),
        "relevance_group": relevance_group_normalized,
        "source_g": resolved_source_g,
        "target_g": resolved_target_g,
        "shadow_direction": resolved_direction,
    }


def screen_shadow_candidates_with_mode(
    df: pd.DataFrame,
    X_cols: Sequence[str],
    t_col: str = "T",
    y_col: str = "Y",
    g_col: str = "G",
    relevance_threshold: float = 0.02,
    independence_threshold: float = 0.02,
    allow_empty_fallback: bool = False,
    relevance_group: Optional[str] = None,
    shadow_direction: Optional[str] = None,
    source_g: GroupLike = None,
    target_g: GroupLike = None,
    screening_mode: str = "screened",
    top_k: Optional[int] = None,
    force_candidate_cols: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Screen shadow candidates under ablation modes while preserving return compatibility."""
    mode = str(screening_mode).lower()
    if mode not in {"screened", "all", "topk"}:
        raise ValueError(f"Unsupported screening_mode={screening_mode}. Expected one of screened/all/topk.")

    x_cols = [str(c) for c in (force_candidate_cols if force_candidate_cols is not None else X_cols)]
    if not x_cols:
        raise ValueError("X_cols must be non-empty after applying force_candidate_cols.")

    if mode == "screened":
        result = screen_shadow_candidates(
            df=df,
            X_cols=x_cols,
            t_col=t_col,
            y_col=y_col,
            g_col=g_col,
            relevance_threshold=relevance_threshold,
            independence_threshold=independence_threshold,
            allow_empty_fallback=allow_empty_fallback,
            relevance_group=relevance_group,
            shadow_direction=shadow_direction,
            source_g=source_g,
            target_g=target_g,
        )
        result["screening_mode"] = mode
        result["top_k"] = None
        result["candidate_cols"] = x_cols
        return result

    required = [*x_cols, t_col, y_col, g_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for shadow screening: {missing}")

    resolved_source_g, resolved_target_g, resolved_direction = _resolve_shadow_direction(
        shadow_direction=shadow_direction,
        source_g=source_g,
        target_g=target_g,
    )
    relevance_group_normalized = None if relevance_group is None else str(relevance_group).lower()
    if relevance_group_normalized not in {None, "target", "source"}:
        raise ValueError("relevance_group must be one of {None, 'target', 'source'}.")

    if relevance_group_normalized is None:
        df_rel = df
    elif relevance_group_normalized == "target":
        df_rel = df[df[g_col] == resolved_target_g]
    else:
        df_rel = df[df[g_col] == resolved_source_g]
    if df_rel.empty:
        raise ValueError(
            f"Shadow relevance subset is empty for relevance_group={relevance_group_normalized}, "
            f"source_g={resolved_source_g}, target_g={resolved_target_g}."
        )

    t_vec_rel = _to_1d_float(df_rel[t_col])
    y_vec_rel = _to_1d_float(df_rel[y_col])
    t_vec = _to_1d_float(df[t_col])
    y_vec = _to_1d_float(df[y_col])
    g_vec = _to_1d_float(df[g_col])

    logs: List[Dict[str, object]] = []
    for col in x_cols:
        remaining = [x for x in x_cols if x != col]
        z_vec_rel = _to_1d_float(df_rel[col])
        z_vec = _to_1d_float(df[col])

        nuisance_rel_parts = [t_vec_rel.reshape(-1, 1)]
        nuisance_ind_parts = [y_vec.reshape(-1, 1), t_vec.reshape(-1, 1)]
        if remaining:
            nuisance_rel_parts.append(df_rel[remaining].to_numpy(dtype=float))
            nuisance_ind_parts.append(df[remaining].to_numpy(dtype=float))

        rel = float(partial_abs_corr(z_vec_rel, y_vec_rel, np.hstack(nuisance_rel_parts)))
        ind = float(partial_abs_corr(z_vec, g_vec, np.hstack(nuisance_ind_parts)))
        logs.append(
            {
                "column": col,
                "relevance_score": rel,
                "independence_score": ind,
                "score": float(rel - ind),
                "selected": False,
                "relevance_threshold": float(relevance_threshold),
                "independence_threshold": float(independence_threshold),
            }
        )

    if mode == "all":
        selected = list(x_cols)
    else:
        n_candidates = len(x_cols)
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
    xc_cols = [x for x in x_cols if x not in selected_set]
    resolved_top_k = None if mode != "topk" else int(top_k)
    return {
        "selected_shadow_cols": selected,
        "Xc_cols": xc_cols,
        "Xz_cols": selected,
        "screening_logs": logs,
        "relevance_threshold": float(relevance_threshold),
        "independence_threshold": float(independence_threshold),
        "allow_empty_fallback": bool(allow_empty_fallback),
        "relevance_group": relevance_group_normalized,
        "source_g": resolved_source_g,
        "target_g": resolved_target_g,
        "shadow_direction": resolved_direction,
        "screening_mode": mode,
        "top_k": resolved_top_k,
        "candidate_cols": x_cols,
    }


def _resolve_xc_xz(
    Xc_cols: Optional[Sequence[str]] = None,
    Xz_cols: Optional[Sequence[str]] = None,
    *,
    X_cols: Optional[Sequence[str]] = None,
    selected_shadow_cols: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[str]]:
    """Resolve canonical ``Xc_cols/Xz_cols`` while keeping old interface compatibility."""
    if X_cols is not None:
        all_cols = [str(c) for c in X_cols]
        z_cols = [str(c) for c in (selected_shadow_cols or [])]
        if not z_cols:
            raise ValueError("selected_shadow_cols must be provided when using X_cols compatibility path.")
        invalid = [c for c in z_cols if c not in all_cols]
        if invalid:
            raise ValueError(f"selected_shadow_cols must be subset of X_cols, invalid={invalid}")
        z_set = set(z_cols)
        c_cols = [c for c in all_cols if c not in z_set]
        return c_cols, z_cols

    if Xc_cols is None or Xz_cols is None:
        raise ValueError("Either provide (Xc_cols, Xz_cols) or (X_cols, selected_shadow_cols).")
    return [str(c) for c in Xc_cols], [str(c) for c in Xz_cols]


def fit_shadow_pipeline(
    df: pd.DataFrame,
    Xc_cols: Optional[Sequence[str]] = None,
    Xz_cols: Optional[Sequence[str]] = None,
    t_col: str = "T",
    y_col: str = "Y",
    g_col: str = "G",
    *,
    X_cols: Optional[Sequence[str]] = None,
    selected_shadow_cols: Optional[Sequence[str]] = None,
    shadow_direction: Optional[str] = None,
    source_g: GroupLike = None,
    target_g: GroupLike = None,
) -> Dict[str, object]:
    """
    Fit all models needed for shadow-based recovery.

    Primary interface uses ``Xc_cols/Xz_cols``.
    Compatibility interface ``X_cols + selected_shadow_cols`` is also supported.
    """
    xc_cols, xz_cols = _resolve_xc_xz(
        Xc_cols=Xc_cols,
        Xz_cols=Xz_cols,
        X_cols=X_cols,
        selected_shadow_cols=selected_shadow_cols,
    )

    required = [*xc_cols, *xz_cols, t_col, y_col, g_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for fit_shadow_pipeline: {missing}")

    resolved_source_g, resolved_target_g, resolved_direction = _resolve_shadow_direction(
        shadow_direction=shadow_direction,
        source_g=source_g,
        target_g=target_g,
    )

    df_source = df[df[g_col] == resolved_source_g].copy()
    if df_source.empty:
        raise ValueError(f"No source rows (G=={resolved_source_g}) available for shadow pipeline.")

    df_source_t1 = df_source[df_source[t_col] == 1].copy()
    df_source_t0 = df_source[df_source[t_col] == 0].copy()
    if df_source_t1.empty or df_source_t0.empty:
        raise ValueError(
            f"Source rows (G=={resolved_source_g}) must contain both treatment arms (T=1 and T=0)."
        )

    dist_cols = list(dict.fromkeys(xc_cols + xz_cols))
    model_f_t1 = fit_conditional_distribution_model(
        X=df_source_t1[dist_cols], y=df_source_t1[y_col], feature_cols=dist_cols
    )
    model_f_t0 = fit_conditional_distribution_model(
        X=df_source_t0[dist_cols], y=df_source_t0[y_col], feature_cols=dist_cols
    )

    gy_cols = list(dict.fromkeys(xc_cols + [t_col, y_col]))
    # Keep classifier semantics explicit: class-1 always means "belongs to source_g".
    source_membership = (df[g_col].to_numpy(dtype=float) == float(resolved_source_g)).astype(float)
    model_g_y = fit_classifier(df[gy_cols], source_membership)

    gxz_cols = list(dict.fromkeys(xc_cols + [t_col] + xz_cols))
    model_g_xz = fit_classifier(df[gxz_cols], source_membership)

    return {
        "model_f_t1": model_f_t1,
        "model_f_t0": model_f_t0,
        "model_g_y": model_g_y,
        "model_g_xz": model_g_xz,
        "Xc_cols": xc_cols,
        "Xz_cols": xz_cols,
        "dist_feature_cols": dist_cols,
        "g_y_feature_cols": gy_cols,
        "g_xz_feature_cols": gxz_cols,
        # backward compatibility alias
        "Xs_cols": xz_cols,
        "t_col": t_col,
        "y_col": y_col,
        "g_col": g_col,
        "source_g": resolved_source_g,
        "target_g": resolved_target_g,
        "shadow_direction": resolved_direction,
    }


def predict_mu_t_shadow(
    models: Dict[str, object],
    xc_vec: ArrayLike,
    xz_vec: ArrayLike,
    t: int,
    M: int = 2000,
    random_state: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    shadow_direction: Optional[str] = None,
    source_g: GroupLike = None,
    target_g: GroupLike = None,
) -> float:
    """
    Predict ``mu_t^shadow(xc, xz)`` for continuous ``Y`` via Monte Carlo.

    Recovery weight for each sampled ``y``:
    ``r_y = (1-s)/s``, where ``s = P(G=source_g|Xc, T=t, Y=y)``.

    The ratio form is direction-invariant. Only source/target group assignment changes.
    """
    if int(t) not in (0, 1):
        raise ValueError(f"t must be 0 or 1, got {t}")
    if M <= 0:
        raise ValueError("M must be positive.")

    if rng is None:
        rng = np.random.default_rng(random_state)

    model_f = models["model_f_t1"] if int(t) == 1 else models["model_f_t0"]
    model_g_y = models["model_g_y"]
    resolved_source_g, resolved_target_g, resolved_direction = _resolve_shadow_direction(
        shadow_direction=shadow_direction,
        source_g=source_g,
        target_g=target_g,
        default_source_g=models.get("source_g", 1),
        default_target_g=models.get("target_g", 0),
    )
    _ = resolved_source_g, resolved_target_g, resolved_direction

    xc = _to_1d_float(xc_vec)
    xz = _to_1d_float(xz_vec)
    y_samples = _to_1d_float(model_f.sample(np.concatenate([xc, xz]), n_samples=M, rng=rng))

    gy_rows = np.column_stack(
        [
            np.repeat(xc.reshape(1, -1), int(M), axis=0),
            np.full((int(M), 1), float(t)),
            y_samples.reshape(-1, 1),
        ]
    )
    s = _to_1d_float(predict_prob(model_g_y, gy_rows))
    s = _to_1d_float(clip_prob(s))
    # Direction-invariant ratio: s is source-membership prob, so (1-s)/s is target/source.
    r_y = (1.0 - s) / s

    denom = float(np.sum(r_y))
    if not np.isfinite(denom) or denom <= 0:
        raise ValueError(f"Invalid Monte Carlo denominator in predict_mu_t_shadow: {denom}")

    numerator = float(np.sum(y_samples * r_y))
    if not np.isfinite(numerator):
        raise ValueError("Invalid Monte Carlo numerator in predict_mu_t_shadow.")

    return float(numerator / denom)


def predict_tau_shadow(
    models: Dict[str, object],
    xc_vec: ArrayLike,
    xz_vec: ArrayLike,
    M: int = 2000,
    random_state: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    shadow_direction: Optional[str] = None,
    source_g: GroupLike = None,
    target_g: GroupLike = None,
) -> Dict[str, float]:
    """Compute ``tau_shadow = mu1_shadow - mu0_shadow`` for one sample."""
    if rng is None:
        rng = np.random.default_rng(random_state)

    mu1 = predict_mu_t_shadow(
        models,
        xc_vec=xc_vec,
        xz_vec=xz_vec,
        t=1,
        M=M,
        rng=rng,
        shadow_direction=shadow_direction,
        source_g=source_g,
        target_g=target_g,
    )
    mu0 = predict_mu_t_shadow(
        models,
        xc_vec=xc_vec,
        xz_vec=xz_vec,
        t=0,
        M=M,
        rng=rng,
        shadow_direction=shadow_direction,
        source_g=source_g,
        target_g=target_g,
    )
    tau = float(mu1 - mu0)
    return {"mu1_shadow": float(mu1), "mu0_shadow": float(mu0), "tau_shadow": tau}


def build_shadow_obs_outcomes_for_cvci(
    df_obs: pd.DataFrame,
    shadow_models: Dict[str, object],
    Xc_cols: Optional[Sequence[str]] = None,
    Xz_cols: Optional[Sequence[str]] = None,
    t_col: str = "T",
    M: int = 2000,
    random_state: int = 2024,
    *,
    X_cols: Optional[Sequence[str]] = None,
    selected_shadow_cols: Optional[Sequence[str]] = None,
    shadow_direction: Optional[str] = None,
    source_g: GroupLike = None,
    target_g: GroupLike = None,
) -> np.ndarray:
    """Build shadow-corrected outcomes for CVCI ``obs_outcomes`` substitution.

    In RCT-target mode this should be configured as obs->rct (source=OBS, target=RCT),
    so pseudo outcomes align OBS samples to the RCT target distribution.
    """
    xc_cols, xz_cols = _resolve_xc_xz(
        Xc_cols=Xc_cols,
        Xz_cols=Xz_cols,
        X_cols=X_cols,
        selected_shadow_cols=selected_shadow_cols,
    )

    resolved_source_g, resolved_target_g, resolved_direction = _resolve_shadow_direction(
        shadow_direction=shadow_direction,
        source_g=source_g,
        target_g=target_g,
        default_source_g=shadow_models.get("source_g", 1),
        default_target_g=shadow_models.get("target_g", 0),
    )
    _ = resolved_source_g, resolved_target_g, resolved_direction

    if t_col not in df_obs.columns:
        raise ValueError(f"Missing treatment column in df_obs: {t_col}")

    rng = np.random.default_rng(random_state)
    out = np.zeros(df_obs.shape[0], dtype=float)
    for idx, (_, row) in enumerate(df_obs.iterrows()):
        out[idx] = predict_mu_t_shadow(
            shadow_models,
            xc_vec=row[xc_cols].to_numpy(dtype=float) if xc_cols else np.asarray([], dtype=float),
            xz_vec=row[xz_cols].to_numpy(dtype=float) if xz_cols else np.asarray([], dtype=float),
            t=int(row[t_col]),
            M=M,
            rng=rng,
            shadow_direction=resolved_direction,
            source_g=resolved_source_g,
            target_g=resolved_target_g,
        )
    return out


def build_shadow_corrected_targets_for_rhc(
    df_rct: pd.DataFrame,
    shadow_models: Dict[str, object],
    w_hat_rct: np.ndarray,
    Xc_cols: Optional[Sequence[str]] = None,
    Xz_cols: Optional[Sequence[str]] = None,
    t_col: str = "T",
    M: int = 2000,
    random_state: int = 2024,
    *,
    X_cols: Optional[Sequence[str]] = None,
    selected_shadow_cols: Optional[Sequence[str]] = None,
    shadow_direction: Optional[str] = None,
    source_g: GroupLike = None,
    target_g: GroupLike = None,
) -> Dict[str, object]:
    """Build RHC second-stage corrected target: ``corrected_target = tau_shadow - w_hat``."""
    xc_cols, xz_cols = _resolve_xc_xz(
        Xc_cols=Xc_cols,
        Xz_cols=Xz_cols,
        X_cols=X_cols,
        selected_shadow_cols=selected_shadow_cols,
    )

    resolved_source_g, resolved_target_g, resolved_direction = _resolve_shadow_direction(
        shadow_direction=shadow_direction,
        source_g=source_g,
        target_g=target_g,
        default_source_g=shadow_models.get("source_g", 1),
        default_target_g=shadow_models.get("target_g", 0),
    )

    w_hat = _to_1d_float(w_hat_rct)
    if w_hat.shape[0] != df_rct.shape[0]:
        raise ValueError(f"w_hat_rct length mismatch: {w_hat.shape[0]} vs {df_rct.shape[0]}")

    rng = np.random.default_rng(random_state)
    mu1_list: List[float] = []
    mu0_list: List[float] = []
    tau_list: List[float] = []

    for _, row in df_rct.iterrows():
        tau_obj = predict_tau_shadow(
            shadow_models,
            xc_vec=row[xc_cols].to_numpy(dtype=float) if xc_cols else np.asarray([], dtype=float),
            xz_vec=row[xz_cols].to_numpy(dtype=float) if xz_cols else np.asarray([], dtype=float),
            M=M,
            rng=rng,
            shadow_direction=resolved_direction,
            source_g=resolved_source_g,
            target_g=resolved_target_g,
        )
        mu1_list.append(float(tau_obj["mu1_shadow"]))
        mu0_list.append(float(tau_obj["mu0_shadow"]))
        tau_list.append(float(tau_obj["tau_shadow"]))

    tau_shadow = _to_1d_float(tau_list)
    corrected_target = tau_shadow - w_hat

    if np.any(~np.isfinite(corrected_target)):
        raise ValueError("Non-finite corrected targets in shadow pipeline.")

    return {
        "corrected_targets": corrected_target,
        "diagnostics": {
            "mu1_shadow": _to_1d_float(mu1_list),
            "mu0_shadow": _to_1d_float(mu0_list),
            "tau_shadow": tau_shadow,
            "mean_tau_shadow": float(np.mean(tau_shadow)),
            "std_tau_shadow": float(np.std(tau_shadow)),
            "mean_corrected_target": float(np.mean(corrected_target)),
            "std_corrected_target": float(np.std(corrected_target)),
            "shadow_direction": resolved_direction,
            "source_g": resolved_source_g,
            "target_g": resolved_target_g,
        },
    }
