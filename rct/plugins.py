from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, LogisticRegression

from methods.iv import fit_iv_pipeline, select_iv_candidates
from methods.shadow import (
    build_shadow_obs_outcomes_for_cvci,
    fit_shadow_pipeline,
    screen_shadow_candidates,
)
from rct.cv import cross_validation


@dataclass
class ObsPluginOutput:
    method: str
    sample_weights: np.ndarray
    pseudo_outcome: Optional[np.ndarray]
    corrected_score: Optional[np.ndarray]
    corrected_loss_component: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if np.any(~np.isfinite(weights)):
        raise ValueError("Plugin produced non-finite weights.")
    if np.any(weights <= 0):
        raise ValueError("Plugin produced non-positive weights.")
    return weights / np.mean(weights)


def _default_lambda_grid(config: Optional[Dict[str, Any]]) -> np.ndarray:
    config = config or {}
    if "lambda_vals" in config and config["lambda_vals"] is not None:
        return np.asarray(config["lambda_vals"], dtype=float)
    lambda_bin = int(config.get("lambda_bin", 5))
    return np.linspace(0.0, 1.0, lambda_bin)


def _extract_design(X: np.ndarray, d: int, include_treatment: bool = True) -> np.ndarray:
    z = X[:, :d]
    if not include_treatment:
        return z
    a = X[:, d].reshape(-1, 1)
    return np.concatenate((a, z), axis=1)


def _safe_column_names(config: Dict[str, Any], d: int) -> list:
    names = config.get("covariate_names")
    if names is None:
        return [f"x{idx}" for idx in range(d)]
    if len(names) != d:
        raise ValueError("Length of covariate_names must match d.")
    return list(names)


def _residualize(target: np.ndarray, design: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=float).reshape(-1)
    if design.size == 0:
        return target - np.mean(target)
    augmented = np.concatenate((np.ones((design.shape[0], 1)), design), axis=1)
    coef, _, _, _ = np.linalg.lstsq(augmented, target, rcond=None)
    fitted = augmented @ coef
    return target - fitted


def _compute_partial_exclusion_pvalue(
    candidate: np.ndarray,
    outcome: np.ndarray,
    nuisance: np.ndarray,
) -> float:
    residual_candidate = _residualize(candidate, nuisance)
    residual_outcome = _residualize(outcome, nuisance)
    if np.std(residual_candidate) < 1e-12 or np.std(residual_outcome) < 1e-12:
        return 1.0
    corr_result = pearsonr(residual_candidate, residual_outcome)
    return float(corr_result.pvalue if hasattr(corr_result, "pvalue") else corr_result[1])


def _compute_relevance_pvalue(candidate: np.ndarray, source: np.ndarray) -> float:
    if np.std(candidate) < 1e-12:
        return 1.0
    corr_result = pearsonr(candidate, source)
    return float(corr_result.pvalue if hasattr(corr_result, "pvalue") else corr_result[1])


def _select_iv_columns(obs_data: np.ndarray, exp_data: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    d_obs = config.get("d_obs")
    d_exp = config.get("d_exp")
    if d_obs is None or d_exp is None or d_obs != d_exp:
        raise ValueError("IV plugin currently requires d_obs == d_exp and both must be provided.")

    column_names = _safe_column_names(config, d_obs)
    source = np.concatenate((np.ones(exp_data.shape[0]), np.zeros(obs_data.shape[0])))
    x_all = np.concatenate((exp_data[:, :d_exp], obs_data[:, :d_obs]), axis=0)
    a_all = np.concatenate((exp_data[:, d_exp], obs_data[:, d_obs]), axis=0).reshape(-1, 1)
    y_all = np.concatenate((exp_data[:, -1], obs_data[:, -1]), axis=0)

    relevance_alpha = float(config.get("iv_relevance_alpha", 0.05))
    exclusion_alpha = float(config.get("iv_exclusion_alpha", 0.10))
    top_k = config.get("iv_max_ivs")

    summaries = []
    for column_idx in range(d_obs):
        candidate = x_all[:, column_idx]
        remaining = np.delete(x_all, column_idx, axis=1)
        nuisance = np.concatenate((a_all, remaining), axis=1) if remaining.size else a_all
        relevance_pvalue = _compute_relevance_pvalue(candidate, source)
        exclusion_pvalue = _compute_partial_exclusion_pvalue(candidate, y_all, nuisance)
        summaries.append(
            {
                "index": column_idx,
                "name": column_names[column_idx],
                "relevance_pvalue": float(relevance_pvalue),
                "exclusion_pvalue": float(exclusion_pvalue),
                "selected": relevance_pvalue < relevance_alpha and exclusion_pvalue > exclusion_alpha,
            }
        )

    selected = [item for item in summaries if item["selected"]]
    selected.sort(key=lambda item: (item["relevance_pvalue"], -item["exclusion_pvalue"]))
    if top_k is not None:
        selected = selected[: int(top_k)]

    if not selected:
        best_fallback = min(summaries, key=lambda item: (item["relevance_pvalue"], -item["exclusion_pvalue"]))
        best_fallback["selected"] = True
        selected = [best_fallback]
        for item in summaries:
            if item["index"] == best_fallback["index"]:
                item["selected"] = True

    selected_indices = [item["index"] for item in selected]
    return {
        "selected_indices": selected_indices,
        "selected_names": [item["name"] for item in selected],
        "screening_summary": summaries,
        "relevance_alpha": relevance_alpha,
        "exclusion_alpha": exclusion_alpha,
    }


def _select_shadow_column(obs_data: np.ndarray, exp_data: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    d_obs = config.get("d_obs")
    d_exp = config.get("d_exp")
    if d_obs is None or d_exp is None or d_obs != d_exp:
        raise ValueError("Shadow plugin currently requires d_obs == d_exp and both must be provided.")

    column_names = _safe_column_names(config, d_obs)
    source = np.concatenate((np.ones(exp_data.shape[0]), np.zeros(obs_data.shape[0])))
    x_all = np.concatenate((exp_data[:, :d_exp], obs_data[:, :d_obs]), axis=0)
    a_all = np.concatenate((exp_data[:, d_exp], obs_data[:, d_obs]), axis=0).reshape(-1, 1)
    y_all = np.concatenate((exp_data[:, -1], obs_data[:, -1]), axis=0)

    x_exp = exp_data[:, :d_exp]
    a_exp = exp_data[:, d_exp].reshape(-1, 1)
    y_exp = exp_data[:, -1]

    relevance_alpha = float(config.get("shadow_relevance_alpha", 0.05))
    independence_alpha = float(config.get("shadow_independence_alpha", 0.10))
    positivity_margin = float(config.get("shadow_positivity_margin", 0.01))
    allow_fallback = bool(config.get("shadow_allow_fallback", False))

    summaries = []
    for column_idx in range(d_obs):
        candidate_all = x_all[:, column_idx]
        controls_all = np.delete(x_all, column_idx, axis=1)
        nuisance_independence = np.concatenate((controls_all, a_all, y_all.reshape(-1, 1)), axis=1) if controls_all.size else np.concatenate((a_all, y_all.reshape(-1, 1)), axis=1)
        independence_pvalue = _compute_partial_exclusion_pvalue(candidate_all, source, nuisance_independence)

        candidate_exp = x_exp[:, column_idx]
        controls_exp = np.delete(x_exp, column_idx, axis=1)
        nuisance_relevance = np.concatenate((controls_exp, a_exp), axis=1) if controls_exp.size else a_exp
        relevance_pvalue = _compute_partial_exclusion_pvalue(candidate_exp, y_exp, nuisance_relevance)

        selection_features = np.concatenate((nuisance_independence, candidate_all.reshape(-1, 1)), axis=1)
        positivity_model = LogisticRegression(max_iter=int(config.get("shadow_max_iter", 1000)))
        positivity_model.fit(selection_features, source)
        source_prob = positivity_model.predict_proba(selection_features)[:, 1]
        positivity_score = float(np.min(np.minimum(source_prob, 1.0 - source_prob)))
        positivity_ok = positivity_score > positivity_margin

        summaries.append(
            {
                "index": column_idx,
                "name": column_names[column_idx],
                "relevance_pvalue": float(relevance_pvalue),
                "independence_pvalue": float(independence_pvalue),
                "positivity_score": positivity_score,
                "selected": relevance_pvalue < relevance_alpha and independence_pvalue > independence_alpha and positivity_ok,
            }
        )

    selected = [item for item in summaries if item["selected"]]
    selected.sort(key=lambda item: (item["relevance_pvalue"], -item["independence_pvalue"], -item["positivity_score"]))

    if not selected and not allow_fallback:
        raise RuntimeError(
            "No candidate shadow variable satisfied the relevance, independence, and positivity assumptions "
            f"under the current thresholds: relevance_alpha={relevance_alpha}, "
            f"independence_alpha={independence_alpha}, positivity_margin={positivity_margin}."
        )

    if not selected:
        best_fallback = min(summaries, key=lambda item: (item["relevance_pvalue"], -item["independence_pvalue"], -item["positivity_score"]))
        best_fallback["selected"] = True
        selected = [best_fallback]
        for item in summaries:
            if item["index"] == best_fallback["index"]:
                item["selected"] = True

    best = selected[0]
    return {
        "selected_index": best["index"],
        "selected_name": best["name"],
        "screening_summary": summaries,
        "relevance_alpha": relevance_alpha,
        "independence_alpha": independence_alpha,
        "positivity_margin": positivity_margin,
        "allow_fallback": allow_fallback,
    }


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def _safe_standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    scale = np.std(values)
    if scale < 1e-12:
        return values - np.mean(values)
    return (values - np.mean(values)) / scale


def _fit_shadow_score(obs_data: np.ndarray, exp_data: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    d_obs = config["d_obs"]
    d_exp = config["d_exp"]
    covariate_names = _safe_column_names(config, d_obs)

    x_exp = exp_data[:, :d_exp]
    x_obs = obs_data[:, :d_obs]
    a_exp = exp_data[:, d_exp].reshape(-1, 1)
    a_obs = obs_data[:, d_obs].reshape(-1, 1)
    y_exp = exp_data[:, -1]
    y_obs = obs_data[:, -1]

    x_all = np.concatenate((x_exp, x_obs), axis=0)
    a_all = np.concatenate((a_exp, a_obs), axis=0)
    y_all = np.concatenate((y_exp, y_obs), axis=0)
    g_all = np.concatenate((np.ones(exp_data.shape[0]), np.zeros(obs_data.shape[0])))
    y_all_std = _safe_standardize(y_all)
    y_exp_std = y_all_std[: exp_data.shape[0]]

    score_features = np.concatenate((np.ones((x_all.shape[0], 1)), a_all, x_all, y_all_std.reshape(-1, 1)), axis=1)
    score_feature_names = ["intercept", "A"] + covariate_names + ["Y_std"]

    relevance_weight = float(config.get("shadow_relevance_weight", 1.0))
    independence_weight = float(config.get("shadow_independence_weight", 1.0))
    positivity_weight = float(config.get("shadow_positivity_weight", 10.0))
    positivity_margin = float(config.get("shadow_positivity_margin", 0.05))
    l2_weight = float(config.get("shadow_l2_weight", 1e-3))
    max_iter = int(config.get("shadow_max_iter", 1000))

    nuisance_rct = np.concatenate((a_exp, x_exp), axis=1)
    nuisance_all = np.concatenate((a_all, x_all, y_all_std.reshape(-1, 1)), axis=1)

    def objective(params: np.ndarray) -> float:
        shadow_score = _sigmoid(score_features @ params)
        shadow_score_exp = shadow_score[: exp_data.shape[0]]

        relevance_corr = abs(
            _compute_partial_exclusion_pvalue(
                shadow_score_exp,
                y_exp_std,
                nuisance_rct,
            )
        )
        independence_corr = abs(
            _compute_partial_exclusion_pvalue(
                shadow_score,
                g_all,
                nuisance_all,
            )
        )

        selection_features = np.concatenate((a_all, x_all, y_all_std.reshape(-1, 1), shadow_score.reshape(-1, 1)), axis=1)
        positivity_model = LogisticRegression(max_iter=max_iter)
        positivity_model.fit(selection_features, g_all)
        source_prob = positivity_model.predict_proba(selection_features)[:, 1]
        lower_violation = np.maximum(0.0, positivity_margin - source_prob)
        upper_violation = np.maximum(0.0, source_prob - (1.0 - positivity_margin))
        positivity_penalty = np.mean(lower_violation**2 + upper_violation**2)

        return relevance_weight * relevance_corr - independence_weight * independence_corr + positivity_weight * positivity_penalty + l2_weight * np.sum(params**2)

    result = minimize(
        objective,
        np.zeros(score_features.shape[1], dtype=float),
        method="L-BFGS-B",
        options={"maxiter": max_iter},
    )
    if not result.success and not np.isfinite(result.fun):
        raise RuntimeError(f"Shadow score optimization failed: {result.message}")

    shadow_score_all = _sigmoid(score_features @ result.x)
    shadow_score_exp = shadow_score_all[: exp_data.shape[0]]
    shadow_score_obs = shadow_score_all[exp_data.shape[0] :]

    calibration_model = LinearRegression()
    calibration_model.fit(shadow_score_exp.reshape(-1, 1), y_exp)
    pseudo_outcome = calibration_model.predict(shadow_score_obs.reshape(-1, 1))

    final_relevance = _compute_partial_exclusion_pvalue(shadow_score_exp, y_exp_std, nuisance_rct)
    final_independence = _compute_partial_exclusion_pvalue(shadow_score_all, g_all, nuisance_all)
    selection_features = np.concatenate((a_all, x_all, y_all_std.reshape(-1, 1), shadow_score_all.reshape(-1, 1)), axis=1)
    selection_model = LogisticRegression(max_iter=max_iter)
    selection_model.fit(selection_features, g_all)
    source_prob = selection_model.predict_proba(selection_features)[:, 1]

    return {
        "shadow_score_obs": shadow_score_obs,
        "pseudo_outcome": pseudo_outcome,
        "score_feature_names": score_feature_names,
        "shadow_score_params": result.x,
        "shadow_score_optimization": result,
        "calibration_model": calibration_model,
        "selection_model": selection_model,
        "selection_probabilities": source_prob[-obs_data.shape[0] :],
        "relevance_pvalue": float(final_relevance),
        "independence_pvalue": float(final_independence),
        "positivity_score": float(np.min(np.minimum(source_prob, 1.0 - source_prob))),
        "positivity_margin": positivity_margin,
        "weights": {
            "relevance_weight": relevance_weight,
            "independence_weight": independence_weight,
            "positivity_weight": positivity_weight,
            "l2_weight": l2_weight,
        },
    }


def _build_source_selection_design(
    obs_data: np.ndarray,
    exp_data: np.ndarray,
    selected_indices: list,
    config: Dict[str, Any],
):
    d_obs = config["d_obs"]
    d_exp = config["d_exp"]
    covariate_names = _safe_column_names(config, d_obs)
    use_treatment = bool(config.get("iv_include_treatment", True))
    use_outcome = bool(config.get("iv_include_outcome", True))
    use_yz_interactions = bool(config.get("iv_include_yz_interactions", True))

    x_exp = exp_data[:, :d_exp]
    x_obs = obs_data[:, :d_obs]
    a_exp = exp_data[:, d_exp].reshape(-1, 1)
    a_obs = obs_data[:, d_obs].reshape(-1, 1)
    y_exp = exp_data[:, -1].reshape(-1, 1)
    y_obs = obs_data[:, -1].reshape(-1, 1)

    x_all = np.concatenate((x_exp, x_obs), axis=0)
    a_all = np.concatenate((a_exp, a_obs), axis=0)
    y_all = np.concatenate((y_exp, y_obs), axis=0)
    source = np.concatenate((np.ones(exp_data.shape[0]), np.zeros(obs_data.shape[0])))

    design_blocks = [x_all]
    feature_names = list(covariate_names)

    if use_treatment:
        design_blocks.append(a_all)
        feature_names.append("A")
    if use_outcome:
        design_blocks.append(y_all)
        feature_names.append("Y")
    if selected_indices:
        z_all = x_all[:, selected_indices]
        design_blocks.append(z_all)
        feature_names.extend([f"Z:{covariate_names[idx]}" for idx in selected_indices])
        if use_outcome and use_yz_interactions:
            yz = y_all * z_all
            design_blocks.append(yz)
            feature_names.extend([f"Y*Z:{covariate_names[idx]}" for idx in selected_indices])

    design_matrix = np.concatenate(design_blocks, axis=1)
    return design_matrix, source, feature_names


def fit_base_cvci(exp_data: np.ndarray, obs_data: np.ndarray, config: Optional[Dict[str, Any]] = None):
    config = config or {}
    lambda_vals = _default_lambda_grid(config)
    mode = config.get("mode", "linear")
    result = cross_validation(
        exp_data,
        obs_data,
        lambda_vals,
        mode=mode,
        k_fold=config.get("k_fold", 5),
        d_exp=config.get("d_exp"),
        d_obs=config.get("d_obs"),
        exp_model=config.get("exp_model", "response_func"),
        stratified_kfold=config.get("stratified_kfold", True),
        random_state=config.get("random_state"),
        rng=config.get("rng"),
    )
    q_values, lambda_opt, theta_opt = result
    return {
        "Q_values": q_values,
        "lambda_opt": lambda_opt,
        "theta_opt": theta_opt,
    }


def build_obs_plugin(obs_data: np.ndarray, exp_data: np.ndarray, method: str, config: Optional[Dict[str, Any]] = None) -> ObsPluginOutput:
    config = config or {}
    method = method.lower()
    if method == "none":
        weights = np.ones(obs_data.shape[0], dtype=float)
        return ObsPluginOutput(
            method="none",
            sample_weights=weights,
            pseudo_outcome=None,
            corrected_score=None,
            corrected_loss_component=None,
            metadata={"description": "Unmodified observational loss."},
        )
    if method == "ipsw":
        return _build_ipsw_plugin(obs_data, config)
    if method == "cw":
        return _build_cw_plugin(obs_data, exp_data, config)
    if method == "iv":
        return _build_iv_plugin(obs_data, exp_data, config)
    if method == "shadow":
        return _build_shadow_plugin(obs_data, exp_data, config)
    raise ValueError(f"Unsupported plugin method: {method}")


def fit_cvci_with_plugin(
    exp_data: np.ndarray,
    obs_data: np.ndarray,
    plugin_output: ObsPluginOutput,
    config: Optional[Dict[str, Any]] = None,
):
    config = config or {}
    lambda_vals = _default_lambda_grid(config)
    mode = config.get("mode", "linear")
    q_values, lambda_opt, theta_opt = cross_validation(
        exp_data,
        obs_data,
        lambda_vals,
        mode=mode,
        k_fold=config.get("k_fold", 5),
        d_exp=config.get("d_exp"),
        d_obs=config.get("d_obs"),
        exp_model=config.get("exp_model", "response_func"),
        stratified_kfold=config.get("stratified_kfold", True),
        random_state=config.get("random_state"),
        rng=config.get("rng"),
        obs_weights=plugin_output.sample_weights,
        obs_outcomes=plugin_output.pseudo_outcome,
    )
    return {
        "plugin_method": plugin_output.method,
        "plugin_output": plugin_output,
        "Q_values": q_values,
        "lambda_opt": lambda_opt,
        "theta_opt": theta_opt,
        "estimate": theta_opt.beta().item() if hasattr(theta_opt.beta(), "item") else theta_opt.beta(),
    }


def _build_ipsw_plugin(obs_data: np.ndarray, config: Dict[str, Any]) -> ObsPluginOutput:
    d_obs = config.get("d_obs")
    if d_obs is None:
        raise ValueError("d_obs must be provided in config for the IPSW plugin.")
    include_treatment = config.get("ipsw_include_treatment", False)
    clip_min = float(config.get("ipsw_clip_min", 1e-3))
    clip_max = float(config.get("ipsw_clip_max", 1 - 1e-3))
    max_iter = int(config.get("ipsw_max_iter", 1000))

    features = _extract_design(obs_data, d_obs, include_treatment=include_treatment)
    treatment = obs_data[:, d_obs]

    propensity_model = LogisticRegression(max_iter=max_iter)
    propensity_model.fit(features, treatment)
    propensity = propensity_model.predict_proba(features)[:, 1]
    propensity = np.clip(propensity, clip_min, clip_max)
    weights = np.where(treatment == 1, 1.0 / propensity, 1.0 / (1.0 - propensity))
    weights = _normalize_weights(weights)

    return ObsPluginOutput(
        method="ipsw",
        sample_weights=weights,
        pseudo_outcome=None,
        corrected_score=None,
        corrected_loss_component={"type": "weighted_obs_loss"},
        metadata={
            "propensity_model": propensity_model,
            "propensity_scores": propensity,
            "include_treatment": include_treatment,
        },
    )


def _build_cw_plugin(obs_data: np.ndarray, exp_data: np.ndarray, config: Dict[str, Any]) -> ObsPluginOutput:
    d_obs = config.get("d_obs")
    d_exp = config.get("d_exp")
    if d_obs is None or d_exp is None:
        raise ValueError("d_exp and d_obs must be provided in config for the CW plugin.")
    include_treatment = config.get("cw_include_treatment", True)
    max_iter = int(config.get("cw_max_iter", 500))

    obs_features = _extract_design(obs_data, d_obs, include_treatment=include_treatment)
    exp_features = _extract_design(exp_data, d_exp, include_treatment=include_treatment)
    obs_aug = np.concatenate((np.ones((obs_features.shape[0], 1)), obs_features), axis=1)
    target_moments = np.mean(np.concatenate((np.ones((exp_features.shape[0], 1)), exp_features), axis=1), axis=0)

    def dual_objective(gamma: np.ndarray) -> float:
        linear_term = obs_aug @ gamma
        return np.mean(np.exp(linear_term)) - target_moments @ gamma

    def dual_gradient(gamma: np.ndarray) -> np.ndarray:
        linear_term = obs_aug @ gamma
        exp_term = np.exp(linear_term).reshape(-1, 1)
        return np.mean(obs_aug * exp_term, axis=0) - target_moments

    result = minimize(
        dual_objective,
        np.zeros(obs_aug.shape[1], dtype=float),
        jac=dual_gradient,
        method="BFGS",
        options={"maxiter": max_iter},
    )
    if not result.success:
        raise RuntimeError(f"CW optimization failed: {result.message}")

    weights = np.exp(obs_aug @ result.x)
    weights = _normalize_weights(weights)

    return ObsPluginOutput(
        method="cw",
        sample_weights=weights,
        pseudo_outcome=None,
        corrected_score=None,
        corrected_loss_component={"type": "weighted_obs_loss"},
        metadata={
            "dual_solution": result.x,
            "optimization_result": result,
            "include_treatment": include_treatment,
            "target_moments": target_moments,
        },
    )


def _build_iv_plugin(obs_data: np.ndarray, exp_data: np.ndarray, config: Dict[str, Any]) -> ObsPluginOutput:
    d_obs = config.get("d_obs")
    d_exp = config.get("d_exp")
    if d_obs is None or d_exp is None or d_obs != d_exp:
        raise ValueError("IV plugin requires d_obs == d_exp and both provided.")
    covariate_names = _safe_column_names(config, d_obs)

    df_exp = pd.DataFrame(exp_data[:, :d_exp], columns=covariate_names)
    df_exp["T"] = exp_data[:, d_exp].astype(float)
    df_exp["Y"] = exp_data[:, -1].astype(float)
    df_exp["G"] = 1.0

    df_obs = pd.DataFrame(obs_data[:, :d_obs], columns=covariate_names)
    df_obs["T"] = obs_data[:, d_obs].astype(float)
    df_obs["Y"] = obs_data[:, -1].astype(float)
    df_obs["G"] = 0.0

    df_all = pd.concat([df_exp, df_obs], axis=0, ignore_index=True)
    relevance_threshold = float(config.get("iv_relevance_alpha", 0.02))
    exclusion_threshold = float(config.get("iv_exclusion_alpha", 0.02))
    allow_fallback = bool(config.get("iv_allow_fallback", True))

    screening = select_iv_candidates(
        df_all,
        candidate_cols=covariate_names,
        t_col="T",
        y_col="Y",
        g_col="G",
        relevance_threshold=relevance_threshold,
        exclusion_threshold=exclusion_threshold,
        allow_empty_fallback=allow_fallback,
    )
    selected_iv_cols = list(screening["selected_iv_cols"])
    xc_cols = list(screening["Xc_cols"])

    # For RCT-target OBS-side loss, estimate odds on reversed source label so w=(1-pi)/pi
    # becomes p(RCT|x)/p(OBS|x) on OBS samples.
    df_all_for_obs_weight = df_all.copy()
    df_all_for_obs_weight["_G_IV"] = 1.0 - df_all_for_obs_weight["G"]
    all_weights = fit_iv_pipeline(
        df_all_for_obs_weight,
        Xc_cols=xc_cols,
        Xz_cols=selected_iv_cols,
        t_col="T",
        y_col="Y",
        g_col="_G_IV",
        weight_clip_min=float(config.get("iv_clip_min", 0.05)),
        weight_clip_max=float(config.get("iv_clip_max", 20.0)),
        max_iter=int(config.get("iv_max_iter", 2000)),
    )
    weights = _normalize_weights(all_weights[-obs_data.shape[0] :])

    return ObsPluginOutput(
        method="iv",
        sample_weights=weights,
        pseudo_outcome=None,
        corrected_score=None,
        corrected_loss_component={"type": "weighted_obs_loss_iv"},
        metadata={
            "selected_iv_names": selected_iv_cols,
            "selected_iv_cols": selected_iv_cols,
            "screening_logs": screening["screening_logs"],
            "screening_summary": screening,
            "Xc_cols": xc_cols,
            "Xz_cols": selected_iv_cols,
            "selection_formula": "w_iv=(1-pi)/pi with pi from eta+lambda logistic decomposition",
        },
    )


def _build_shadow_plugin(obs_data: np.ndarray, exp_data: np.ndarray, config: Dict[str, Any]) -> ObsPluginOutput:
    d_obs = config.get("d_obs")
    d_exp = config.get("d_exp")
    if d_obs is None or d_exp is None:
        raise ValueError("d_exp and d_obs must be provided in config for the shadow plugin.")

    if d_obs != d_exp:
        raise ValueError("Shadow plugin currently requires d_obs == d_exp.")
    covariate_names = _safe_column_names(config, d_obs)
    relevance_threshold = float(config.get("shadow_association_threshold", 0.02))
    independence_threshold = float(config.get("shadow_residual_independence_threshold", 0.02))
    allow_empty_fallback = bool(config.get("shadow_allow_fallback", False))
    mc_samples = int(config.get("shadow_mc_samples", 2000))
    random_state = int(config.get("random_state", 2024))

    df_exp = pd.DataFrame(exp_data[:, :d_exp], columns=covariate_names)
    df_exp["T"] = exp_data[:, d_exp].astype(float)
    df_exp["Y"] = exp_data[:, -1].astype(float)
    df_exp["G"] = 1.0

    df_obs = pd.DataFrame(obs_data[:, :d_obs], columns=covariate_names)
    df_obs["T"] = obs_data[:, d_obs].astype(float)
    df_obs["Y"] = obs_data[:, -1].astype(float)
    df_obs["G"] = 0.0

    df_all = pd.concat([df_exp, df_obs], axis=0, ignore_index=True)
    screening = screen_shadow_candidates(
        df_all,
        X_cols=covariate_names,
        t_col="T",
        y_col="Y",
        g_col="G",
        relevance_threshold=relevance_threshold,
        independence_threshold=independence_threshold,
        allow_empty_fallback=allow_empty_fallback,
    )
    selected_shadow_cols = list(screening["selected_shadow_cols"])
    xc_cols = list(screening["Xc_cols"])
    shadow_models = fit_shadow_pipeline(
        df_all,
        Xc_cols=xc_cols,
        Xz_cols=selected_shadow_cols,
        t_col="T",
        y_col="Y",
        g_col="G",
    )
    pseudo_outcome = build_shadow_obs_outcomes_for_cvci(
        df_obs=df_obs,
        shadow_models=shadow_models,
        Xc_cols=xc_cols,
        Xz_cols=selected_shadow_cols,
        t_col="T",
        M=mc_samples,
        random_state=random_state,
    )

    return ObsPluginOutput(
        method="shadow",
        sample_weights=np.ones(obs_data.shape[0], dtype=float),
        pseudo_outcome=pseudo_outcome,
        corrected_score=None,
        corrected_loss_component={"type": "shadow_obs_outcome_substitution"},
        metadata={
            "selected_shadow_cols": selected_shadow_cols,
            "Xc_cols": shadow_models["Xc_cols"],
            "Xz_cols": shadow_models["Xz_cols"],
            "screening_logs": screening["screening_logs"],
            "screening_summary": screening,
            "pseudo_outcome_mean": float(np.mean(pseudo_outcome)),
            "pseudo_outcome_std": float(np.std(pseudo_outcome)),
            "shadow_mc_samples": mc_samples,
            "description": "Shadow correction replaces observational outcomes used by L_obs.",
        },
    )
