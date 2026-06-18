"""Unified experiment runner for LaLonde/ACTG/JTPA with RCT-target (CVCI) and OBS-target methods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from methods import CWPlugin, IPSWPlugin, SelectionCorrectionPlugin, SelectionIVPlugin, ShadowPlugin, ShadowSourceEPPlugin
from obs.estimator import IntegrativeObsEstimator, IntegrativeRLearnerObsEstimator, RHCObsEstimator
from utils.dataset_utils import load_dataset_split


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="lalonde", choices=["lalonde", "actg", "jtpa", "sim_obs_iv", "sim_obs_shadow"])
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--target-mode", type=str, required=True, choices=["rct", "obs"])
    parser.add_argument("--method", type=str, required=True, choices=["cvci", "rhc", "integrative", "integrative_rlearner"])
    parser.add_argument(
        "--plugin",
        type=str,
        default="none",
        choices=["none", "shadow", "shadow_source_ep", "iv", "ipsw", "cw", "oracle_source"],
    )

    parser.add_argument("--obs-source", type=str, default="cps", choices=["cps", "psid"])
    parser.add_argument("--x-cols", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--truth-mode", type=str, default="auto", choices=["auto", "none", "proxy_ate", "synthetic_truth"])
    parser.add_argument("--truth-value", type=float, default=None)
    parser.add_argument("--lalonde-path", type=str, default="lalonde.csv")
    parser.add_argument("--data-mode", type=str, default="real", choices=["real", "semi_synthetic"])
    parser.add_argument(
        "--construction-mode",
        type=str,
        default="current",
        choices=["current", "y_dependent_source", "tau_dependent_source"],
    )
    parser.add_argument("--semisynth-effect-mode", type=str, default="linear", choices=["constant", "linear", "nonlinear"])
    parser.add_argument("--semisynth-truth-source", type=str, default="rct", choices=["rct", "pooled"])
    parser.add_argument("--semisynth-noise-mode", type=str, default="groupwise", choices=["shared", "groupwise"])
    parser.add_argument("--semisynth-tau0", type=float, default=1.0)
    parser.add_argument("--semisynth-tau-scale", type=float, default=0.5)
    parser.add_argument("--semisynth-mu0", type=float, default=0.0)
    parser.add_argument("--semisynth-mu-scale", type=float, default=1.0)
    parser.add_argument("--semisynth-g-shift", type=float, default=0.2)
    parser.add_argument("--semisynth-noise-std", type=float, default=1.0)
    parser.add_argument(
        "--semisynth-bias-mode",
        type=str,
        default="none",
        choices=["none", "obs_treatment_bias", "nonlinear_obs_treatment_bias", "localized_obs_treatment_bias"],
    )
    parser.add_argument("--semisynth-bias-scale", type=float, default=0.0)
    parser.add_argument("--semisynth-seed", type=int, default=2024)
    parser.add_argument("--source-alpha-x", type=float, default=0.5)
    parser.add_argument("--source-alpha-t", type=float, default=0.2)
    parser.add_argument("--source-alpha-y", type=float, default=1.0)
    parser.add_argument("--source-alpha-tau", type=float, default=1.0)
    parser.add_argument("--source-rct-frac", type=float, default=0.3)
    parser.add_argument("--correction-strength", type=float, default=1.0)
    parser.add_argument("--source-correction-cv", action="store_true")

    parser.add_argument("--lambda-bin", type=int, default=5)
    parser.add_argument("--k-fold", type=int, default=5)

    parser.add_argument("--iv-relevance-threshold", type=float, default=0.02)
    parser.add_argument("--iv-exclusion-threshold", type=float, default=0.02)
    parser.add_argument("--iv-clip-min", type=float, default=0.05)
    parser.add_argument("--iv-clip-max", type=float, default=20.0)

    parser.add_argument("--ipsw-clip-min", type=float, default=0.05)
    parser.add_argument("--ipsw-clip-max", type=float, default=20.0)

    parser.add_argument("--cw-degree", type=int, default=1)
    parser.add_argument("--cw-interactions", action="store_true")
    parser.add_argument("--cw-max-iter", type=int, default=2000)

    parser.add_argument("--shadow-association-threshold", type=float, default=0.02)
    parser.add_argument("--shadow-residual-independence-threshold", type=float, default=0.02)
    parser.add_argument("--shadow-allow-fallback", action="store_true")
    parser.add_argument("--shadow-source-ep-clip", type=float, default=0.05)
    parser.add_argument("--shadow-mc-samples", type=int, default=2000)
    parser.add_argument("--shadow-direction", type=str, default="auto", choices=["auto", "obs_to_rct", "rct_to_obs"])
    parser.add_argument("--shadow-relevance-group", type=str, default="none", choices=["none", "target", "source"])
    parser.add_argument("--screening-mode", type=str, default="screened", choices=["screened", "all", "topk"])
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--force-candidate-cols", nargs="*", default=None)

    parser.add_argument("--treatment-col", type=str, default=None)
    parser.add_argument("--outcome-col", type=str, default=None)
    parser.add_argument("--site-col", type=str, default=None)
    parser.add_argument("--jtpa-rct-site", type=str, default="Coosa Valley")

    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    return parser.parse_args(argv)


def _normalize_plugin_name(plugin: str) -> str:
    return "iv" if plugin == "selection_iv" else plugin


def _normalize_shadow_relevance_group(arg_value: str) -> Optional[str]:
    value = str(arg_value).lower()
    if value == "none":
        return None
    return value


def _resolve_shadow_direction(arg_direction: str, target_mode: str) -> str:
    direction = str(arg_direction).lower()
    mode = str(target_mode).lower()
    if direction != "auto":
        return direction
    if mode == "rct":
        return "obs_to_rct"
    if mode == "obs":
        return "rct_to_obs"
    raise ValueError(f"Unsupported target_mode for shadow direction auto resolution: {target_mode}")


def _resolve_truth_mode(arg_truth_mode: str, data_mode: str) -> str:
    mode = str(arg_truth_mode).lower()
    data_mode = str(data_mode).lower()
    if mode != "auto":
        return mode
    if data_mode == "semi_synthetic":
        return "synthetic_truth"
    return "proxy_ate"


def _weight_summary(weights: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(weights, dtype=float).reshape(-1)
    return {
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
    }


class WeightShrinkagePlugin(SelectionCorrectionPlugin):
    def __init__(self, base_plugin: SelectionCorrectionPlugin, strength: float):
        super().__init__(name=base_plugin.name)
        self.base_plugin = base_plugin
        self.strength = float(strength)

    def fit(self, df_rct, df_obs, x_cols, a_col, y_col, g_col):
        self.base_plugin.fit(df_rct=df_rct, df_obs=df_obs, x_cols=x_cols, a_col=a_col, y_col=y_col, g_col=g_col)
        self.fitted_ = True
        return self

    def get_rct_weights(self, df_rct: pd.DataFrame) -> Optional[np.ndarray]:
        weights = self.base_plugin.get_rct_weights(df_rct)
        if weights is None:
            return None
        arr = np.asarray(weights, dtype=float).reshape(-1)
        shrunk = 1.0 + self.strength * (arr - 1.0)
        return np.clip(shrunk, 1e-8, None)

    def get_corrected_bias_target(self, df_rct: pd.DataFrame, base_w_hat: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        return self.base_plugin.get_corrected_bias_target(df_rct=df_rct, base_w_hat=base_w_hat)

    def summary(self) -> Dict[str, object]:
        summary = dict(self.base_plugin.summary())
        weights = getattr(self.base_plugin, "weights_", None)
        shrunk = self.get_rct_weights(pd.DataFrame(index=range(len(weights)))) if weights is not None else None
        summary["correction_strength"] = float(self.strength)
        summary["weight_shrinkage"] = "w_gamma=1+gamma*(w-1)"
        if shrunk is not None:
            summary["shrunk_weight_summary"] = _weight_summary(shrunk)
        return summary


class OracleSourcePlugin(SelectionCorrectionPlugin):
    def __init__(self, clip_min: float = 0.05, clip_max: float = 20.0):
        super().__init__(name="oracle_source")
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)

    def fit(self, df_rct, df_obs, x_cols, a_col, y_col, g_col):
        _ = df_obs, x_cols, a_col, y_col, g_col
        if "source_prob_true" not in df_rct.columns:
            raise ValueError("oracle_source requires source_prob_true; use y/tau-dependent semi_synthetic construction.")
        p = np.asarray(df_rct["source_prob_true"], dtype=float).reshape(-1)
        p = np.clip(p, self.clip_min, 1.0 - self.clip_min)
        weights = (1.0 - p) / p
        weights = np.clip(weights, self.clip_min, self.clip_max)
        weights = weights / max(float(np.mean(weights)), 1e-12)
        self.weights_ = weights
        self.fitted_ = True
        self.diagnostics_ = {
            "plugin": self.name,
            "weight_summary": _weight_summary(weights),
            "oracle_source_prob_min": float(np.min(p)),
            "oracle_source_prob_mean": float(np.mean(p)),
            "oracle_source_prob_max": float(np.max(p)),
        }
        return self

    def get_rct_weights(self, df_rct: pd.DataFrame) -> Optional[np.ndarray]:
        _ = df_rct
        return self.weights_ if self.fitted_ else None


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_to_jsonable(v) for v in value.tolist()]
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, pd.Series):
        return [_to_jsonable(v) for v in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return _to_jsonable(value.to_dict(orient="list"))
    return repr(value)


def _naive_gap(df: pd.DataFrame) -> float:
    t = df["T"].to_numpy(dtype=float)
    y = df["Y"].to_numpy(dtype=float)
    if np.sum(t == 1) == 0 or np.sum(t == 0) == 0:
        raise ValueError("Naive gap requires both treated and control groups.")
    return float(y[t == 1].mean() - y[t == 0].mean())


def _truth_and_rmse(
    truth_mode: str,
    effect_pred: np.ndarray,
    eval_df: pd.DataFrame,
    target_mode: str,
    df_rct: pd.DataFrame,
    df_obs: pd.DataFrame,
    truth_value_arg: Optional[float],
    data_mode: str,
) -> Tuple[str, Optional[float], Optional[str], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    effect_pred = np.asarray(effect_pred, dtype=float).reshape(-1)
    rct_naive_gap = _naive_gap(df_rct)
    obs_naive_gap = _naive_gap(df_obs)

    if truth_mode == "none":
        return "none", None, None, None, rct_naive_gap, obs_naive_gap, None, None

    if truth_mode == "proxy_ate":
        if str(target_mode).lower() == "rct":
            proxy_ate, proxy_name = rct_naive_gap, "rct_naive_gap"
        else:
            proxy_ate, proxy_name = obs_naive_gap, "obs_naive_gap"
        proxy_abs_error = float(abs(float(np.mean(effect_pred)) - proxy_ate))
        rmse = proxy_abs_error
        return "proxy_ate", proxy_ate, proxy_name, proxy_ate, rct_naive_gap, obs_naive_gap, proxy_abs_error, rmse

    if truth_mode == "synthetic_truth":
        if "tau_true" in eval_df.columns:
            tau_true = eval_df["tau_true"].to_numpy(dtype=float)
            if tau_true.shape[0] != effect_pred.shape[0]:
                raise ValueError("tau_true length mismatch for synthetic_truth rmse.")
            rmse = float(np.sqrt(np.mean((effect_pred - tau_true) ** 2)))
            return (
                "synthetic_truth",
                float(np.mean(tau_true)),
                None,
                None,
                rct_naive_gap,
                obs_naive_gap,
                None,
                rmse,
            )
        if truth_value_arg is not None:
            rmse = float(np.sqrt((np.mean(effect_pred) - float(truth_value_arg)) ** 2))
            return (
                "synthetic_truth",
                float(truth_value_arg),
                None,
                None,
                rct_naive_gap,
                obs_naive_gap,
                None,
                rmse,
            )
        raise ValueError("truth_mode='synthetic_truth' requires tau_true column or --truth-value.")

    raise ValueError(f"Unsupported truth_mode: {truth_mode}")


def _build_rhc_plugin(
    args: argparse.Namespace,
    x_cols: Sequence[str],
    shadow_direction: str,
    shadow_relevance_group: Optional[str],
) -> SelectionCorrectionPlugin:
    if args.plugin == "none":
        plugin = SelectionCorrectionPlugin(name="none")
    elif args.plugin == "oracle_source":
        plugin = OracleSourcePlugin(clip_min=args.iv_clip_min, clip_max=args.iv_clip_max)
    elif args.plugin == "ipsw":
        plugin = IPSWPlugin(clip_min=args.ipsw_clip_min, clip_max=args.ipsw_clip_max)
    elif args.plugin == "cw":
        plugin = CWPlugin(degree=args.cw_degree, include_interactions=args.cw_interactions, max_iter=args.cw_max_iter)
    elif args.plugin == "iv":
        plugin = SelectionIVPlugin(
            iv_candidate_cols=list(x_cols),
            relevance_threshold=args.iv_relevance_threshold,
            exclusion_threshold=args.iv_exclusion_threshold,
            clip_min=args.iv_clip_min,
            clip_max=args.iv_clip_max,
            screening_mode=args.screening_mode,
            top_k=args.top_k,
            force_candidate_cols=args.force_candidate_cols,
            verbose=False,
        )
    elif args.plugin == "shadow":
        if shadow_direction != "rct_to_obs":
            raise ValueError(
                "OBS-target RHC shadow plugin uses rct_to_obs direction. "
                f"Received shadow_direction={shadow_direction}."
            )
        plugin = ShadowPlugin(
            association_threshold=args.shadow_association_threshold,
            residual_independence_threshold=args.shadow_residual_independence_threshold,
            shadow_mc_samples=args.shadow_mc_samples,
            allow_empty_fallback=args.shadow_allow_fallback,
            shadow_relevance_group=shadow_relevance_group,
            random_state=args.seed,
            screening_mode=args.screening_mode,
            top_k=args.top_k,
            force_candidate_cols=args.force_candidate_cols,
            verbose=False,
        )
    elif args.plugin == "shadow_source_ep":
        if shadow_direction != "rct_to_obs":
            raise ValueError(
                "OBS-target RHC shadow_source_ep plugin uses rct_to_obs direction. "
                f"Received shadow_direction={shadow_direction}."
            )
        plugin = ShadowSourceEPPlugin(
            association_threshold=args.shadow_association_threshold,
            residual_independence_threshold=args.shadow_residual_independence_threshold,
            clip_min=args.shadow_source_ep_clip,
            clip_max=args.iv_clip_max,
            allow_empty_fallback=args.shadow_allow_fallback,
            shadow_relevance_group=shadow_relevance_group,
            random_state=args.seed,
            screening_mode=args.screening_mode,
            top_k=args.top_k,
            force_candidate_cols=args.force_candidate_cols,
            verbose=False,
        )
    else:
        raise ValueError(f"Unsupported plugin for RHC: {args.plugin}")

    if args.correction_strength != 1.0 and args.plugin != "none":
        plugin = WeightShrinkagePlugin(plugin, strength=args.correction_strength)
    return plugin


def _run_cvci(
    args: argparse.Namespace,
    df_rct: pd.DataFrame,
    df_obs: pd.DataFrame,
    data_split_summary: Dict[str, object],
) -> Dict[str, object]:
    from rct.plugins import build_obs_plugin, fit_cvci_with_plugin

    x_cols = list(args.x_cols)
    x_exp = np.concatenate([df_rct[x_cols].to_numpy(dtype=float), df_rct[["T"]].to_numpy(dtype=float), df_rct[["Y"]].to_numpy(dtype=float)], axis=1)
    x_obs = np.concatenate([df_obs[x_cols].to_numpy(dtype=float), df_obs[["T"]].to_numpy(dtype=float), df_obs[["Y"]].to_numpy(dtype=float)], axis=1)

    resolved_shadow_direction = _resolve_shadow_direction(args.shadow_direction, target_mode="rct")
    shadow_relevance_group = _normalize_shadow_relevance_group(args.shadow_relevance_group)

    config = {
        "mode": "linear",
        "lambda_bin": int(args.lambda_bin),
        "k_fold": int(args.k_fold),
        "d_exp": len(x_cols),
        "d_obs": len(x_cols),
        "covariate_names": x_cols,
        "exp_model": "response_func",
        "stratified_kfold": True,
        "random_state": int(args.seed),
        "shadow_association_threshold": float(args.shadow_association_threshold),
        "shadow_residual_independence_threshold": float(args.shadow_residual_independence_threshold),
        "shadow_allow_fallback": bool(args.shadow_allow_fallback),
        "shadow_source_ep_clip": float(args.shadow_source_ep_clip),
        "shadow_mc_samples": int(args.shadow_mc_samples),
        "shadow_direction": resolved_shadow_direction,
        "shadow_relevance_group": shadow_relevance_group,
        "iv_relevance_alpha": float(args.iv_relevance_threshold),
        "iv_exclusion_alpha": float(args.iv_exclusion_threshold),
        "ipsw_clip_min": float(args.ipsw_clip_min),
        "ipsw_clip_max": float(args.ipsw_clip_max),
        "iv_clip_min": float(args.iv_clip_min),
        "iv_clip_max": float(args.iv_clip_max),
        "cw_max_iter": int(args.cw_max_iter),
        "screening_mode": str(args.screening_mode),
        "top_k": None if args.top_k is None else int(args.top_k),
        "force_candidate_cols": None if args.force_candidate_cols is None else [str(c) for c in args.force_candidate_cols],
    }

    plugin_output = build_obs_plugin(x_obs, x_exp, args.plugin, config)
    cvci_result = fit_cvci_with_plugin(x_exp, x_obs, plugin_output, config)
    theta_opt = cvci_result["theta_opt"]

    X_eval = df_rct[x_cols].to_numpy(dtype=float)
    if hasattr(theta_opt, "predict_tau"):
        effect_pred = np.asarray(theta_opt.predict_tau(X_eval), dtype=float).reshape(-1)
    else:
        beta = theta_opt.beta().item() if hasattr(theta_opt.beta(), "item") else float(theta_opt.beta())
        effect_pred = np.repeat(beta, X_eval.shape[0])

    ate_hat = float(np.mean(effect_pred))
    truth_type, truth_value, proxy_name, proxy_ate, rct_naive_gap, obs_naive_gap, proxy_abs_error, rmse = _truth_and_rmse(
        truth_mode=args.truth_mode,
        effect_pred=effect_pred,
        eval_df=df_rct,
        target_mode=args.target_mode,
        df_rct=df_rct,
        df_obs=df_obs,
        truth_value_arg=args.truth_value,
        data_mode=args.data_mode,
    )

    metadata = dict(plugin_output.metadata)
    selected_shadow_cols = metadata.get("selected_shadow_cols")
    selected_iv_cols = metadata.get("selected_iv_cols", metadata.get("selected_iv_names"))
    plugin_summary = {**metadata, "weight_summary": _weight_summary(plugin_output.sample_weights)}
    screening_logs = plugin_summary.get("screening_logs")
    if screening_logs is None and isinstance(plugin_summary.get("screening"), dict):
        screening_logs = plugin_summary["screening"].get("screening_logs")

    return {
        "method": "cvci",
        "plugin_method": str(plugin_output.method),
        "selection_method": str(plugin_output.method),
        "target_mode": "rct",
        "obs_source": args.obs_source,
        "x_cols": x_cols,
        "ate_hat": ate_hat,
        "rmse": rmse,
        "lambda_opt": float(cvci_result["lambda_opt"]),
        "plugin_name": _normalize_plugin_name(args.plugin),
        "plugin_summary": plugin_summary,
        "plugin_summary_json": json.dumps(_to_jsonable(plugin_summary), ensure_ascii=False),
        "screening_logs_json": json.dumps(_to_jsonable(screening_logs), ensure_ascii=False),
        "selected_shadow_cols": selected_shadow_cols,
        "selected_iv_cols": selected_iv_cols,
        "probability_features": plugin_summary.get("probability_features"),
        "pi_shadow_mean": plugin_summary.get("pi_shadow_mean"),
        "pi_shadow_min": plugin_summary.get("pi_shadow_min"),
        "pi_shadow_max": plugin_summary.get("pi_shadow_max"),
        "mean_weight": plugin_summary.get("mean_weight"),
        "n_selected_shadow": int(len(selected_shadow_cols or [])),
        "n_selected_iv": int(len(selected_iv_cols or [])),
        "screening_mode": str(args.screening_mode),
        "top_k": None if args.top_k is None else int(args.top_k),
        "force_candidate_cols": None if args.force_candidate_cols is None else [str(c) for c in args.force_candidate_cols],
        "shadow_direction": resolved_shadow_direction,
        "shadow_relevance_group": shadow_relevance_group,
        "truth_type": truth_type,
        "truth_value": truth_value,
        "proxy_name": proxy_name,
        "proxy_ate": proxy_ate,
        "rct_naive_gap": rct_naive_gap,
        "obs_naive_gap": obs_naive_gap,
        "proxy_abs_error": proxy_abs_error,
        "data_split_summary": data_split_summary,
    }


def _run_obs_method(
    args: argparse.Namespace,
    df_rct: pd.DataFrame,
    df_obs: pd.DataFrame,
    data_split_summary: Dict[str, object],
) -> Dict[str, object]:
    x_cols = list(args.x_cols)
    resolved_shadow_direction = _resolve_shadow_direction(args.shadow_direction, target_mode="obs")
    shadow_relevance_group = _normalize_shadow_relevance_group(args.shadow_relevance_group)
    plugin = _build_rhc_plugin(args, x_cols=x_cols, shadow_direction=resolved_shadow_direction, shadow_relevance_group=shadow_relevance_group)
    source_correction_cv = bool(args.source_correction_cv)
    source_correction_cv_plugins = {"iv", "shadow_source_ep"}
    if source_correction_cv and args.plugin not in source_correction_cv_plugins:
        raise ValueError(
            "--source-correction-cv is only available for modified plugin-corrected integrative methods; "
            "base integrative/integrative_rlearner and other baseline plugins must not use gamma CV."
        )

    if args.method == "rhc":
        estimator = RHCObsEstimator(plugin=plugin, model_type="linear", random_state=args.seed)
    elif args.method == "integrative":
        estimator = IntegrativeObsEstimator(
            plugin=plugin,
            model_type="linear",
            random_state=args.seed,
            source_correction_cv=source_correction_cv,
            source_correction_cv_folds=args.k_fold,
        )
    elif args.method == "integrative_rlearner":
        estimator = IntegrativeRLearnerObsEstimator(
            plugin=plugin,
            model_type="linear",
            random_state=args.seed,
            source_correction_cv=source_correction_cv,
            source_correction_cv_folds=args.k_fold,
        )
    else:
        raise ValueError(f"Unsupported OBS-target method: {args.method}")
    estimator.fit(df_rct=df_rct, df_obs=df_obs, x_cols=x_cols, a_col="T", y_col="Y", g_col="G")

    X_eval = df_obs[x_cols].copy()
    effect_pred = np.asarray(estimator.predict_tau(X_eval), dtype=float).reshape(-1)
    ate_hat = float(np.mean(effect_pred))

    truth_type, truth_value, proxy_name, proxy_ate, rct_naive_gap, obs_naive_gap, proxy_abs_error, rmse = _truth_and_rmse(
        truth_mode=args.truth_mode,
        effect_pred=effect_pred,
        eval_df=df_obs,
        target_mode=args.target_mode,
        df_rct=df_rct,
        df_obs=df_obs,
        truth_value_arg=args.truth_value,
        data_mode=args.data_mode,
    )

    estimator_summary = estimator.summary()
    plugin_summary = estimator_summary.get("plugin_summary", {})
    selected_shadow_cols = plugin_summary.get("selected_shadow_cols")
    selected_iv_cols = plugin_summary.get("selected_iv_cols")
    screening_logs = plugin_summary.get("screening_logs")
    if screening_logs is None and isinstance(plugin_summary.get("screening"), dict):
        screening_logs = plugin_summary["screening"].get("screening_logs")

    return {
        "method": str(args.method),
        "selection_method": str(plugin.name),
        "target_mode": "obs",
        "obs_source": args.obs_source,
        "x_cols": x_cols,
        "original_x_cols": estimator_summary.get("original_x_cols"),
        "effect_x_cols": estimator_summary.get("effect_x_cols"),
        "excluded_iv_cols": estimator_summary.get("excluded_iv_cols"),
        "source_correction_cv": estimator_summary.get("source_correction_cv"),
        "source_correction_grid": estimator_summary.get("source_correction_grid"),
        "selected_source_correction_strength": estimator_summary.get("selected_source_correction_strength"),
        "source_correction_cv_results": estimator_summary.get("source_correction_cv_results"),
        "ate_hat": ate_hat,
        "rmse": rmse,
        "lambda_opt": None,
        "plugin_name": _normalize_plugin_name(args.plugin),
        "plugin_summary": plugin_summary,
        "anchor_summary": estimator_summary.get("anchor_summary"),
        "plugin_summary_json": json.dumps(_to_jsonable(plugin_summary), ensure_ascii=False),
        "screening_logs_json": json.dumps(_to_jsonable(screening_logs), ensure_ascii=False),
        "selected_shadow_cols": selected_shadow_cols,
        "selected_iv_cols": selected_iv_cols,
        "probability_features": plugin_summary.get("probability_features"),
        "pi_shadow_mean": plugin_summary.get("pi_shadow_mean"),
        "pi_shadow_min": plugin_summary.get("pi_shadow_min"),
        "pi_shadow_max": plugin_summary.get("pi_shadow_max"),
        "mean_weight": plugin_summary.get("mean_weight"),
        "n_selected_shadow": int(len(selected_shadow_cols or [])),
        "n_selected_iv": int(len(selected_iv_cols or [])),
        "screening_mode": str(args.screening_mode),
        "top_k": None if args.top_k is None else int(args.top_k),
        "force_candidate_cols": None if args.force_candidate_cols is None else [str(c) for c in args.force_candidate_cols],
        "shadow_direction": resolved_shadow_direction,
        "shadow_relevance_group": shadow_relevance_group,
        "truth_type": truth_type,
        "truth_value": truth_value,
        "proxy_name": proxy_name,
        "proxy_ate": proxy_ate,
        "rct_naive_gap": rct_naive_gap,
        "obs_naive_gap": obs_naive_gap,
        "proxy_abs_error": proxy_abs_error,
        "data_split_summary": data_split_summary,
        "tau0_mean": estimator_summary.get("tau0_mean"),
        "bG_mean": estimator_summary.get("bG_mean"),
        "bT_mean": estimator_summary.get("bT_mean"),
        "raw_rct_tau_mean": estimator_summary.get("raw_rct_tau_mean"),
        "tau0_anchor_mean": estimator_summary.get("tau0_anchor_mean"),
        "obs_biased_tau_mean": estimator_summary.get("obs_biased_tau_mean"),
    }


def _save_outputs(result: Dict[str, object], output_json: Optional[str], output_csv: Optional[str]) -> None:
    json_safe_result = _to_jsonable(result)

    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(json_safe_result, indent=2, ensure_ascii=False), encoding="utf-8")

    if output_csv:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        row = dict(json_safe_result)
        row["x_cols"] = json.dumps(_to_jsonable(result.get("x_cols", [])), ensure_ascii=False)
        row["selected_iv_cols"] = json.dumps(_to_jsonable(result.get("selected_iv_cols")), ensure_ascii=False)
        row["selected_shadow_cols"] = json.dumps(_to_jsonable(result.get("selected_shadow_cols")), ensure_ascii=False)
        row["force_candidate_cols"] = json.dumps(_to_jsonable(result.get("force_candidate_cols")), ensure_ascii=False)
        row["plugin_summary"] = json.dumps(json_safe_result.get("plugin_summary", {}), ensure_ascii=False)
        row["data_split_summary"] = json.dumps(json_safe_result.get("data_split_summary", {}), ensure_ascii=False)
        row["semisynth_config"] = json.dumps(json_safe_result.get("semisynth_config", None), ensure_ascii=False)
        existing = pd.read_csv(output_path) if output_path.exists() else pd.DataFrame()
        updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
        updated.to_csv(output_path, index=False)


def run_experiment(args: argparse.Namespace) -> Dict[str, object]:
    np.random.seed(args.seed)
    if args.construction_mode == "tau_dependent_source" and args.data_mode != "semi_synthetic":
        raise ValueError("tau_dependent_source requires --data-mode semi_synthetic.")

    semisynth_config = None
    if args.data_mode == "semi_synthetic":
        semisynth_config = {
            "effect_mode": str(args.semisynth_effect_mode),
            "truth_source": str(args.semisynth_truth_source),
            "noise_mode": str(args.semisynth_noise_mode),
            "tau0": float(args.semisynth_tau0),
            "tau_scale": float(args.semisynth_tau_scale),
            "mu0": float(args.semisynth_mu0),
            "mu_scale": float(args.semisynth_mu_scale),
            "g_shift": float(args.semisynth_g_shift),
            "noise_std": float(args.semisynth_noise_std),
            "bias_mode": str(args.semisynth_bias_mode),
            "bias_scale": float(args.semisynth_bias_scale),
            "seed": int(args.semisynth_seed),
        }
    source_config = {
        "source_alpha_x": float(args.source_alpha_x),
        "source_alpha_t": float(args.source_alpha_t),
        "source_alpha_y": float(args.source_alpha_y),
        "source_alpha_tau": float(args.source_alpha_tau),
        "source_rct_frac": float(args.source_rct_frac),
    }

    args.truth_mode = _resolve_truth_mode(args.truth_mode, data_mode=args.data_mode)

    if args.target_mode == "rct" and args.method != "cvci":
        raise ValueError("For target_mode='rct', method must be 'cvci'.")
    if args.target_mode == "obs" and args.method not in {"rhc", "integrative", "integrative_rlearner"}:
        raise ValueError("For target_mode='obs', method must be one of rhc/integrative/integrative_rlearner.")

    effective_data_path = args.data_path
    if args.dataset == "lalonde":
        effective_data_path = args.lalonde_path

    df_rct, df_obs, split_summary = load_dataset_split(
        dataset=args.dataset,
        data_path=effective_data_path,
        target_mode=args.target_mode,
        x_cols=args.x_cols,
        seed=args.seed,
        obs_source=args.obs_source,
        lalonde_path=args.lalonde_path,
        data_mode=args.data_mode,
        semisynth_config=semisynth_config,
        construction_mode=args.construction_mode,
        source_config=source_config,
        treatment_col=args.treatment_col,
        outcome_col=args.outcome_col,
        site_col=args.site_col,
        jtpa_rct_site=args.jtpa_rct_site,
    )
    args.x_cols = list(split_summary["x_cols"])

    if args.method == "cvci":
        result = _run_cvci(args, df_rct=df_rct, df_obs=df_obs, data_split_summary=split_summary)
    else:
        result = _run_obs_method(args, df_rct=df_rct, df_obs=df_obs, data_split_summary=split_summary)

    rct_true_ate = float(df_rct["tau_true"].mean()) if "tau_true" in df_rct.columns else None
    obs_true_ate = float(df_obs["tau_true"].mean()) if "tau_true" in df_obs.columns else None
    target_true_ate = rct_true_ate if args.target_mode == "rct" else obs_true_ate

    result["dataset"] = args.dataset
    result["data_path"] = effective_data_path
    result["data_mode"] = args.data_mode
    result["construction_mode"] = args.construction_mode
    result["semisynth_config"] = semisynth_config
    result["source_config"] = source_config
    result["source_alpha_y"] = float(args.source_alpha_y)
    result["source_alpha_tau"] = float(args.source_alpha_tau)
    result["actual_rct_frac"] = split_summary.get("actual_rct_frac")
    result["corr_G_Y"] = split_summary.get("corr_G_Y")
    result["corr_G_tau_true"] = split_summary.get("corr_G_tau_true")
    result["target_true_ate"] = target_true_ate
    result["rct_true_ate"] = rct_true_ate
    result["obs_true_ate"] = obs_true_ate

    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    result = run_experiment(args)
    _save_outputs(result, output_json=args.output_json, output_csv=args.output_csv)
    print(json.dumps(_to_jsonable(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
