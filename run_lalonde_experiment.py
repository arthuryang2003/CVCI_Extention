"""Unified Lalonde experiment runner for RCT-target (CVCI) and OBS-target (RHC)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from methods import CWPlugin, IPSWPlugin, SelectionCorrectionPlugin, SelectionIVPlugin, ShadowPlugin
from obs.estimator import ObsTargetBaseEstimator
from utils.lalonde_utils import get_lalonde_default_covariates, load_lalonde_csv, load_lalonde_split


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-mode", type=str, required=True, choices=["rct", "obs"])
    parser.add_argument("--method", type=str, required=True, choices=["cvci", "rhc"])
    parser.add_argument("--plugin", type=str, default="none", choices=["none", "shadow", "iv", "ipsw", "cw"])
    parser.add_argument("--obs-source", type=str, default="cps", choices=["cps", "psid"])
    parser.add_argument("--x-cols", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--truth-mode", type=str, default="auto", choices=["auto", "none", "proxy_ate", "synthetic_truth"])
    parser.add_argument("--truth-value", type=float, default=None)
    parser.add_argument("--lalonde-path", type=str, default="lalonde.csv")
    parser.add_argument("--data-mode", type=str, default="real", choices=["real", "semi_synthetic"])
    parser.add_argument("--semisynth-effect-mode", type=str, default="constant", choices=["constant", "linear"])
    parser.add_argument("--semisynth-truth-source", type=str, default="rct", choices=["rct", "pooled"])
    parser.add_argument("--semisynth-noise-mode", type=str, default="groupwise", choices=["shared", "groupwise"])
    parser.add_argument("--semisynth-seed", type=int, default=2024)

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
    parser.add_argument("--shadow-mc-samples", type=int, default=2000)
    parser.add_argument(
        "--shadow-direction",
        type=str,
        default="auto",
        choices=["auto", "obs_to_rct", "rct_to_obs"],
    )
    parser.add_argument(
        "--shadow-relevance-group",
        type=str,
        default="none",
        choices=["none", "target", "source"],
    )
    parser.add_argument("--screening-mode", type=str, default="screened", choices=["screened", "all", "topk"])
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--force-candidate-cols", nargs="*", default=None)

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


def _to_jsonable(value: Any) -> Any:
    """Recursively convert experiment outputs into JSON-serializable structures."""
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


def _ate_proxy_from_rct(df_rct: pd.DataFrame) -> float:
    t = df_rct["T"].to_numpy(dtype=float)
    y = df_rct["Y"].to_numpy(dtype=float)
    if np.sum(t == 1) == 0 or np.sum(t == 0) == 0:
        raise ValueError("RCT data must contain both treatment arms for proxy_ate.")
    return float(y[t == 1].mean() - y[t == 0].mean())


def _truth_and_rmse(
    truth_mode: str,
    effect_pred: np.ndarray,
    eval_df: pd.DataFrame,
    df_rct: pd.DataFrame,
    truth_value_arg: Optional[float],
) -> Tuple[str, Optional[float], Optional[float]]:
    effect_pred = np.asarray(effect_pred, dtype=float).reshape(-1)
    if truth_mode == "none":
        return "none", None, None

    if truth_mode == "proxy_ate":
        truth = _ate_proxy_from_rct(df_rct)
        rmse = float(np.sqrt((np.mean(effect_pred) - truth) ** 2))
        return "proxy_ate", float(truth), rmse

    if truth_mode == "synthetic_truth":
        if "tau_true" in eval_df.columns:
            tau_true = eval_df["tau_true"].to_numpy(dtype=float)
            if tau_true.shape[0] != effect_pred.shape[0]:
                raise ValueError("tau_true length mismatch for synthetic_truth rmse.")
            rmse = float(np.sqrt(np.mean((effect_pred - tau_true) ** 2)))
            return "synthetic_truth", float(np.mean(tau_true)), rmse
        if truth_value_arg is not None:
            rmse = float(np.sqrt((np.mean(effect_pred) - float(truth_value_arg)) ** 2))
            return "synthetic_truth", float(truth_value_arg), rmse
        raise ValueError("truth_mode='synthetic_truth' requires tau_true column or --truth-value.")

    raise ValueError(f"Unsupported truth_mode: {truth_mode}")


def _build_rhc_plugin(
    args: argparse.Namespace,
    x_cols: Sequence[str],
    shadow_direction: str,
    shadow_relevance_group: Optional[str],
) -> SelectionCorrectionPlugin:
    if args.plugin == "none":
        return SelectionCorrectionPlugin(name="none")
    if args.plugin == "ipsw":
        return IPSWPlugin(clip_min=args.ipsw_clip_min, clip_max=args.ipsw_clip_max)
    if args.plugin == "cw":
        return CWPlugin(degree=args.cw_degree, include_interactions=args.cw_interactions, max_iter=args.cw_max_iter)
    if args.plugin == "iv":
        return SelectionIVPlugin(
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
    if args.plugin == "shadow":
        if shadow_direction != "rct_to_obs":
            raise ValueError(
                "OBS-target RHC shadow plugin uses rct_to_obs direction. "
                f"Received shadow_direction={shadow_direction}."
            )
        return ShadowPlugin(
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
    raise ValueError(f"Unsupported plugin for RHC: {args.plugin}")


def _run_cvci(
    args: argparse.Namespace,
    df_rct: pd.DataFrame,
    df_obs: pd.DataFrame,
    data_split_summary: Dict[str, object],
) -> Dict[str, object]:
    from rct.plugins import build_obs_plugin, fit_cvci_with_plugin

    x_cols = list(args.x_cols)
    x_exp = np.concatenate(
        [
            df_rct[x_cols].to_numpy(dtype=float),
            df_rct[["T"]].to_numpy(dtype=float),
            df_rct[["Y"]].to_numpy(dtype=float),
        ],
        axis=1,
    )
    x_obs = np.concatenate(
        [
            df_obs[x_cols].to_numpy(dtype=float),
            df_obs[["T"]].to_numpy(dtype=float),
            df_obs[["Y"]].to_numpy(dtype=float),
        ],
        axis=1,
    )

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
    truth_type, truth_value, rmse = _truth_and_rmse(
        truth_mode=args.truth_mode,
        effect_pred=effect_pred,
        eval_df=df_rct,
        df_rct=df_rct,
        truth_value_arg=args.truth_value,
    )

    metadata = dict(plugin_output.metadata)
    selected_shadow_cols = metadata.get("selected_shadow_cols")
    selected_iv_cols = metadata.get("selected_iv_cols", metadata.get("selected_iv_names"))
    plugin_summary = {
        **metadata,
        "weight_summary": _weight_summary(plugin_output.sample_weights),
    }
    screening_logs = plugin_summary.get("screening_logs")
    if screening_logs is None and isinstance(plugin_summary.get("screening"), dict):
        screening_logs = plugin_summary["screening"].get("screening_logs")

    return {
        "method": "cvci",
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
        "n_selected_shadow": int(len(selected_shadow_cols or [])),
        "n_selected_iv": int(len(selected_iv_cols or [])),
        "screening_mode": str(args.screening_mode),
        "top_k": None if args.top_k is None else int(args.top_k),
        "force_candidate_cols": None if args.force_candidate_cols is None else [str(c) for c in args.force_candidate_cols],
        "shadow_direction": resolved_shadow_direction,
        "shadow_relevance_group": shadow_relevance_group,
        "truth_type": truth_type,
        "truth_value": truth_value,
        "data_split_summary": data_split_summary,
    }


def _run_rhc(
    args: argparse.Namespace,
    df_rct: pd.DataFrame,
    df_obs: pd.DataFrame,
    data_split_summary: Dict[str, object],
) -> Dict[str, object]:
    x_cols = list(args.x_cols)
    resolved_shadow_direction = _resolve_shadow_direction(args.shadow_direction, target_mode="obs")
    shadow_relevance_group = _normalize_shadow_relevance_group(args.shadow_relevance_group)
    plugin = _build_rhc_plugin(
        args,
        x_cols=x_cols,
        shadow_direction=resolved_shadow_direction,
        shadow_relevance_group=shadow_relevance_group,
    )

    estimator = ObsTargetBaseEstimator(plugin=plugin, model_type="linear", random_state=args.seed)
    estimator.fit(df_rct=df_rct, df_obs=df_obs, x_cols=x_cols, a_col="T", y_col="Y", g_col="G")

    X_eval = df_obs[x_cols].to_numpy(dtype=float)
    effect_pred = np.asarray(estimator.predict_tau(X_eval), dtype=float).reshape(-1)
    ate_hat = float(np.mean(effect_pred))

    truth_type, truth_value, rmse = _truth_and_rmse(
        truth_mode=args.truth_mode,
        effect_pred=effect_pred,
        eval_df=df_obs,
        df_rct=df_rct,
        truth_value_arg=args.truth_value,
    )

    plugin_summary = estimator.summary().get("plugin_summary", {})
    selected_shadow_cols = plugin_summary.get("selected_shadow_cols")
    selected_iv_cols = plugin_summary.get("selected_iv_cols")
    screening_logs = plugin_summary.get("screening_logs")
    if screening_logs is None and isinstance(plugin_summary.get("screening"), dict):
        screening_logs = plugin_summary["screening"].get("screening_logs")

    return {
        "method": "rhc",
        "target_mode": "obs",
        "obs_source": args.obs_source,
        "x_cols": x_cols,
        "ate_hat": ate_hat,
        "rmse": rmse,
        "lambda_opt": None,
        "plugin_name": _normalize_plugin_name(args.plugin),
        "plugin_summary": plugin_summary,
        "plugin_summary_json": json.dumps(_to_jsonable(plugin_summary), ensure_ascii=False),
        "screening_logs_json": json.dumps(_to_jsonable(screening_logs), ensure_ascii=False),
        "selected_shadow_cols": selected_shadow_cols,
        "selected_iv_cols": selected_iv_cols,
        "n_selected_shadow": int(len(selected_shadow_cols or [])),
        "n_selected_iv": int(len(selected_iv_cols or [])),
        "screening_mode": str(args.screening_mode),
        "top_k": None if args.top_k is None else int(args.top_k),
        "force_candidate_cols": None if args.force_candidate_cols is None else [str(c) for c in args.force_candidate_cols],
        "shadow_direction": resolved_shadow_direction,
        "shadow_relevance_group": shadow_relevance_group,
        "truth_type": truth_type,
        "truth_value": truth_value,
        "data_split_summary": data_split_summary,
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
        row["x_cols"] = ",".join(result.get("x_cols", []))
        row["selected_iv_cols"] = json.dumps(_to_jsonable(result.get("selected_iv_cols")), ensure_ascii=False)
        row["selected_shadow_cols"] = json.dumps(_to_jsonable(result.get("selected_shadow_cols")), ensure_ascii=False)
        row["force_candidate_cols"] = json.dumps(_to_jsonable(result.get("force_candidate_cols")), ensure_ascii=False)
        row["plugin_summary"] = json.dumps(json_safe_result.get("plugin_summary", {}), ensure_ascii=False)
        row["data_split_summary"] = json.dumps(json_safe_result.get("data_split_summary", {}), ensure_ascii=False)
        existing = pd.read_csv(output_path) if output_path.exists() else pd.DataFrame()
        updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
        updated.to_csv(output_path, index=False)


def run_experiment(args: argparse.Namespace) -> Dict[str, object]:
    np.random.seed(args.seed)
    semisynth_config = None
    if args.data_mode == "semi_synthetic":
        semisynth_config = {
            "effect_mode": str(args.semisynth_effect_mode),
            "truth_source": str(args.semisynth_truth_source),
            "noise_mode": str(args.semisynth_noise_mode),
            "seed": int(args.semisynth_seed),
        }

    args.truth_mode = _resolve_truth_mode(args.truth_mode, data_mode=args.data_mode)
    if args.x_cols is None:
        raw_df = load_lalonde_csv(args.lalonde_path)
        args.x_cols = get_lalonde_default_covariates(raw_df)

    if args.target_mode == "rct" and args.method != "cvci":
        raise ValueError("For target_mode='rct', method must be 'cvci'.")
    if args.target_mode == "obs" and args.method != "rhc":
        raise ValueError("For target_mode='obs', method must be 'rhc'.")

    df_rct, df_obs, split_summary = load_lalonde_split(
        target_mode=args.target_mode,
        obs_source=args.obs_source,
        x_cols=args.x_cols,
        lalonde_path=args.lalonde_path,
        data_mode=args.data_mode,
        semisynth_config=semisynth_config,
    )

    if args.method == "cvci":
        result = _run_cvci(args, df_rct=df_rct, df_obs=df_obs, data_split_summary=split_summary)
    else:
        result = _run_rhc(args, df_rct=df_rct, df_obs=df_obs, data_split_summary=split_summary)

    rct_true_ate = float(df_rct["tau_true"].mean()) if "tau_true" in df_rct.columns else None
    obs_true_ate = float(df_obs["tau_true"].mean()) if "tau_true" in df_obs.columns else None
    target_true_ate = rct_true_ate if args.target_mode == "rct" else obs_true_ate
    result["data_mode"] = args.data_mode
    result["semisynth_config"] = semisynth_config
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
