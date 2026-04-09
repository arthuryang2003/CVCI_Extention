"""
Run OBS-target unified framework on LaLonde data.

Examples:
    python experiments/lalonde_obs_target.py --method base
    python experiments/lalonde_obs_target.py --method ipsw --obs-source cps
    python experiments/lalonde_obs_target.py --method cw --obs-source psid
    python experiments/lalonde_obs_target.py --method selection_iv --obs-source cps --iv-candidate-cols re74 re75
    python experiments/lalonde_obs_target.py --method shadow --obs-source psid --shadow-candidate-cols re74 re75
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Sequence

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from obs import (
    CWPlugin,
    IPSWPlugin,
    ObsTargetBaseEstimator,
    SelectionCorrectionPlugin,
    SelectionIVPlugin,
    ShadowPlugin,
    load_lalonde_obs_target_data,
)


DEFAULT_VARIABLES = ["re75"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="base", choices=["base", "ipsw", "cw", "selection_iv", "shadow", "iv"])
    parser.add_argument("--obs-source", type=str, default="cps", choices=["cps", "psid"])
    parser.add_argument("--model-type", type=str, default="linear", choices=["linear"])
    parser.add_argument("--variables", nargs="*", type=str, default=None)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--lalonde-path", type=str, default="lalonde.csv")

    parser.add_argument("--ipsw-clip-min", type=float, default=0.05)
    parser.add_argument("--ipsw-clip-max", type=float, default=20.0)

    parser.add_argument("--cw-degree", type=int, default=1)
    parser.add_argument("--cw-interactions", action="store_true")
    parser.add_argument("--cw-max-iter", type=int, default=2000)

    parser.add_argument("--iv-candidate-cols", nargs="*", default=None)
    parser.add_argument("--x-cols-for-iv-screen", nargs="*", default=None)
    parser.add_argument("--use-treatment-in-iv-screen", action="store_true")
    parser.add_argument("--iv-relevance-threshold", type=float, default=0.02)
    parser.add_argument("--iv-exclusion-threshold", type=float, default=0.02)
    parser.add_argument("--iv-clip-min", type=float, default=0.05)
    parser.add_argument("--iv-clip-max", type=float, default=20.0)

    parser.add_argument("--shadow-candidate-cols", nargs="*", default=None)
    parser.add_argument("--x-cols-for-shadow-screen", nargs="*", default=None)
    parser.add_argument("--use-treatment-in-shadow-screen", action="store_true")
    parser.add_argument("--shadow-association-threshold", type=float, default=0.02)
    parser.add_argument("--shadow-residual-independence-threshold", type=float, default=0.02)
    parser.add_argument("--shadow-or-model-type", type=str, default="exp_linear", choices=["exp_linear", "exp_y_only"])
    parser.add_argument("--shadow-clip-min", type=float, default=0.05)
    parser.add_argument("--shadow-clip-max", type=float, default=20.0)
    return parser.parse_args()


def build_plugin(method: str, args, x_cols: Sequence[str]) -> SelectionCorrectionPlugin:
    normalized_method = "selection_iv" if method == "iv" else method

    if normalized_method == "base":
        return SelectionCorrectionPlugin(name="base")
    if normalized_method == "ipsw":
        return IPSWPlugin(clip_min=args.ipsw_clip_min, clip_max=args.ipsw_clip_max)
    if normalized_method == "cw":
        return CWPlugin(
            degree=args.cw_degree,
            include_interactions=args.cw_interactions,
            max_iter=args.cw_max_iter,
        )
    if normalized_method == "selection_iv":
        iv_candidates = args.iv_candidate_cols if args.iv_candidate_cols is not None else list(x_cols)
        return SelectionIVPlugin(
            iv_candidate_cols=iv_candidates,
            x_cols_for_iv_screen=args.x_cols_for_iv_screen,
            use_treatment_in_screen=args.use_treatment_in_iv_screen,
            relevance_threshold=args.iv_relevance_threshold,
            exclusion_threshold=args.iv_exclusion_threshold,
            clip_min=args.iv_clip_min,
            clip_max=args.iv_clip_max,
            verbose=True,
        )
    if normalized_method == "shadow":
        shadow_candidates = args.shadow_candidate_cols if args.shadow_candidate_cols is not None else list(x_cols)
        return ShadowPlugin(
            shadow_candidate_cols=shadow_candidates,
            x_cols_for_shadow_screen=args.x_cols_for_shadow_screen,
            use_treatment_in_screen=args.use_treatment_in_shadow_screen,
            association_threshold=args.shadow_association_threshold,
            residual_independence_threshold=args.shadow_residual_independence_threshold,
            or_model_type=args.shadow_or_model_type,
            clip_min=args.shadow_clip_min,
            clip_max=args.shadow_clip_max,
            verbose=True,
        )
    raise ValueError(f"Unsupported method: {method}")


def main():
    args = parse_args()
    np.random.seed(args.seed)

    method_name = "selection_iv" if args.method == "iv" else args.method
    x_cols = DEFAULT_VARIABLES if args.variables is None else list(args.variables)
    extra_feature_cols = []
    if args.iv_candidate_cols:
        extra_feature_cols.extend(list(args.iv_candidate_cols))
    if args.shadow_candidate_cols:
        extra_feature_cols.extend(list(args.shadow_candidate_cols))
    extra_feature_cols = list(dict.fromkeys(extra_feature_cols))

    bundle = load_lalonde_obs_target_data(
        lalonde_path=args.lalonde_path,
        obs_source=args.obs_source,
        x_cols=x_cols,
        extra_feature_cols=extra_feature_cols,
    )
    plugin = build_plugin(method_name, args, x_cols=x_cols)
    estimator = ObsTargetBaseEstimator(plugin=plugin, model_type=args.model_type, random_state=args.seed)

    start = time.perf_counter()
    estimator.fit(bundle.df_rct, bundle.df_obs, x_cols=x_cols, a_col="A", y_col="Y", g_col="G")
    elapsed_sec = time.perf_counter() - start

    X_obs = bundle.df_obs[x_cols].to_numpy(dtype=float)
    tau_hat_obs = estimator.predict_tau(X_obs)
    ate_hat_obs = estimator.estimate_ate(X_obs)
    plugin_summary = estimator.summary().get("plugin_summary", {})

    result_payload = {
        "settings": {
            "method": method_name,
            "obs_source": args.obs_source,
            "model_type": args.model_type,
            "seed": args.seed,
            "variables": x_cols,
        },
        "data": bundle.metadata,
        "results": {
            "method_name": method_name,
            "obs_target_ate_estimate": float(ate_hat_obs),
            "eta_model_train_size": int(estimator.eta_train_size_),
            "runtime_seconds": float(elapsed_sec),
            "tau_hat_examples_first5": tau_hat_obs[:5].tolist(),
        },
        "plugin_info": {
            "selected_iv_cols": plugin_summary.get("selected_iv_cols"),
            "selected_shadow_cols": plugin_summary.get("selected_shadow_cols"),
            "weight_summary": plugin_summary.get("weight_summary"),
            "screening": plugin_summary.get("screening"),
            "gamma0": plugin_summary.get("gamma0"),
            "gamma": plugin_summary.get("gamma"),
            "full": plugin_summary,
        },
        "model_info": {
            "w_model": estimator.summary().get("w_model", {}),
            "eta_model": estimator.summary().get("eta_model", {}),
        },
    }

    print(json.dumps(result_payload, indent=2))


if __name__ == "__main__":
    main()
