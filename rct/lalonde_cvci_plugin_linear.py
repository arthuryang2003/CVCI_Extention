"""
Run the CVCI plugin framework on the LaLonde dataset.

Default setting:
- RCT: NSW treated + NSW control
- OBS: CPS controls together with NSW treated
- Model: linear
- Covariates: re75
- Lambda selection: 5-fold cross-validation

Examples:
    python lalonde_cvci_plugin_linear.py
    python lalonde_cvci_plugin_linear.py --method ipsw
    python lalonde_cvci_plugin_linear.py --method cw --variables re75 age education
"""

import argparse
import json
import os
from datetime import date

import numpy as np
import pandas as pd

from rct.causal_sim import compute_exp_minmizer, lalonde_get_data
from rct.cvci_plugins import build_obs_plugin, fit_base_cvci, fit_cvci_with_plugin


DEFAULT_VARIABLES = ["re75"]


def variable_set_label(variables):
    return "re75" if list(variables) == ["re75"] else "all"


def predict_linear(theta_opt, x_data, d):
    theta = theta_opt.theta_model.detach().cpu().numpy().reshape(-1)
    z = x_data[:, :d]
    a = x_data[:, d]
    return theta[0] * a + z @ theta[1:-1] + theta[-1]


def compute_rmse_summary(theta_opt, x_exp, x_obs, d, obs_target=None):
    pred_exp = predict_linear(theta_opt, x_exp, d)
    pred_obs = predict_linear(theta_opt, x_obs, d)
    y_exp = x_exp[:, -1]
    y_obs = x_obs[:, -1]

    summary = {
        "exp_actual_rmse": float(np.sqrt(np.mean((pred_exp - y_exp) ** 2))),
        "obs_actual_rmse": float(np.sqrt(np.mean((pred_obs - y_obs) ** 2))),
        "all_actual_rmse": float(np.sqrt(np.mean((np.concatenate((pred_exp, pred_obs)) - np.concatenate((y_exp, y_obs))) ** 2))),
    }
    if obs_target is not None:
        obs_target = np.asarray(obs_target, dtype=float).reshape(-1)
        summary["obs_target_rmse"] = float(np.sqrt(np.mean((pred_obs - obs_target) ** 2)))
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="none", choices=["none", "ipsw", "cw", "iv", "shadow"])
    parser.add_argument("--group", type=str, default="cps", choices=["cps", "cps2", "cps3", "psid", "psid2", "psid3"])
    parser.add_argument("--variables", nargs="*", type=str, default=None)
    parser.add_argument("--ground-truth-group", type=str, default="nsw", choices=["none", "nsw"])
    parser.add_argument("--lambda-bin", type=int, default=5)
    parser.add_argument("--k-fold", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=2024)
    return parser.parse_args()


def compute_ground_truth_payload(df: pd.DataFrame, variables, ground_truth_group: str):
    if ground_truth_group == "none":
        return None
    if ground_truth_group != "nsw":
        raise ValueError(f"Unsupported ground truth group: {ground_truth_group}")

    df_gt = df[df["group"].isin(["control", "treated"])].copy()
    x_gt, _ = lalonde_get_data(df_gt, "control", variables)
    treatment = x_gt[:, len(variables)]
    outcome = x_gt[:, -1]
    treated_outcome = outcome[treatment == 1]
    control_outcome = outcome[treatment == 0]

    response_func_estimate = float(
        compute_exp_minmizer(
            x_gt,
            mode="linear",
            exp_model="response_func",
            d_exp=len(variables),
        )
    )

    gt_theta = np.zeros(len(variables) + 2, dtype=float)
    gt_theta[0] = response_func_estimate
    pred_gt = gt_theta[0] * treatment + x_gt[:, : len(variables)] @ gt_theta[1:-1] + gt_theta[-1]
    gt_exp_rmse = float(np.sqrt(np.mean((pred_gt - outcome) ** 2)))

    return {
        "group": ground_truth_group,
        "variables": list(variables),
        "n_total": int(x_gt.shape[0]),
        "n_treated": int(np.sum(treatment == 1)),
        "n_control": int(np.sum(treatment == 0)),
        "treated_mean_outcome": float(np.mean(treated_outcome)),
        "control_mean_outcome": float(np.mean(control_outcome)),
        "mean_diff_estimate": float(np.mean(treated_outcome) - np.mean(control_outcome)),
        "response_func_estimate": response_func_estimate,
        "exp_rmse": gt_exp_rmse,
    }


def main():
    args = parse_args()
    df = pd.read_csv("lalonde.csv")
    df["age2"] = df["age"] ** 2

    variables = DEFAULT_VARIABLES if args.variables is None else args.variables

    x_exp, x_obs = lalonde_get_data(df, args.group, variables)
    ground_truth_payload = compute_ground_truth_payload(df, variables, args.ground_truth_group)
    config = {
        "mode": "linear",
        "lambda_bin": args.lambda_bin,
        "k_fold": args.k_fold,
        "d_exp": len(variables),
        "d_obs": len(variables),
        "covariate_names": variables,
        "exp_model": "response_func",
        "stratified_kfold": True,
        "random_state": args.random_state,
    }

    base_result = fit_base_cvci(x_exp, x_obs, config)
    plugin_output = build_obs_plugin(x_obs, x_exp, args.method, config)
    plugin_result = fit_cvci_with_plugin(x_exp, x_obs, plugin_output, config)
    base_rmse = compute_rmse_summary(base_result["theta_opt"], x_exp, x_obs, len(variables))
    plugin_rmse = compute_rmse_summary(
        plugin_result["theta_opt"],
        x_exp,
        x_obs,
        len(variables),
        obs_target=plugin_output.pseudo_outcome,
    )

    result_payload = {
        "settings": {
            "method": args.method,
            "group": args.group,
            "variables": variables,
            "ground_truth_group": args.ground_truth_group,
            "lambda_bin": args.lambda_bin,
            "k_fold": args.k_fold,
            "random_state": args.random_state,
        },
        "ground_truth": ground_truth_payload,
        "base_lambda_opt": float(base_result["lambda_opt"]),
        "base_estimate": float(base_result["theta_opt"].beta().item()),
        "base_rmse": base_rmse,
        "plugin_lambda_opt": float(plugin_result["lambda_opt"]),
        "plugin_estimate": float(plugin_result["estimate"]),
        "plugin_rmse": plugin_rmse,
        "obs_weight_summary": {
            "min": float(np.min(plugin_output.sample_weights)),
            "mean": float(np.mean(plugin_output.sample_weights)),
            "max": float(np.max(plugin_output.sample_weights)),
        },
        "plugin_metadata": {
            "screening_summary": plugin_output.metadata.get("screening_summary"),
            "selected_iv_names": plugin_output.metadata.get("selected_iv_names"),
            "selected_iv_indices": plugin_output.metadata.get("selected_iv_indices"),
            "selected_shadow_name": plugin_output.metadata.get("selected_shadow_name"),
            "selected_shadow_index": plugin_output.metadata.get("selected_shadow_index"),
            "selection_feature_names": plugin_output.metadata.get("selection_feature_names"),
            "shadow_feature_names": plugin_output.metadata.get("shadow_feature_names"),
            "pseudo_outcome_mean": plugin_output.metadata.get("pseudo_outcome_mean"),
            "pseudo_outcome_std": plugin_output.metadata.get("pseudo_outcome_std"),
        },
    }

    today = str(date.today())
    output_dir = f"./{today}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = f"lalonde_cvci_plugin_linear_{args.group}_{variable_set_label(variables)}_{args.method}_rs{args.random_state}"
    with open(output_dir + filename + ".json", "w", encoding="utf-8") as file:
        json.dump(result_payload, file, indent=2)

    print(json.dumps(result_payload, indent=2))
    print("saved file", output_dir + filename + ".json")


if __name__ == "__main__":
    main()
