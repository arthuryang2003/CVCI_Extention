import csv
import json
import os
import subprocess
import sys
from datetime import date

import numpy as np


PYTHON_EXE = sys.executable
METHODS = ["none", "ipsw", "cw", "iv", "shadow"]
GROUPS = ["cps", "psid"]
VARIABLE_SETS = [
    ("re75", ["re75"]),
    ("all", ["age", "age2", "education", "black", "hispanic", "married", "nodegree", "re74", "re75"]),
]
DEFAULT_RANDOM_STATES = [2024, 2025, 2026]


def _extract_json_from_stdout(stdout: str):
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Could not locate JSON payload in subprocess output.")
    return json.loads(stdout[start : end + 1])


def _run_one(method: str, group: str, variables, random_state: int):
    cmd = [
        PYTHON_EXE,
        "-m",
        "rct.experiments.lalonde_cvci_plugin_linear",
        "--method",
        method,
        "--group",
        group,
        "--random-state",
        str(random_state),
        "--variables",
        *variables,
    ]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.getcwd() if not existing_pythonpath else f"{os.getcwd()}:{existing_pythonpath}"
    completed = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
    return _extract_json_from_stdout(completed.stdout)


def _mean_var(values):
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), float(np.var(arr))


def main():
    rows = []
    detailed = []
    today = str(date.today())

    for group in GROUPS:
        for variable_label, variables in VARIABLE_SETS:
            for method in METHODS:
                run_payloads = []
                for random_state in DEFAULT_RANDOM_STATES:
                    payload = _run_one(method, group, variables, random_state)
                    run_payloads.append(payload)

                base_estimates = [item["base_estimate"] for item in run_payloads]
                plugin_estimates = [item["plugin_estimate"] for item in run_payloads]
                base_lambdas = [item["base_lambda_opt"] for item in run_payloads]
                plugin_lambdas = [item["plugin_lambda_opt"] for item in run_payloads]
                gt_response = [item["ground_truth"]["response_func_estimate"] for item in run_payloads]
                gt_mean_diff = [item["ground_truth"]["mean_diff_estimate"] for item in run_payloads]
                base_exp_rmse = [item["base_rmse"]["exp_actual_rmse"] for item in run_payloads]
                base_obs_rmse = [item["base_rmse"]["obs_actual_rmse"] for item in run_payloads]
                base_all_rmse = [item["base_rmse"]["all_actual_rmse"] for item in run_payloads]
                plugin_exp_rmse = [item["plugin_rmse"]["exp_actual_rmse"] for item in run_payloads]
                plugin_obs_rmse = [item["plugin_rmse"]["obs_actual_rmse"] for item in run_payloads]
                plugin_all_rmse = [item["plugin_rmse"]["all_actual_rmse"] for item in run_payloads]
                plugin_obs_target_rmse = [
                    item["plugin_rmse"].get("obs_target_rmse") for item in run_payloads if item["plugin_rmse"].get("obs_target_rmse") is not None
                ]

                base_estimate_mean, base_estimate_var = _mean_var(base_estimates)
                plugin_estimate_mean, plugin_estimate_var = _mean_var(plugin_estimates)
                base_lambda_mean, base_lambda_var = _mean_var(base_lambdas)
                plugin_lambda_mean, plugin_lambda_var = _mean_var(plugin_lambdas)
                gt_response_mean, gt_response_var = _mean_var(gt_response)
                gt_mean_diff_mean, gt_mean_diff_var = _mean_var(gt_mean_diff)
                base_exp_rmse_mean, base_exp_rmse_var = _mean_var(base_exp_rmse)
                base_obs_rmse_mean, base_obs_rmse_var = _mean_var(base_obs_rmse)
                base_all_rmse_mean, base_all_rmse_var = _mean_var(base_all_rmse)
                plugin_exp_rmse_mean, plugin_exp_rmse_var = _mean_var(plugin_exp_rmse)
                plugin_obs_rmse_mean, plugin_obs_rmse_var = _mean_var(plugin_obs_rmse)
                plugin_all_rmse_mean, plugin_all_rmse_var = _mean_var(plugin_all_rmse)

                error_vs_gt_response = [abs(pe - gt) for pe, gt in zip(plugin_estimates, gt_response)]
                abs_err_mean, abs_err_var = _mean_var(error_vs_gt_response)

                row = {
                    "group": group,
                    "variables": variable_label,
                    "method": method,
                    "n_runs": len(DEFAULT_RANDOM_STATES),
                    "random_states": ",".join(str(x) for x in DEFAULT_RANDOM_STATES),
                    "base_estimate_mean": base_estimate_mean,
                    "base_estimate_var": base_estimate_var,
                    "plugin_estimate_mean": plugin_estimate_mean,
                    "plugin_estimate_var": plugin_estimate_var,
                    "base_lambda_mean": base_lambda_mean,
                    "base_lambda_var": base_lambda_var,
                    "plugin_lambda_mean": plugin_lambda_mean,
                    "plugin_lambda_var": plugin_lambda_var,
                    "gt_response_func_mean": gt_response_mean,
                    "gt_response_func_var": gt_response_var,
                    "gt_mean_diff_mean": gt_mean_diff_mean,
                    "gt_mean_diff_var": gt_mean_diff_var,
                    "base_exp_actual_rmse_mean": base_exp_rmse_mean,
                    "base_exp_actual_rmse_var": base_exp_rmse_var,
                    "base_obs_actual_rmse_mean": base_obs_rmse_mean,
                    "base_obs_actual_rmse_var": base_obs_rmse_var,
                    "base_all_actual_rmse_mean": base_all_rmse_mean,
                    "base_all_actual_rmse_var": base_all_rmse_var,
                    "plugin_exp_actual_rmse_mean": plugin_exp_rmse_mean,
                    "plugin_exp_actual_rmse_var": plugin_exp_rmse_var,
                    "plugin_obs_actual_rmse_mean": plugin_obs_rmse_mean,
                    "plugin_obs_actual_rmse_var": plugin_obs_rmse_var,
                    "plugin_all_actual_rmse_mean": plugin_all_rmse_mean,
                    "plugin_all_actual_rmse_var": plugin_all_rmse_var,
                    "plugin_obs_target_rmse_mean": None if not plugin_obs_target_rmse else _mean_var(plugin_obs_target_rmse)[0],
                    "plugin_obs_target_rmse_var": None if not plugin_obs_target_rmse else _mean_var(plugin_obs_target_rmse)[1],
                    "abs_err_vs_gt_response_mean": abs_err_mean,
                    "abs_err_vs_gt_response_var": abs_err_var,
                }
                rows.append(row)
                detailed.append(
                    {
                        "group": group,
                        "variables": variable_label,
                        "method": method,
                        "random_states": DEFAULT_RANDOM_STATES,
                        "runs": run_payloads,
                        "summary": row,
                    }
                )

    output_dir = f"./{today}/"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "lalonde_cvci_plugin_repeat_summary.csv")
    json_path = os.path.join(output_dir, "lalonde_cvci_plugin_repeat_summary.json")

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(detailed, file, indent=2)

    print(csv_path)
    print(json_path)
    for row in rows:
        print(
            "{group:4} {variables:4} {method:6} plugin_mean={plugin_estimate_mean:10.3f} "
            "plugin_var={plugin_estimate_var:10.3f} rmse_all_mean={plugin_all_actual_rmse_mean:10.3f} "
            "err_mean={abs_err_vs_gt_response_mean:10.3f}".format(**row)
        )


if __name__ == "__main__":
    main()
