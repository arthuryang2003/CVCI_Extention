"""Unified benchmark runner for OBS/RCT LaLonde experiments.

This script runs a shared method panel:
- base
- base+ipsw
- base+cw
- base+iv
- base+shadow

for target data in {obs, rct} and aggregates results across multiple seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]


METHOD_SPECS: Dict[str, Dict[str, str]] = {
    "base": {"obs": "base", "rct": "none", "label": "base"},
    "base+ipsw": {"obs": "ipsw", "rct": "ipsw", "label": "base+ipsw"},
    "base+cw": {"obs": "cw", "rct": "cw", "label": "base+cw"},
    "base+iv": {"obs": "selection_iv", "rct": "iv", "label": "base+iv"},
    "base+shadow": {"obs": "shadow", "rct": "shadow", "label": "base+shadow"},
}

METHOD_ALIASES = {
    "ipsw": "base+ipsw",
    "cw": "base+cw",
    "iv": "base+iv",
    "shadow": "base+shadow",
    "selection_iv": "base+iv",
    "none": "base",
}


@dataclass
class RunResult:
    target: str
    dataset: str
    method_label: str
    seed: int
    success: bool
    estimate: Optional[float]
    runtime_sec: float
    error: Optional[str]
    raw_payload: Optional[dict]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified benchmark table for OBS/RCT methods.")
    parser.add_argument("--target", type=str, default="obs", choices=["obs", "rct", "both"])
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["base", "base+ipsw", "base+cw", "base+iv", "base+shadow"],
        help="Method list from {base, base+ipsw, base+cw, base+iv, base+shadow}.",
    )
    parser.add_argument("--datasets", nargs="*", default=None, help="Dataset groups. obs default: cps psid; rct default: cps psid")
    parser.add_argument("--seeds", nargs="*", type=int, default=[2024, 2025, 2026])
    parser.add_argument("--lalonde-path", type=str, default="lalonde.csv")
    parser.add_argument("--variables", nargs="*", default=["re75"])
    parser.add_argument("--timeout-sec", type=int, default=1200)
    parser.add_argument("--output-csv", type=str, default="experiments/benchmark_summary.csv")
    parser.add_argument("--output-json", type=str, default="experiments/benchmark_raw_runs.json")
    return parser.parse_args()


def normalize_methods(methods: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    for m in methods:
        key = m.strip().lower()
        canonical = METHOD_ALIASES.get(key, key)
        if canonical not in METHOD_SPECS:
            raise ValueError(f"Unsupported method: {m}. Supported: {sorted(METHOD_SPECS)}")
        if canonical not in normalized:
            normalized.append(canonical)
    return normalized


def targets_from_arg(target: str) -> List[str]:
    return ["obs", "rct"] if target == "both" else [target]


def datasets_for_target(target: str, datasets_arg: Optional[Sequence[str]]) -> List[str]:
    if datasets_arg:
        return list(dict.fromkeys([d.strip().lower() for d in datasets_arg]))
    if target == "obs":
        return ["cps", "psid"]
    return ["cps", "psid"]


def extract_first_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end < start:
        raise ValueError("Cannot locate JSON payload in command output.")
    return json.loads(text[start : end + 1])


def build_obs_cmd(dataset: str, seed: int, method_label: str, lalonde_path: str, variables: Sequence[str]) -> List[str]:
    method_arg = METHOD_SPECS[method_label]["obs"]
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "lalonde_obs_target.py"),
        "--method",
        method_arg,
        "--obs-source",
        dataset,
        "--seed",
        str(seed),
        "--lalonde-path",
        lalonde_path,
    ]
    if variables:
        cmd.extend(["--variables", *variables])
    return cmd


def build_rct_cmd(dataset: str, seed: int, method_label: str, variables: Sequence[str]) -> List[str]:
    method_arg = METHOD_SPECS[method_label]["rct"]
    cmd = [
        sys.executable,
        "-m",
        "rct.experiments.lalonde_cvci_plugin_linear",
        "--method",
        method_arg,
        "--group",
        dataset,
        "--random-state",
        str(seed),
    ]
    if variables:
        cmd.extend(["--variables", *variables])
    return cmd


def run_one(target: str, dataset: str, method_label: str, seed: int, args: argparse.Namespace) -> RunResult:
    if target == "obs":
        cmd = build_obs_cmd(dataset, seed, method_label, args.lalonde_path, args.variables)
    else:
        cmd = build_rct_cmd(dataset, seed, method_label, args.variables)

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(PROJECT_ROOT) if not existing_pythonpath else f"{PROJECT_ROOT}:{existing_pythonpath}"

    start = time.perf_counter()
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=args.timeout_sec,
            check=True,
        )
        runtime = time.perf_counter() - start
        payload = extract_first_json(completed.stdout)
        if target == "obs":
            estimate = float(payload["results"]["obs_target_ate_estimate"])
        else:
            estimate = float(payload["plugin_estimate"])
        return RunResult(
            target=target,
            dataset=dataset,
            method_label=method_label,
            seed=seed,
            success=True,
            estimate=estimate,
            runtime_sec=runtime,
            error=None,
            raw_payload=payload,
        )
    except Exception as exc:  # noqa: BLE001
        runtime = time.perf_counter() - start
        return RunResult(
            target=target,
            dataset=dataset,
            method_label=method_label,
            seed=seed,
            success=False,
            estimate=None,
            runtime_sec=runtime,
            error=str(exc),
            raw_payload=None,
        )


def aggregate_results(results: Sequence[RunResult]) -> List[dict]:
    grouped: Dict[Tuple[str, str, str], List[RunResult]] = {}
    for r in results:
        grouped.setdefault((r.target, r.dataset, r.method_label), []).append(r)

    rows: List[dict] = []
    for (target, dataset, method_label), group in sorted(grouped.items()):
        estimates = [x.estimate for x in group if x.success and x.estimate is not None]
        runtimes = [x.runtime_sec for x in group if x.success]
        failed = [x.seed for x in group if not x.success]

        est_mean = float(statistics.mean(estimates)) if estimates else math.nan
        est_std = float(statistics.pstdev(estimates)) if len(estimates) > 1 else (0.0 if len(estimates) == 1 else math.nan)
        rt_mean = float(statistics.mean(runtimes)) if runtimes else math.nan
        rt_std = float(statistics.pstdev(runtimes)) if len(runtimes) > 1 else (0.0 if len(runtimes) == 1 else math.nan)

        rows.append(
            {
                "target": target,
                "dataset": dataset,
                "method": method_label,
                "n_runs": len(group),
                "n_success": len(estimates),
                "estimate_mean": est_mean,
                "estimate_std": est_std,
                "runtime_mean_sec": rt_mean,
                "runtime_std_sec": rt_std,
                "failed_seeds": ",".join(str(x) for x in failed),
                "status": "ok" if len(estimates) == len(group) else "partial_or_failed",
            }
        )
    return rows


def print_table(rows: Sequence[dict]) -> None:
    headers = [
        "target",
        "dataset",
        "method",
        "n_success/n_runs",
        "estimate_mean",
        "estimate_std",
        "runtime_mean_sec",
        "status",
    ]
    print("\t".join(headers))
    for r in rows:
        print(
            "\t".join(
                [
                    str(r["target"]),
                    str(r["dataset"]),
                    str(r["method"]),
                    f"{r['n_success']}/{r['n_runs']}",
                    f"{r['estimate_mean']:.6f}" if not math.isnan(r["estimate_mean"]) else "nan",
                    f"{r['estimate_std']:.6f}" if not math.isnan(r["estimate_std"]) else "nan",
                    f"{r['runtime_mean_sec']:.3f}" if not math.isnan(r["runtime_mean_sec"]) else "nan",
                    str(r["status"]),
                ]
            )
        )


def save_csv(rows: Sequence[dict], path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target",
        "dataset",
        "method",
        "n_runs",
        "n_success",
        "estimate_mean",
        "estimate_std",
        "runtime_mean_sec",
        "runtime_std_sec",
        "failed_seeds",
        "status",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_raw(results: Sequence[RunResult], path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "target": r.target,
            "dataset": r.dataset,
            "method": r.method_label,
            "seed": r.seed,
            "success": r.success,
            "estimate": r.estimate,
            "runtime_sec": r.runtime_sec,
            "error": r.error,
            "raw_payload": r.raw_payload,
        }
        for r in results
    ]
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    methods = normalize_methods(args.methods)

    run_results: List[RunResult] = []
    for target in targets_from_arg(args.target):
        datasets = datasets_for_target(target, args.datasets)
        for dataset in datasets:
            for method_label in methods:
                for seed in args.seeds:
                    result = run_one(target, dataset, method_label, seed, args)
                    run_results.append(result)

    summary_rows = aggregate_results(run_results)
    print_table(summary_rows)
    save_csv(summary_rows, args.output_csv)
    save_raw(run_results, args.output_json)
    print(f"\nSaved summary CSV: {args.output_csv}")
    print(f"Saved raw JSON: {args.output_json}")


if __name__ == "__main__":
    main()
