"""Batch runner for screening ablation experiments on Lalonde."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_experiment import parse_args as parse_unified_args
from run_experiment import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lalonde-path", type=str, default="lalonde.csv")
    parser.add_argument("--output-csv", type=str, default="experiments/screening_ablation_results.csv")
    parser.add_argument("--obs-source", nargs="*", default=["cps"])
    parser.add_argument("--target-mode", nargs="*", default=["rct", "obs"])
    parser.add_argument("--plugin", nargs="*", default=["iv", "shadow"])
    parser.add_argument("--screening-mode", nargs="*", default=["screened", "all", "topk"])
    parser.add_argument("--top-k-grid", nargs="*", type=int, default=[1, 2, 3])
    parser.add_argument("--seed", nargs="*", type=int, default=[2024, 2025, 2026])
    return parser.parse_args()


def _normalize_unique(values: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        key = str(value).lower()
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _run_one(
    target_mode: str,
    plugin: str,
    screening_mode: str,
    top_k: Optional[int],
    seed: int,
    obs_source: str,
    lalonde_path: str,
) -> Dict[str, object]:
    method = "cvci" if target_mode == "rct" else "rhc"
    cli = [
        "--dataset",
        "lalonde",
        "--target-mode",
        target_mode,
        "--method",
        method,
        "--plugin",
        plugin,
        "--obs-source",
        obs_source,
        "--seed",
        str(seed),
        "--lalonde-path",
        lalonde_path,
        "--screening-mode",
        screening_mode,
    ]
    if screening_mode == "topk" and top_k is not None:
        cli.extend(["--top-k", str(top_k)])
    args = parse_unified_args(cli)
    result = run_experiment(args)
    return {
        "target_mode": result.get("target_mode"),
        "method": result.get("method"),
        "plugin": result.get("plugin_name"),
        "obs_source": result.get("obs_source"),
        "seed": seed,
        "screening_mode": result.get("screening_mode"),
        "top_k": result.get("top_k"),
        "ate_hat": result.get("ate_hat"),
        "rmse": result.get("rmse"),
        "lambda_opt": result.get("lambda_opt"),
        "selected_iv_cols": json.dumps(result.get("selected_iv_cols"), ensure_ascii=False),
        "selected_shadow_cols": json.dumps(result.get("selected_shadow_cols"), ensure_ascii=False),
        "n_selected_iv": result.get("n_selected_iv"),
        "n_selected_shadow": result.get("n_selected_shadow"),
        "plugin_summary_json": result.get("plugin_summary_json"),
        "screening_logs_json": result.get("screening_logs_json"),
    }


def main() -> None:
    args = parse_args()
    target_modes = _normalize_unique([str(x) for x in args.target_mode])
    plugins = _normalize_unique([str(x) for x in args.plugin])
    screening_modes = _normalize_unique([str(x) for x in args.screening_mode])
    obs_sources = _normalize_unique([str(x) for x in args.obs_source])
    seeds = [int(s) for s in args.seed]
    top_k_grid = [int(k) for k in args.top_k_grid]

    rows: List[Dict[str, object]] = []
    errors: List[Dict[str, object]] = []
    total = 0

    for target_mode in target_modes:
        for plugin in plugins:
            for screening_mode in screening_modes:
                topk_values: List[Optional[int]] = [None]
                if screening_mode == "topk":
                    topk_values = [int(k) for k in top_k_grid]
                for top_k in topk_values:
                    for seed in seeds:
                        for obs_source in obs_sources:
                            total += 1
                            try:
                                print(
                                    f"[run {total}] target={target_mode} plugin={plugin} "
                                    f"mode={screening_mode} top_k={top_k} seed={seed} obs={obs_source}"
                                )
                                row = _run_one(
                                    target_mode=target_mode,
                                    plugin=plugin,
                                    screening_mode=screening_mode,
                                    top_k=top_k,
                                    seed=seed,
                                    obs_source=obs_source,
                                    lalonde_path=args.lalonde_path,
                                )
                                rows.append(row)
                            except Exception as exc:  # noqa: BLE001
                                tb = traceback.format_exc(limit=3)
                                error_row = {
                                    "target_mode": target_mode,
                                    "plugin": plugin,
                                    "screening_mode": screening_mode,
                                    "top_k": top_k,
                                    "seed": seed,
                                    "obs_source": obs_source,
                                    "error": str(exc),
                                    "traceback": tb,
                                }
                                errors.append(error_row)
                                print(f"[error] {error_row}")

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"[saved] {output_path} rows={len(df)}")

    if errors:
        error_path = output_path.with_suffix(".errors.csv")
        pd.DataFrame(errors).to_csv(error_path, index=False)
        print(f"[saved] {error_path} rows={len(errors)}")


if __name__ == "__main__":
    main()
