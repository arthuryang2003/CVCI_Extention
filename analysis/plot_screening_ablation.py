"""Plot screening ablation results for presentation-ready figures."""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=str, default="experiments/screening_ablation_results.csv")
    parser.add_argument("--output-dir", type=str, default="experiments/figures")
    return parser.parse_args()


def _parse_list_cell(value) -> List[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    text = str(value).strip()
    if not text or text.lower() == "none":
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:  # noqa: BLE001
            continue
    return [text]


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"saved: {path}")


def _plot_grouped_bar(df: pd.DataFrame, y_col: str, y_label: str, out_path: Path) -> None:
    modes = ["screened", "all", "topk"]
    grouped = (
        df.dropna(subset=[y_col])
        .groupby(["target_mode", "screening_mode", "plugin"], as_index=False)[y_col]
        .mean()
    )
    targets = sorted(grouped["target_mode"].unique().tolist())
    if not targets:
        return

    fig, axes = plt.subplots(1, len(targets), figsize=(6 * len(targets), 4), squeeze=False)
    for idx, target in enumerate(targets):
        ax = axes[0, idx]
        sub = grouped[grouped["target_mode"] == target]
        plugins = sorted(sub["plugin"].dropna().unique().tolist())
        x = np.arange(len(modes))
        width = 0.8 / max(len(plugins), 1)
        for p_idx, plugin in enumerate(plugins):
            vals = []
            for mode in modes:
                match = sub[(sub["screening_mode"] == mode) & (sub["plugin"] == plugin)][y_col]
                vals.append(float(match.iloc[0]) if not match.empty else np.nan)
            ax.bar(x + p_idx * width - 0.4 + width / 2, vals, width=width, label=plugin)
        ax.set_xticks(x)
        ax.set_xticklabels(modes)
        ax.set_title(f"target_mode={target}")
        ax.set_xlabel("screening_mode")
        ax.set_ylabel(y_label)
        ax.legend()
    _save(fig, out_path)


def _plot_selection_frequency(df: pd.DataFrame, col: str, title: str, out_path: Path) -> None:
    counter: Counter[str] = Counter()
    for values in df[col].tolist():
        for name in _parse_list_cell(values):
            counter[name] += 1
    if not counter:
        return
    keys = sorted(counter.keys())
    vals = [counter[k] for k in keys]
    fig, ax = plt.subplots(figsize=(max(8, len(keys) * 0.8), 4))
    ax.bar(keys, vals)
    ax.set_title(title)
    ax.set_xlabel("variable")
    ax.set_ylabel("selection_count")
    ax.tick_params(axis="x", rotation=45)
    _save(fig, out_path)


def _plot_topk_curve(df: pd.DataFrame, out_path: Path) -> None:
    topk_df = df[(df["screening_mode"] == "topk") & df["top_k"].notna() & df["rmse"].notna()].copy()
    if topk_df.empty:
        return
    topk_df["top_k"] = topk_df["top_k"].astype(int)
    grouped = topk_df.groupby(["target_mode", "plugin", "top_k"], as_index=False)["rmse"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    for (target_mode, plugin), sub in grouped.groupby(["target_mode", "plugin"]):
        sub = sub.sort_values("top_k")
        ax.plot(sub["top_k"], sub["rmse"], marker="o", label=f"{target_mode}-{plugin}")
    ax.set_xlabel("top_k")
    ax.set_ylabel("mean_rmse")
    ax.set_title("Top-k RMSE Curve")
    ax.legend()
    _save(fig, out_path)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    out_dir = Path(args.output_dir)

    _plot_grouped_bar(
        df=df,
        y_col="rmse",
        y_label="mean_rmse",
        out_path=out_dir / "rmse_by_screening_mode.png",
    )
    _plot_grouped_bar(
        df=df,
        y_col="ate_hat",
        y_label="mean_ate_hat",
        out_path=out_dir / "ate_by_screening_mode.png",
    )
    _plot_selection_frequency(
        df=df[df["plugin"] == "iv"],
        col="selected_iv_cols",
        title="IV Selection Frequency",
        out_path=out_dir / "iv_selection_frequency.png",
    )
    _plot_selection_frequency(
        df=df[df["plugin"] == "shadow"],
        col="selected_shadow_cols",
        title="Shadow Selection Frequency",
        out_path=out_dir / "shadow_selection_frequency.png",
    )
    _plot_topk_curve(df=df, out_path=out_dir / "topk_curve.png")


if __name__ == "__main__":
    main()
