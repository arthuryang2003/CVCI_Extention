"""Generate OBS-target simulation datasets for IV and shadow selection-bias tests.

This module only generates data and metadata. It does not implement or call any
selection-correction method.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"Correlation length mismatch: {a.shape[0]} vs {b.shape[0]}.")
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _ensure_group_overlap(df: pd.DataFrame) -> None:
    n_rct = int((df["G"] == 1).sum())
    n_obs = int((df["G"] == 0).sum())
    if n_rct == 0 or n_obs == 0:
        raise ValueError(f"Simulation produced degenerate source split: n_rct={n_rct}, n_obs={n_obs}.")


def _ensure_treatment_overlap(df: pd.DataFrame, group_value: int, group_name: str) -> None:
    subset = df[df["G"] == group_value]
    treated = int((subset["T"] == 1).sum())
    control = int((subset["T"] == 0).sum())
    if treated == 0 or control == 0:
        raise ValueError(
            f"{group_name} subset lost treatment overlap: treated={treated}, control={control}. "
            "Adjust simulation parameters."
        )


def generate_obs_target_simulation(
    sim_type: str,
    n: int = 10000,
    seed: int = 42,
    sigma_y: float = 1.0,
    sigma_z: float = 0.1,
    beta0: float = 0.0,
    beta1: float = 0.6,
    beta2: float = -0.4,
    beta_u: float = 1.0,
    alpha0: float = -0.2,
    alpha_x: float = 0.5,
    alpha_u: float = 0.6,
    alpha_t: float = 0.3,
    alpha_y: float = 0.6,
    alpha_iv: float = 0.8,
    alpha_tau: float = 0.0,
    delta0: float = 0.0,
    delta1: float = 0.7,
) -> Dict[str, object]:
    sim_type_normalized = str(sim_type).lower()
    if sim_type_normalized not in {"iv", "shadow"}:
        raise ValueError(f"sim_type must be one of ['iv', 'shadow'], got {sim_type}.")
    if int(n) <= 0:
        raise ValueError(f"n must be positive, got {n}.")

    rng = np.random.default_rng(int(seed))
    n = int(n)

    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    u = rng.normal(size=n)
    eps0 = rng.normal(scale=float(sigma_y), size=n)

    tau_true = 1.0 + 0.5 * x1 - 0.3 * x2
    y0 = 1.0 + x1 + 0.5 * x2 + 0.8 * u + eps0
    y1 = y0 + tau_true

    p_t_obs = _sigmoid(beta0 + beta1 * x1 + beta2 * x2 + beta_u * u)
    t_obs = rng.binomial(1, p_t_obs, size=n).astype(float)
    y_obs = t_obs * y1 + (1.0 - t_obs) * y0

    z_iv: Optional[np.ndarray] = None
    z_shadow: Optional[np.ndarray] = None
    iv_cols = []
    shadow_cols = []
    extra_term = np.zeros(n, dtype=float)

    if sim_type_normalized == "iv":
        z_iv_prob = _sigmoid(delta0 + delta1 * x1)
        z_iv = rng.binomial(1, z_iv_prob, size=n).astype(float)
        extra_term = alpha_iv * z_iv
        iv_cols = ["Z_iv"]
    else:
        z_shadow = y_obs + rng.normal(scale=float(sigma_z), size=n)
        shadow_cols = ["Z_shadow"]

    p_g = _sigmoid(
        alpha0
        + alpha_x * x1
        + alpha_u * u
        + alpha_t * t_obs
        + alpha_y * y_obs
        + alpha_tau * tau_true
        + extra_term
    )
    g = rng.binomial(1, p_g, size=n).astype(float)

    t_rct = rng.binomial(1, 0.5, size=n).astype(float)
    y_rct = t_rct * y1 + (1.0 - t_rct) * y0
    t = np.where(g == 0.0, t_obs, t_rct)
    y = np.where(g == 0.0, y_obs, y_rct)

    data = {
        "X1": x1.astype(float),
        "X2": x2.astype(float),
        "X3": x3.astype(float),
        "T": t.astype(float),
        "Y": y.astype(float),
        "G": g.astype(float),
        "U": u.astype(float),
        "Y0": y0.astype(float),
        "Y1": y1.astype(float),
        "tau_true": tau_true.astype(float),
        "p_g": p_g.astype(float),
        "T_obs": t_obs.astype(float),
        "Y_obs": y_obs.astype(float),
        "p_t_obs": p_t_obs.astype(float),
    }
    if z_iv is not None:
        data["Z_iv"] = z_iv.astype(float)
    if z_shadow is not None:
        data["Z_shadow"] = z_shadow.astype(float)

    df = pd.DataFrame(data)
    _ensure_group_overlap(df)
    _ensure_treatment_overlap(df, group_value=1, group_name="RCT")
    _ensure_treatment_overlap(df, group_value=0, group_name="OBS")

    ate_true_obs = float(df.loc[df["G"] == 0.0, "tau_true"].mean())
    if not np.isfinite(ate_true_obs):
        raise ValueError("ate_true_obs is not finite.")

    metadata = {
        "sim_type": sim_type_normalized,
        "target": "obs",
        "ate_true_obs": ate_true_obs,
        "x_cols": ["X1", "X2", "X3"],
        "candidate_feature_cols": ["X1", "X2", "X3", *iv_cols, *shadow_cols],
        "treatment_col": "T",
        "outcome_col": "Y",
        "source_col": "G",
        "iv_cols": iv_cols,
        "shadow_cols": shadow_cols,
        "target_estimand": "ATE_OBS = E[Y(1) - Y(0) | G = 0]",
        "rct_propensity": 0.5,
        "g_semantics": {"1": "RCT", "0": "OBS"},
        "ground_truth_cols": ["U", "Y0", "Y1", "tau_true", "p_g", "T_obs", "Y_obs", "p_t_obs"],
        "dgp_params": {
            "n": n,
            "seed": int(seed),
            "beta0": float(beta0),
            "beta1": float(beta1),
            "beta2": float(beta2),
            "beta_u": float(beta_u),
            "alpha0": float(alpha0),
            "alpha_x": float(alpha_x),
            "alpha_u": float(alpha_u),
            "alpha_t": float(alpha_t),
            "alpha_y": float(alpha_y),
            "alpha_iv": float(alpha_iv),
            "alpha_tau": float(alpha_tau),
            "delta0": float(delta0),
            "delta1": float(delta1),
            "sigma_y": float(sigma_y),
            "sigma_z": float(sigma_z),
        },
    }

    diagnostics = {
        "sim_type": sim_type_normalized,
        "N": int(df.shape[0]),
        "n_rct": int((df["G"] == 1.0).sum()),
        "n_obs": int((df["G"] == 0.0).sum()),
        "ate_true_obs": ate_true_obs,
        "corr_G_Y": _corr(df["G"].to_numpy(), df["Y"].to_numpy()),
        "corr_G_Y_obs": _corr(df["G"].to_numpy(), df["Y_obs"].to_numpy()),
        "corr_Tobs_U_in_obs": _corr(
            df.loc[df["G"] == 0.0, "T_obs"].to_numpy(),
            df.loc[df["G"] == 0.0, "U"].to_numpy(),
        ),
    }
    if z_iv is not None:
        diagnostics["corr_Z_iv_G"] = _corr(df["Z_iv"].to_numpy(), df["G"].to_numpy())
    if z_shadow is not None:
        diagnostics["corr_Z_shadow_Y"] = _corr(df["Z_shadow"].to_numpy(), df["Y"].to_numpy())

    return {
        "df": df,
        "metadata": metadata,
        "diagnostics": diagnostics,
    }


def _write_simulation_outputs(sim_result: Dict[str, object], out_dir: Path) -> Dict[str, Path]:
    df = sim_result["df"]
    metadata = sim_result["metadata"]
    sim_type = metadata["sim_type"]
    prefix = f"sim_obs_{sim_type}"

    out_dir.mkdir(parents=True, exist_ok=True)
    combined_path = out_dir / f"{prefix}_combined.csv"
    rct_path = out_dir / f"{prefix}_rct.csv"
    obs_path = out_dir / f"{prefix}_obs.csv"
    metadata_path = out_dir / f"{prefix}_metadata.json"

    df.to_csv(combined_path, index=False)
    df[df["G"] == 1.0].copy().to_csv(rct_path, index=False)
    df[df["G"] == 0.0].copy().to_csv(obs_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "combined_csv": combined_path,
        "rct_csv": rct_path,
        "obs_csv": obs_path,
        "metadata_json": metadata_path,
    }


def _print_diagnostics(diag: Dict[str, object]) -> None:
    ordered_keys = [
        "sim_type",
        "N",
        "n_rct",
        "n_obs",
        "ate_true_obs",
        "corr_G_Y",
        "corr_G_Y_obs",
        "corr_Z_shadow_Y",
        "corr_Z_iv_G",
        "corr_Tobs_U_in_obs",
    ]
    for key in ordered_keys:
        if key in diag:
            print(f"{key}: {diag[key]}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_type", type=str, default="both", choices=["iv", "shadow", "both"])
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="results/sim_data")
    parser.add_argument("--sigma_y", type=float, default=1.0)
    parser.add_argument("--sigma_z", type=float, default=0.1)
    parser.add_argument("--alpha_y", type=float, default=0.6)
    parser.add_argument("--alpha_u", type=float, default=0.6)
    parser.add_argument("--alpha_iv", type=float, default=0.8)
    parser.add_argument("--alpha_tau", type=float, default=0.0)
    parser.add_argument("--beta_u", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    sim_types = ["iv", "shadow"] if args.sim_type == "both" else [str(args.sim_type)]
    out_dir = Path(args.out_dir)

    for sim_type in sim_types:
        sim_result = generate_obs_target_simulation(
            sim_type=sim_type,
            n=int(args.n),
            seed=int(args.seed),
            sigma_y=float(args.sigma_y),
            sigma_z=float(args.sigma_z),
            alpha_y=float(args.alpha_y),
            alpha_u=float(args.alpha_u),
            alpha_iv=float(args.alpha_iv),
            alpha_tau=float(args.alpha_tau),
            beta_u=float(args.beta_u),
        )
        written = _write_simulation_outputs(sim_result, out_dir=out_dir)
        _print_diagnostics(sim_result["diagnostics"])
        print(f"combined_csv: {written['combined_csv']}")
        print(f"rct_csv: {written['rct_csv']}")
        print(f"obs_csv: {written['obs_csv']}")
        print(f"metadata_json: {written['metadata_json']}")


if __name__ == "__main__":
    main()
