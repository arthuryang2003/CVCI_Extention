"""Shadow-variable inspired plugin for OBS-target eta-model reweighting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from methods.plugins.base import SelectionCorrectionPlugin
from utils.screening_utils import partial_abs_corr
from utils.weight_utils import ensure_1d_float, finalize_weights, weight_summary


@dataclass
class ShadowPlugin(SelectionCorrectionPlugin):
    """
    Engineering first version of shadow-variable selection correction.

    Notes:
    - This is a shadow-variable inspired approximation for source-selection correction.
    - It does not fully solve shadow identification (e.g., Fredholm equations / semiparametric efficiency).
    - It converts shadow information into practical eta-stage weights in the OBS-target framework.
    - Future versions can replace this with stricter odds-ratio identification modules.
    """

    name: str = "shadow"
    shadow_candidate_cols: Optional[Sequence[str]] = None
    x_cols_for_shadow_screen: Optional[Sequence[str]] = None
    use_treatment_in_screen: bool = True
    association_threshold: float = 0.02
    residual_independence_threshold: float = 0.02
    or_model_type: str = "exp_linear"
    clip_min: float = 0.05
    clip_max: float = 20.0
    verbose: bool = True

    def _log(self, message: str):
        if self.verbose:
            print(f"[ShadowPlugin] {message}")

    def screen_shadow_candidates(
        self,
        df_all: pd.DataFrame,
        x_cols: Sequence[str],
        a_col: str,
        y_col: str,
        g_col: str,
    ) -> Dict[str, object]:
        """
        Heuristic screening (not a formal identification test):
        1) association score: |partial corr(Z, Y | X, A?)| should be high
        2) residual-independence score: |partial corr(Z, G | X, Y, A?)| should be low
        """
        candidates = list(self.shadow_candidate_cols or [])
        if not candidates:
            raise ValueError("ShadowPlugin requires non-empty shadow_candidate_cols.")

        screen_x = list(self.x_cols_for_shadow_screen or x_cols)
        missing_screen_x = [col for col in screen_x if col not in df_all.columns]
        if missing_screen_x:
            raise ValueError(f"Missing x_cols_for_shadow_screen columns: {missing_screen_x}")

        X_screen = df_all[screen_x].to_numpy(dtype=float)
        y_vec = df_all[y_col].to_numpy(dtype=float)
        g_vec = df_all[g_col].to_numpy(dtype=float)

        nuisance_assoc = [X_screen]
        nuisance_indep = [X_screen, y_vec.reshape(-1, 1)]
        if self.use_treatment_in_screen:
            a_vec = df_all[a_col].to_numpy(dtype=float).reshape(-1, 1)
            nuisance_assoc.append(a_vec)
            nuisance_indep.append(a_vec)

        assoc_nuisance = np.hstack(nuisance_assoc)
        indep_nuisance = np.hstack(nuisance_indep)

        logs: List[Dict[str, object]] = []
        selected: List[str] = []
        for col in candidates:
            if col not in df_all.columns:
                logs.append(
                    {
                        "column": col,
                        "association_score": None,
                        "residual_independence_score": None,
                        "selected": False,
                        "reason": "missing_column",
                    }
                )
                continue

            z = df_all[col].to_numpy(dtype=float)
            association_score = partial_abs_corr(z, y_vec, assoc_nuisance)
            residual_independence_score = partial_abs_corr(z, g_vec, indep_nuisance)
            is_selected = bool(
                association_score >= self.association_threshold
                and residual_independence_score <= self.residual_independence_threshold
            )
            logs.append(
                {
                    "column": col,
                    "association_score": float(association_score),
                    "residual_independence_score": float(residual_independence_score),
                    "selected": is_selected,
                    "reason": "pass" if is_selected else "threshold_not_met",
                }
            )
            if is_selected:
                selected.append(col)

        if not selected:
            fallback = [col for col in candidates if col in df_all.columns][:2]
            if not fallback:
                raise ValueError("No shadow candidates available after screening; please pass valid shadow_candidate_cols.")
            selected = fallback
            for entry in logs:
                if entry["column"] in selected:
                    entry["selected"] = True
                    entry["reason"] = "fallback_selected"

        self._log(f"Heuristic shadow screening selected columns: {selected}")
        return {
            "screening_type": "heuristic",
            "screening_logs": logs,
            "selected_shadow_cols": selected,
            "association_threshold": float(self.association_threshold),
            "residual_independence_threshold": float(self.residual_independence_threshold),
        }

    def fit_complete_case_outcome_model(
        self,
        df_rct: pd.DataFrame,
        x_cols: Sequence[str],
        a_col: str,
        y_col: str,
    ) -> LinearRegression:
        """
        Engineering approximation:
        Fit the outcome model on RCT (treated as reliable selection-observed subset)
        to build a stable Y-proxy used by downstream distortion modeling.
        """
        X = df_rct[[a_col] + list(x_cols)].to_numpy(dtype=float)
        y = df_rct[y_col].to_numpy(dtype=float)
        model = LinearRegression()
        model.fit(X, y)
        return model

    def fit_shadow_or_model(
        self,
        df_all: pd.DataFrame,
        x_cols: Sequence[str],
        g_col: str,
        y_proxy_all: np.ndarray,
        selected_shadow_cols: Sequence[str],
    ) -> Dict[str, object]:
        """
        Engineering approximation for OR/distortion fitting.

        Step A: Fit auxiliary selection model G ~ X + Y_proxy + Shadow.
        Step B: Project auxiliary log-odds onto OR parameterization:
            exp_linear: OR(X, Y) = exp(gamma0 + gamma1*Y + gamma_x^T X)
            exp_y_only: OR(X, Y) = exp(gamma0 + gamma1*Y)
        This is a practical proxy objective, not full theory identification.
        """
        y_proxy = ensure_1d_float(y_proxy_all)
        X_core = df_all[list(x_cols)].to_numpy(dtype=float)
        X_shadow = df_all[list(selected_shadow_cols)].to_numpy(dtype=float)
        X_aux = np.hstack([X_core, y_proxy.reshape(-1, 1), X_shadow])
        g = df_all[g_col].to_numpy(dtype=float)

        aux_model = LogisticRegression(max_iter=2000)
        aux_model.fit(X_aux, g)

        p_hat = np.clip(aux_model.predict_proba(X_aux)[:, 1], 1e-6, 1.0 - 1e-6)
        logit_hat = np.log(p_hat / (1.0 - p_hat))

        if self.or_model_type == "exp_linear":
            or_feature_cols = ["y_proxy"] + list(x_cols)
            or_design = np.hstack([y_proxy.reshape(-1, 1), X_core])
        elif self.or_model_type == "exp_y_only":
            or_feature_cols = ["y_proxy"]
            or_design = y_proxy.reshape(-1, 1)
        else:
            raise ValueError("Unsupported or_model_type. Use 'exp_linear' or 'exp_y_only'.")

        reg = LinearRegression()
        reg.fit(or_design, logit_hat)

        gamma0 = float(reg.intercept_)
        gamma = reg.coef_.reshape(-1)
        return {
            "aux_selection_model": aux_model,
            "or_feature_cols": or_feature_cols,
            "gamma0": gamma0,
            "gamma": gamma,
            "selected_shadow_cols": list(selected_shadow_cols),
        }

    def compute_shadow_weights(
        self,
        df_rct: pd.DataFrame,
        x_cols: Sequence[str],
        y_proxy_rct: np.ndarray,
        gamma0: float,
        gamma: np.ndarray,
    ) -> np.ndarray:
        y_proxy = ensure_1d_float(y_proxy_rct)
        X_rct = df_rct[list(x_cols)].to_numpy(dtype=float)

        if self.or_model_type == "exp_linear":
            or_design = np.hstack([y_proxy.reshape(-1, 1), X_rct])
        else:
            or_design = y_proxy.reshape(-1, 1)

        distortion = np.exp(np.clip(gamma0 + or_design @ gamma.reshape(-1), -20.0, 20.0))
        raw_weights = 1.0 / np.clip(distortion, 1e-6, None)
        return finalize_weights(raw_weights, clip_min=self.clip_min, clip_max=self.clip_max)

    def fit(self, df_rct, df_obs, x_cols, a_col, y_col, g_col):
        df_all = pd.concat([df_rct, df_obs], axis=0, ignore_index=True)

        screening_result = self.screen_shadow_candidates(
            df_all=df_all,
            x_cols=x_cols,
            a_col=a_col,
            y_col=y_col,
            g_col=g_col,
        )
        selected_shadow_cols = screening_result["selected_shadow_cols"]

        outcome_model = self.fit_complete_case_outcome_model(
            df_rct=df_rct,
            x_cols=x_cols,
            a_col=a_col,
            y_col=y_col,
        )

        y_proxy_all = outcome_model.predict(df_all[[a_col] + list(x_cols)].to_numpy(dtype=float))
        or_model = self.fit_shadow_or_model(
            df_all=df_all,
            x_cols=x_cols,
            g_col=g_col,
            y_proxy_all=y_proxy_all,
            selected_shadow_cols=selected_shadow_cols,
        )

        y_proxy_rct = outcome_model.predict(df_rct[[a_col] + list(x_cols)].to_numpy(dtype=float))
        weights = self.compute_shadow_weights(
            df_rct=df_rct,
            x_cols=x_cols,
            y_proxy_rct=y_proxy_rct,
            gamma0=or_model["gamma0"],
            gamma=or_model["gamma"],
        )
        if np.any(~np.isfinite(weights)):
            raise ValueError("ShadowPlugin produced non-finite weights.")

        self.selected_shadow_cols_ = list(selected_shadow_cols)
        self.outcome_model_ = outcome_model
        self.or_model_ = or_model
        self.weights_ = weights
        self.x_cols_ = list(x_cols)
        self.a_col_ = a_col
        self.fitted_ = True
        self.diagnostics_ = {
            "plugin": self.name,
            "screening": screening_result,
            "selected_shadow_cols": self.selected_shadow_cols_,
            "or_model_type": self.or_model_type,
            "gamma0": float(or_model["gamma0"]),
            "gamma": ensure_1d_float(or_model["gamma"]).tolist(),
            "weight_summary": weight_summary(self.weights_),
        }
        return self

    def get_rct_weights(self, df_rct: pd.DataFrame) -> Optional[np.ndarray]:
        if not self.fitted_:
            return None
        y_proxy_rct = self.outcome_model_.predict(df_rct[[self.a_col_] + self.x_cols_].to_numpy(dtype=float))
        return self.compute_shadow_weights(
            df_rct=df_rct,
            x_cols=self.x_cols_,
            y_proxy_rct=y_proxy_rct,
            gamma0=float(self.or_model_["gamma0"]),
            gamma=ensure_1d_float(self.or_model_["gamma"]),
        )

    def get_corrected_bias_target(self, df_rct: pd.DataFrame, base_w_hat: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        _ = df_rct, base_w_hat
        return None
