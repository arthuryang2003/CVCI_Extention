"""Selection-IV inspired plugin for OBS-target eta-model reweighting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from methods.plugins.base import SelectionCorrectionPlugin
from utils.screening_utils import partial_abs_corr
from utils.weight_utils import ensure_1d_float, finalize_weights, normalize_weights, weight_summary


@dataclass
class SelectionIVPlugin(SelectionCorrectionPlugin):
    """
    Engineering first version of a selection-IV plugin.

    Notes:
    - This uses selection IV ideas to correct source selection bias G.
    - It is not a treatment-IV implementation.
    - It provides IV-informed RCT reweighting for eta(x) fitting.
    - Full nonparametric identification / semiparametric efficiency is intentionally deferred.
    """

    name: str = "selection_iv"
    iv_candidate_cols: Optional[Sequence[str]] = None
    x_cols_for_iv_screen: Optional[Sequence[str]] = None
    use_treatment_in_screen: bool = True
    relevance_threshold: float = 0.02
    exclusion_threshold: float = 0.02
    clip_min: float = 0.05
    clip_max: float = 20.0
    model_type: str = "logistic"
    verbose: bool = True

    def _log(self, message: str):
        if self.verbose:
            print(f"[SelectionIVPlugin] {message}")

    def screen_selection_iv_candidates(
        self,
        df_all: pd.DataFrame,
        x_cols: Sequence[str],
        a_col: str,
        y_col: str,
        g_col: str,
    ) -> Dict[str, object]:
        """
        Heuristic screening (not a formal causal test):
        1) relevance score: |partial corr(Z, G | X)|
        2) exclusion score: |partial corr(Z, Y | X, A?)|
        """
        candidates = list(self.iv_candidate_cols or [])
        if not candidates:
            raise ValueError("SelectionIVPlugin requires non-empty iv_candidate_cols.")

        screen_x = list(self.x_cols_for_iv_screen or x_cols)
        missing_screen_x = [col for col in screen_x if col not in df_all.columns]
        if missing_screen_x:
            raise ValueError(f"Missing x_cols_for_iv_screen columns: {missing_screen_x}")

        X_screen = df_all[screen_x].to_numpy(dtype=float)
        nuisance_exclusion = [X_screen]
        if self.use_treatment_in_screen:
            nuisance_exclusion.append(df_all[a_col].to_numpy(dtype=float).reshape(-1, 1))
        exclusion_nuisance = np.hstack(nuisance_exclusion)

        y_vec = df_all[y_col].to_numpy(dtype=float)
        g_vec = df_all[g_col].to_numpy(dtype=float)

        logs: List[Dict[str, object]] = []
        selected: List[str] = []
        for col in candidates:
            if col not in df_all.columns:
                logs.append(
                    {
                        "column": col,
                        "relevance_score": None,
                        "exclusion_score": None,
                        "selected": False,
                        "reason": "missing_column",
                    }
                )
                continue

            z = df_all[col].to_numpy(dtype=float)
            relevance_score = partial_abs_corr(z, g_vec, X_screen)
            exclusion_score = partial_abs_corr(z, y_vec, exclusion_nuisance)
            is_selected = bool(relevance_score >= self.relevance_threshold and exclusion_score <= self.exclusion_threshold)

            logs.append(
                {
                    "column": col,
                    "relevance_score": float(relevance_score),
                    "exclusion_score": float(exclusion_score),
                    "selected": is_selected,
                    "reason": "pass" if is_selected else "threshold_not_met",
                }
            )
            if is_selected:
                selected.append(col)

        # Engineering fallback when heuristic screening is too strict.
        if not selected:
            fallback = [col for col in candidates if col in df_all.columns][:2]
            if not fallback:
                raise ValueError("No IV candidates available after screening; please pass valid iv_candidate_cols.")
            selected = fallback
            for entry in logs:
                if entry["column"] in selected:
                    entry["selected"] = True
                    entry["reason"] = "fallback_selected"

        self._log(f"Heuristic IV screening selected columns: {selected}")
        return {
            "screening_type": "heuristic",
            "screening_logs": logs,
            "selected_iv_cols": selected,
            "relevance_threshold": float(self.relevance_threshold),
            "exclusion_threshold": float(self.exclusion_threshold),
        }

    def fit_selection_iv_model(
        self,
        df_all: pd.DataFrame,
        x_cols: Sequence[str],
        g_col: str,
        selected_iv_cols: Sequence[str],
    ) -> Dict[str, object]:
        if self.model_type != "logistic":
            raise ValueError("SelectionIVPlugin currently supports model_type='logistic' only.")

        feature_cols = list(dict.fromkeys(list(x_cols) + list(selected_iv_cols)))
        X = df_all[feature_cols].to_numpy(dtype=float)
        g = df_all[g_col].to_numpy(dtype=float)

        selection_model = LogisticRegression(max_iter=2000)
        selection_model.fit(X, g)
        return {
            "selection_model": selection_model,
            "feature_cols": feature_cols,
        }

    def compute_iv_weights(
        self,
        selection_model: LogisticRegression,
        df_rct: pd.DataFrame,
        feature_cols: Sequence[str],
    ) -> np.ndarray:
        X_rct = df_rct[list(feature_cols)].to_numpy(dtype=float)
        p_g1 = selection_model.predict_proba(X_rct)[:, 1]
        p_g1 = np.clip(p_g1, 1e-6, 1.0 - 1e-6)

        # OBS-target correction: upweight RCT units resembling OBS source distribution.
        raw_weights = (1.0 - p_g1) / p_g1
        weights = finalize_weights(raw_weights, clip_min=self.clip_min, clip_max=self.clip_max, normalize=True)
        weights = normalize_weights(weights)
        return ensure_1d_float(weights)

    def fit(self, df_rct, df_obs, x_cols, a_col, y_col, g_col):
        df_all = pd.concat([df_rct, df_obs], axis=0, ignore_index=True)

        screening_result = self.screen_selection_iv_candidates(
            df_all=df_all,
            x_cols=x_cols,
            a_col=a_col,
            y_col=y_col,
            g_col=g_col,
        )
        selected_iv_cols = screening_result["selected_iv_cols"]

        model_result = self.fit_selection_iv_model(
            df_all=df_all,
            x_cols=x_cols,
            g_col=g_col,
            selected_iv_cols=selected_iv_cols,
        )

        weights = self.compute_iv_weights(
            selection_model=model_result["selection_model"],
            df_rct=df_rct,
            feature_cols=model_result["feature_cols"],
        )

        if np.any(~np.isfinite(weights)):
            raise ValueError("SelectionIVPlugin produced non-finite weights.")

        self.selected_iv_cols_ = list(selected_iv_cols)
        self.selection_model_ = model_result["selection_model"]
        self.feature_cols_ = model_result["feature_cols"]
        self.weights_ = weights
        self.fitted_ = True
        self.diagnostics_ = {
            "plugin": self.name,
            "screening": screening_result,
            "selected_iv_cols": self.selected_iv_cols_,
            "selection_feature_cols": self.feature_cols_,
            "weight_summary": weight_summary(self.weights_),
        }
        return self

    def get_rct_weights(self, df_rct: pd.DataFrame) -> Optional[np.ndarray]:
        if not self.fitted_:
            return None
        # Recompute by current row order to guarantee alignment.
        return self.compute_iv_weights(self.selection_model_, df_rct=df_rct, feature_cols=self.feature_cols_)

    def get_corrected_bias_target(self, df_rct: pd.DataFrame, base_w_hat: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        _ = df_rct, base_w_hat
        return None
