"""IPSW plugin for OBS-target eta-model correction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from methods.plugins.base import SelectionCorrectionPlugin
from utils.weight_utils import finalize_weights, weight_summary


@dataclass
class IPSWPlugin(SelectionCorrectionPlugin):
    name: str = "ipsw"
    clip_min: float = 0.05
    clip_max: float = 20.0
    max_iter: int = 1000

    def fit(self, df_rct, df_obs, x_cols: Sequence[str], a_col, y_col, g_col):
        _ = a_col, y_col
        df_all = pd.concat([df_rct, df_obs], axis=0, ignore_index=True)
        model = LogisticRegression(max_iter=self.max_iter)
        model.fit(df_all[list(x_cols)].to_numpy(dtype=float), df_all[g_col].to_numpy(dtype=float))

        p_rct = model.predict_proba(df_rct[list(x_cols)].to_numpy(dtype=float))[:, 1]
        p_rct = np.clip(p_rct, 1e-6, 1.0 - 1e-6)
        raw_weights = (1.0 - p_rct) / p_rct
        self.weights_ = finalize_weights(raw_weights, clip_min=self.clip_min, clip_max=self.clip_max)
        self.selection_model_ = model
        self.fitted_ = True
        self.diagnostics_ = {
            "plugin": self.name,
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
            "weight_summary": weight_summary(self.weights_),
        }
        return self

    def get_rct_weights(self, df_rct: pd.DataFrame):
        _ = df_rct
        return self.weights_
