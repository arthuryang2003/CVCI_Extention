"""Base interface for OBS-target selection bias correction plugins."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class SelectionCorrectionPlugin:
    """
    Unified plugin contract for eta(x) training-time correction.

    Plugins may provide either:
    - RCT sample weights, or
    - a corrected bias target.
    """

    name: str = "base"
    diagnostics_: Dict[str, object] = field(default_factory=dict)
    fitted_: bool = False

    def fit(
        self,
        df_rct: pd.DataFrame,
        df_obs: pd.DataFrame,
        x_cols: Sequence[str],
        a_col: str,
        y_col: str,
        g_col: str,
    ):
        _ = df_rct, df_obs, x_cols, a_col, y_col, g_col
        self.fitted_ = True
        self.diagnostics_ = {"plugin": self.name}
        return self

    def get_rct_weights(self, df_rct: pd.DataFrame) -> Optional[np.ndarray]:
        _ = df_rct
        return None

    def get_corrected_bias_target(
        self,
        df_rct: pd.DataFrame,
        base_w_hat: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        _ = df_rct, base_w_hat
        return None

    def summary(self) -> Dict[str, object]:
        return {"plugin": self.name, **self.diagnostics_}
