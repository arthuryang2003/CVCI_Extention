"""Unified data interface for RCT experiment scripts."""

from __future__ import annotations

import pandas as pd

from rct.data import load_lalonde_csv


def get_lalonde_dataframe(lalonde_path: str = "lalonde.csv") -> pd.DataFrame:
    """Load LaLonde CSV with shared feature engineering (age2/u74/u75)."""
    return load_lalonde_csv(lalonde_path)
