"""Shared helpers for cross-sectional factor scores (0–1) and 0–5 ratings."""

from typing import Union

import numpy as np
import pandas as pd


def rating_to_grade(x: float) -> Union[str, float]:
    if pd.isna(x):
        return np.nan
    if x >= 4.5:
        return "A+"
    if x >= 4.0:
        return "A"
    if x >= 3.67:
        return "A-"
    if x >= 3.33:
        return "B+"
    if x >= 3.0:
        return "B"
    if x >= 2.67:
        return "B-"
    if x >= 2.33:
        return "C+"
    if x >= 2.0:
        return "C"
    if x >= 1.67:
        return "C-"
    if x >= 1.33:
        return "D+"
    if x >= 1.0:
        return "D"
    return "F"


def score_to_rating(score: pd.Series) -> pd.Series:
    """Map percentile score in [0, 1] to ~0.1–5.0 rating (Seeking Alpha style)."""
    return (score * 5).clip(lower=0.1).round(2)


def combine_two_rank_scores(
    s1: pd.Series,
    s2: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Average two percentile rank scores; use whichever exists if only one is present.
    Returns (component_count, combined_score, combined_rating).
    """
    count = s1.notna().astype(int) + s2.notna().astype(int)
    combined = np.where(
        s1.notna() & s2.notna(),
        (s1 + s2) / 2,
        np.where(s1.notna(), s1, s2),
    )
    combined = pd.Series(combined, index=s1.index, dtype=float)
    combined.loc[count == 0] = np.nan
    rating = score_to_rating(combined)
    return count, combined, rating
