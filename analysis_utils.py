"""Shared utilities for schema validation, sentiment columns, and safe date parsing."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def validate_required_columns(df: pd.DataFrame, required: Iterable[str], df_name: str) -> None:
    """Raise a clear error when required columns are missing."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def parse_date_safe(series: pd.Series) -> pd.Series:
    """Parse mixed date formats safely with explicit fallback passes."""
    as_str = series.astype(str).str.strip()
    # Normalize blank-ish values so they become NaT across all passes.
    as_str = as_str.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA})

    parsed = pd.to_datetime(as_str, format="%Y-%m-%d", errors="coerce")
    needs_fallback = parsed.isna() & as_str.notna()
    if needs_fallback.any():
        parsed.loc[needs_fallback] = pd.to_datetime(
            as_str.loc[needs_fallback], format="%Y-%m-%d %H:%M:%S", errors="coerce"
        )

    needs_fallback = parsed.isna() & as_str.notna()
    if needs_fallback.any():
        try:
            parsed.loc[needs_fallback] = pd.to_datetime(
                as_str.loc[needs_fallback], format="mixed", errors="coerce"
            )
        except (TypeError, ValueError):
            parsed.loc[needs_fallback] = pd.to_datetime(
                as_str.loc[needs_fallback], errors="coerce"
            )
    return parsed


def standardize_date_column(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    """Return dataframe copy with normalized ISO date strings in `col`."""
    validate_required_columns(df, [col], "DataFrame")
    out = df.copy()
    parsed = parse_date_safe(out[col])
    out[col] = parsed.dt.strftime("%Y-%m-%d").fillna("")
    return out


def ensure_sentiment_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure compatibility sentiment schema:
    - `sentiment` numeric
    - `compound` numeric alias
    - `sentiment_label` categorical
    """
    out = df.copy()
    if "sentiment" not in out.columns and "compound" not in out.columns:
        raise ValueError("Expected at least one of: sentiment, compound")

    if "sentiment" not in out.columns and "compound" in out.columns:
        out["sentiment"] = pd.to_numeric(out["compound"], errors="coerce")
    if "compound" not in out.columns and "sentiment" in out.columns:
        out["compound"] = pd.to_numeric(out["sentiment"], errors="coerce")

    out["sentiment"] = pd.to_numeric(out["sentiment"], errors="coerce")
    out["compound"] = pd.to_numeric(out["compound"], errors="coerce")

    if "sentiment_label" not in out.columns:
        out["sentiment_label"] = "neutral"

    # Rebuild/patch labels wherever missing.
    def _label(score: float) -> str:
        if pd.isna(score):
            return "neutral"
        if score >= 0.05:
            return "positive"
        if score <= -0.05:
            return "negative"
        return "neutral"

    missing_mask = out["sentiment_label"].isna() | (out["sentiment_label"].astype(str).str.strip() == "")
    if missing_mask.any():
        out.loc[missing_mask, "sentiment_label"] = out.loc[missing_mask, "sentiment"].apply(_label)

    return out
