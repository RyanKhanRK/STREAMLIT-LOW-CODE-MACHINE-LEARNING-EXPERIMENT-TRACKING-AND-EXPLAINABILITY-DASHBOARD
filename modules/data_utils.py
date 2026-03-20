"""
data_utils.py – Dataset loading, analysis and preprocessing helpers.
"""

from __future__ import annotations

import io
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


# ── Loading ──────────────────────────────────────────────────────────────────

def load_csv(uploaded_file) -> pd.DataFrame:
    """Read an uploaded CSV (or TSV) file into a DataFrame."""
    content = uploaded_file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        df = pd.read_csv(io.BytesIO(content), sep="\t")
    return df


# ── Analysis ─────────────────────────────────────────────────────────────────

def analyze_dataset(df: pd.DataFrame) -> dict:
    """Return a summary dict describing the dataset."""
    numeric_cols  = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols      = df.select_dtypes(include=["object", "category"]).columns.tolist()
    missing_count = df.isnull().sum().sum()
    missing_pct   = round(missing_count / (df.shape[0] * df.shape[1]) * 100, 2)

    return {
        "rows":          df.shape[0],
        "cols":          df.shape[1],
        "numeric_cols":  numeric_cols,
        "cat_cols":      cat_cols,
        "missing_count": int(missing_count),
        "missing_pct":   missing_pct,
        "dtypes":        df.dtypes.astype(str).to_dict(),
        "missing_by_col": df.isnull().sum().to_dict(),
        "nunique":       df.nunique().to_dict(),
    }


def infer_task_type(df: pd.DataFrame, target_col: str) -> str:
    """Heuristically infer whether this looks like classification or regression."""
    series = df[target_col].dropna()
    if series.dtype == object or series.nunique() <= 20:
        return "classification"
    return "regression"


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str | None,
    cfg: dict,
) -> tuple[pd.DataFrame, dict]:
    """
    Apply preprocessing steps defined in *cfg*.

    Returns
    -------
    df_out : pd.DataFrame
        Processed dataframe containing feature_cols (+ target_col if given).
    report : dict
        Human-readable log of what was applied.
    """
    report: list[str] = []
    cols = feature_cols + ([target_col] if target_col else [])
    df_out = df[cols].copy()

    original_rows = len(df_out)

    # ── Missing values ────────────────────────────────────────────────────
    if cfg.get("drop_na"):
        df_out = df_out.dropna()
        dropped = original_rows - len(df_out)
        report.append(f"Dropped {dropped} rows with missing values.")
    elif cfg.get("fill_mean"):
        num_cols = df_out.select_dtypes(include=np.number).columns
        for c in num_cols:
            filled = df_out[c].isnull().sum()
            if filled:
                df_out[c] = df_out[c].fillna(df_out[c].mean())
                report.append(f"Filled {filled} missing values in '{c}' with mean ({df_out[c].mean():.4f}).")

    # ── Encode categoricals ───────────────────────────────────────────────
    if cfg.get("encode_cats"):
        cat_cols = df_out[feature_cols].select_dtypes(include=["object", "category"]).columns
        for c in cat_cols:
            le = LabelEncoder()
            df_out[c] = le.fit_transform(df_out[c].astype(str))
            report.append(f"Label-encoded column '{c}'.")

    # ── Scale numeric features ────────────────────────────────────────────
    if cfg.get("scale_features"):
        scaler_type = cfg.get("scaler", "standard")
        num_feats   = [c for c in feature_cols if c in df_out.columns and
                       df_out[c].dtype in [np.float64, np.int64, float, int]]
        if num_feats:
            scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
            df_out[num_feats] = scaler.fit_transform(df_out[num_feats])
            report.append(f"Applied {scaler_type} scaling to {len(num_feats)} numeric feature(s).")

    if not report:
        report.append("No preprocessing steps applied.")

    return df_out, {"steps": report, "final_shape": df_out.shape}
