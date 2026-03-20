"""
session.py – Centralised session-state initialisation.

All keys used across the app are declared here so every module can safely
read them without KeyError.
"""

import streamlit as st


DEFAULTS = {
    # ── Data ────────────────────────────────────────────────────────────────
    "df":              None,   # raw uploaded DataFrame
    "df_processed":    None,   # preprocessed DataFrame
    "dataset_name":    "",     # original filename (no extension)
    "target_col":      None,   # name of the target column
    "feature_cols":    [],     # list of selected feature column names
    "task_type":       None,   # "classification" | "regression" | "clustering"

    # ── Preprocessing ───────────────────────────────────────────────────────
    "preprocess_cfg": {
        "drop_na":        False,
        "fill_mean":      False,
        "encode_cats":    True,
        "scale_features": False,
    },

    # ── Training ────────────────────────────────────────────────────────────
    "trained_models":  {},     # {run_id: {"model": obj, "meta": dict}}
    "experiments":     [],     # list of experiment-result dicts (for tracker)
    "last_run_id":     None,

    # ── Explainability ──────────────────────────────────────────────────────
    "shap_values":     None,
    "shap_explainer":  None,
    "shap_X":          None,
}


def init_session():
    for key, default in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default
