"""
pages/preprocessing.py – Step 2: Configure & apply data preprocessing.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from modules.data_utils import preprocess


def render():
    st.title("⚙️ Data Preprocessing")

    # ── Guard ────────────────────────────────────────────────────────────────
    if st.session_state.get("df") is None:
        st.warning("⬅️  Please upload a dataset first (Data Upload page).")
        return

    df           = st.session_state["df"]
    feature_cols = st.session_state.get("feature_cols", list(df.columns))
    target_col   = st.session_state.get("target_col")
    task_type    = st.session_state.get("task_type", "classification")

    if not feature_cols:
        st.warning("⬅️  Please select features on the Data Upload page.")
        return

    st.markdown(
        f"Dataset: **{st.session_state['dataset_name']}** · "
        f"Task: **{task_type}** · Features: {len(feature_cols)}"
    )
    st.markdown("---")

    # ── Config panel ─────────────────────────────────────────────────────────
    col_cfg, col_preview = st.columns([1, 2])

    with col_cfg:
        st.subheader("🔧 Options")

        missing_strategy = st.radio(
            "Missing value strategy",
            ["Keep as-is", "Drop rows with NaN", "Fill with column mean"],
            help="How to handle missing values in numeric columns.",
        )

        encode_cats = st.checkbox(
            "Encode categorical columns (Label Encoding)",
            value=True,
            help="Converts string/object columns to integers.",
        )

        scale_features = st.checkbox(
            "Scale numeric features",
            value=False,
        )

        scaler_type = "standard"
        if scale_features:
            scaler_type = st.selectbox(
                "Scaler",
                ["standard", "minmax"],
                format_func=lambda x: "StandardScaler (z-score)" if x == "standard" else "MinMaxScaler (0-1)",
            )

    # ── Build config dict ────────────────────────────────────────────────────
    cfg = {
        "drop_na":        missing_strategy == "Drop rows with NaN",
        "fill_mean":      missing_strategy == "Fill with column mean",
        "encode_cats":    encode_cats,
        "scale_features": scale_features,
        "scaler":         scaler_type,
    }
    st.session_state["preprocess_cfg"] = cfg

    # ── Live preview ─────────────────────────────────────────────────────────
    with col_preview:
        st.subheader("👁️ Preview (first 5 rows)")

        try:
            df_prev, report = preprocess(
                df,
                feature_cols[:],
                target_col,
                cfg,
            )
            st.dataframe(df_prev.head(), use_container_width=True)

            # Shape delta
            dc1, dc2 = st.columns(2)
            dc1.metric("Rows after preprocessing",  df_prev.shape[0],
                       delta=df_prev.shape[0] - df.shape[0])
            dc2.metric("Columns", df_prev.shape[1])

            with st.expander("📋 Preprocessing log"):
                for step in report["steps"]:
                    st.write(f"• {step}")

        except Exception as e:
            st.error(f"Preview error: {e}")

    st.markdown("---")

    # ── Correlation heatmap ───────────────────────────────────────────────────
    if st.checkbox("Show feature correlation heatmap", value=False):
        try:
            df_cor, _ = preprocess(df, feature_cols[:], target_col, cfg)
            num_f = [c for c in feature_cols if df_cor[c].dtype in
                     ["float64", "float32", "int64", "int32"]]
            if len(num_f) > 1:
                corr = df_cor[num_f].corr()
                fig  = px.imshow(
                    corr, text_auto=".2f",
                    color_continuous_scale="RdBu_r",
                    title="Feature Correlation Matrix",
                    aspect="auto",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need at least 2 numeric features for a correlation heatmap.")
        except Exception as e:
            st.error(f"Correlation error: {e}")

    # ── Apply button ─────────────────────────────────────────────────────────
    if st.button("✅  Apply & Save Preprocessing", type="primary"):
        with st.spinner("Processing…"):
            try:
                df_processed, report = preprocess(
                    df, feature_cols[:], target_col, cfg
                )
                st.session_state["df_processed"] = df_processed
                st.success(
                    f"✅ Preprocessing applied!  "
                    f"Final shape: {df_processed.shape[0]} rows × {df_processed.shape[1]} cols"
                )
                for step in report["steps"]:
                    st.write(f"• {step}")
            except Exception as e:
                st.error(f"Preprocessing failed: {e}")
