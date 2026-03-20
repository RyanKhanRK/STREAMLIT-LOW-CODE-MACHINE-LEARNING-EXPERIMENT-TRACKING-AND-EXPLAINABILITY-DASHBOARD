"""
pages/explainability.py – Step 5: SHAP-based model explainability.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from modules.explainability import (
    get_explainer,
    compute_shap_values,
    plot_summary,
    plot_bar,
    plot_waterfall,
    feature_importance_df,
    SHAP_AVAILABLE,
)


def render():
    st.title("🔍 Explainability (SHAP)")
    st.markdown(
        "Understand **why** your model makes predictions using SHAP "
        "(SHapley Additive exPlanations). Supports global and local interpretability."
    )

    # ── Dependency check ─────────────────────────────────────────────────────
    if not SHAP_AVAILABLE:
        st.error(
            "SHAP is not installed.  Run: `pip install shap` then restart the app."
        )
        return

    # ── Guard ────────────────────────────────────────────────────────────────
    trained = st.session_state.get("trained_models", {})
    if not trained:
        st.warning("⬅️  Please train at least one model on the **Train & Evaluate** page.")
        return

    df_proc = st.session_state.get("df_processed")
    feats   = st.session_state.get("feature_cols", [])
    if df_proc is None or not feats:
        st.warning("⬅️  No processed data found – go to the Preprocessing page first.")
        return

    # ── Model selector ───────────────────────────────────────────────────────
    left, right = st.columns([1, 2])

    with left:
        st.subheader("⚙️ Configuration")

        run_id = st.selectbox(
            "Select trained model",
            list(trained.keys()),
            format_func=lambda r: (
                f"{trained[r]['meta']['model_name']}  ({r})"
            ),
        )
        entry      = trained[run_id]
        model      = entry["model"]
        meta       = entry["meta"]
        task_type  = meta.get("task_type", "classification")

        if task_type == "clustering":
            st.info(
                "SHAP explainability is not available for clustering models."
            )
            return

        # Prepare feature matrix
        valid_feats = [c for c in feats if c in df_proc.columns]
        X_full = df_proc[valid_feats].astype(float)

        max_sample = min(500, len(X_full))
        sample_n   = st.slider(
            "Background sample size",
            min_value=20,
            max_value=max_sample,
            value=min(200, max_sample),
            help=(
                "Smaller = faster. KernelExplainer uses this as background data. "
                "TreeExplainer ignores this setting."
            ),
        )

        st.markdown("---")
        st.markdown(f"**Model:** `{meta['model_name']}`")
        st.markdown(f"**Task:** `{task_type}`")
        st.markdown(f"**Features:** {len(valid_feats)}")
        st.markdown(f"**Rows available:** {len(X_full)}")

    with right:
        st.subheader("ℹ️ What is SHAP?")
        st.markdown(
            """
            SHAP assigns each feature an importance value for a particular
            prediction, grounded in cooperative game theory.

            | Plot | What it shows |
            |------|---------------|
            | **Summary (dot)** | Each dot = one sample. X = SHAP value, colour = feature value. |
            | **Bar (global)** | Mean abs(SHAP) per feature — overall importance ranking. |
            | **Waterfall** | Single prediction breakdown: each feature's push up or down. |
            | **Dependence** | SHAP value vs raw feature value for one feature. |
            """
        )

    st.markdown("---")

    # ── Compute button ───────────────────────────────────────────────────────
    if st.button("⚡  Compute SHAP Values", type="primary"):
        X_sample = X_full.sample(sample_n, random_state=42).reset_index(drop=True)

        with st.spinner("Building explainer and computing SHAP values…"):
            try:
                explainer = get_explainer(model, X_sample, task_type)
                shap_vals = compute_shap_values(explainer, X_sample)

                st.session_state["shap_values"]   = shap_vals
                st.session_state["shap_explainer"] = explainer
                st.session_state["shap_X"]         = X_sample

                st.success(
                    f"✅  SHAP values computed for {len(X_sample)} samples "
                    f"× {len(valid_feats)} features."
                )
            except Exception as e:
                st.error(f"SHAP computation failed: {e}")
                st.info(
                    "Tip: try reducing the sample size, or use a tree-based model "
                    "(Random Forest, Gradient Boosting) for fastest results."
                )
                return

    # ── Check if values are available ────────────────────────────────────────
    shap_vals = st.session_state.get("shap_values")
    X_sample  = st.session_state.get("shap_X")

    if shap_vals is None or X_sample is None:
        st.info("👆  Click **Compute SHAP Values** to generate explainability charts.")
        return

    # Guard against stale values from a different model
    if X_sample.shape[1] != len(valid_feats):
        st.warning(
            "Cached SHAP values are from a different feature set. "
            "Please click **Compute SHAP Values** again."
        )
        return

    st.markdown("---")

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_summary, tab_bar, tab_waterfall, tab_dependence, tab_table = st.tabs(
        ["Summary (dot)", "Bar Importance", "Waterfall", "Dependence Plot", "Feature Table"]
    )

    with tab_summary:
        st.markdown("**Global feature importance** — each dot is one observation.")
        try:
            fig = plot_summary(shap_vals, X_sample, plot_type="dot")
            st.pyplot(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Summary plot error: {e}")

    with tab_bar:
        st.markdown("**Mean |SHAP| per feature** — higher = more globally influential.")
        max_disp = st.slider("Max features to display", 5, len(valid_feats), min(15, len(valid_feats)))
        try:
            fig = plot_bar(shap_vals, X_sample, max_display=max_disp)
            st.pyplot(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Bar plot error: {e}")

    with tab_waterfall:
        st.markdown(
            "**Single-prediction breakdown** — how each feature contributed to this prediction."
        )
        obs_idx = st.number_input(
            "Observation index",
            min_value=0,
            max_value=len(X_sample) - 1,
            value=0,
            help="Choose which row of the sampled data to explain.",
        )

        obs_row = X_sample.iloc[int(obs_idx)]
        st.markdown("**Selected observation:**")
        st.dataframe(obs_row.to_frame(name="Value").T, use_container_width=True)

        try:
            fig = plot_waterfall(
                st.session_state["shap_explainer"],
                X_sample,
                idx=int(obs_idx),
            )
            st.pyplot(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Waterfall plot error: {e}")

    with tab_dependence:
        st.markdown(
            "**Dependence plot** — SHAP value vs raw feature value for one feature. "
            "Reveals non-linear relationships and interaction effects."
        )
        dep_feat = st.selectbox("Feature", valid_feats, key="dep_feat_sel")
        colour_feat = st.selectbox(
            "Colour by (interaction)",
            ["None"] + [f for f in valid_feats if f != dep_feat],
            key="dep_color_sel",
        )

        feat_idx = valid_feats.index(dep_feat)
        shap_col = shap_vals[:, feat_idx]
        raw_col  = X_sample[dep_feat].values

        dep_df = pd.DataFrame({"Feature value": raw_col, "SHAP value": shap_col})

        if colour_feat != "None":
            dep_df["Colour"] = X_sample[colour_feat].values
            fig = px.scatter(
                dep_df, x="Feature value", y="SHAP value",
                color="Colour",
                color_continuous_scale="RdBu",
                title=f"SHAP Dependence: {dep_feat}  (coloured by {colour_feat})",
                opacity=0.7,
            )
        else:
            fig = px.scatter(
                dep_df, x="Feature value", y="SHAP value",
                title=f"SHAP Dependence: {dep_feat}",
                opacity=0.7,
                trendline="lowess",
            )

        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

    with tab_table:
        st.markdown("**Ranked feature importance table** (mean |SHAP| across all samples).")
        fi_df = feature_importance_df(shap_vals, valid_feats)

        # Add a small bar chart column via plotly
        fig = px.bar(
            fi_df.head(20),
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            title="Top Features by Mean |SHAP|",
            color="mean_abs_shap",
            color_continuous_scale="Blues",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(fi_df, use_container_width=True)

        csv = fi_df.to_csv(index=False).encode()
        st.download_button(
            "⬇️  Export importance table",
            data=csv,
            file_name="shap_feature_importance.csv",
            mime="text/csv",
        )
