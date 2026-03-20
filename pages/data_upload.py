"""
pages/data_upload.py – Step 1: Upload & explore a CSV dataset.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from modules.data_utils import load_csv, analyze_dataset, infer_task_type


def render():
    st.title("📂 Data Upload & Exploration")
    st.markdown(
        "Upload any **CSV dataset** to begin. The dashboard will auto-analyse "
        "column types, missing values, and suggest a task type."
    )

    # ── Upload widget ────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Drop a CSV file here or click to browse",
        type=["csv", "tsv"],
        help="Supported formats: CSV, TSV",
    )

    if uploaded is None:
        st.info("⬆️  Upload a dataset to continue.")
        _show_sample_datasets()
        return

    # ── Load ────────────────────────────────────────────────────────────────
    with st.spinner("Reading dataset…"):
        try:
            df = load_csv(uploaded)
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            return

    st.session_state["df"]           = df
    st.session_state["dataset_name"] = uploaded.name.rsplit(".", 1)[0]

    info = analyze_dataset(df)

    # ── KPI strip ────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows",          info["rows"])
    c2.metric("Columns",       info["cols"])
    c3.metric("Numeric cols",  len(info["numeric_cols"]))
    c4.metric("Categorical",   len(info["cat_cols"]))
    c5.metric("Missing cells", f"{info['missing_count']} ({info['missing_pct']}%)")

    st.markdown("---")

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_preview, tab_stats, tab_dtypes, tab_missing, tab_distributions = st.tabs(
        ["Preview", "Statistics", "Column Types", "Missing Values", "Distributions"]
    )

    with tab_preview:
        st.markdown(f"**First {min(100, len(df))} rows**")
        st.dataframe(df.head(100), use_container_width=True)

    with tab_stats:
        desc = df.describe(include="all").T
        st.dataframe(desc, use_container_width=True)

    with tab_dtypes:
        dtype_df = pd.DataFrame({
            "Column": df.columns,
            "Dtype":  df.dtypes.values.astype(str),
            "Unique": [df[c].nunique() for c in df.columns],
            "Sample": [str(df[c].dropna().iloc[0]) if not df[c].dropna().empty else "—"
                       for c in df.columns],
        })
        st.dataframe(dtype_df, use_container_width=True)

    with tab_missing:
        miss = pd.DataFrame({
            "Column":  df.columns,
            "Missing": df.isnull().sum().values,
            "Pct (%)": (df.isnull().sum().values / len(df) * 100).round(2),
        }).query("Missing > 0")
        if miss.empty:
            st.success("✅ No missing values found!")
        else:
            fig = px.bar(
                miss, x="Column", y="Pct (%)",
                title="Missing Value % per Column",
                color="Pct (%)", color_continuous_scale="Reds",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(miss, use_container_width=True)

    with tab_distributions:
        num_cols = info["numeric_cols"]
        if not num_cols:
            st.warning("No numeric columns to plot.")
        else:
            sel = st.selectbox("Select numeric column", num_cols)
            fig = px.histogram(df, x=sel, nbins=40, marginal="box",
                               title=f"Distribution of '{sel}'")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Column role selection ─────────────────────────────────────────────────
    st.subheader("🎯 Define Target & Features")

    col_a, col_b = st.columns([1, 2])

    with col_a:
        task_type = st.selectbox(
            "Task type",
            ["classification", "regression", "clustering"],
            index=0,
            help="Auto-detected based on target column—change if needed.",
        )

        if task_type != "clustering":
            target_col = st.selectbox("Target column", df.columns.tolist())
            auto_task  = infer_task_type(df, target_col)
            if auto_task != task_type:
                st.caption(f"ℹ️ Auto-detected task type: **{auto_task}**")
        else:
            target_col = None
            st.info("Clustering is unsupervised – no target column needed.")

    with col_b:
        exclude = [target_col] if target_col else []
        feature_cols = st.multiselect(
            "Feature columns",
            [c for c in df.columns if c not in exclude],
            default=[c for c in df.columns if c not in exclude],
        )

    if st.button("✅  Confirm Column Selection", type="primary"):
        if not feature_cols:
            st.error("Please select at least one feature column.")
            return

        st.session_state["target_col"]   = target_col
        st.session_state["feature_cols"] = feature_cols
        st.session_state["task_type"]    = task_type
        # Reset downstream state
        st.session_state["df_processed"] = None
        st.session_state["experiments"]  = st.session_state.get("experiments", [])
        st.success(
            f"✅  Saved: {len(feature_cols)} features · "
            f"Target: `{target_col}` · Task: **{task_type}**"
        )


def _show_sample_datasets():
    """Show quick info about what kinds of datasets work well."""
    with st.expander("💡 What kind of datasets can I use?"):
        st.markdown(
            """
            - Any **CSV** or **TSV** file with a header row.
            - Works with **classification**, **regression**, or **clustering** tasks.
            - Missing values are handled in the Preprocessing step.
            - Categorical columns are automatically encoded before training.

            **Example datasets to try:**
            - Iris / Wine / Breast Cancer (classification)
            - Boston Housing / Diamonds (regression)
            - Customer Segmentation (clustering)
            """
        )
