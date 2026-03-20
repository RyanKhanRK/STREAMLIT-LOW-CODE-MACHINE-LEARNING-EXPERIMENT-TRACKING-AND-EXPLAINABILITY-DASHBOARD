"""
pages/experiment_tracker.py – Step 4: Browse, compare & export experiment runs.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def render():
    st.title("📊 Experiment Tracker")

    experiments: list[dict] = st.session_state.get("experiments", [])

    if not experiments:
        st.info(
            "No experiments recorded yet.  "
            "Train at least one model on the **Train & Evaluate** page."
        )
        return

    # ── Build summary table ──────────────────────────────────────────────────
    rows = []
    for e in experiments:
        row = {
            "Run ID":     e.get("run_id"),
            "Model":      e.get("model_name"),
            "Task":       e.get("task_type"),
            "Dataset":    e.get("dataset_name"),
            "Timestamp":  e.get("timestamp"),
            "Time (s)":   e.get("train_time_s"),
        }
        row.update(e.get("metrics", {}))
        rows.append(row)

    df_runs = pd.DataFrame(rows)

    # ── KPI strip ────────────────────────────────────────────────────────────
    k1, k2, k3 = st.columns(3)
    k1.metric("Total runs",    len(experiments))
    k2.metric("Unique models", df_runs["Model"].nunique())
    k3.metric("Unique datasets", df_runs["Dataset"].nunique())

    st.markdown("---")

    # ── Filters ──────────────────────────────────────────────────────────────
    with st.expander("🔎 Filters", expanded=True):
        fc1, fc2, fc3 = st.columns(3)
        dataset_filter = fc1.multiselect("Dataset", df_runs["Dataset"].unique(),
                                          default=df_runs["Dataset"].unique().tolist())
        task_filter    = fc2.multiselect("Task",    df_runs["Task"].unique(),
                                          default=df_runs["Task"].unique().tolist())
        model_filter   = fc3.multiselect("Model",   df_runs["Model"].unique(),
                                          default=df_runs["Model"].unique().tolist())

    mask = (
        df_runs["Dataset"].isin(dataset_filter) &
        df_runs["Task"].isin(task_filter) &
        df_runs["Model"].isin(model_filter)
    )
    df_filtered = df_runs[mask].reset_index(drop=True)

    if df_filtered.empty:
        st.warning("No runs match the current filters.")
        return

    # ── Runs table ───────────────────────────────────────────────────────────
    st.subheader("All Runs")
    _style_table(df_filtered)

    st.markdown("---")

    # ── Metric charts ────────────────────────────────────────────────────────
    numeric_metrics = [
        c for c in df_filtered.columns
        if c not in ("Run ID", "Model", "Task", "Dataset", "Timestamp")
        and pd.api.types.is_numeric_dtype(df_filtered[c])
    ]

    if numeric_metrics:
        st.subheader("📈 Metric Comparison")

        mc1, mc2 = st.columns([1, 3])
        with mc1:
            selected_metric = st.selectbox("Metric to plot", numeric_metrics)
            chart_type      = st.radio("Chart type", ["Bar", "Box", "Line"])

        with mc2:
            if chart_type == "Bar":
                fig = px.bar(
                    df_filtered, x="Model", y=selected_metric,
                    color="Dataset", barmode="group",
                    title=f"{selected_metric} by Model",
                    text_auto=".4f",
                )
            elif chart_type == "Box":
                fig = px.box(
                    df_filtered, x="Model", y=selected_metric,
                    color="Dataset",
                    title=f"{selected_metric} Distribution",
                )
            else:
                fig = px.line(
                    df_filtered.reset_index(), x="index", y=selected_metric,
                    color="Model", markers=True,
                    title=f"{selected_metric} over Runs",
                )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Radar / spider chart (multi-metric) ──────────────────────────────────
    if len(numeric_metrics) >= 3:
        st.subheader("🕸️ Multi-Metric Radar")
        radar_metrics = st.multiselect(
            "Metrics for radar",
            numeric_metrics,
            default=numeric_metrics[:min(5, len(numeric_metrics))],
        )
        if len(radar_metrics) >= 3:
            _render_radar(df_filtered, radar_metrics)

    st.markdown("---")

    # ── Run-vs-Run comparison ────────────────────────────────────────────────
    st.subheader("⚖️ Head-to-Head Run Comparison")
    run_ids = df_filtered["Run ID"].tolist()
    if len(run_ids) >= 2:
        r_col1, r_col2 = st.columns(2)
        run_a = r_col1.selectbox("Run A", run_ids, key="cmp_a")
        run_b = r_col2.selectbox("Run B", [r for r in run_ids if r != run_a], key="cmp_b")

        row_a = df_filtered[df_filtered["Run ID"] == run_a].iloc[0]
        row_b = df_filtered[df_filtered["Run ID"] == run_b].iloc[0]

        cmp_rows = []
        for col in numeric_metrics:
            va = row_a.get(col, float("nan"))
            vb = row_b.get(col, float("nan"))
            try:
                delta = round(float(vb) - float(va), 4)
            except Exception:
                delta = None
            cmp_rows.append({
                "Metric": col,
                f"Run A ({row_a['Model']})": va,
                f"Run B ({row_b['Model']})": vb,
                "Δ (B - A)":               delta,
            })

        cmp_df = pd.DataFrame(cmp_rows)
        st.dataframe(cmp_df, use_container_width=True)
    else:
        st.info("Train at least 2 runs to enable comparison.")

    st.markdown("---")

    # ── Best run highlight ────────────────────────────────────────────────────
    st.subheader("🥇 Best Run (per primary metric)")
    task_types = df_filtered["Task"].unique()
    for task in task_types:
        df_task = df_filtered[df_filtered["Task"] == task]
        primary = "accuracy" if task == "classification" else (
                  "r2"       if task == "regression"     else "silhouette_score")
        if primary in df_task.columns and not df_task[primary].isna().all():
            best = df_task.loc[df_task[primary].idxmax()]
            st.success(
                f"**{task.title()}** – Best: **{best['Model']}**  "
                f"| {primary}: **{best[primary]}**  "
                f"| Run `{best['Run ID']}`"
            )

    # ── Export ───────────────────────────────────────────────────────────────
    st.markdown("---")
    csv = df_filtered.to_csv(index=False).encode()
    st.download_button(
        "⬇️  Export runs as CSV",
        data=csv,
        file_name="experiment_runs.csv",
        mime="text/csv",
    )

    # ── Clear all ────────────────────────────────────────────────────────────
    if st.button("🗑️  Clear all experiments", type="secondary"):
        st.session_state["experiments"]    = []
        st.session_state["trained_models"] = {}
        st.rerun()


# ── Chart helpers ─────────────────────────────────────────────────────────────

def _style_table(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        st.dataframe(
            df.style.background_gradient(subset=numeric_cols, cmap="Blues"),
            use_container_width=True,
        )
    else:
        st.dataframe(df, use_container_width=True)


def _render_radar(df: pd.DataFrame, metrics: list[str]):
    from sklearn.preprocessing import MinMaxScaler

    models = df["Model"].unique()
    fig    = go.Figure()

    # Normalise metrics 0-1 per column for fair comparison
    scaler = MinMaxScaler()
    df_n   = df[metrics].copy()
    try:
        df_n[metrics] = scaler.fit_transform(df_n[metrics].fillna(0))
    except Exception:
        pass
    df_n["Model"] = df["Model"].values

    for model in models:
        rows = df_n[df_n["Model"] == model]
        if rows.empty:
            continue
        values = rows[metrics].mean().tolist()
        values += [values[0]]  # close the polygon
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics + [metrics[0]],
            fill="toself",
            name=model,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Normalised Metric Radar",
    )
    st.plotly_chart(fig, use_container_width=True)
