"""
pages/model_train.py – Step 3: Configure, train and evaluate models.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from modules.models      import get_model_registry, train_and_evaluate
from modules.data_utils  import preprocess
from modules.mlflow_utils import log_run, setup_tracking


def render():
    st.title("🤖 Train & Evaluate Models")

    # ── Guards ───────────────────────────────────────────────────────────────
    if st.session_state.get("df") is None:
        st.warning("⬅️  Please upload a dataset first.")
        return

    df_proc = st.session_state.get("df_processed")
    if df_proc is None:
        st.info("⚠️  No preprocessed data found – running with defaults.")
        try:
            df_proc, _ = preprocess(
                st.session_state["df"],
                st.session_state.get("feature_cols", []),
                st.session_state.get("target_col"),
                st.session_state.get("preprocess_cfg", {}),
            )
        except Exception as e:
            st.error(f"Could not auto-preprocess: {e}")
            return

    task_type    = st.session_state.get("task_type", "classification")
    feature_cols = st.session_state.get("feature_cols", [])
    target_col   = st.session_state.get("target_col")
    dataset_name = st.session_state.get("dataset_name", "dataset")

    registry = get_model_registry(task_type)
    if not registry:
        st.error(f"No models registered for task '{task_type}'.")
        return

    # ── Prepare X / y ────────────────────────────────────────────────────────
    try:
        valid_feats = [c for c in feature_cols if c in df_proc.columns]
        X = df_proc[valid_feats].values.astype(float)
        y = df_proc[target_col].values if target_col and target_col in df_proc.columns else None
    except Exception as e:
        st.error(f"Could not extract feature matrix: {e}")
        return

    st.markdown(
        f"Dataset: **{dataset_name}** · Task: **{task_type}** · "
        f"Shape: {X.shape[0]} × {X.shape[1]}"
    )
    st.markdown("---")

    # ── Model selector ───────────────────────────────────────────────────────
    left, right = st.columns([1, 2])

    with left:
        st.subheader("Model Selection")

        run_mode = st.radio(
            "Training mode",
            ["Single model", "Compare all models"],
            help="Compare all trains every available model in one click.",
        )

        selected_models = []
        if run_mode == "Single model":
            m = st.selectbox("Choose a model", list(registry.keys()))
            selected_models = [m]
        else:
            selected_models = list(registry.keys())
            st.write(f"Will train {len(selected_models)} models.")

        st.subheader("Training options")
        test_size = st.slider("Test split (%)", 10, 40, 20) / 100
        cv_folds  = st.slider("CV folds", 2, 10, 5)

        use_mlflow = st.checkbox("Log to MLflow", value=True)
        if use_mlflow:
            tracking_uri = st.text_input("Tracking URI", value="mlruns",
                                         help="Local path or remote server URL.")

    with right:
        st.subheader("Hyperparameters")

        all_hyperparams: dict[str, dict] = {}

        if run_mode == "Single model":
            _render_hyperparam_ui(selected_models[0], registry, all_hyperparams)
        else:
            st.info(
                "Batch mode uses each model's **default** hyperparameters. "
                "Switch to Single model to customise."
            )
            for m in selected_models:
                all_hyperparams[m] = {}  # use defaults

    st.markdown("---")

    # ── Train button ─────────────────────────────────────────────────────────
    if st.button("🚀  Run Training", type="primary"):
        if use_mlflow:
            try:
                setup_tracking(tracking_uri)
            except Exception:
                pass

        progress = st.progress(0, text="Starting…")
        results  = []
        errors   = []

        for i, model_name in enumerate(selected_models):
            progress.progress(
                int((i / len(selected_models)) * 100),
                text=f"Training {model_name}…",
            )
            try:
                hp = all_hyperparams.get(model_name, {})
                res = train_and_evaluate(
                    model_name, task_type, hp, X, y,
                    test_size=test_size, cv_folds=cv_folds,
                    dataset_name=dataset_name,
                )
                # MLflow logging
                if use_mlflow:
                    res["mlflow_run_id"] = log_run(res, experiment_name=dataset_name)

                results.append(res)
                # Persist in session
                st.session_state["experiments"].append(res)
                st.session_state["trained_models"][res["run_id"]] = {
                    "model": res["model"],
                    "meta":  res,
                }
            except Exception as e:
                errors.append(f"{model_name}: {e}")

        progress.progress(100, text="Done!")

        if errors:
            for err in errors:
                st.error(err)

        if results:
            st.session_state["last_run_id"] = results[-1]["run_id"]
            st.success(f"✅  Trained {len(results)} model(s).")
            _render_results(results, task_type, valid_feats)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _render_hyperparam_ui(model_name: str, registry: dict, sink: dict):
    """Render dynamic hyperparameter sliders and store chosen values in *sink*."""
    params_meta = registry[model_name].get("params", {})
    defaults    = registry[model_name].get("defaults", {})

    if not params_meta:
        st.caption(f"{model_name} has no tunable hyperparameters.")
        sink[model_name] = {}
        return

    chosen = {}
    for param, meta in params_meta.items():
        default = defaults.get(param, meta.get("min", 1))
        if meta["type"] == "int":
            chosen[param] = st.slider(
                f"{param}",
                min_value=meta["min"], max_value=meta["max"],
                value=int(default), step=meta.get("step", 1),
                help=meta.get("help", ""),
            )
        else:
            chosen[param] = st.slider(
                f"{param}",
                min_value=float(meta["min"]), max_value=float(meta["max"]),
                value=float(default), step=float(meta.get("step", 0.01)),
                help=meta.get("help", ""),
            )

    sink[model_name] = chosen


def _render_results(results: list[dict], task_type: str, feature_names: list[str]):
    """Display training results inline after a run."""
    st.markdown("---")
    st.subheader("📊 Training Results")

    for res in results:
        with st.expander(
            f"**{res['model_name']}**  —  "
            f"Run ID `{res['run_id']}`  |  "
            f"⏱ {res['train_time_s']}s",
            expanded=len(results) == 1,
        ):
            metrics = res.get("metrics", {})
            cols = st.columns(len(metrics))
            for col, (k, v) in zip(cols, metrics.items()):
                col.metric(k.replace("_", " ").title(), v)

            # Classification: confusion matrix
            if task_type == "classification" and "confusion_matrix" in res:
                cm  = np.array(res["confusion_matrix"])
                labels = sorted(set(res["y_test"]))
                fig = px.imshow(
                    cm, text_auto=True,
                    x=[str(l) for l in labels],
                    y=[str(l) for l in labels],
                    color_continuous_scale="Blues",
                    title="Confusion Matrix",
                    labels={"x": "Predicted", "y": "Actual"},
                )
                st.plotly_chart(fig, use_container_width=True)

            # Regression: actual vs predicted scatter
            if task_type == "regression" and "y_test" in res:
                fig = px.scatter(
                    x=res["y_test"], y=res["y_pred"],
                    labels={"x": "Actual", "y": "Predicted"},
                    title="Actual vs Predicted",
                    opacity=0.6,
                )
                fig.add_shape(
                    type="line",
                    x0=res["y_test"].min(), y0=res["y_test"].min(),
                    x1=res["y_test"].max(), y1=res["y_test"].max(),
                    line=dict(color="red", dash="dash"),
                )
                st.plotly_chart(fig, use_container_width=True)

            # Feature importances (tree-based)
            model = res.get("model")
            if hasattr(model, "feature_importances_") and feature_names:
                fi = pd.DataFrame({
                    "feature": feature_names,
                    "importance": model.feature_importances_,
                }).sort_values("importance", ascending=True).tail(15)
                fig = px.bar(fi, x="importance", y="feature", orientation="h",
                             title="Feature Importances")
                st.plotly_chart(fig, use_container_width=True)

    # ── Comparison table (if multiple) ───────────────────────────────────────
    if len(results) > 1:
        st.subheader("🏆 Model Comparison")
        rows = []
        for r in results:
            row = {"Model": r["model_name"], "Run ID": r["run_id"],
                   "Time (s)": r["train_time_s"]}
            row.update(r.get("metrics", {}))
            rows.append(row)
        df_cmp = pd.DataFrame(rows)

        # Highlight best
        primary = "accuracy" if task_type == "classification" else "r2"
        if primary in df_cmp.columns:
            best_idx = df_cmp[primary].idxmax()
            st.dataframe(
                df_cmp.style.highlight_max(subset=[primary], color="#166534"),
                use_container_width=True,
            )
            st.success(
                f"🥇 Best model: **{df_cmp.loc[best_idx, 'Model']}** "
                f"({primary}={df_cmp.loc[best_idx, primary]})"
            )
        else:
            st.dataframe(df_cmp, use_container_width=True)
