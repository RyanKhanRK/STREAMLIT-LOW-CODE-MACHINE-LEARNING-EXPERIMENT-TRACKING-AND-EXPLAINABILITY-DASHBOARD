"""
explainability.py – SHAP-based model explainability helpers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def check_shap():
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not installed.  Run: pip install shap")


def get_explainer(model, X: pd.DataFrame | np.ndarray, task_type: str = "classification"):
    """
    Return the most appropriate SHAP Explainer for *model*.
    Uses TreeExplainer for tree-based models, LinearExplainer for linear
    models, and KernelExplainer (sampled) as a fallback.
    """
    check_shap()

    model_cls = type(model).__name__

    tree_classes = {
        "RandomForestClassifier", "RandomForestRegressor",
        "GradientBoostingClassifier", "GradientBoostingRegressor",
        "DecisionTreeClassifier", "DecisionTreeRegressor",
        "ExtraTreesClassifier", "ExtraTreesRegressor",
    }
    linear_classes = {
        "LogisticRegression", "LinearRegression",
        "Ridge", "Lasso", "ElasticNet",
    }

    if model_cls in tree_classes:
        explainer = shap.TreeExplainer(model)
    elif model_cls in linear_classes:
        X_bg = shap.maskers.Independent(X, max_samples=min(100, len(X)))
        explainer = shap.LinearExplainer(model, X_bg)
    else:
        # Kernel explainer – sample background data for speed
        background = shap.sample(X, min(50, len(X)))
        def predict_fn(x):
            if task_type == "classification" and hasattr(model, "predict_proba"):
                return model.predict_proba(x)
            return model.predict(x)
        explainer = shap.KernelExplainer(predict_fn, background)

    return explainer


def compute_shap_values(explainer, X: pd.DataFrame | np.ndarray):
    """Compute SHAP values; handles multi-output by taking class-1 slice."""
    check_shap()
    vals = explainer.shap_values(X)
    # For binary classifiers TreeExplainer returns a list [neg_class, pos_class]
    if isinstance(vals, list):
        vals = vals[1] if len(vals) == 2 else vals[0]
    return vals


def plot_summary(shap_values: np.ndarray, X: pd.DataFrame, plot_type: str = "dot"):
    """Return a matplotlib Figure of the SHAP summary plot."""
    check_shap()
    fig, ax = plt.subplots(figsize=(9, max(4, min(len(X.columns) * 0.4, 10))))
    shap.summary_plot(shap_values, X, plot_type=plot_type, show=False)
    fig = plt.gcf()
    plt.tight_layout()
    return fig


def plot_bar(shap_values: np.ndarray, X: pd.DataFrame, max_display: int = 15):
    """Return a matplotlib Figure of the SHAP bar (mean |SHAP|) plot."""
    check_shap()
    shap.summary_plot(shap_values, X, plot_type="bar",
                      max_display=max_display, show=False)
    fig = plt.gcf()
    plt.tight_layout()
    return fig


def plot_waterfall(explainer, X: pd.DataFrame | np.ndarray, idx: int = 0):
    """Return a waterfall plot for a single observation at *idx*."""
    check_shap()
    try:
        expl_obj = explainer(X)
        # Multi-output → take positive class
        if expl_obj.values.ndim == 3:
            vals   = expl_obj.values[idx, :, 1]
            base   = expl_obj.base_values[idx, 1]
        else:
            vals   = expl_obj.values[idx]
            base   = expl_obj.base_values[idx] if np.ndim(expl_obj.base_values) > 0 else expl_obj.expected_value

        exp = shap.Explanation(
            values      = vals,
            base_values = base,
            data        = X.iloc[idx] if hasattr(X, "iloc") else X[idx],
            feature_names = list(X.columns) if hasattr(X, "columns") else None,
        )
        shap.waterfall_plot(exp, show=False)
        fig = plt.gcf()
        plt.tight_layout()
        return fig
    except Exception as e:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"Waterfall unavailable:\n{e}", ha="center", va="center")
        return fig


def feature_importance_df(shap_values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Return a DataFrame with mean |SHAP| importance per feature."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
