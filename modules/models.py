"""
models.py – Model registry, hyperparameter defaults, training & evaluation.
"""

from __future__ import annotations

import uuid
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix,
)

# Classification
from sklearn.linear_model      import LogisticRegression
from sklearn.tree              import DecisionTreeClassifier
from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm               import SVC
from sklearn.neighbors         import KNeighborsClassifier
from sklearn.naive_bayes       import GaussianNB

# Regression
from sklearn.linear_model      import LinearRegression, Ridge, Lasso
from sklearn.tree              import DecisionTreeRegressor
from sklearn.ensemble          import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm               import SVR

# Clustering
from sklearn.cluster           import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics           import silhouette_score, davies_bouldin_score


# ── Registries ───────────────────────────────────────────────────────────────

CLASSIFICATION_MODELS = {
    "Logistic Regression": {
        "cls": LogisticRegression,
        "defaults": {"C": 1.0, "max_iter": 200, "solver": "lbfgs"},
        "params": {
            "C":        {"type": "float",  "min": 0.01, "max": 10.0, "step": 0.01, "help": "Inverse regularisation strength"},
            "max_iter": {"type": "int",    "min": 50,   "max": 1000, "step": 50,   "help": "Maximum iterations"},
        },
    },
    "Decision Tree": {
        "cls": DecisionTreeClassifier,
        "defaults": {"max_depth": 5, "min_samples_split": 2},
        "params": {
            "max_depth":         {"type": "int", "min": 1, "max": 30, "step": 1, "help": "Maximum tree depth"},
            "min_samples_split": {"type": "int", "min": 2, "max": 20, "step": 1, "help": "Min samples to split a node"},
        },
    },
    "Random Forest": {
        "cls": RandomForestClassifier,
        "defaults": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
        "params": {
            "n_estimators": {"type": "int",   "min": 10,  "max": 500, "step": 10,  "help": "Number of trees"},
            "max_depth":    {"type": "int",   "min": 1,   "max": 30,  "step": 1,   "help": "Maximum tree depth"},
        },
    },
    "Gradient Boosting": {
        "cls": GradientBoostingClassifier,
        "defaults": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
        "params": {
            "n_estimators":  {"type": "int",   "min": 10, "max": 300, "step": 10,   "help": "Boosting stages"},
            "learning_rate": {"type": "float", "min": 0.001, "max": 1.0, "step": 0.01, "help": "Learning rate"},
            "max_depth":     {"type": "int",   "min": 1, "max": 10, "step": 1, "help": "Max tree depth"},
        },
    },
    "SVM": {
        "cls": SVC,
        "defaults": {"C": 1.0, "kernel": "rbf", "probability": True},
        "params": {
            "C": {"type": "float", "min": 0.01, "max": 10.0, "step": 0.01, "help": "Regularisation parameter"},
        },
    },
    "K-Nearest Neighbours": {
        "cls": KNeighborsClassifier,
        "defaults": {"n_neighbors": 5},
        "params": {
            "n_neighbors": {"type": "int", "min": 1, "max": 30, "step": 1, "help": "Number of neighbours"},
        },
    },
    "Naive Bayes": {
        "cls": GaussianNB,
        "defaults": {},
        "params": {},
    },
}

REGRESSION_MODELS = {
    "Linear Regression": {
        "cls": LinearRegression,
        "defaults": {},
        "params": {},
    },
    "Ridge Regression": {
        "cls": Ridge,
        "defaults": {"alpha": 1.0},
        "params": {
            "alpha": {"type": "float", "min": 0.001, "max": 100.0, "step": 0.1, "help": "Regularisation strength"},
        },
    },
    "Lasso Regression": {
        "cls": Lasso,
        "defaults": {"alpha": 1.0},
        "params": {
            "alpha": {"type": "float", "min": 0.001, "max": 100.0, "step": 0.1, "help": "Regularisation strength"},
        },
    },
    "Decision Tree": {
        "cls": DecisionTreeRegressor,
        "defaults": {"max_depth": 5},
        "params": {
            "max_depth": {"type": "int", "min": 1, "max": 30, "step": 1, "help": "Maximum tree depth"},
        },
    },
    "Random Forest": {
        "cls": RandomForestRegressor,
        "defaults": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
        "params": {
            "n_estimators": {"type": "int", "min": 10, "max": 500, "step": 10, "help": "Number of trees"},
            "max_depth":    {"type": "int", "min": 1,  "max": 30,  "step": 1,  "help": "Maximum tree depth"},
        },
    },
    "Gradient Boosting": {
        "cls": GradientBoostingRegressor,
        "defaults": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
        "params": {
            "n_estimators":  {"type": "int",   "min": 10,    "max": 300,  "step": 10,   "help": "Boosting stages"},
            "learning_rate": {"type": "float", "min": 0.001, "max": 1.0,  "step": 0.01, "help": "Learning rate"},
        },
    },
    "SVR": {
        "cls": SVR,
        "defaults": {"C": 1.0, "kernel": "rbf"},
        "params": {
            "C": {"type": "float", "min": 0.01, "max": 10.0, "step": 0.01, "help": "Regularisation parameter"},
        },
    },
}

CLUSTERING_MODELS = {
    "K-Means": {
        "cls": KMeans,
        "defaults": {"n_clusters": 3, "random_state": 42, "n_init": 10},
        "params": {
            "n_clusters": {"type": "int", "min": 2, "max": 15, "step": 1, "help": "Number of clusters"},
        },
    },
    "DBSCAN": {
        "cls": DBSCAN,
        "defaults": {"eps": 0.5, "min_samples": 5},
        "params": {
            "eps":         {"type": "float", "min": 0.01, "max": 5.0, "step": 0.01, "help": "Neighbourhood radius"},
            "min_samples": {"type": "int",   "min": 1,   "max": 20,  "step": 1,    "help": "Min samples in neighbourhood"},
        },
    },
    "Agglomerative": {
        "cls": AgglomerativeClustering,
        "defaults": {"n_clusters": 3},
        "params": {
            "n_clusters": {"type": "int", "min": 2, "max": 15, "step": 1, "help": "Number of clusters"},
        },
    },
}


def get_model_registry(task_type: str) -> dict:
    return {
        "classification": CLASSIFICATION_MODELS,
        "regression":     REGRESSION_MODELS,
        "clustering":     CLUSTERING_MODELS,
    }.get(task_type, {})


# ── Training ──────────────────────────────────────────────────────────────────

def train_and_evaluate(
    model_name:   str,
    task_type:    str,
    hyperparams:  dict,
    X:            np.ndarray | pd.DataFrame,
    y:            np.ndarray | pd.Series | None,
    test_size:    float = 0.2,
    cv_folds:     int   = 5,
    dataset_name: str   = "dataset",
) -> dict:
    """
    Train a model and return a full experiment-result dict.

    Returned dict keys
    ------------------
    run_id, model_name, task_type, dataset_name, hyperparams,
    metrics, model, X_test, y_test, y_pred, train_time_s, timestamp
    """
    registry = get_model_registry(task_type)
    if model_name not in registry:
        raise ValueError(f"Unknown model '{model_name}' for task '{task_type}'.")

    cls    = registry[model_name]["cls"]
    params = {**registry[model_name]["defaults"], **hyperparams}

    t0    = time.time()
    model = cls(**params)

    result: dict = {
        "run_id":       str(uuid.uuid4())[:8],
        "model_name":   model_name,
        "task_type":    task_type,
        "dataset_name": dataset_name,
        "hyperparams":  params,
        "timestamp":    pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ── Clustering (unsupervised) ────────────────────────────────────────
    if task_type == "clustering":
        labels = model.fit_predict(X)
        result["train_time_s"] = round(time.time() - t0, 4)
        result["model"]        = model
        result["X_train"]      = X
        result["labels"]       = labels
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        metrics: dict = {"n_clusters_found": n_clusters}
        if n_clusters > 1:
            try:
                metrics["silhouette_score"]     = round(silhouette_score(X, labels), 4)
                metrics["davies_bouldin_score"] = round(davies_bouldin_score(X, labels), 4)
            except Exception:
                pass
        result["metrics"] = metrics
        return result

    # ── Supervised ──────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    result["train_time_s"] = round(time.time() - t0, 4)
    result["model"]  = model
    result["X_test"] = X_test
    result["y_test"] = y_test
    result["y_pred"] = y_pred

    if task_type == "classification":
        cv = cross_val_score(model, X, y, cv=cv_folds, scoring="accuracy")
        metrics = {
            "accuracy":        round(accuracy_score(y_test, y_pred), 4),
            "f1_weighted":     round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "precision":       round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "recall":          round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "cv_accuracy_mean": round(cv.mean(), 4),
            "cv_accuracy_std":  round(cv.std(), 4),
        }
        try:
            if hasattr(model, "predict_proba"):
                classes = np.unique(y)
                if len(classes) == 2:
                    proba = model.predict_proba(X_test)[:, 1]
                    metrics["roc_auc"] = round(roc_auc_score(y_test, proba), 4)
        except Exception:
            pass
        result["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()

    else:  # regression
        cv = cross_val_score(model, X, y, cv=cv_folds, scoring="r2")
        metrics = {
            "r2":             round(r2_score(y_test, y_pred), 4),
            "rmse":           round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
            "mae":            round(mean_absolute_error(y_test, y_pred), 4),
            "cv_r2_mean":     round(cv.mean(), 4),
            "cv_r2_std":      round(cv.std(), 4),
        }

    result["metrics"] = metrics
    return result
