"""
mlflow_utils.py – Thin wrapper around MLflow for experiment / run logging.

Falls back gracefully when MLflow tracking server is unavailable (uses local
file-based tracking in ./mlruns by default).
"""

from __future__ import annotations

import os
import warnings

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def _check():
    if not MLFLOW_AVAILABLE:
        raise ImportError("MLflow is not installed.  Run: pip install mlflow")


def setup_tracking(tracking_uri: str = "mlruns"):
    """Configure the MLflow tracking URI (local dir or remote server)."""
    _check()
    mlflow.set_tracking_uri(tracking_uri)


def log_run(result: dict, experiment_name: str | None = None) -> str | None:
    """
    Log an experiment result dict to MLflow.

    Returns the MLflow run_id string, or None on failure.
    """
    if not MLFLOW_AVAILABLE:
        return None

    try:
        name = experiment_name or result.get("dataset_name", "default_experiment")
        mlflow.set_experiment(name)

        with mlflow.start_run(run_name=result.get("model_name", "run")) as run:
            # ── Parameters ──────────────────────────────────────────────
            for k, v in result.get("hyperparams", {}).items():
                try:
                    mlflow.log_param(k, v)
                except Exception:
                    pass

            mlflow.log_param("task_type",    result.get("task_type"))
            mlflow.log_param("dataset_name", result.get("dataset_name"))
            mlflow.log_param("train_time_s", result.get("train_time_s"))

            # ── Metrics ─────────────────────────────────────────────────
            for k, v in result.get("metrics", {}).items():
                try:
                    mlflow.log_metric(k, float(v))
                except Exception:
                    pass

            # ── Model artifact ───────────────────────────────────────────
            model = result.get("model")
            if model is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        mlflow.sklearn.log_model(model, "model")
                    except Exception:
                        pass

            return run.info.run_id

    except Exception as exc:
        # Don't crash the app if MLflow isn't reachable
        print(f"[mlflow_utils] Warning: could not log run – {exc}")
        return None


def list_runs(experiment_name: str) -> list[dict]:
    """Return all runs for a given experiment as a list of dicts."""
    if not MLFLOW_AVAILABLE:
        return []
    try:
        client = mlflow.tracking.MlflowClient()
        exp    = client.get_experiment_by_name(experiment_name)
        if exp is None:
            return []
        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        return [
            {
                "mlflow_run_id": r.info.run_id,
                "model_name":    r.data.params.get("task_type", ""),
                "metrics":       r.data.metrics,
                "params":        r.data.params,
                "start_time":    r.info.start_time,
            }
            for r in runs
        ]
    except Exception:
        return []
