"""Stage 2 - model engineering: features, training, evaluation, packaging.

Input : data/processed/train.csv, data/processed/test.csv
Output: models/model.pkl, models/metrics.json, models/model_meta.json
        plus params/metrics/model logged to MLflow.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.append(str(Path(__file__).resolve().parent))
from features import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_pipeline,
)

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODELS_DIR / "model.pkl"
METRICS_FILE = MODELS_DIR / "metrics.json"
META_FILE = MODELS_DIR / "model_meta.json"

EXPERIMENT_NAME = "pmldl-penguins"
MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 8,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1,
}


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    for path in (train_path, test_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing - run the data engineering stage first"
            )
    return pd.read_csv(train_path), pd.read_csv(test_path)


def evaluate(pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = pipeline.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "precision_macro": float(
            precision_score(y_test, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_test, y_pred, average="macro", zero_division=0)
        ),
    }
    log.info("test metrics: %s", json.dumps(metrics))
    log.info(
        "classification report:\n%s",
        classification_report(y_test, y_pred, zero_division=0),
    )
    return metrics


def _log_sklearn_model(mlflow, pipeline) -> None:
    """Log the fitted pipeline, tolerating the API differences across MLflow versions.

    MLflow 3 serialises sklearn models with skops and needs the custom
    transformer declared as trusted; MLflow 2 uses ``artifact_path`` instead of
    ``name`` and knows neither keyword.
    """
    # Types skops has to be told are safe to load: our own transformer and the
    # compiled tree structure behind RandomForestClassifier.
    trusted = ["features.DerivedFeatures", "sklearn.tree._tree.Tree"]
    attempts = (
        {"name": "model", "skops_trusted_types": trusted},
        {"name": "model"},
        {"artifact_path": "model"},
    )
    for kwargs in attempts:
        try:
            mlflow.sklearn.log_model(pipeline, **kwargs)
            return
        except TypeError:
            continue  # this MLflow version does not know that keyword
        except Exception as exc:  # noqa: BLE001 - params/metrics are worth keeping
            log.warning("could not log the model artifact: %s", exc)
            return
    log.warning("could not log the model with any known MLflow signature")


def log_to_mlflow(pipeline, metrics: dict, train_rows: int, test_rows: int) -> None:
    """Log params, metrics and the model. Never fail the pipeline on MLflow issues."""
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        log.warning("mlflow is not installed - skipping experiment tracking")
        return

    # Inside Docker this points at the MLflow server; locally it falls back to a
    # SQLite store next to the project (the file store is deprecated in MLflow 3).
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or (
        f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
    )
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(EXPERIMENT_NAME)
        run_name = datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(MODEL_PARAMS)
            mlflow.log_param("model_type", "RandomForestClassifier")
            mlflow.log_param("train_rows", train_rows)
            mlflow.log_param("test_rows", test_rows)
            mlflow.log_metrics(metrics)
            _log_sklearn_model(mlflow, pipeline)
        log.info("logged run %s to MLflow at %s", run_name, tracking_uri)
    except Exception as exc:  # noqa: BLE001 - tracking must not break the pipeline
        log.warning("MLflow logging failed (%s): %s", type(exc).__name__, exc)


def run_model_engineering() -> dict:
    """Entry point used by the Airflow DAG."""
    train_df, test_df = load_splits()
    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    pipeline = build_pipeline(RandomForestClassifier(**MODEL_PARAMS))
    pipeline.fit(X_train, y_train)
    log.info("trained on %d rows", len(X_train))

    metrics = evaluate(pipeline, X_test, y_test)
    log_to_mlflow(pipeline, metrics, len(X_train), len(X_test))

    # Packaging: the fitted pipeline plus everything the API/app need to know.
    #
    # The standard-library pickle, not joblib: this stage runs inside an Airflow
    # worker, where dill (an Airflow dependency) patches the pure-Python pickler
    # that joblib uses and writes dill._dill references into the artifact. The API
    # image has no dill and could not load such a file. pickle.dump uses the C
    # pickler, which dill leaves alone, so model.pkl loads anywhere scikit-learn is.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with MODEL_FILE.open("wb") as fh:
        pickle.dump(pipeline, fh, protocol=pickle.HIGHEST_PROTOCOL)

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_type": "RandomForestClassifier",
        "target": TARGET,
        "classes": sorted(pipeline.classes_.tolist()),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "categories": {
            col: sorted(train_df[col].dropna().unique().tolist())
            for col in CATEGORICAL_FEATURES
        },
        "numeric_ranges": {
            col: {
                "min": float(train_df[col].min()),
                "max": float(train_df[col].max()),
                "mean": float(train_df[col].mean()),
            }
            for col in NUMERIC_FEATURES
        },
        "metrics": metrics,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
    }
    METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    META_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("saved model to %s", MODEL_FILE)
    return {"model_file": str(MODEL_FILE), **metrics}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(json.dumps(run_model_engineering(), indent=2))
