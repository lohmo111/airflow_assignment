"""End-to-end MLOps pipeline: data -> model -> deployment, every 5 minutes.

Stage 1  data_engineering  : load, clean, split          (services/airflow/dags/data_engineering.py)
Stage 2  model_engineering : features, train, evaluate   (code/models/train.py)
Stage 3  deployment        : docker compose build + up   (code/deployment/docker-compose.yml)
         smoke_test        : call the running API and check it answers
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import pendulum
from airflow import DAG
from airflow.exceptions import AirflowException
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/opt/project"))
COMPOSE_FILE = PROJECT_ROOT / "code" / "deployment" / "docker-compose.yml"
COMPOSE_PROJECT = "pmldl-deploy"
# The API port is published on the host, so the Airflow container reaches it
# through the host gateway rather than through the deployment network.
API_SMOKE_URL = os.environ.get("API_SMOKE_URL", "http://host.docker.internal:8000")

SAMPLE_REQUEST = {
    "bill_length_mm": 44.5,
    "bill_depth_mm": 17.3,
    "flipper_length_mm": 197.0,
    "body_mass_g": 4200.0,
    "island": "Biscoe",
    "sex": "MALE",
}


def stage_data_engineering(**_) -> dict:
    # Imported lazily: heavy imports must not slow down DAG parsing.
    from data_engineering import run_data_engineering

    summary = run_data_engineering()
    log.info("stage 1 summary: %s", json.dumps(summary, indent=2))
    return summary


def stage_model_engineering(**_) -> dict:
    models_code = str(PROJECT_ROOT / "code" / "models")
    if models_code not in sys.path:
        sys.path.insert(0, models_code)
    from train import run_model_engineering

    summary = run_model_engineering()
    log.info("stage 2 summary: %s", json.dumps(summary, indent=2))
    return summary


def stage_smoke_test(**_) -> dict:
    """Wait for the freshly (re)deployed API and verify a real prediction."""
    import requests

    deadline = time.time() + 180
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            health = requests.get(f"{API_SMOKE_URL}/health", timeout=5)
            if health.ok and health.json().get("status") == "ok":
                break
            last_error = RuntimeError(f"health returned {health.status_code}")
        except Exception as exc:  # noqa: BLE001 - the container may still be starting
            last_error = exc
        log.info("API not ready yet (%s), retrying...", last_error)
        time.sleep(5)
    else:
        raise AirflowException(f"API did not become healthy: {last_error}")

    response = requests.post(f"{API_SMOKE_URL}/predict", json=SAMPLE_REQUEST, timeout=10)
    response.raise_for_status()
    result = response.json()
    log.info("smoke test prediction: %s", json.dumps(result, indent=2))
    if "species" not in result:
        raise AirflowException(f"unexpected API response: {result}")
    return result


default_args = {
    "owner": "pmldl",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="pmldl_pipeline",
    description="Data engineering -> model engineering -> Docker deployment",
    default_args=default_args,
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    schedule="*/5 * * * *",
    catchup=False,
    max_active_runs=1,
    # Start scheduling as soon as the stack comes up - no manual toggle needed.
    is_paused_upon_creation=False,
    dagrun_timeout=timedelta(minutes=20),
    tags=["pmldl", "mlops", "assignment1"],
) as dag:
    data_engineering = PythonOperator(
        task_id="data_engineering",
        python_callable=stage_data_engineering,
    )

    model_engineering = PythonOperator(
        task_id="model_engineering",
        python_callable=stage_model_engineering,
    )

    deployment = BashOperator(
        task_id="deployment",
        bash_command=(
            "set -euo pipefail\n"
            f"cd {PROJECT_ROOT / 'code' / 'deployment'}\n"
            f"docker compose -p {COMPOSE_PROJECT} -f {COMPOSE_FILE} "
            "up -d --build --remove-orphans\n"
            f"docker compose -p {COMPOSE_PROJECT} -f {COMPOSE_FILE} ps\n"
        ),
    )

    smoke_test = PythonOperator(
        task_id="smoke_test",
        python_callable=stage_smoke_test,
    )

    data_engineering >> model_engineering >> deployment >> smoke_test
