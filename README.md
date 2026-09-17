# PMLDL Assignment 1 — Deployment

An automated MLOps pipeline that runs **every 5 minutes** in Airflow and covers all
three required stages: data engineering → model engineering → deployment of a model
API and a web app in **separate Docker containers**.

**Task:** predict the species of a penguin (Adelie / Chinstrap / Gentoo) from its
body measurements — [Palmer Penguins](https://github.com/mwaskom/seaborn-data/blob/master/penguins.csv),
344 rows, 6 features. Current test accuracy: **0.986** (macro F1 0.983).

| Service | URL | Notes |
| --- | --- | --- |
| Streamlit app | http://localhost:8501 | input fields + *Predict* button + prediction |
| Model API (FastAPI) | http://localhost:8000/docs | `/health`, `/metadata`, `/predict` |
| Airflow UI | http://localhost:8080 | login `admin` / `admin` |
| MLflow UI | http://localhost:5000 | experiment `pmldl-penguins` |

## Repository structure

```
├── code
│   ├── datasets                # raw data download helper
│   ├── deployment
│   │   ├── api                 # FastAPI service + Dockerfile
│   │   ├── app                 # Streamlit app + Dockerfile
│   │   └── docker-compose.yml  # API + app, two separate containers
│   └── models                  # feature engineering + training/evaluation
├── data
│   ├── raw                     # penguins.csv (input artifact)
│   └── processed               # train.csv / test.csv (stage 1 output)
├── notebooks                   # EDA
├── models                      # model.pkl, metrics.json, model_meta.json
├── services
│   └── airflow
│       ├── dags                # pipeline DAG + data engineering code
│       ├── logs                # task logs
│       ├── Dockerfile          # Airflow + docker CLI + pipeline libs
│       └── docker-compose.yml  # Airflow + MLflow tracking server
└── requirements.txt
```

## How to run

Requirements: Docker Desktop (running), ~10 GB of free disk and ~4 GB free RAM.
Nothing else is needed — Python, Airflow, MLflow and both deployment containers all
run in Docker.

```bash
git clone https://github.com/lohmo111/airflow_assignment.git
cd airflow_assignment

# Start the orchestrator (Airflow) and the MLflow tracking server.
docker compose -f services/airflow/docker-compose.yml up -d --build
```

The first build takes a few minutes. The **`pmldl_pipeline`** DAG is created
*unpaused* and scheduled at `*/5 * * * *`, so the first run starts by itself within
five minutes — open http://localhost:8080 (`admin` / `admin`) to watch it, or press ▶
to trigger it immediately. The first run also builds the two deployment images, so it
takes longer than the following ones.

After the first successful run:

* the app is at **http://localhost:8501** — fill in the measurements, press
  **Predict species**, and the prediction returned by the API appears below;
* the API is at **http://localhost:8000/docs**;
* the run, its parameters and the test metrics are in MLflow at **http://localhost:5000**;
* `docker ps` shows four containers: `pmldl-airflow`, `pmldl-mlflow` and the two
  deployment containers `pmldl-api` and `pmldl-app`.

Stop everything:

```bash
docker compose -f services/airflow/docker-compose.yml down                  # orchestration
docker compose -p pmldl-deploy -f code/deployment/docker-compose.yml down   # API + app
```

## The pipeline

`services/airflow/dags/pmldl_pipeline.py` — four tasks, run in order, every 5 minutes
(`max_active_runs=1`, `catchup=False`, so a slow run is never overlapped by the next one):

1. **`data_engineering`** (`services/airflow/dags/data_engineering.py`)
   Loads `data/raw/penguins.csv` (downloading it if absent), drops unlabelled rows and
   duplicates, normalises the `sex` column, imputes missing numeric values with the
   median and categorical values with the mode, removes outliers with an IQR rule, and
   writes a stratified 80/20 split to `data/processed/train.csv` and `test.csv`.

2. **`model_engineering`** (`code/models/train.py`, `code/models/features.py`)
   Adds the derived features `bill_ratio` and `mass_per_flipper`, scales the numeric
   columns and one-hot encodes the categorical ones inside a scikit-learn `Pipeline`,
   trains a `RandomForestClassifier`, evaluates it on the test split
   (accuracy, macro F1/precision/recall), logs parameters, metrics and the model to
   **MLflow**, and packages the fitted pipeline into `models/model.pkl` together with
   `models/metrics.json` and `models/model_meta.json`.

3. **`deployment`**
   Runs `docker compose -p pmldl-deploy -f code/deployment/docker-compose.yml up -d --build`
   on the host through the mounted Docker socket. The API image copies the freshly
   trained `models/model.pkl`, so every run redeploys the newest model into two
   containers: `pmldl-api` (FastAPI, port 8000) and `pmldl-app` (Streamlit, port 8501).

4. **`smoke_test`**
   Waits for `/health` and sends a real request to `/predict`, so a run is only green
   when the deployed API actually answers.

## API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"bill_length_mm": 50.0, "bill_depth_mm": 15.0, "flipper_length_mm": 220,
       "body_mass_g": 5500, "island": "Biscoe", "sex": "MALE"}'
```

```json
{"species": "Gentoo", "confidence": 0.9998,
 "probabilities": {"Adelie": 0.0002, "Chinstrap": 0.0, "Gentoo": 0.9998}}
```

`GET /metadata` returns the feature schema, the training-set categories and the test
metrics; the Streamlit app uses it to build its form and its sidebar.

## Running the stages without Airflow

Useful for debugging a single stage:

```bash
python -m venv .venv && . .venv/Scripts/activate   # Linux/macOS: . .venv/bin/activate
pip install -r requirements.txt

python code/datasets/download_data.py              # fetch raw data
python services/airflow/dags/data_engineering.py   # stage 1
python code/models/train.py                        # stage 2 (MLflow -> ./mlflow.db)
docker compose -p pmldl-deploy -f code/deployment/docker-compose.yml up -d --build  # stage 3
```

## Notes

* The Airflow container runs as `root` and mounts `/var/run/docker.sock` so the
  deployment task can drive Docker on the host ("docker-outside-of-docker"). On a Linux
  host where root is not enough, add the `docker` group instead:
  `group_add: ["<docker GID>"]` in `services/airflow/docker-compose.yml`.
* The project root is mounted into the Airflow container at `/opt/project`, so the DAG
  reads and writes the same `data/` and `models/` folders you see in the repository.
* `smoke_test` reaches the published API port through `host.docker.internal`; the
  address can be overridden with the `API_SMOKE_URL` environment variable.
* Ports used: 8080 (Airflow), 5000 (MLflow), 8000 (API), 8501 (app). On macOS port 5000
  is often taken by AirPlay — remap it to `"5001:5000"` if so.
* `scikit-learn`, `pandas`, `numpy` and `joblib` are pinned to the same versions in
  `services/airflow/requirements.txt` and `code/deployment/api/requirements.txt`, because
  the model is trained in one container and unpickled in the other — change them together.
* Airflow state lives in named volumes (`airflow-home`, `mlflow-data`), so DAG history
  and MLflow runs survive a restart.
* `models/model.pkl` is committed (380 KB) so that stage 3 can also be run on its
  own; every pipeline run overwrites it.
