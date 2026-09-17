"""Stage 3a - FastAPI service wrapping the trained model.

The model file is baked into the image at build time (see Dockerfile), so every
pipeline run that rebuilds the image ships the freshly trained model.
"""
from __future__ import annotations

import json
import logging
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("api")

APP_DIR = Path(__file__).resolve().parent
MODEL_FILE = APP_DIR / "artifacts" / "model.pkl"
META_FILE = APP_DIR / "artifacts" / "model_meta.json"

model = None
meta: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the baked-in model once, before the first request is served."""
    global model, meta
    with MODEL_FILE.open("rb") as fh:
        model = pickle.load(fh)
    if META_FILE.exists():
        meta = json.loads(META_FILE.read_text(encoding="utf-8"))
    log.info("model loaded from %s (trained_at=%s)", MODEL_FILE, meta.get("trained_at"))
    yield


app = FastAPI(
    title="Penguin Species API",
    description="Predicts the species of a penguin from its measurements.",
    version="1.0.0",
    lifespan=lifespan,
)


class PenguinFeatures(BaseModel):
    bill_length_mm: float = Field(..., gt=0, le=100, examples=[44.5])
    bill_depth_mm: float = Field(..., gt=0, le=50, examples=[17.3])
    flipper_length_mm: float = Field(..., gt=0, le=350, examples=[197.0])
    body_mass_g: float = Field(..., gt=0, le=10000, examples=[4200.0])
    island: Literal["Biscoe", "Dream", "Torgersen"] = "Biscoe"
    sex: Literal["MALE", "FEMALE"] = "MALE"


class Prediction(BaseModel):
    species: str
    confidence: float
    probabilities: dict[str, float]


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if model is not None else "loading"}


@app.get("/metadata")
def metadata() -> dict:
    """Schema and training info the Streamlit app uses to build its form."""
    if not meta:
        raise HTTPException(status_code=503, detail="metadata not available")
    return meta


@app.post("/predict", response_model=Prediction)
def predict(features: PenguinFeatures) -> Prediction:
    if model is None:
        raise HTTPException(status_code=503, detail="model is not loaded yet")
    row = pd.DataFrame([features.model_dump()])
    try:
        proba = model.predict_proba(row)[0]
    except Exception as exc:  # noqa: BLE001
        log.exception("prediction failed")
        raise HTTPException(status_code=400, detail=f"prediction failed: {exc}") from exc
    classes = list(model.classes_)
    probabilities = {cls: float(p) for cls, p in zip(classes, proba)}
    species = max(probabilities, key=probabilities.get)
    return Prediction(
        species=species,
        confidence=probabilities[species],
        probabilities=probabilities,
    )
