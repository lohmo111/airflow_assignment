"""Stage 3b - Streamlit app that talks to the model API."""
from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = 10

SPECIES_EMOJI = {"Adelie": "🐧", "Chinstrap": "🎩", "Gentoo": "🐤"}

DEFAULTS = {
    "bill_length_mm": 44.5,
    "bill_depth_mm": 17.3,
    "flipper_length_mm": 197.0,
    "body_mass_g": 4200.0,
}

st.set_page_config(page_title="Penguin Species Predictor", page_icon="🐧")


@st.cache_data(ttl=30)
def fetch_metadata() -> dict:
    response = requests.get(f"{API_URL}/metadata", timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


st.title("🐧 Penguin Species Predictor")
st.caption(f"Predictions come from the model API at `{API_URL}`.")

try:
    meta = fetch_metadata()
except Exception as exc:  # noqa: BLE001
    meta = {}
    st.warning(f"Could not reach the API metadata endpoint ({exc}). Using defaults.")

islands = meta.get("categories", {}).get("island", ["Biscoe", "Dream", "Torgersen"])
sexes = meta.get("categories", {}).get("sex", ["MALE", "FEMALE"])

with st.sidebar:
    st.subheader("Model")
    if meta:
        st.write(f"**Type:** {meta.get('model_type', 'n/a')}")
        st.write(f"**Trained at:** {meta.get('trained_at', 'n/a')}")
        metrics = meta.get("metrics", {})
        if metrics:
            st.metric("Test accuracy", f"{metrics.get('accuracy', 0):.3f}")
            st.metric("Test F1 (macro)", f"{metrics.get('f1_macro', 0):.3f}")
        st.write(f"**Train / test rows:** {meta.get('train_rows')} / {meta.get('test_rows')}")
    else:
        st.write("Metadata unavailable.")

st.subheader("Measurements")
col1, col2 = st.columns(2)
with col1:
    bill_length = st.number_input(
        "Bill length (mm)", min_value=20.0, max_value=80.0,
        value=DEFAULTS["bill_length_mm"], step=0.1,
    )
    flipper_length = st.number_input(
        "Flipper length (mm)", min_value=150.0, max_value=260.0,
        value=DEFAULTS["flipper_length_mm"], step=1.0,
    )
    island = st.selectbox("Island", islands)
with col2:
    bill_depth = st.number_input(
        "Bill depth (mm)", min_value=10.0, max_value=25.0,
        value=DEFAULTS["bill_depth_mm"], step=0.1,
    )
    body_mass = st.number_input(
        "Body mass (g)", min_value=2000.0, max_value=7000.0,
        value=DEFAULTS["body_mass_g"], step=25.0,
    )
    sex = st.selectbox("Sex", sexes)

if st.button("Predict species", type="primary", use_container_width=True):
    payload = {
        "bill_length_mm": bill_length,
        "bill_depth_mm": bill_depth,
        "flipper_length_mm": flipper_length,
        "body_mass_g": body_mass,
        "island": island,
        "sex": sex,
    }
    try:
        response = requests.post(
            f"{API_URL}/predict", json=payload, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        result = response.json()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Request to the API failed: {exc}")
    else:
        species = result["species"]
        st.success(
            f"## {SPECIES_EMOJI.get(species, '🐧')} {species}\n"
            f"confidence: **{result['confidence']:.1%}**"
        )
        probs = pd.DataFrame(
            sorted(result["probabilities"].items(), key=lambda kv: -kv[1]),
            columns=["species", "probability"],
        ).set_index("species")
        st.bar_chart(probs)
        with st.expander("Request sent to the API"):
            st.json(payload)
